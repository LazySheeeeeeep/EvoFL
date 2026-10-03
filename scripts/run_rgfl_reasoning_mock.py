"""Run all four author reasoning/ranking stages on real source with a fake model."""
from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from prepare_rgfl_reasoning import AUTHOR, BASE, source_files
from run_rgfl_mock_smoke import INPUT, load_public_case


def run() -> dict:
    case = load_public_case(INPUT)
    output = BASE / "rgfl_pilot/reasoning_mock"
    output.mkdir(exist_ok=True)
    structure = json.loads((INPUT / "repo_structures" / f"{case['instance_id']}.json").read_text())
    files = source_files(structure["structure"])
    found = json.loads((BASE / "rgfl_pilot/live_file_smoke_path_adapter/loc_outputs.jsonl").read_text())
    selected = []
    for path in found["found_files"]:
        matches = [candidate for candidate in files if candidate == path or candidate.endswith("/" + path)]
        if len(matches) != 1:
            raise ValueError("Ambiguous file candidate")
        selected.append(matches[0])
    combined = output / "combined_locs.jsonl"
    combined.write_text(json.dumps({"instance_id": case["instance_id"], "found_files": selected}) + "\n")
    calls = []
    def create(**kwargs):
        text = kwargs["messages"][0]["content"]
        calls.append({"stage": current_stage, "input_chars": len(text)})
        if current_stage in ("file_ranking", "element_ranking"):
            data = json.loads(text.split("\n\n")[1])
            content = json.dumps(list(data))
        else:
            content = "Offline mock: code inspected; no localization inference performed."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def dataset(name, **kwargs):
        if name != "frozen-public-input":
            raise ValueError("Unexpected dataset lookup")
        return [case]
    stages = [
        ("file_reasoning", "--combined_locs", combined),
        ("file_ranking", "--reasoning_file", output / "file_reasoning.json"),
        ("element_reasoning", "--input", output / "file_ranking.json"),
        ("element_ranking", "--input", output / "element_reasoning.json"),
    ]
    with tempfile.TemporaryDirectory(prefix="evolutefl-rgfl-") as temporary:
        root = Path(temporary)
        for name, source in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source path")
            path = root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        def repo_for_case(repo, commit):
            if (repo, commit) != (case["repo"], case["base_commit"]):
                raise ValueError("Repository revision mismatch")
            return str(root)
        for current_stage, input_flag, input_path in stages:
            source_path = AUTHOR / f"{current_stage}.py"
            tree = ast.parse(source_path.read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "clone_and_checkout":
                    node.body = ast.parse("return frozen_repo(repo_name, commit)").body
                if isinstance(node, ast.FunctionDef) and node.name == "extract_file_list":
                    node.body = ast.parse("return safe_file_list(llm_output)").body
            def safe_file_list(text):
                try:
                    value = ast.literal_eval(text)
                    if isinstance(value, list) and all(isinstance(item, str) for item in value):
                        return value
                except (SyntaxError, ValueError):
                    pass
                return re.findall(r"[\w\-/]+\.py", text)
            arguments = [str(source_path), "--model", "deepseek-v4-flash", "--backend", "openai",
                         "--dataset", "frozen-public-input", input_flag, str(input_path),
                         "--output", str(output / f"{current_stage}.json"), "--target_id", case["instance_id"]]
            with patch.object(sys, "argv", arguments), patch.dict(sys.modules, {
                    "datasets": SimpleNamespace(load_dataset=dataset),
                    "openai": SimpleNamespace(OpenAI=lambda: client)}):
                exec(compile(ast.fix_missing_locations(tree), str(source_path), "exec"),
                     {"__name__": "__main__", "frozen_repo": repo_for_case, "safe_file_list": safe_file_list})
    from collections import Counter
    result = json.loads((output / "element_ranking.json").read_text())[0]
    if not result.get("similar_elements_file1"):
        raise ValueError("No mock ranked elements")
    report = {"status": "completed", "instance_id": case["instance_id"], "model_calls": 0,
              "mock_calls_by_stage": dict(Counter(call["stage"] for call in calls)),
              "total_mock_calls": len(calls), "max_request_chars": max(call["input_chars"] for call in calls),
              "adapters": ["frozen public dataset", "audited local base source", "literal-only list parser"],
              "note": "All four author stages executed. Fake predictions are not scored; full RGFL still needs retrieval/merge, live inference, and function normalization."}
    (output / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
