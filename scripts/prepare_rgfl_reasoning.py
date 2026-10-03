"""Audit RGFL's remaining author stages and generate exact file reasoning requests."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "runs/external_baseline_sources/RGFL/rgfl/fl"
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924"


def author_function(module: str, name: str):
    source = AUTHOR / f"{module}.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(definitions) != 1:
        raise ValueError("Expected one author function")
    namespace = {"backend": "openai", "MODEL_ID": "deepseek-v4-flash"}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def exact_prompt(module: str, *args) -> list[dict]:
    captured = []
    def create(**kwargs):
        captured.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="offline"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=lambda: client)}):
        author_function(module, "get_reasoning")(*args)
    return captured[0]


def source_files(structure: dict, prefix: str = "") -> dict[str, str]:
    result = {}
    for name, node in structure.items():
        path = f"{prefix}/{name}".lstrip("/")
        if isinstance(node, dict) and "text" in node and name.endswith(".py"):
            result[path] = "\n".join(node["text"]) if isinstance(node["text"], list) else node["text"]
        elif isinstance(node, dict):
            result.update(source_files(node, path))
    return result


def elements(source: str) -> list[dict]:
    tree = ast.parse(source)
    rows = []
    # Match the author's sync-function/class/global traversal for cost accounting.
    globals_ = {id(node) for node in tree.body if isinstance(node, ast.Assign)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            rows.append({"kind": "function" if isinstance(node, ast.FunctionDef) else "class",
                         "name": node.name, "line": node.lineno})
        elif id(node) in globals_:
            rows.extend({"kind": "global", "name": target.id, "line": node.lineno}
                        for target in node.targets if isinstance(target, ast.Name))
    return rows


def run(prepared: Path, file_output: Path, output: Path) -> dict:
    public = json.loads((prepared / "case.jsonl").read_text())
    case_id = public["instance_id"]
    structure_path = prepared / "repo_structures" / f"{case_id}.json"
    structure = json.loads(structure_path.read_text())
    candidates = [json.loads(line) for line in file_output.read_text().splitlines() if line]
    if len(candidates) != 1 or candidates[0]["instance_id"] != case_id:
        raise ValueError("File-stage case mismatch")
    files = source_files(structure["structure"])
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for candidate in candidates[0]["found_files"]:
        matches = [path for path in files if path == candidate or path.endswith("/" + candidate)]
        if len(matches) != 1:
            raise ValueError(f"Non-unique candidate path: {candidate}")
        path = matches[0]
        entries = elements(files[path])
        counts = Counter((entry["kind"], entry["name"]) for entry in entries)
        rows.append({"path": path, "element_calls": len(entries),
                     "ambiguous_author_keys": [f"{kind}: {name}" for (kind, name), count in counts.items() if count > 1],
                     "element_counts": dict(Counter(entry["kind"] for entry in entries)),
                     "source_chars": len(files[path]),
                     "file_reasoning_messages": exact_prompt("file_reasoning", files[path], public["problem_statement"])})
    report = {"instance_id": case_id, "status": "offline_plan_ready", "model_calls": 0,
              "author_files_sha256": {name: hashlib.sha256((AUTHOR / f"{name}.py").read_bytes()).hexdigest()
                                      for name in ("file_reasoning", "file_ranking", "element_reasoning", "element_ranking")},
              "file_reasoning_calls": len(rows), "file_ranking_calls": 1,
              "element_reasoning_calls_before_rerank": sum(row["element_calls"] for row in rows[:3]),
              "element_ranking_calls": min(3, len(rows)),
              "candidates": rows,
              "remaining": ["embedding retrieval and file merge", "live file reasoning/ranking",
                            "live element reasoning/ranking", "strict function output normalization"],
              "note": "Author element keys omit class qualification; collisions require explicit handling. No inference or score produced."}
    (output / "reasoning_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "candidates"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=BASE / "rgfl_pilot/astropy__astropy-13033")
    parser.add_argument("--file-output", type=Path, default=BASE / "rgfl_pilot/live_file_smoke_path_adapter/loc_outputs.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "rgfl_pilot/reasoning_preparation")
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.file_output, args.output), indent=2))
