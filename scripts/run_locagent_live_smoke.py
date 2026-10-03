"""Run one audited LocAgent acquisition case with the official DeepSeek API."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
from queue import Queue
import re
import signal
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924"
AUTHOR = ROOT / "runs/external_baseline_sources/LocAgent"
MODEL = "openai/deepseek-v4-flash"
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")


def load_public_case(case_id: str, source: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("Invalid case ID")
    if (source / "evaluation_cases.json").is_file():
        cases = json.loads((source / "evaluation_cases.json").read_text(encoding="utf-8"))
        matches = [case for case in cases if case.get("instance_id") == case_id]
        if len(matches) != 1:
            raise ValueError("Frozen evaluation case not uniquely found")
        public = {field: matches[0][field] for field in PUBLIC_FIELDS}
    else:
        public = json.loads((BASE / "cosil_acquisition_smoke/case.jsonl").read_text(encoding="utf-8"))
        if public.get("instance_id") != case_id:
            raise ValueError("Acquisition smoke case ID mismatch")
    if set(public) != set(PUBLIC_FIELDS) or not public["problem_statement"].strip():
        raise ValueError("Only nonempty public localization inputs are accepted")
    return public


def run(case_id: str, source: Path, graph_dir: Path, output: Path) -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY must be supplied to this process")
    public = load_public_case(case_id, source)
    graph_dir = graph_dir.resolve()
    output = output.resolve()
    if not (graph_dir / f"{public['instance_id']}.pkl").is_file():
        raise FileNotFoundError("Author graph must be built before the live smoke")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPH_INDEX_DIR"] = str(graph_dir)
    os.environ["BM25_INDEX_DIR"] = str(output / "bm25")
    sys.path.insert(0, str(AUTHOR))

    # The author's unused metrics module requires Python 3.12 syntax; WSL runs 3.10.
    metrics_stub = ModuleType("evaluation.eval_metric")
    metrics_stub.filtered_instances = []
    sys.modules["evaluation.eval_metric"] = metrics_stub
    try:
        import auto_search_main as author
    finally:
        sys.modules.pop("evaluation.eval_metric", None)
    import litellm
    from plugins.location_tools.repo_ops import repo_ops
    from util.prompts.pipelines import auto_search_prompt
    from util.runtime import function_calling

    original_completion = litellm.completion
    model_calls = 0

    def completion(**kwargs):
        nonlocal model_calls
        model_calls += 1
        if model_calls > 14:
            raise RuntimeError("LocAgent live smoke exceeded 14 model calls")
        return original_completion(
            **{**kwargs, "model": MODEL, "api_base": "https://api.deepseek.com/v1",
               "api_key": key, "timeout": 75, "max_tokens": 4096}
        )

    def time_limit(_signum, _frame):
        raise TimeoutError("LocAgent live smoke exceeded 600 seconds")

    previous = Path.cwd()
    os.chdir(output)
    old_handler = signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(600)
    try:
        repo_ops.set_current_issue(instance_data=public)
        messages = [
            {"role": "system", "content": function_calling.SYSTEM_PROMPT},
            {"role": "user", "content": author.get_task_instruction(public, include_pr=True, include_hint=True)},
        ]
        result = Queue()
        tools = function_calling.get_tools(
            codeact_enable_search_keyword=True,
            codeact_enable_search_entity=True,
            codeact_enable_tree_structure_traverser=True,
            simple_desc=True,
        )
        with (output / "author_loop.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), patch.object(litellm, "completion", completion):
                # This alias bypasses the author's text-to-tool conversion for
                # DeepSeek; the actual request above still uses DeepSeek native tools.
                author.auto_search_process(
                    result, "openai/gpt-4o-2024-05-13", messages,
                    auto_search_prompt.FAKE_USER_MSG_FOR_LOC,
                    tools=tools, max_iteration_num=12,
                )
        final, _, trajectory = result.get_nowait()
        observations = [m for m in trajectory["messages"] if m.get("role") == "tool"]
        has_location = bool(re.search(r"\b[\w./-]+\.py\b", final))
        (output / "author_trajectory.json").write_text(
            json.dumps(trajectory, indent=2, default=str) + "\n", encoding="utf-8"
        )
        report = {
            "status": "completed" if has_location else "empty_prediction",
            "instance_id": public["instance_id"],
            "model": MODEL, "model_calls": model_calls,
            "tool_observations": len(observations), "final_output": final,
            "observation_errors": sum("Traceback (most recent call last):" in m.get("content", "")
                                      for m in observations),
            "graph_source": str(graph_dir / f"{public['instance_id']}.pkl"),
            "note": ("FL-only live smoke on frozen evaluation case; not a full 500-case baseline."
                     if (source / "evaluation_cases.json").is_file() else
                     "FL-only live smoke on acquisition case; not a frozen-test score."),
        }
        (output / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="psf__requests-1376")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/rq1_temporal_deepseek_20260915")
    parser.add_argument("--graph-dir", type=Path, default=BASE / "locagent_acquisition_graph")
    parser.add_argument("--output", type=Path, default=BASE / "locagent_live_smoke_author_prompt")
    args = parser.parse_args()
    print(json.dumps(run(args.case_id, args.source, args.graph_dir, args.output), indent=2))
