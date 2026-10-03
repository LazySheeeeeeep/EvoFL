"""Exercise LocAgent's author native-tool loop on an audited graph, without API calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924"
AUTHOR = ROOT / "runs/external_baseline_sources/LocAgent"


def run() -> dict:
    public = json.loads((BASE / "cosil_acquisition_smoke/case.jsonl").read_text())
    if set(public) != {"instance_id", "repo", "base_commit", "problem_statement"}:
        raise ValueError("Only public localization inputs are accepted")
    graph_dir = BASE / "locagent_acquisition_graph"
    output = BASE / "locagent_mock_smoke"
    output.mkdir(exist_ok=True)
    os.environ["GRAPH_INDEX_DIR"] = str(graph_dir)
    os.environ["BM25_INDEX_DIR"] = str(output / "bm25")
    sys.path.insert(0, str(AUTHOR))
    # The author's metrics module uses Python 3.12 f-string syntax but is not
    # involved in the native search loop exercised by this Python 3.10 smoke.
    metrics_stub = ModuleType("evaluation.eval_metric")
    metrics_stub.filtered_instances = []
    sys.modules["evaluation.eval_metric"] = metrics_stub
    try:
        import auto_search_main as author
    finally:
        sys.modules.pop("evaluation.eval_metric", None)
    import litellm
    from util.runtime import function_calling
    from plugins.location_tools.repo_ops import repo_ops

    previous = Path.cwd()
    os.chdir(output)
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            name, arguments = "search_code_snippets", {"search_terms": ["Session"], "file_path_or_pattern": "requests/*.py"}
        elif len(calls) == 2:
            if not any(message.get("role") == "tool" for message in kwargs["messages"]):
                raise ValueError("Author loop did not return a tool observation")
            name, arguments = "finish", {"thought": "requests/sessions.py\nfunction: Session.request"}
        else:
            raise ValueError("Mock loop did not terminate")
        return litellm.ModelResponse(model="mock", choices=[{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": f"Offline protocol step {len(calls)}.", "tool_calls": [
                {"id": f"call_{len(calls)}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    try:
        repo_ops.set_current_issue(instance_data=public)
        messages = [{"role": "system", "content": function_calling.SYSTEM_PROMPT},
                    {"role": "user", "content": author.get_task_instruction(public, include_pr=True, include_hint=True)}]
        result = Queue()
        with patch.object(litellm, "completion", completion):
            author.auto_search_process(result, "openai/gpt-4o-2024-05-13", messages,
                "Complete localization.", tools=function_calling.get_tools(codeact_enable_search_keyword=True, simple_desc=True), max_iteration_num=4)
        final, _, trajectory = result.get_nowait()
        observations = [message["content"] for message in trajectory["messages"] if message.get("role") == "tool"]
        if len(observations) != 1 or any(marker in observations[0] for marker in ("Traceback", "NameError", "Exception:")):
            raise ValueError(f"LocAgent search observation failed: {observations[:1]}")
        report = {"status": "completed", "instance_id": public["instance_id"], "model_calls": 0,
                  "mock_responses": len(calls), "tool_observations": len(observations),
                  "observation_excerpt": observations[0][:500],
                  "final_output": final, "note": "Author loop and graph search only; mock predictions are not scored. Live DeepSeek adaptation remains."}
        (output / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
