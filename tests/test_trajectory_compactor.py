from __future__ import annotations

import json

from evolutefl.reflection.trajectory_compactor import build_compacted_trajectory


def _tool_call(step: int, name: str, arguments: str = "{}") -> dict:
    return {
        "event": "assistant_tool_calls",
        "step": step,
        "tool_calls": [
            {
                "id": f"call_{step}",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_compactor_preserves_head_and_tail_with_middle_marker() -> None:
    trajectory = [{"event": "skill_context", "step": 0}]
    trajectory.extend(_tool_call(step, "grep") for step in range(1, 30))
    trajectory.append({"event": "finish", "step": 30, "result": {"action": {"ranked_functions": ["x::y"]}}})

    compacted = build_compacted_trajectory(trajectory, head_events=3, tail_events=4)

    assert compacted["trajectory_compaction"]["strategy"] == "field_aware_head_tail_v1"
    assert compacted["trajectory_compaction"]["omitted_middle_event_count"] > 0
    assert compacted["trajectory_summary"][0]["event"] == "skill_context"
    assert any(event["event"] == "middle_truncated" for event in compacted["trajectory_summary"])
    assert compacted["trajectory_summary"][-1]["event"] == "finish"
    assert "raw_trajectory" not in compacted


def test_compactor_excludes_embedding_trace_from_skill_context() -> None:
    trajectory = [
        {
            "event": "skill_context",
            "issue_abstraction": {"abstract_problem_signature": "Configured value is ignored."},
            "skill_search_trace": {"embedding": [0.1] * 10000},
            "matched_skill_ids": ["project_config_v1"],
            "assembled_context": {"sections": {"project_skill": []}},
        }
    ]

    compacted = build_compacted_trajectory(trajectory)
    details = compacted["trajectory_summary"][0]["details"]

    assert details["matched_skill_ids"] == ["project_config_v1"]
    assert "skill_search_trace" not in details


def test_compactor_clips_tool_observation_with_head_and_tail() -> None:
    long_content = "HEAD-" + ("x" * 2000) + "-TAIL"
    trajectory = [
        _tool_call(1, "read_file", "{\"file_path\":\"a.py\"}"),
        {"event": "tool_result", "step": 1, "name": "read_file", "tool_call_id": "call_1", "content": long_content},
    ]

    compacted = build_compacted_trajectory(trajectory)
    preview = compacted["trajectory_summary"][1]["content_preview"]

    assert preview.startswith("HEAD-")
    assert preview.endswith("-TAIL")
    assert "[middle truncated:" in preview


def test_compactor_counts_tools_and_errors() -> None:
    error_content = json.dumps({"ok": False, "error": "Tool grep failed"})
    trajectory = [
        _tool_call(1, "grep"),
        {"event": "tool_result", "step": 1, "name": "grep", "content": error_content},
        _tool_call(2, "read_file"),
        {"event": "tool_result", "step": 2, "name": "read_file", "content": json.dumps({"ok": True})},
    ]

    stats = build_compacted_trajectory(trajectory)["tool_call_statistics"]

    assert stats["total_tool_calls"] == 2
    assert stats["by_tool"] == {"grep": 1, "read_file": 1}
    assert stats["errors_by_tool"] == {"grep": 1}
