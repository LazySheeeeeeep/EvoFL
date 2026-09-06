from __future__ import annotations

from evolutefl.reflection.trajectory_compactor import build_compacted_trajectory


def test_compactor_preserves_middle_stage_events_and_neighbors() -> None:
    trajectory = [{"event": "tool_result", "step": index, "name": "grep", "content": str(index)} for index in range(30)]
    trajectory[12] = {"event": "project_skill_request", "step": 12, "query": "boundary"}
    trajectory[13] = {"event": "project_skill_loaded", "step": 12, "loaded_skill_id": "p"}
    trajectory[16] = {"event": "fault_skill_request", "step": 16, "request": {"fault_family": "state_assignment"}}
    trajectory[17] = {"event": "fault_skill_loaded", "step": 16, "loaded_skill_id": "f"}
    trajectory[20] = {"event": "strategy_skill_request", "step": 20, "query": "contrast"}
    trajectory[21] = {"event": "strategy_skill_loaded", "step": 20, "loaded_skill_id": "s"}
    compacted = build_compacted_trajectory(trajectory, head_events=2, tail_events=2)
    events = [item["event"] for item in compacted["trajectory_summary"]]
    assert compacted["trajectory_compaction"]["strategy"] == "stage_aware_field_clipping_v2"
    for event in (
        "project_skill_request", "project_skill_loaded",
        "fault_skill_request", "fault_skill_loaded",
        "strategy_skill_request", "strategy_skill_loaded",
    ):
        assert event in events


def test_compactor_counts_control_tools() -> None:
    trajectory = [{"event": "assistant_tool_calls", "step": 1, "tool_calls": [{"function": {"name": "load_project_skill", "arguments": "{}"}}]}]
    assert build_compacted_trajectory(trajectory)["tool_call_statistics"]["by_tool"]["load_project_skill"] == 1
