from __future__ import annotations

from collections import Counter
from typing import Any


FIELD_LIMITS = {
    "assistant_content": 900,
    "tool_arguments": 600,
    "tool_result": 1200,
    "tool_result_head": 700,
    "tool_result_tail": 400,
    "details": 900,
    "error": 700,
    "finish": 1400,
}


def build_compacted_trajectory(
    trajectory: list[dict[str, Any]],
    *,
    head_events: int = 8,
    tail_events: int = 16,
) -> dict[str, Any]:
    """Create a SkillOpt-inspired evidence packet from raw Explorer events.

    The compactor is intentionally deterministic: it selects and clips evidence,
    but never rewrites it into new conclusions.
    """
    selected, compaction = _select_head_tail(trajectory, head_events=head_events, tail_events=tail_events)
    summary = [_compact_event(event) for event in selected if isinstance(event, dict)]
    return {
        "trajectory_summary": summary,
        "trajectory_timeline": [_timeline_event(event) for event in summary],
        "tool_call_statistics": _tool_call_statistics(trajectory),
        "trajectory_compaction": compaction,
    }


def _select_head_tail(
    trajectory: list[dict[str, Any]],
    *,
    head_events: int,
    tail_events: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = len(trajectory)
    stage_events = {
        "project_skill_request",
        "project_skill_loaded",
        "issue_revealed",
        "fault_skill_request",
        "fault_skill_loaded",
        "strategy_skill_request",
        "strategy_skill_loaded",
    }
    selected_indices = set(range(min(head_events, total)))
    selected_indices.update(range(max(0, total - tail_events), total))
    stage_indices: list[int] = []
    for index, event in enumerate(trajectory):
        if isinstance(event, dict) and event.get("event") in stage_events:
            stage_indices.append(index)
            # Preserve the local repository observation/control exchange around
            # each stage transition without summarizing it with another model.
            selected_indices.update(range(max(0, index - 2), min(total, index + 3)))

    ordered_indices = sorted(selected_indices)
    selected: list[dict[str, Any]] = []
    previous = -1
    for index in ordered_indices:
        gap = index - previous - 1
        if gap > 0 and previous >= 0:
            selected.append({"event": "middle_truncated", "omitted_event_count": gap})
        selected.append(trajectory[index])
        previous = index
    omitted = total - len(ordered_indices)
    return selected, {
        "strategy": "stage_aware_field_clipping_v2",
        "raw_event_count": total,
        "selected_event_count": len(selected),
        "head_events": min(head_events, total),
        "tail_events": min(tail_events, max(0, total - min(head_events, total))),
        "omitted_middle_event_count": omitted,
        "preserved_stage_event_indices": stage_indices,
        "field_limits": FIELD_LIMITS,
        "notes": [
            "Head events preserve initial repository exploration.",
            "Project, Fault, and Strategy requests, loads, and nearby observations are always retained.",
            "Tail events preserve final ranking, repair, forced finish, or failure behavior.",
            "Tool observations are clipped by field budget; the compactor does not infer new knowledge.",
        ],
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = event.get("event", "unknown")
    entry: dict[str, Any] = {"event": kind}
    if "step" in event:
        entry["step"] = event.get("step")
    if kind == "middle_truncated":
        entry["omitted_event_count"] = event.get("omitted_event_count", 0)
    elif kind == "assistant_tool_calls":
        entry["tool_calls"] = [_compact_tool_call(call) for call in event.get("tool_calls", []) or []]
        content = event.get("content")
        if content:
            entry["content_preview"] = _clip_text(content, FIELD_LIMITS["assistant_content"])
    elif kind == "tool_result":
        entry["name"] = event.get("name")
        entry["tool_call_id"] = event.get("tool_call_id")
        entry["content_preview"] = _clip_observation(event.get("content", ""))
    elif kind == "finish":
        entry["details"] = _clip_mapping({key: value for key, value in event.items() if key != "event"}, FIELD_LIMITS["finish"])
    elif kind in (
        "project_skill_request",
        "project_skill_loaded",
        "issue_revealed",
        "fault_skill_request",
        "fault_skill_loaded",
        "strategy_skill_request",
        "strategy_skill_loaded",
    ):
        entry["details"] = _clip_mapping(
            {key: value for key, value in event.items() if key not in {"event", "step"}},
            FIELD_LIMITS["details"],
        )
    elif kind in ("forced_finish", "response_repair", "stage_finalization"):
        entry["details"] = _clip_mapping({key: value for key, value in event.items() if key != "event"}, FIELD_LIMITS["details"])
    else:
        entry["details"] = _clip_mapping({key: value for key, value in event.items() if key != "event"}, FIELD_LIMITS["details"])
    return entry


def _compact_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "id": call.get("id"),
        "name": function.get("name"),
        "arguments": _clip_text(function.get("arguments", ""), FIELD_LIMITS["tool_arguments"]),
    }


def _tool_call_statistics(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    result_count = 0
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        if event.get("event") == "assistant_tool_calls":
            for call in event.get("tool_calls", []) or []:
                function = call.get("function") or {}
                name = function.get("name") or "unknown"
                calls[name] += 1
        elif event.get("event") == "tool_result":
            result_count += 1
            name = event.get("name") or "unknown"
            content = str(event.get("content", ""))
            if '"ok": false' in content.lower() or '"ok":false' in content.lower() or '"error"' in content.lower():
                errors[name] += 1
    return {
        "total_tool_calls": sum(calls.values()),
        "total_tool_results": result_count,
        "by_tool": dict(calls),
        "errors_by_tool": dict(errors),
    }


def _timeline_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = event.get("event", "unknown")
    item: dict[str, Any] = {"event": kind}
    if "step" in event:
        item["step"] = event.get("step")
    if kind == "assistant_tool_calls":
        item["tools"] = [call.get("name") for call in event.get("tool_calls", []) or [] if call.get("name")]
    elif kind == "tool_result":
        item["tool"] = event.get("name")
    elif kind == "middle_truncated":
        item["omitted_event_count"] = event.get("omitted_event_count", 0)
    return item


def _clip_observation(value: Any) -> str:
    text = str(value)
    limit = FIELD_LIMITS["tool_result"]
    if len(text) <= limit:
        return text
    head = FIELD_LIMITS["tool_result_head"]
    tail = FIELD_LIMITS["tool_result_tail"]
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[middle truncated: {omitted} chars]...\n{text[-tail:]}"


def _clip_mapping(mapping: dict[str, Any], limit: int) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, str):
            compact[key] = _clip_text(value, limit)
        elif isinstance(value, dict):
            compact[key] = _clip_mapping(value, limit)
        elif isinstance(value, list):
            compact[key] = [_clip_mapping(item, limit) if isinstance(item, dict) else _clip_text(item, limit) for item in value]
        else:
            compact[key] = value
    return compact


def _clip_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated: {omitted} chars]..."
