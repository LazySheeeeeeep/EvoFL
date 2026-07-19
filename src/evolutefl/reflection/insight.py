from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.skills.schema import SKILL_TYPES


def build_insight(
    *,
    case_input: dict[str, Any],
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(case_input, ensure_ascii=False, indent=2)},
    ]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response = llm_client.chat(messages=messages, tool_choice="none")
        content = response.get("content") or ""
        try:
            insight = validate_insight(extract_json_object(content))
            if output_path:
                write_json(output_path, {"attempt": attempt, "insight": insight, "raw_response": content})
            return insight
        except Exception as exc:  # noqa: BLE001 - fixed-input retry records last parse error.
            last_error = exc
    raise RuntimeError(f"Insight failed after {attempts} fixed-input attempts: {last_error}") from last_error


def validate_insight(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("instance_id"):
        raise ValueError("Insight missing instance_id.")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Insight missing analysis object.")
    outcome_type = analysis.get("outcome_type", "failure")
    if outcome_type not in ("success", "failure"):
        raise ValueError(f"Invalid outcome_type: {outcome_type!r}")
    skill_type = analysis.get("missing_or_reinforced_skill_type")
    if skill_type not in SKILL_TYPES:
        raise ValueError(f"Invalid missing_or_reinforced_skill_type: {skill_type!r}")
    analysis["outcome_type"] = outcome_type
    analysis["missing_or_reinforced_skill_type"] = skill_type
    for key in ("miss_or_success_reason", "patch_lesson", "target_value_hint", "transferable_lesson"):
        analysis.setdefault(key, "")
    if "miss_reason" in analysis and not analysis["miss_or_success_reason"]:
        analysis["miss_or_success_reason"] = analysis["miss_reason"]
    evidence = analysis.get("evidence")
    if not isinstance(evidence, list):
        analysis["evidence"] = [str(evidence)] if evidence else []
    return payload
