from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.skills.schema import SKILL_SCOPES, SKILL_TYPES, slugify


UPDATE_DECISIONS = ("create_new", "rewrite_existing", "preserve_existing", "no_update")


def run_reflector(
    *,
    insight: dict[str, Any] | None = None,
    trajectory_evidence: dict[str, Any] | None = None,
    issue_abstraction: dict[str, Any] | None = None,
    skill_search_context: dict[str, Any],
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if insight is None and trajectory_evidence is None:
        raise ValueError("run_reflector requires insight or trajectory_evidence.")
    payload = {
        "issue_abstraction": issue_abstraction,
        "trajectory_evidence": trajectory_evidence,
        "insight": insight,
        "skill_search_context": skill_search_context,
        "evolution_policy": {
            "success": "Prefer preserve_existing or no_update unless the trajectory adds reusable knowledge.",
            "failure": "Rewrite only a candidate with the same semantic identity; otherwise create a distinct skill.",
            "system_failure": "System-level execution failures normally map to no_update.",
        },
    }
    base_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    messages = list(base_messages)
    last_error: Exception | None = None
    attempts_debug: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        response = llm_client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
        )
        content = response.get("content") or ""
        try:
            reflector_output = validate_reflector_output(
                extract_json_object(content),
                insight,
                trajectory_evidence,
                skill_search_context,
            )
            if output_path:
                write_json(
                    output_path,
                    {"attempt": attempt, "reflector_output": reflector_output, "raw_response": content},
                )
            return reflector_output
        except Exception as exc:  # noqa: BLE001 - retry one fixed schema after protocol failure.
            last_error = exc
            attempts_debug.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = base_messages + [
                {
                    "role": "user",
                    "content": (
                        "The previous JSON failed validation. Return a corrected strict JSON object using the same schema. "
                        f"Validation feedback: {exc}"
                    ),
                }
            ]
    fallback = _no_update_fallback(insight, trajectory_evidence, last_error)
    if output_path:
        write_json(output_path, {"failed": True, "attempts": attempts_debug, "reflector_output": fallback})
    return fallback


def validate_reflector_output(
    payload: dict[str, Any],
    insight: dict[str, Any] | None = None,
    trajectory_evidence: dict[str, Any] | None = None,
    skill_search_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_updates = payload.get("skill_updates")
    if not isinstance(raw_updates, dict):
        raise ValueError("Reflector output missing skill_updates object.")

    analysis = (insight or {}).get("analysis", {})
    outcome = (trajectory_evidence or {}).get("outcome", {})
    outcome_type = str(analysis.get("outcome_type") or outcome.get("label") or "failure")
    context = skill_search_context or {}
    candidate_skills = _candidate_skills_by_id(context)
    project_type = context.get("project_type") if isinstance(context.get("project_type"), dict) else {}
    source_case = (insight or {}).get("instance_id") or (trajectory_evidence or {}).get("case", {}).get("instance_id")

    normalized_updates: dict[str, dict[str, Any]] = {}
    materialized_updates: list[dict[str, Any]] = []
    for skill_type in SKILL_TYPES:
        raw_update = raw_updates.get(skill_type)
        if not isinstance(raw_update, dict):
            raise ValueError(f"skill_updates.{skill_type} must be an object.")
        update, materialized = _validate_skill_update(
            skill_type=skill_type,
            raw_update=raw_update,
            candidate_skills=candidate_skills,
            outcome_type=outcome_type,
            source_case=source_case,
            update_index=len(materialized_updates) + 1,
            project_type=project_type,
        )
        normalized_updates[skill_type] = update
        if materialized is not None:
            materialized_updates.append(materialized)

    payload["skill_updates"] = normalized_updates
    payload["materialized_updates"] = materialized_updates
    payload.setdefault("case_summary", "")
    payload["outcome_type"] = outcome_type
    payload.setdefault("optimization_intent", "correct" if outcome_type == "failure" else "preserve")
    payload.setdefault(
        "no_update_reason",
        None if materialized_updates else _combined_no_update_reason(normalized_updates),
    )
    return payload


def _validate_skill_update(
    *,
    skill_type: str,
    raw_update: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
    outcome_type: str,
    source_case: str | None,
    update_index: int,
    project_type: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decision = str(raw_update.get("decision") or "no_update").strip()
    if decision not in UPDATE_DECISIONS:
        raise ValueError(f"skill_updates.{skill_type}.decision is invalid: {decision!r}")
    target_skill_id = raw_update.get("target_skill_id")
    target_skill_id = str(target_skill_id).strip() if target_skill_id else None
    rationale = str(raw_update.get("rationale") or "").strip()
    no_update_reason = str(raw_update.get("no_update_reason") or "").strip()
    raw_skill = raw_update.get("skill")

    if decision == "no_update":
        if target_skill_id:
            raise ValueError(f"skill_updates.{skill_type}.target_skill_id must be null for no_update.")
        if raw_skill not in (None, {}):
            raise ValueError(f"skill_updates.{skill_type}.skill must be null for no_update.")
        if not no_update_reason and not rationale:
            raise ValueError(f"skill_updates.{skill_type} requires a no_update reason.")
        normalized = {
            "decision": decision,
            "target_skill_id": None,
            "rationale": rationale,
            "skill": None,
            "no_update_reason": no_update_reason or rationale,
        }
        return normalized, None

    if not rationale:
        raise ValueError(f"skill_updates.{skill_type} requires rationale for {decision}.")

    candidate: dict[str, Any] | None = None
    if decision == "create_new":
        if target_skill_id:
            raise ValueError(f"skill_updates.{skill_type}.target_skill_id must be null for create_new.")
        complete_skill = _validate_complete_skill(raw_skill, skill_type)
        if skill_type == "project_skill":
            expected = str(project_type.get("key") or "unknown").strip()
            if expected != "unknown" and slugify(complete_skill["value"]) != slugify(expected):
                raise ValueError(
                    "skill_updates.project_skill.skill.value must use issue_abstraction.project_type_key "
                    f"{expected!r}."
                )
            duplicate = next(
                (
                    skill
                    for skill in candidate_skills.values()
                    if skill.get("skill_type") == "project_skill"
                    and slugify(str(skill.get("value") or "")) == slugify(complete_skill["value"])
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(
                    "A ProjectSkill container already exists for this project_type_key; use "
                    f"rewrite_existing or preserve_existing with target_skill_id={duplicate.get('skill_id')!r}."
                )
        operation = "create"
    else:
        if not target_skill_id:
            raise ValueError(f"skill_updates.{skill_type}.target_skill_id is required for {decision}.")
        candidate = candidate_skills.get(target_skill_id)
        if candidate is None:
            raise ValueError(
                f"skill_updates.{skill_type}.target_skill_id={target_skill_id!r} was not among this case's retrieved skills."
            )
        if candidate.get("skill_type") != skill_type:
            raise ValueError(f"skill_updates.{skill_type} must target a skill of the same type.")
        if decision == "preserve_existing":
            if raw_skill not in (None, {}):
                raise ValueError(f"skill_updates.{skill_type}.skill must be null for preserve_existing.")
            normalized = {
                "decision": decision,
                "target_skill_id": target_skill_id,
                "rationale": rationale,
                "skill": None,
                "no_update_reason": None,
            }
            return normalized, _materialize_update(
                update_index=update_index,
                operation="preserve",
                skill_type=skill_type,
                target_skill_id=target_skill_id,
                skill=None,
                rationale=rationale,
                outcome_type=outcome_type,
                source_case=source_case,
            )
        complete_skill = _validate_complete_skill(raw_skill, skill_type)
        operation = "rewrite"

    normalized = {
        "decision": decision,
        "target_skill_id": target_skill_id,
        "rationale": rationale,
        "skill": complete_skill,
        "no_update_reason": None,
    }
    materialized = _materialize_update(
        update_index=update_index,
        operation=operation,
        skill_type=skill_type,
        target_skill_id=target_skill_id,
        skill=complete_skill,
        rationale=rationale,
        outcome_type=outcome_type,
        source_case=source_case,
    )
    return normalized, materialized


def _validate_complete_skill(raw_skill: Any, skill_type: str) -> dict[str, str]:
    if not isinstance(raw_skill, dict):
        raise ValueError(f"skill_updates.{skill_type}.skill must be a complete skill object.")
    scope = _validate_scope(skill_type, raw_skill.get("scope"))
    normalized = {
        "scope": scope,
        "value": str(raw_skill.get("value") or "").strip(),
        "title": str(raw_skill.get("title") or "").strip(),
        "trigger": str(raw_skill.get("trigger") or "").strip(),
        "knowledge": str(raw_skill.get("knowledge") or "").strip(),
    }
    missing = [key for key in ("value", "title", "trigger", "knowledge") if not normalized[key]]
    if missing:
        raise ValueError(f"skill_updates.{skill_type}.skill missing required fields: {', '.join(missing)}")
    return normalized


def _materialize_update(
    *,
    update_index: int,
    operation: str,
    skill_type: str,
    target_skill_id: str | None,
    skill: dict[str, str] | None,
    rationale: str,
    outcome_type: str,
    source_case: str | None,
) -> dict[str, Any]:
    return {
        "update_id": f"update_{update_index:03d}_{skill_type}",
        "operation": operation,
        "skill_type": skill_type,
        "target_skill_id": target_skill_id,
        "skill": skill,
        "rationale": rationale,
        "outcome_type": outcome_type,
        "source_cases": [source_case] if source_case else [],
    }


def _validate_scope(skill_type: str, scope: Any) -> str:
    value = str(scope or "").strip()
    if not value:
        return "architecture_family" if skill_type == "project_skill" else "contextual"
    if value not in SKILL_SCOPES[skill_type]:
        raise ValueError(f"Invalid scope {value!r} for {skill_type}.")
    return value


def _candidate_skills_by_id(skill_search_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for key in (
        "candidate_target_skills",
        "matched_case_skills",
        "same_type_skills",
        "project_type_candidates",
        "strategy_dedup_candidates",
    ):
        for skill in skill_search_context.get(key, []) or []:
            if isinstance(skill, dict) and skill.get("skill_id"):
                candidates[str(skill["skill_id"])] = skill
    for slot in (skill_search_context.get("skill_type_slots") or {}).values():
        if not isinstance(slot, dict):
            continue
        for key in ("matched_skills", "weak_candidates"):
            for skill in slot.get(key, []) or []:
                if isinstance(skill, dict) and skill.get("skill_id"):
                    candidates[str(skill["skill_id"])] = skill
    return candidates


def _combined_no_update_reason(skill_updates: dict[str, dict[str, Any]]) -> str:
    reasons = []
    for skill_type in SKILL_TYPES:
        update = skill_updates[skill_type]
        if update.get("decision") == "no_update":
            reason = update.get("no_update_reason") or update.get("rationale")
            if reason:
                reasons.append(f"{skill_type}: {reason}")
    return "; ".join(reasons) or "Both skill types selected no_update."


def _no_update_fallback(
    insight: dict[str, Any] | None,
    trajectory_evidence: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    outcome = (insight or {}).get("analysis", {}).get("outcome_type")
    outcome = outcome or (trajectory_evidence or {}).get("outcome", {}).get("label") or "failure"
    reason = f"Reflector protocol failed: {error}" if error else "Reflector protocol failed."
    return {
        "case_summary": "No SkillBank update was applied.",
        "outcome_type": outcome,
        "skill_updates": {
            skill_type: {
                "decision": "no_update",
                "target_skill_id": None,
                "rationale": reason,
                "skill": None,
                "no_update_reason": reason,
            }
            for skill_type in SKILL_TYPES
        },
        "materialized_updates": [],
        "optimization_intent": "no_update",
        "no_update_reason": reason,
    }
