from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.skills.schema import DIMENSIONS


EDIT_OPERATIONS = ("add", "replace", "delete", "preserve")
EDIT_FIELDS = ("skill.knowledge", "skill.trigger", "skill.anti_patterns", "retrieval_text")
UPDATE_DECISIONS = ("update_existing", "create_new", "no_update", "preserve_existing")
LEARNABLE_LEVELS = ("none", "weak", "strong")


def run_reflector(
    *,
    insight: dict[str, Any] | None = None,
    trajectory_evidence: dict[str, Any] | None = None,
    skill_search_context: dict[str, Any],
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if insight is None and trajectory_evidence is None:
        raise ValueError("run_reflector requires insight or trajectory_evidence.")
    payload = {
        "trajectory_evidence": trajectory_evidence,
        "insight": insight,
        "skill_search_context": skill_search_context,
        "evolution_policy": {
            "success": "Prefer preserve or no_update. Add or replace only when the successful trajectory reveals reusable knowledge not already covered.",
            "failure": "Prefer correct/refine existing relevant skills. Create new skills only when no relevant skill exists and the lesson is transferable.",
            "system_failure": "System-level failures normally map to no_update. With force enabled, edit only when the trajectory contains a clear localization lesson.",
        },
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    last_error: Exception | None = None
    attempts_debug: list[dict[str, Any]] = []
    base_messages = list(messages)
    for attempt in range(1, attempts + 1):
        response = llm_client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            reflector_output = validate_reflector_output(
                extract_json_object(content),
                insight,
                trajectory_evidence,
                skill_search_context,
            )
            if output_path:
                write_json(output_path, {"attempt": attempt, "reflector_output": reflector_output, "raw_response": content})
            return reflector_output
        except Exception as exc:  # noqa: BLE001
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
    if not isinstance(payload.get("dimension_updates"), dict):
        raise ValueError("Reflector output missing dimension_updates object.")
    return _validate_dimension_updates_output(payload, insight, trajectory_evidence, skill_search_context)


def _validate_dimension_updates_output(
    payload: dict[str, Any],
    insight: dict[str, Any] | None = None,
    trajectory_evidence: dict[str, Any] | None = None,
    skill_search_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis = (insight or {}).get("analysis", {})
    outcome = (trajectory_evidence or {}).get("outcome", {})
    outcome_type = analysis.get("outcome_type") or outcome.get("label") or "failure"
    context = skill_search_context or {}
    candidate_skills = _candidate_skills_by_id(context)
    dimension_slots = _dimension_slots_from_context(context)
    raw_updates = payload.get("dimension_updates")
    if not isinstance(raw_updates, dict):
        raise ValueError("Reflector output missing dimension_updates object.")

    normalized_updates: dict[str, dict[str, Any]] = {}
    normalized_edits: list[dict[str, Any]] = []
    source_case = (insight or {}).get("instance_id") or (trajectory_evidence or {}).get("case", {}).get("instance_id")
    default_evidence = analysis.get("evidence") or (trajectory_evidence or {}).get("evidence", [])
    for dimension in DIMENSIONS:
        raw_update = raw_updates.get(dimension)
        if not isinstance(raw_update, dict):
            raise ValueError(f"dimension_updates.{dimension} must be an object.")
        update, edit = _validate_single_dimension_update(
            dimension=dimension,
            raw_update=raw_update,
            candidate_skills=candidate_skills,
            dimension_slots=dimension_slots,
            outcome_type=outcome_type,
            source_case=source_case,
            default_evidence=default_evidence,
            edit_index=len(normalized_edits) + 1,
        )
        normalized_updates[dimension] = update
        if edit is not None:
            normalized_edits.append(edit)

    payload["dimension_updates"] = normalized_updates
    payload["dimension_assessment"] = _normalize_multi_dimension_assessment(payload, normalized_updates)
    payload["materialized_edits"] = normalized_edits
    payload.setdefault("case_summary", "")
    payload.setdefault("outcome_type", outcome_type)
    payload.setdefault("optimization_intent", "correct" if outcome_type == "failure" else "preserve")
    payload.setdefault("no_update_reason", None if normalized_edits else _combined_no_update_reason(normalized_updates))
    return payload


def _validate_single_dimension_update(
    *,
    dimension: str,
    raw_update: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
    dimension_slots: dict[str, dict[str, Any]],
    outcome_type: str,
    source_case: str | None,
    default_evidence: list[Any],
    edit_index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decision = str(raw_update.get("decision") or "no_update").strip()
    if decision not in UPDATE_DECISIONS:
        raise ValueError(f"dimension_updates.{dimension}.decision is invalid: {decision!r}")
    target_value = str(raw_update.get("target_value") or "unknown").strip() or "unknown"
    target_skill_id = raw_update.get("target_skill_id")
    if target_skill_id is not None:
        target_skill_id = str(target_skill_id).strip() or None
    rationale = str(raw_update.get("rationale") or raw_update.get("why_this_dimension") or "").strip()
    no_update_reason = str(raw_update.get("no_update_reason") or "").strip()

    if decision == "no_update":
        if raw_update.get("edit") not in (None, {}):
            raise ValueError(f"dimension_updates.{dimension}.edit must be null/empty when decision='no_update'.")
        if not no_update_reason and not rationale:
            raise ValueError(f"dimension_updates.{dimension} requires rationale or no_update_reason for no_update.")
        return (
            {
                "decision": decision,
                "target_value": target_value,
                "target_skill_id": None,
                "rationale": rationale,
                "edit": None,
                "no_update_reason": no_update_reason or rationale,
            },
            None,
        )

    if decision == "create_new":
        if target_skill_id:
            raise ValueError(f"dimension_updates.{dimension}.target_skill_id must be null for create_new.")
        if dimension != "general" and target_value == "unknown":
            raise ValueError(f"dimension_updates.{dimension}.target_value must be concrete for create_new.")
    else:
        if not target_skill_id:
            raise ValueError(f"dimension_updates.{dimension}.target_skill_id is required for {decision}.")
        candidate = candidate_skills.get(target_skill_id)
        if candidate is None:
            raise ValueError(
                f"dimension_updates.{dimension}.target_skill_id={target_skill_id!r} was not among this case's retrieved skills."
            )
        if candidate.get("dimension") != dimension:
            raise ValueError(f"dimension_updates.{dimension}.target skill dimension must equal {dimension}.")
        target_value = str(candidate.get("value") or target_value or "unknown")

    raw_edit = raw_update.get("edit")
    if decision == "preserve_existing":
        if raw_edit in (None, {}):
            raw_edit = {"operation": "preserve", "field": "skill.knowledge", "content": {}}
    if not isinstance(raw_edit, dict):
        raise ValueError(f"dimension_updates.{dimension}.edit must be an object for {decision}.")

    operation = str(raw_edit.get("operation") or ("add" if decision == "create_new" else "")).strip()
    if operation not in EDIT_OPERATIONS:
        raise ValueError(f"Invalid edit operation for {dimension}: {operation!r}")
    if decision == "create_new" and operation != "add":
        raise ValueError(f"dimension_updates.{dimension} create_new requires edit.operation='add'.")
    if decision == "update_existing" and operation == "preserve":
        raise ValueError(f"dimension_updates.{dimension} update_existing requires add/replace/delete.")
    if decision == "preserve_existing" and operation != "preserve":
        raise ValueError(f"dimension_updates.{dimension} preserve_existing requires edit.operation='preserve'.")

    raw_target = raw_edit.get("target") if isinstance(raw_edit.get("target"), dict) else {}
    field = str(raw_edit.get("field") or raw_target.get("field") or "skill.knowledge")
    target = {
        "skill_id": None if decision == "create_new" else target_skill_id,
        "dimension": dimension,
        "value": target_value,
        "field": field,
    }
    content = raw_edit.get("content")
    if not isinstance(content, dict):
        content = {}
    edit = {
        "edit_id": raw_edit.get("edit_id") or f"edit_{edit_index:03d}_{dimension}",
        "operation": operation,
        "target": target,
        "rationale": rationale or str(raw_edit.get("rationale") or "").strip(),
        "evidence": raw_edit.get("evidence") or raw_update.get("evidence") or default_evidence,
        "content": content,
        "expected_effect": raw_edit.get("expected_effect") or raw_update.get("expected_effect") or "",
        "risk": raw_edit.get("risk") or raw_update.get("risk") or "",
        "outcome_type": outcome_type,
        "source_cases": [source_case] if source_case else [],
    }
    _validate_dimension_edit(dimension, decision, edit, candidate_skills, dimension_slots)
    normalized_update = {
        "decision": decision,
        "target_value": target_value,
        "target_skill_id": target_skill_id,
        "rationale": rationale,
        "edit": edit,
        "no_update_reason": no_update_reason or None,
    }
    return normalized_update, edit


def _validate_dimension_edit(
    dimension: str,
    decision: str,
    edit: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
    dimension_slots: dict[str, dict[str, Any]],
) -> None:
    target = edit["target"]
    operation = edit["operation"]
    if target["dimension"] != dimension:
        raise ValueError("dimension update edit target.dimension must match its dimension key.")
    if target["field"] not in EDIT_FIELDS:
        raise ValueError(f"Invalid target.field: {target['field']!r}")
    content = edit["content"]
    if operation == "add" and not any(content.get(key) for key in ("text", "new_text")):
        raise ValueError("add edit requires content.text or content.new_text.")
    if decision == "create_new":
        _validate_create_new_target(target, content, edit, dimension_slots)
        return
    _bind_existing_target_to_candidate_skill(operation, target, candidate_skills)
    if operation == "replace":
        _validate_precise_replace_edit(target, content, candidate_skills)
    if operation == "delete":
        _validate_precise_delete_edit(target, content, candidate_skills)


def _normalize_multi_dimension_assessment(
    payload: dict[str, Any],
    dimension_updates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    raw_assessment = payload.get("dimension_assessment")
    if not isinstance(raw_assessment, dict):
        raw_assessment = {}
    normalized: dict[str, dict[str, str]] = {}
    for dimension in DIMENSIONS:
        raw = raw_assessment.get(dimension)
        if not isinstance(raw, dict):
            raw = {}
        update = dimension_updates[dimension]
        learnable = str(raw.get("learnable") or "").strip().lower()
        if learnable not in LEARNABLE_LEVELS:
            learnable = "none" if update["decision"] == "no_update" else "strong"
        candidate_value = str(raw.get("candidate_value") or "").strip()
        if not candidate_value:
            candidate_value = update.get("target_value") or "unknown"
        what_can_be_learned = str(raw.get("what_can_be_learned") or "").strip()
        if not what_can_be_learned:
            what_can_be_learned = update.get("rationale") or update.get("no_update_reason") or (
                "No reusable lesson selected for this dimension."
            )
        normalized[dimension] = {
            "learnable": learnable,
            "candidate_value": candidate_value,
            "what_can_be_learned": what_can_be_learned,
        }
    return normalized


def _combined_no_update_reason(dimension_updates: dict[str, dict[str, Any]]) -> str:
    reasons = []
    for dimension in DIMENSIONS:
        update = dimension_updates[dimension]
        if update["decision"] == "no_update":
            reason = update.get("no_update_reason") or update.get("rationale")
            if reason:
                reasons.append(f"{dimension}: {reason}")
    return "; ".join(reasons) or "All dimensions selected no_update."


def _validate_precise_replace_edit(
    target: dict[str, Any],
    content: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
) -> None:
    old_text = str(content.get("old_text") or "").strip()
    new_text = str(content.get("new_text") or "").strip()
    if not old_text or not new_text:
        raise ValueError("replace edit requires exact content.old_text and content.new_text.")
    current = _current_target_text(target, candidate_skills)
    if old_text not in current:
        raise ValueError("replace edit content.old_text was not found in the target skill field.")


def _validate_precise_delete_edit(
    target: dict[str, Any],
    content: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
) -> None:
    old_text = str(content.get("old_text") or content.get("text") or "").strip()
    if not old_text:
        raise ValueError("delete edit requires exact content.old_text or content.text.")
    current = _current_target_text(target, candidate_skills)
    if old_text not in current:
        raise ValueError("delete edit text was not found in the target skill field.")


def _current_target_text(target: dict[str, Any], candidate_skills: dict[str, dict[str, Any]]) -> str:
    skill_id = target.get("skill_id")
    if not skill_id:
        raise ValueError("replace/delete edit must target an existing retrieved skill.")
    skill = candidate_skills.get(str(skill_id))
    if skill is None:
        raise ValueError(f"target.skill_id={skill_id!r} was not among this case's retrieved dimension skills.")
    field = str(target.get("field") or "skill.knowledge")
    if field == "skill.knowledge":
        return str(skill.get("knowledge") or "")
    if field == "skill.trigger":
        return str(skill.get("trigger") or "")
    if field == "retrieval_text":
        return str(skill.get("retrieval_text") or "")
    if field == "skill.anti_patterns":
        return "\n".join(str(item) for item in skill.get("anti_patterns", []) or [])
    raise ValueError(f"Unsupported edit target field: {field!r}")


def _no_update_fallback(
    insight: dict[str, Any] | None,
    trajectory_evidence: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    analysis = (insight or {}).get("analysis", {})
    outcome = (trajectory_evidence or {}).get("outcome", {})
    outcome_type = analysis.get("outcome_type") or outcome.get("label") or "failure"
    return {
        "case_summary": "Reflector did not return a valid bounded skill edit.",
        "outcome_type": outcome_type,
        "optimization_intent": "no_update",
        "dimension_assessment": {
            dimension: {
                "learnable": "none",
                "candidate_value": "unknown",
                "what_can_be_learned": "No dimension can be safely updated after protocol failure.",
            }
            for dimension in DIMENSIONS
        },
        "dimension_updates": {
            dimension: {
                "decision": "no_update",
                "target_value": "unknown",
                "target_skill_id": None,
                "rationale": "Reflector output could not be validated.",
                "edit": None,
                "no_update_reason": "Reflector output could not be validated.",
            }
            for dimension in DIMENSIONS
        },
        "materialized_edits": [],
        "no_update_reason": f"Reflector protocol failure; SkillBank left unchanged. Last error: {error}",
    }


def _candidate_skills_by_id(skill_search_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for key in ("candidate_target_skills", "matched_case_skills", "same_dimension_skills"):
        for skill in skill_search_context.get(key, []) or []:
            if isinstance(skill, dict) and skill.get("skill_id"):
                candidates[str(skill["skill_id"])] = skill
    for slot in (skill_search_context.get("dimension_slots") or {}).values():
        if not isinstance(slot, dict):
            continue
        for key in ("matched_skills", "weak_candidates"):
            for skill in slot.get(key, []) or []:
                if isinstance(skill, dict) and skill.get("skill_id"):
                    candidates[str(skill["skill_id"])] = skill
    return candidates


def _dimension_slots_from_context(skill_search_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slots = skill_search_context.get("dimension_slots") or {}
    slots: dict[str, dict[str, Any]] = {}
    if isinstance(raw_slots, dict):
        for dimension in DIMENSIONS:
            slot = raw_slots.get(dimension) or {}
            matched = slot.get("matched_skills", []) if isinstance(slot, dict) else []
            weak = slot.get("weak_candidates", []) if isinstance(slot, dict) else []
            status = slot.get("status") if isinstance(slot, dict) else None
            slots[dimension] = {
                "dimension": dimension,
                "status": status or ("matched" if matched else "weak" if weak else "missing"),
                "matched_skills": matched if isinstance(matched, list) else [],
                "weak_candidates": weak if isinstance(weak, list) else [],
            }
    if slots:
        return slots

    slots = {
        dimension: {"dimension": dimension, "status": "missing", "matched_skills": [], "weak_candidates": []}
        for dimension in DIMENSIONS
    }
    fallback_context = {key: skill_search_context.get(key, []) for key in ("candidate_target_skills", "matched_case_skills", "same_dimension_skills")}
    for skill in _candidate_skills_by_id(fallback_context).values():
        dimension = skill.get("dimension") if skill.get("dimension") in DIMENSIONS else "general"
        slots[dimension]["matched_skills"].append(skill)
        slots[dimension]["status"] = "matched"
    return slots


def _bind_existing_target_to_candidate_skill(
    operation: str,
    target: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
) -> None:
    target_skill_id = target.get("skill_id")
    if operation == "add" and not target_skill_id:
        return
    if not target_skill_id:
        raise ValueError(f"{operation} edit must target a retrieved dimension skill or use no_update.")
    target_skill_id = str(target_skill_id)
    candidate = candidate_skills.get(target_skill_id)
    if candidate is None:
        raise ValueError(
            f"{operation} edit target.skill_id={target_skill_id!r} was not among this case's retrieved dimension skills."
        )
    target["skill_id"] = target_skill_id
    target["dimension"] = candidate.get("dimension") or target.get("dimension")
    target["value"] = candidate.get("value") or target.get("value")


def _validate_create_new_target(
    target: dict[str, Any],
    content: dict[str, Any],
    edit: dict[str, Any],
    dimension_slots: dict[str, dict[str, Any]],
) -> None:
    dimension = str(target.get("dimension") or "")
    if dimension not in DIMENSIONS:
        raise ValueError(f"create_new target.dimension must be one of {DIMENSIONS}.")
    value = str(target.get("value") or "").strip()
    if dimension != "general" and value in ("", "unknown"):
        raise ValueError("create_new for non-general dimension requires a concrete target.value.")
    if target.get("field") != "skill.knowledge":
        raise ValueError("create_new must target field='skill.knowledge'.")
    for key in ("title", "trigger", "retrieval_text"):
        if not str(content.get(key) or "").strip():
            raise ValueError(f"create_new requires content.{key}.")
    slot = dimension_slots.get(dimension) or {"status": "missing", "matched_skills": [], "weak_candidates": []}
    target["slot_status"] = slot.get("status", "missing")
    existing_candidates = list(slot.get("matched_skills", []) or []) + list(slot.get("weak_candidates", []) or [])
    if existing_candidates and not str(edit.get("rationale") or "").strip():
        raise ValueError(
            "create_new in a dimension with matched/weak candidates requires rationale explaining why no existing skill is being updated."
        )
