from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.skills.schema import SKILL_SCOPES, SKILL_TYPES, slugify


EDIT_OPERATIONS = ("add", "replace", "delete", "preserve")
EDIT_FIELDS = ("skill.knowledge", "skill.trigger", "retrieval_text")
UPDATE_DECISIONS = ("update_existing", "create_new", "no_update", "preserve_existing")


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
            "success": "Prefer preserve or no_update. Edit only when the successful trajectory reveals reusable knowledge not already covered.",
            "failure": "Refine a relevant skill when it represents the same concept. Create a skill only for an independent transferable lesson.",
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
        except Exception as exc:  # noqa: BLE001 - protocol failures get one fixed-schema retry.
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
        write_json(
            output_path,
            {"failed": True, "attempts": attempts_debug, "reflector_output": fallback},
        )
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
    skill_type_slots = _skill_type_slots_from_context(context)
    project_type = context.get("project_type") if isinstance(context.get("project_type"), dict) else {}
    source_case = (insight or {}).get("instance_id") or (trajectory_evidence or {}).get("case", {}).get("instance_id")

    normalized_updates: dict[str, dict[str, Any]] = {}
    materialized_edits: list[dict[str, Any]] = []
    for skill_type in SKILL_TYPES:
        raw_update = raw_updates.get(skill_type)
        if not isinstance(raw_update, dict):
            raise ValueError(f"skill_updates.{skill_type} must be an object.")
        update, edit = _validate_skill_update(
            skill_type=skill_type,
            raw_update=raw_update,
            candidate_skills=candidate_skills,
            skill_type_slots=skill_type_slots,
            outcome_type=outcome_type,
            source_case=source_case,
            edit_index=len(materialized_edits) + 1,
            project_type=project_type,
        )
        normalized_updates[skill_type] = update
        if edit is not None:
            materialized_edits.append(edit)

    payload["skill_updates"] = normalized_updates
    payload["materialized_edits"] = materialized_edits
    payload.setdefault("case_summary", "")
    payload["outcome_type"] = outcome_type
    payload.setdefault("optimization_intent", "correct" if outcome_type == "failure" else "preserve")
    payload.setdefault(
        "no_update_reason",
        None if materialized_edits else _combined_no_update_reason(normalized_updates),
    )
    return payload


def _validate_skill_update(
    *,
    skill_type: str,
    raw_update: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
    skill_type_slots: dict[str, dict[str, Any]],
    outcome_type: str,
    source_case: str | None,
    edit_index: int,
    project_type: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decision = str(raw_update.get("decision") or "no_update").strip()
    if decision not in UPDATE_DECISIONS:
        raise ValueError(f"skill_updates.{skill_type}.decision is invalid: {decision!r}")
    target_value = str(raw_update.get("target_value") or "unknown").strip() or "unknown"
    target_skill_id = raw_update.get("target_skill_id")
    target_skill_id = str(target_skill_id).strip() if target_skill_id else None
    rationale = str(raw_update.get("rationale") or "").strip()
    no_update_reason = str(raw_update.get("no_update_reason") or "").strip()
    scope = raw_update.get("scope")

    if decision == "no_update":
        if raw_update.get("edit") not in (None, {}):
            raise ValueError(f"skill_updates.{skill_type}.edit must be null for no_update.")
        if not no_update_reason and not rationale:
            raise ValueError(f"skill_updates.{skill_type} requires a no_update reason.")
        return (
            {
                "decision": decision,
                "scope": None,
                "target_value": target_value,
                "target_skill_id": None,
                "rationale": rationale,
                "edit": None,
                "no_update_reason": no_update_reason or rationale,
            },
            None,
        )

    if not rationale:
        raise ValueError(f"skill_updates.{skill_type} requires rationale for {decision}.")

    candidate: dict[str, Any] | None = None
    if decision == "create_new":
        if target_skill_id:
            raise ValueError(f"skill_updates.{skill_type}.target_skill_id must be null for create_new.")
        if target_value == "unknown":
            raise ValueError(f"skill_updates.{skill_type}.target_value must be concrete for create_new.")
        scope = _validate_scope(skill_type, scope)
        if skill_type == "project_skill":
            expected_project_type = str(project_type.get("key") or "unknown").strip()
            if expected_project_type != "unknown" and slugify(target_value) != slugify(expected_project_type):
                raise ValueError(
                    "skill_updates.project_skill.target_value must use issue_abstraction.project_type_key "
                    f"{expected_project_type!r}."
                )
            duplicate = next(
                (
                    skill
                    for skill in candidate_skills.values()
                    if skill.get("skill_type") == "project_skill"
                    and slugify(str(skill.get("value") or "")) == slugify(target_value)
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(
                    "A ProjectSkill container already exists for this project_type_key; use "
                    f"update_existing or preserve_existing with target_skill_id={duplicate.get('skill_id')!r}."
                )
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
        target_value = str(candidate.get("value") or target_value)
        scope = _validate_scope(skill_type, candidate.get("scope") or scope)

    if decision == "preserve_existing":
        if raw_update.get("edit") not in (None, {}):
            raise ValueError(f"skill_updates.{skill_type}.edit must be null for preserve_existing.")
        edit = _materialize_edit(
            edit_index=edit_index,
            operation="preserve",
            skill_type=skill_type,
            scope=scope,
            target_value=target_value,
            target_skill_id=target_skill_id,
            field="skill.knowledge",
            content={},
            rationale=rationale,
            outcome_type=outcome_type,
            source_case=source_case,
            slot_status=(skill_type_slots.get(skill_type) or {}).get("status", "missing"),
        )
        return (
            {
                "decision": decision,
                "scope": scope,
                "target_value": target_value,
                "target_skill_id": target_skill_id,
                "rationale": rationale,
                "edit": None,
                "no_update_reason": None,
            },
            edit,
        )

    raw_edit = raw_update.get("edit")
    if not isinstance(raw_edit, dict):
        raise ValueError(f"skill_updates.{skill_type}.edit must be an object for {decision}.")
    operation = str(raw_edit.get("operation") or ("add" if decision == "create_new" else "")).strip()
    if operation not in EDIT_OPERATIONS:
        raise ValueError(f"Invalid edit operation for {skill_type}: {operation!r}")
    if decision == "create_new" and operation != "add":
        raise ValueError(f"skill_updates.{skill_type} create_new requires operation='add'.")
    if decision == "update_existing" and operation not in ("add", "replace", "delete"):
        raise ValueError(f"skill_updates.{skill_type} update_existing requires add/replace/delete.")
    field = str(raw_edit.get("field") or "skill.knowledge")
    if field not in EDIT_FIELDS:
        raise ValueError(f"Invalid target field: {field!r}")
    content = raw_edit.get("content")
    if not isinstance(content, dict):
        raise ValueError(f"skill_updates.{skill_type}.edit.content must be an object.")

    slot_status = (skill_type_slots.get(skill_type) or {}).get("status", "missing")
    if decision == "create_new":
        _validate_create_content(content)
        if slot_status in ("matched", "weak") and not rationale:
            raise ValueError(
                f"Creating a new {skill_type} while candidates exist requires rationale explaining its independent identity."
            )
    elif candidate is not None:
        _validate_precise_existing_edit(operation, field, content, candidate)

    edit = _materialize_edit(
        edit_index=edit_index,
        operation=operation,
        skill_type=skill_type,
        scope=scope,
        target_value=target_value,
        target_skill_id=target_skill_id,
        field=field,
        content=content,
        rationale=rationale,
        outcome_type=outcome_type,
        source_case=source_case,
        slot_status=slot_status,
    )
    return (
        {
            "decision": decision,
            "scope": scope,
            "target_value": target_value,
            "target_skill_id": target_skill_id,
            "rationale": rationale,
            "edit": edit,
            "no_update_reason": no_update_reason or None,
        },
        edit,
    )


def _materialize_edit(
    *,
    edit_index: int,
    operation: str,
    skill_type: str,
    scope: str,
    target_value: str,
    target_skill_id: str | None,
    field: str,
    content: dict[str, Any],
    rationale: str,
    outcome_type: str,
    source_case: str | None,
    slot_status: str,
) -> dict[str, Any]:
    return {
        "edit_id": f"edit_{edit_index:03d}_{skill_type}",
        "operation": operation,
        "target": {
            "skill_id": target_skill_id,
            "skill_type": skill_type,
            "scope": scope,
            "value": target_value,
            "field": field,
            "slot_status": slot_status,
        },
        "rationale": rationale,
        "content": content,
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


def _validate_create_content(content: dict[str, Any]) -> None:
    required = ("text", "title", "trigger", "retrieval_text")
    missing = [key for key in required if not str(content.get(key) or "").strip()]
    if missing:
        raise ValueError(f"create_new content missing required fields: {', '.join(missing)}")


def _validate_precise_existing_edit(
    operation: str,
    field: str,
    content: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    current = _candidate_field_text(candidate, field)
    if operation == "add":
        if not str(content.get("text") or content.get("new_text") or "").strip():
            raise ValueError("add edit requires content.text or content.new_text.")
        return
    if operation == "replace":
        old_text = str(content.get("old_text") or "")
        new_text = str(content.get("new_text") or "")
        if not old_text or not new_text:
            raise ValueError("replace edit requires exact old_text and new_text.")
        if old_text not in current:
            raise ValueError("replace edit old_text was not found in the target skill field.")
        return
    if operation == "delete":
        target = str(content.get("text") or content.get("old_text") or "")
        if not target:
            raise ValueError("delete edit requires exact content.text or content.old_text.")
        if target not in current:
            raise ValueError("delete edit text was not found in the target skill field.")


def _candidate_field_text(candidate: dict[str, Any], field: str) -> str:
    if field == "skill.knowledge":
        return str(candidate.get("knowledge") or "")
    if field == "skill.trigger":
        return str(candidate.get("trigger") or "")
    return str(candidate.get("retrieval_text") or "")


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


def _skill_type_slots_from_context(skill_search_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slots = skill_search_context.get("skill_type_slots") or {}
    slots: dict[str, dict[str, Any]] = {}
    for skill_type in SKILL_TYPES:
        raw = raw_slots.get(skill_type) if isinstance(raw_slots, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        matched = raw.get("matched_skills") if isinstance(raw.get("matched_skills"), list) else []
        weak = raw.get("weak_candidates") if isinstance(raw.get("weak_candidates"), list) else []
        slots[skill_type] = {
            "skill_type": skill_type,
            "status": str(raw.get("status") or ("matched" if matched else "weak" if weak else "missing")),
            "matched_skills": matched,
            "weak_candidates": weak,
        }
    return slots


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
                "scope": None,
                "target_value": "unknown",
                "target_skill_id": None,
                "rationale": reason,
                "edit": None,
                "no_update_reason": reason,
            }
            for skill_type in SKILL_TYPES
        },
        "materialized_edits": [],
        "optimization_intent": "no_update",
        "no_update_reason": reason,
    }
