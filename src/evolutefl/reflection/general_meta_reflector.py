from __future__ import annotations

from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.json_utils import extract_json_object, read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank


GENERAL_DECISIONS = ("create_new", "update_existing", "no_update")
GENERAL_EDIT_OPERATIONS = ("add", "replace", "delete")
GENERAL_EDIT_FIELDS = ("skill.knowledge", "skill.trigger", "skill.anti_patterns", "retrieval_text")


def collect_residual_cards(input_run_dir: str | Path, *, window_size: int | None = None) -> list[dict[str, Any]]:
    root = Path(input_run_dir)
    cards: list[dict[str, Any]] = []
    for path in sorted(root.rglob("residual_card.json")):
        try:
            card = read_json(path)
        except Exception:  # noqa: BLE001 - one bad card should not block the batch.
            continue
        if isinstance(card, dict):
            card["_source_path"] = str(path)
            cards.append(card)
    if window_size is not None and window_size > 0:
        return cards[-window_size:]
    return cards


def run_general_reflection(
    *,
    input_run_dir: str | Path,
    config: dict[str, Any],
    llm_client: Any | None = None,
    output_dir: str | Path | None = None,
    window_size: int | None = None,
    apply_updates: bool = True,
    attempts: int = 2,
) -> dict[str, Any]:
    cards = collect_residual_cards(input_run_dir, window_size=window_size)
    out_dir = Path(output_dir) if output_dir else Path(input_run_dir) / "general_reflection"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "residual_cards.json", cards)

    bank = make_skill_bank(config)
    general_skills = [_compact_general_skill(skill) for skill in bank.active_skills() if skill.dimension == "general"]
    reflection_cfg = config.get("reflection", {})
    prompt_path = reflection_cfg.get(
        "general_reflector_prompt_path",
        "prompt_records/reflection/general_meta_reflector_v0.txt",
    )
    prompt = resolve_path(prompt_path).read_text(encoding="utf-8")
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))
    payload = {
        "task": "batch_level_general_residual_reflection",
        "residual_cards": cards,
        "existing_general_skills": general_skills,
        "policy": {
            "general_definition": (
                "General skills capture batch-level residual localization knowledge that is repeatedly observed "
                "across cases but cannot be naturally attributed to project_type, fault_mode, or strategy_type."
            ),
            "create_threshold": "Prefer create/update only when multiple residual cards support the same cross-dimensional pattern.",
            "scope": "Only general dimension skills may be edited by this meta-reflector.",
        },
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _json_dumps(payload)},
    ]
    last_error: Exception | None = None
    debug_attempts: list[dict[str, Any]] = []
    reflector_output: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        response = client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            reflector_output = validate_general_reflector_output(
                extract_json_object(content),
                existing_general_skills=general_skills,
                residual_cards=cards,
            )
            debug_attempts.append({"attempt": attempt, "raw_response": content})
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            debug_attempts.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = [
                *messages[:2],
                {
                    "role": "user",
                    "content": (
                        "The previous JSON failed validation. Return corrected strict JSON only. "
                        f"Validation feedback: {exc}"
                    ),
                },
            ]

    if reflector_output is None:
        reflector_output = _general_no_update_fallback(last_error)

    write_json(out_dir / "general_reflector_output.json", reflector_output)
    write_json(out_dir / "general_reflector_debug.json", {"attempts": debug_attempts})

    applied_edits: list[dict[str, Any]] = []
    if apply_updates:
        for edit in reflector_output.get("materialized_edits", []):
            applied_edits.append(bank.apply_update(edit))
    write_json(out_dir / "general_skill_bank_delta.json", applied_edits)

    summary = {
        "input_run_dir": str(input_run_dir),
        "output_dir": str(out_dir),
        "residual_card_count": len(cards),
        "existing_general_skill_count": len(general_skills),
        "proposed_update_count": len(reflector_output.get("proposed_general_updates", [])),
        "materialized_edit_count": len(reflector_output.get("materialized_edits", [])),
        "applied_edits": applied_edits,
        "updated_skill_ids": [item.get("updated_skill_id") for item in applied_edits if item.get("updated_skill_id")],
        "no_update_reason": reflector_output.get("no_update_reason"),
        "reflector_output_path": str(out_dir / "general_reflector_output.json"),
    }
    write_json(out_dir / "general_reflection_summary.json", summary)
    return summary


def validate_general_reflector_output(
    payload: dict[str, Any],
    *,
    existing_general_skills: list[dict[str, Any]],
    residual_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    updates = payload.get("proposed_general_updates")
    if not isinstance(updates, list):
        raise ValueError("General meta-reflector output missing proposed_general_updates list.")
    existing_by_id = {str(skill.get("skill_id")): skill for skill in existing_general_skills if skill.get("skill_id")}
    source_cases = _source_cases_from_cards(residual_cards)
    normalized_updates: list[dict[str, Any]] = []
    materialized_edits: list[dict[str, Any]] = []
    for index, raw_update in enumerate(updates, start=1):
        if not isinstance(raw_update, dict):
            raise ValueError("Each proposed_general_updates item must be an object.")
        update, edit = _validate_general_update(raw_update, existing_by_id, source_cases, index)
        normalized_updates.append(update)
        if edit is not None:
            materialized_edits.append(edit)
    payload["proposed_general_updates"] = normalized_updates
    payload["materialized_edits"] = materialized_edits
    payload.setdefault("batch_summary", "")
    payload.setdefault("dimension_coverage_summary", "")
    payload.setdefault("residual_patterns", [])
    payload.setdefault("no_update_reason", None if materialized_edits else "No general residual update selected.")
    return payload


def _validate_general_update(
    raw_update: dict[str, Any],
    existing_by_id: dict[str, dict[str, Any]],
    source_cases: list[str],
    index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decision = str(raw_update.get("decision") or "no_update")
    if decision not in GENERAL_DECISIONS:
        raise ValueError(f"Invalid general update decision: {decision!r}")
    target_skill_id = raw_update.get("target_skill_id")
    target_skill_id = str(target_skill_id).strip() if target_skill_id is not None else None
    target_value = str(raw_update.get("target_value") or "general_residual").strip() or "general_residual"
    rationale = str(raw_update.get("rationale") or "").strip()
    supporting_cases = _as_list(raw_update.get("supporting_cases")) or source_cases
    if decision == "no_update":
        if raw_update.get("edit") not in (None, {}):
            raise ValueError("no_update general update must not include an edit.")
        return (
            {
                "decision": decision,
                "target_value": target_value,
                "target_skill_id": None,
                "rationale": rationale or str(raw_update.get("no_update_reason") or ""),
                "supporting_cases": supporting_cases,
                "edit": None,
                "no_update_reason": raw_update.get("no_update_reason") or rationale,
            },
            None,
        )
    if decision == "create_new" and target_skill_id:
        raise ValueError("create_new general update must use target_skill_id=null.")
    if decision == "update_existing":
        if not target_skill_id or target_skill_id not in existing_by_id:
            raise ValueError("update_existing general update must target an existing general skill.")
        target_value = str(existing_by_id[target_skill_id].get("value") or target_value)
    edit = raw_update.get("edit")
    if not isinstance(edit, dict):
        raise ValueError(f"{decision} general update requires edit object.")
    operation = str(edit.get("operation") or ("add" if decision == "create_new" else ""))
    if operation not in GENERAL_EDIT_OPERATIONS:
        raise ValueError(f"Invalid general edit operation: {operation!r}")
    if decision == "create_new" and operation != "add":
        raise ValueError("create_new general update requires edit.operation='add'.")
    field = str(edit.get("field") or "skill.knowledge")
    if field not in GENERAL_EDIT_FIELDS:
        raise ValueError(f"Invalid general edit field: {field!r}")
    content = edit.get("content")
    if not isinstance(content, dict):
        content = {}
    if decision == "create_new":
        for key in ("text", "title", "trigger", "retrieval_text"):
            if not str(content.get(key) or "").strip():
                raise ValueError(f"create_new general update requires content.{key}.")
    elif operation == "replace":
        if not str(content.get("old_text") or "").strip() or not str(content.get("new_text") or "").strip():
            raise ValueError("replace general update requires content.old_text and content.new_text.")
    elif operation == "delete":
        if not str(content.get("old_text") or content.get("text") or "").strip():
            raise ValueError("delete general update requires content.old_text or content.text.")
    materialized = {
        "edit_id": edit.get("edit_id") or f"general_meta_edit_{index:03d}",
        "operation": operation,
        "target": {
            "skill_id": None if decision == "create_new" else target_skill_id,
            "dimension": "general",
            "value": target_value,
            "field": field,
        },
        "rationale": rationale or str(edit.get("rationale") or ""),
        "evidence": edit.get("evidence") or raw_update.get("evidence") or [],
        "content": content,
        "expected_effect": edit.get("expected_effect") or raw_update.get("expected_effect") or "",
        "risk": edit.get("risk") or raw_update.get("risk") or "",
        "outcome_type": "batch_residual",
        "source_cases": supporting_cases,
    }
    return (
        {
            "decision": decision,
            "target_value": target_value,
            "target_skill_id": target_skill_id,
            "rationale": rationale,
            "supporting_cases": supporting_cases,
            "edit": materialized,
        },
        materialized,
    )


def _compact_general_skill(skill: Any) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "dimension": skill.dimension,
        "value": skill.value,
        "title": skill.title,
        "trigger": skill.trigger,
        "knowledge": skill.knowledge,
        "anti_patterns": skill.anti_patterns,
        "retrieval_text": skill.retrieval_text,
        "provenance": {
            "supported_by_cases": skill.supported_by_cases,
            "supported_by_successes": skill.supported_by_successes,
            "supported_by_failures": skill.supported_by_failures,
        },
    }


def _source_cases_from_cards(cards: list[dict[str, Any]]) -> list[str]:
    cases = []
    for card in cards:
        instance_id = card.get("instance_id")
        if instance_id:
            cases.append(str(instance_id))
    return list(dict.fromkeys(cases))


def _general_no_update_fallback(error: Exception | None) -> dict[str, Any]:
    return {
        "batch_summary": "General meta-reflector did not return a valid residual update.",
        "dimension_coverage_summary": "",
        "residual_patterns": [],
        "proposed_general_updates": [],
        "materialized_edits": [],
        "no_update_reason": f"General meta-reflector protocol failure; SkillBank left unchanged. Last error: {error}",
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
