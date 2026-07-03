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
DEFAULT_GENERAL_MINIBATCH_SIZE = 8
DEFAULT_GENERAL_EDIT_BUDGET = 2
DEFAULT_GENERAL_MERGE_BUDGET = 2


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
    attempts: int = 3,
    minibatch_size: int | None = None,
    edit_budget: int | None = None,
    merge_budget: int | None = None,
) -> dict[str, Any]:
    cards = collect_residual_cards(input_run_dir, window_size=window_size)
    out_dir = Path(output_dir) if output_dir else Path(input_run_dir) / "general_reflection"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "residual_cards.json", cards)
    compact_cards = [_compact_residual_card(card) for card in cards]
    write_json(out_dir / "compact_residual_cards.json", compact_cards)

    bank = make_skill_bank(config)
    general_skills = [_compact_general_skill(skill) for skill in bank.active_skills() if skill.dimension == "general"]
    reflection_cfg = config.get("reflection", {})
    minibatch_size = int(reflection_cfg.get("general_minibatch_size", DEFAULT_GENERAL_MINIBATCH_SIZE) if minibatch_size is None else minibatch_size)
    edit_budget = int(reflection_cfg.get("general_edit_budget", DEFAULT_GENERAL_EDIT_BUDGET) if edit_budget is None else edit_budget)
    merge_budget = int(reflection_cfg.get("general_merge_budget", DEFAULT_GENERAL_MERGE_BUDGET) if merge_budget is None else merge_budget)
    analyst_prompt_path = reflection_cfg.get(
        "general_analyst_prompt_path",
        "prompt_records/reflection/general_analyst_v0.txt",
    )
    merge_prompt_path = reflection_cfg.get(
        "general_merge_prompt_path",
        "prompt_records/reflection/general_merge_v0.txt",
    )
    analyst_prompt = resolve_path(analyst_prompt_path).read_text(encoding="utf-8")
    merge_prompt = resolve_path(merge_prompt_path).read_text(encoding="utf-8")
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))

    analyst_outputs: list[dict[str, Any]] = []
    analyst_debug: list[dict[str, Any]] = []
    for batch_index, batch_cards in enumerate(_split_batches(compact_cards, minibatch_size), start=1):
        analyst_output, batch_debug = _run_general_analyst_minibatch(
            client=client,
            prompt=analyst_prompt,
            batch_cards=batch_cards,
            existing_general_skills=general_skills,
            edit_budget=edit_budget,
            attempts=attempts,
            batch_index=batch_index,
        )
        analyst_outputs.append(analyst_output)
        analyst_debug.append(batch_debug)

    write_json(out_dir / "general_analyst_outputs.json", analyst_outputs)

    reflector_output, merge_debug = _merge_general_candidates(
        client=client,
        prompt=merge_prompt,
        analyst_outputs=analyst_outputs,
        existing_general_skills=general_skills,
        residual_cards=cards,
        merge_budget=merge_budget,
        attempts=attempts,
    )

    write_json(out_dir / "general_reflector_output.json", reflector_output)
    write_json(
        out_dir / "general_reflector_debug.json",
        {
            "analyst_attempts": analyst_debug,
            "merge_attempts": merge_debug,
        },
    )

    applied_edits: list[dict[str, Any]] = []
    if apply_updates:
        for edit in reflector_output.get("materialized_edits", []):
            applied_edits.append(bank.apply_update(edit))
    write_json(out_dir / "general_skill_bank_delta.json", applied_edits)

    summary = {
        "input_run_dir": str(input_run_dir),
        "output_dir": str(out_dir),
        "residual_card_count": len(cards),
        "compact_residual_card_count": len(compact_cards),
        "existing_general_skill_count": len(general_skills),
        "general_minibatch_size": minibatch_size,
        "general_edit_budget": edit_budget,
        "general_merge_budget": merge_budget,
        "analyst_batch_count": len(analyst_outputs),
        "analyst_candidate_count": sum(len(output.get("general_patch_candidates", [])) for output in analyst_outputs),
        "proposed_update_count": len(reflector_output.get("proposed_general_updates", [])),
        "materialized_edit_count": len(reflector_output.get("materialized_edits", [])),
        "applied_edits": applied_edits,
        "updated_skill_ids": [item.get("updated_skill_id") for item in applied_edits if item.get("updated_skill_id")],
        "no_update_reason": reflector_output.get("no_update_reason"),
        "reflector_output_path": str(out_dir / "general_reflector_output.json"),
    }
    write_json(out_dir / "general_reflection_summary.json", summary)
    return summary


def _run_general_analyst_minibatch(
    *,
    client: Any,
    prompt: str,
    batch_cards: list[dict[str, Any]],
    existing_general_skills: list[dict[str, Any]],
    edit_budget: int,
    attempts: int,
    batch_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "task": "general_residual_minibatch_analysis",
        "batch_index": batch_index,
        "edit_budget": edit_budget,
        "compact_residual_cards": batch_cards,
        "existing_general_skills": existing_general_skills,
        "policy": _general_policy(),
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _json_dumps(payload)},
    ]
    last_error: Exception | None = None
    attempts_debug: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        response = client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            output = validate_general_analyst_output(
                extract_json_object(content),
                existing_general_skills=existing_general_skills,
                residual_cards=batch_cards,
                edit_budget=edit_budget,
            )
            attempts_debug.append({"attempt": attempt, "raw_response": content})
            return output, {"batch_index": batch_index, "attempts": attempts_debug}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts_debug.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = _repair_messages(messages, exc)
    return (
        _general_analyst_no_update_fallback(batch_cards, last_error),
        {"batch_index": batch_index, "attempts": attempts_debug},
    )


def _merge_general_candidates(
    *,
    client: Any,
    prompt: str,
    analyst_outputs: list[dict[str, Any]],
    existing_general_skills: list[dict[str, Any]],
    residual_cards: list[dict[str, Any]],
    merge_budget: int,
    attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _candidate_updates_from_analyst_outputs(analyst_outputs)
    if not candidates:
        return _general_no_update_from_analyst_outputs(analyst_outputs), []
    if len(analyst_outputs) == 1:
        final = {
            "batch_summary": analyst_outputs[0].get("batch_summary", ""),
            "dimension_coverage_summary": analyst_outputs[0].get("dimension_coverage_summary", ""),
            "residual_patterns": analyst_outputs[0].get("common_residual_patterns", []),
            "proposed_general_updates": candidates[:merge_budget],
            "no_update_reason": None,
        }
        return (
            validate_general_reflector_output(
                final,
                existing_general_skills=existing_general_skills,
                residual_cards=residual_cards,
                update_budget=merge_budget,
            ),
            [],
        )

    payload = {
        "task": "merge_general_patch_candidates",
        "merge_budget": merge_budget,
        "existing_general_skills": existing_general_skills,
        "general_patch_candidates": candidates,
        "analyst_batch_summaries": [
            {
                "batch_index": output.get("batch_index"),
                "batch_summary": output.get("batch_summary", ""),
                "no_update_reason": output.get("no_update_reason"),
            }
            for output in analyst_outputs
        ],
        "policy": _general_policy(),
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _json_dumps(payload)},
    ]
    last_error: Exception | None = None
    debug_attempts: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        response = client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            output = validate_general_reflector_output(
                extract_json_object(content),
                existing_general_skills=existing_general_skills,
                residual_cards=residual_cards,
                update_budget=merge_budget,
            )
            debug_attempts.append({"attempt": attempt, "raw_response": content})
            return output, debug_attempts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            debug_attempts.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = _repair_messages(messages, exc)
    return _general_no_update_fallback(last_error), debug_attempts


def validate_general_analyst_output(
    payload: dict[str, Any],
    *,
    existing_general_skills: list[dict[str, Any]],
    residual_cards: list[dict[str, Any]],
    edit_budget: int,
) -> dict[str, Any]:
    raw_candidates = payload.get("general_patch_candidates")
    if raw_candidates is None:
        raw_candidates = payload.get("proposed_general_updates")
    if not isinstance(raw_candidates, list):
        raise ValueError("General analyst output missing general_patch_candidates list.")
    existing_by_id = {str(skill.get("skill_id")): skill for skill in existing_general_skills if skill.get("skill_id")}
    source_cases = _source_cases_from_cards(residual_cards)
    normalized: list[dict[str, Any]] = []
    for index, raw_update in enumerate(raw_candidates, start=1):
        if not isinstance(raw_update, dict):
            raise ValueError("Each general_patch_candidates item must be an object.")
        update, _edit = _validate_general_update(raw_update, existing_by_id, source_cases, index)
        normalized.append(update)
    payload["general_patch_candidates"] = normalized[: max(0, edit_budget)]
    payload.setdefault("batch_summary", "")
    payload.setdefault("dimension_coverage_summary", "")
    payload.setdefault("common_residual_patterns", payload.get("residual_patterns", []))
    payload.setdefault("no_update_reason", None if _candidate_updates_from_analyst_outputs([payload]) else "No general patch candidate selected.")
    return payload


def validate_general_reflector_output(
    payload: dict[str, Any],
    *,
    existing_general_skills: list[dict[str, Any]],
    residual_cards: list[dict[str, Any]],
    update_budget: int | None = None,
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
    if update_budget is not None and update_budget >= 0:
        normalized_updates = normalized_updates[:update_budget]
        materialized_edits = [update["edit"] for update in normalized_updates if update.get("edit") is not None]
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


def _compact_residual_card(card: dict[str, Any]) -> dict[str, Any]:
    coverage = {}
    raw_coverage = card.get("dimension_coverage") if isinstance(card.get("dimension_coverage"), dict) else {}
    for dimension in ("project_type", "fault_mode", "strategy_type"):
        value = raw_coverage.get(dimension) if isinstance(raw_coverage, dict) else {}
        value = value if isinstance(value, dict) else {}
        coverage[dimension] = {
            "verdict": _clip_text(value.get("verdict", ""), 24),
            "value": _clip_text(value.get("value", "unknown"), 80),
            "evidence": _clip_text(value.get("evidence", ""), 120),
        }
    return {
        "instance_id": str(card.get("instance_id") or ""),
        "outcome_type": _clip_text(card.get("outcome_type", ""), 24),
        "best_explaining_dimension": _clip_text(card.get("best_explaining_dimension", ""), 32),
        "residual_reason": _clip_text(card.get("residual_reason", ""), 180),
        "residual_lesson": _clip_text(card.get("residual_lesson", ""), 260),
        "dimension_coverage": coverage,
        "supporting_evidence": [_clip_text(item, 180) for item in _as_list(card.get("supporting_evidence"))[:2]],
    }


def _compact_general_skill(skill: Any, *, text_limit: int = 1200) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "dimension": skill.dimension,
        "value": skill.value,
        "title": _clip_text(skill.title, 160),
        "trigger": _clip_text(skill.trigger, 360),
        "knowledge": _clip_text(skill.knowledge, text_limit),
        "anti_patterns": [_clip_text(item, 220) for item in skill.anti_patterns[:5]],
        "retrieval_text": _clip_text(skill.retrieval_text, 600),
        "provenance": {
            "supported_by_cases": skill.supported_by_cases,
            "supported_by_successes": skill.supported_by_successes,
            "supported_by_failures": skill.supported_by_failures,
        },
    }


def _candidate_updates_from_analyst_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for output in outputs:
        for update in output.get("general_patch_candidates", []):
            if update.get("decision") == "no_update" or update.get("edit") is None:
                continue
            candidate = dict(update)
            candidate["source_batch_index"] = output.get("batch_index")
            candidates.append(candidate)
    return candidates


def _general_no_update_from_analyst_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = [str(output.get("no_update_reason") or "").strip() for output in outputs if output.get("no_update_reason")]
    return {
        "batch_summary": "General analyst minibatches did not produce materialized general patch candidates.",
        "dimension_coverage_summary": "",
        "residual_patterns": [],
        "proposed_general_updates": [],
        "materialized_edits": [],
        "no_update_reason": "; ".join(reasons) or "No general patch candidate selected.",
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


def _general_analyst_no_update_fallback(cards: list[dict[str, Any]], error: Exception | None) -> dict[str, Any]:
    return {
        "batch_summary": "General analyst did not return a valid patch candidate.",
        "dimension_coverage_summary": "",
        "common_residual_patterns": [],
        "general_patch_candidates": [],
        "no_update_reason": f"General analyst protocol failure. Last error: {error}",
        "source_cases": _source_cases_from_cards(cards),
    }


def _general_policy() -> dict[str, str]:
    return {
        "general_definition": (
            "General skills capture batch-level residual localization knowledge that is repeatedly observed "
            "across cases but cannot be naturally attributed to project_type, fault_mode, or strategy_type."
        ),
        "create_threshold": "Prefer create/update when multiple residual cards support the same cross-dimensional pattern.",
        "scope": "Only general dimension skills may be edited by this meta-reflector.",
    }


def _split_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    batch_size = max(1, int(batch_size or DEFAULT_GENERAL_MINIBATCH_SIZE))
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _repair_messages(messages: list[dict[str, str]], error: Exception) -> list[dict[str, str]]:
    return [
        *messages[:2],
        {
            "role": "user",
            "content": (
                "The previous JSON failed validation. Return corrected strict JSON only. "
                f"Validation feedback: {error}"
            ),
        },
    ]


def _clip_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
