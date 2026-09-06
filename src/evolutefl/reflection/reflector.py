from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.skills.schema import (
    FINAL_MAX_STATEMENT_CHARS,
    SKILL_TYPES,
    validate_atomic_knowledge,
    validate_fault_knowledge,
    validate_portable_skill_card,
)
from evolutefl.skills.fault_taxonomy import validate_fault_family


UPDATE_DECISIONS = ("create_new", "rewrite_existing", "preserve_existing", "no_update")
LEGACY_REFLECTOR_SKILL_TYPES = ("project_skill", "strategy_skill")
# Queries should stay retrieval-oriented, but a single generic relationship
# sometimes needs more than 500 characters to preserve its diagnostic roles.
MAX_EVOLUTION_QUERY_CHARS = 800


def generate_evolution_queries(
    *,
    trajectory_evidence: dict[str, Any],
    strategy_catalog: list[dict[str, Any]],
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a Project query and select a Strategy trigger from one case."""
    catalog_ids = {
        str(item.get("skill_id") or "")
        for item in strategy_catalog
        if isinstance(item, dict) and item.get("skill_id")
    }
    selector_input = {
        "trajectory_evidence": trajectory_evidence,
        "strategy_skill_catalog": strategy_catalog,
    }
    base_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(selector_input, ensure_ascii=False, indent=2)},
    ]
    messages = list(base_messages)
    debug: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        response = llm_client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
        )
        content = response.get("content") or ""
        try:
            payload = extract_json_object(content)
            raw_selected = payload.get("selected_strategy_skill_id")
            selected_strategy_skill_id = (
                None
                if raw_selected in (None, "", "none", "null")
                else str(raw_selected).strip()
            )
            unavailable_strategy_skill_id = None
            if (
                selected_strategy_skill_id is not None
                and selected_strategy_skill_id not in catalog_ids
            ):
                # A hallucinated catalog ID should not discard otherwise valid
                # evolution queries. Treat it as no existing Strategy match so
                # the downstream Reflector may still create a new card.
                unavailable_strategy_skill_id = selected_strategy_skill_id
                selected_strategy_skill_id = None
            result = {
                "project_skill_query": str(payload.get("project_skill_query") or "").strip(),
                "strategy_diagnostic_context": str(
                    payload.get("strategy_diagnostic_context") or ""
                ).strip(),
                "selected_strategy_skill_id": selected_strategy_skill_id,
                "strategy_selection_reason": str(
                    payload.get("strategy_selection_reason") or ""
                ).strip(),
            }
            if unavailable_strategy_skill_id:
                result["unavailable_strategy_skill_id"] = unavailable_strategy_skill_id
                result["strategy_selection_reason"] = (
                    result["strategy_selection_reason"]
                    + " No supplied Strategy Skill matched this diagnostic context."
                ).strip()
            if not result["project_skill_query"] or not result["strategy_diagnostic_context"]:
                raise ValueError(
                    "project_skill_query and strategy_diagnostic_context must be non-empty strings."
                )
            if not result["strategy_selection_reason"]:
                raise ValueError("strategy_selection_reason must be non-empty.")
            oversized = [
                key
                for key in (
                    "project_skill_query",
                    "strategy_diagnostic_context",
                    "strategy_selection_reason",
                )
                if len(str(result[key])) > MAX_EVOLUTION_QUERY_CHARS
            ]
            if oversized:
                raise ValueError(
                    f"Evolution queries exceed {MAX_EVOLUTION_QUERY_CHARS} characters: {', '.join(oversized)}."
                )
            if output_path:
                write_json(output_path, {"attempt": attempt, "evolution_queries": result, "raw_response": content})
            return result
        except Exception as exc:  # noqa: BLE001 - retry the same strict contract.
            last_error = exc
            debug.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = base_messages + [
                {
                    "role": "user",
                    "content": (
                        "Return the same JSON schema with each text field compressed "
                        f"to at most {MAX_EVOLUTION_QUERY_CHARS} characters. Select only a "
                        "Strategy Skill ID present in the supplied trigger catalog, or null. "
                        f"Validation feedback: {exc}"
                    ),
                }
            ]
    if output_path:
        write_json(output_path, {"failed": True, "attempts": debug})
    raise ValueError(f"Evolution query generation failed: {last_error}")


def run_reflector(
    *,
    trajectory_evidence: dict[str, Any],
    evolution_queries: dict[str, str],
    skill_search_context: dict[str, Any],
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
    skill_types: tuple[str, ...] | None = None,
    strict_final_cards: bool = False,
    isolate_skill_errors_on_final_attempt: bool = False,
) -> dict[str, Any]:
    payload = {
        "trajectory_evidence": trajectory_evidence,
        "evolution_queries": evolution_queries,
        "skill_search_context": skill_search_context,
    }
    base_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    messages = list(base_messages)
    debug: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        response = llm_client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
        )
        content = response.get("content") or ""
        try:
            result = validate_reflector_output(
                extract_json_object(content),
                trajectory_evidence=trajectory_evidence,
                skill_search_context=skill_search_context,
                skill_types=skill_types,
                strict_final_cards=strict_final_cards,
                isolate_skill_errors=(
                    isolate_skill_errors_on_final_attempt
                    and attempt == max(1, attempts)
                ),
            )
            if output_path:
                write_json(output_path, {"attempt": attempt, "reflector_output": result, "raw_response": content})
            return result
        except Exception as exc:  # noqa: BLE001 - one protocol repair is isolated here.
            last_error = exc
            debug.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = base_messages + [
                {
                    "role": "user",
                    "content": _reflector_repair_instruction(skill_types, exc),
                }
            ]
    fallback = _no_update_fallback(trajectory_evidence, last_error, skill_types=skill_types)
    if output_path:
        write_json(output_path, {"failed": True, "attempts": debug, "reflector_output": fallback})
    return fallback


def _reflector_repair_instruction(
    skill_types: tuple[str, ...] | None, error: Exception
) -> str:
    requested = tuple(skill_types or LEGACY_REFLECTOR_SKILL_TYPES)
    if requested == ("fault_skill",):
        contract = (
            "Return only skill_updates.fault_skill. create_new and rewrite_existing provide a "
            "complete card with fault_family, fault_subtype, title, trigger, and knowledge."
        )
    elif requested == ("strategy_skill",):
        contract = (
            "Return only skill_updates.strategy_skill. A create_new or rewrite_existing card "
            f"has exactly three knowledge statements at or below {FINAL_MAX_STATEMENT_CHARS} characters."
        )
    else:
        contract = (
            "Fault create_new or rewrite_existing provides a complete card. Project and Strategy "
            "create_new or rewrite_existing "
            f"cards provide exactly three knowledge statements at or below {FINAL_MAX_STATEMENT_CHARS} characters."
        )
    return (
        "Return a corrected JSON object using the exact schema. " + contract + " "
        "For create_new/no_update use target_skill_id null; for rewrite_existing/"
        "preserve_existing use an exact same-type candidate ID. "
        f"Validation feedback: {error}"
    )


def finalize_skill_cards(
    *,
    reflector_output: dict[str, Any],
    llm_client: Any,
    prompt: str,
    semantic_center_prompt: str | None = None,
    semantic_center_attempts: int = 3,
    source_terms: list[str] | None = None,
    attempts: int = 1,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rewrite materialized cards into portable role-level language.

    The Reflector decides whether a Skill should be created or rewritten. This
    narrowly-scoped pass only edits the complete card that will be persisted,
    keeping source-specific case vocabulary out of the runtime SkillBank. When
    ``semantic_center_prompt`` is provided, a small prior call distills the
    draft; the card writer then receives only that source-independent center.
    """

    result = deepcopy(reflector_output)
    debug: list[dict[str, Any]] = []
    retained_updates: list[dict[str, Any]] = []
    for update in result.get("materialized_updates", []) or []:
        if update.get("operation") not in {"create", "rewrite"}:
            retained_updates.append(update)
            continue
        skill_type = str(update.get("skill_type") or "")
        draft = update.get("skill")
        if skill_type not in SKILL_TYPES or not isinstance(draft, dict):
            retained_updates.append(update)
            continue

        # Concrete symbols are an output-safety check, not generation context.
        # Supplying a long list here caused the finalizer to recite source code
        # details while trying to "translate" each item, which made cards both
        # longer and less portable.
        card_source_terms = sorted(
            set(_source_symbol_terms(draft)).union(source_terms or []),
            key=lambda item: (-len(item), item),
        )[:48]

        semantic_center: dict[str, Any] | None = None
        semantic_center_debug: dict[str, Any] | None = None
        if semantic_center_prompt:
            try:
                semantic_center, semantic_center_debug = _derive_semantic_center(
                    skill_type=skill_type,
                    draft_skill=draft,
                    llm_client=llm_client,
                    prompt=semantic_center_prompt,
                    attempts=max(1, int(semantic_center_attempts)),
                )
            except Exception as exc:  # noqa: BLE001 - finalizer remains the persistence gate.
                debug.append(
                    {
                        "update_id": update.get("update_id"),
                        "skill_type": skill_type,
                        "semantic_center": None,
                        "semantic_center_error": str(exc),
                        "semantic_center_fallback": "draft_skill",
                    }
                )
                # The semantic center is a compact authoring aid, not a
                # correctness boundary.  A transient provider empty response
                # must not discard an otherwise useful strategy update.  The
                # finalizer still receives the draft, then enforces the same
                # portable-card schema, source-term checks, and retry policy
                # before anything can enter the SkillBank.
                semantic_center = None

        base_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "skill_type": skill_type,
                        **(
                            {
                                "semantic_center": {
                                    "semantic_center": semantic_center["semantic_center"],
                                    "roles": semantic_center["roles"],
                                    "contract": semantic_center["contract"],
                                }
                            }
                            if semantic_center is not None
                            else {"draft_skill": draft}
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        messages = list(base_messages)
        finalized: dict[str, Any] | None = None
        attempts_log: list[dict[str, Any]] = []
        for attempt in range(1, max(1, attempts) + 1):
            response = llm_client.chat(
                messages=messages,
                tool_choice="none",
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            content = response.get("content") or ""
            try:
                payload = extract_json_object(content)
                candidate = payload.get("skill", payload)
                validated = validate_portable_skill_card(candidate, skill_type=skill_type)
                remaining_terms = _remaining_source_terms(validated, card_source_terms)
                remaining_terms.extend(_portable_card_identifier_violations(validated))
                remaining_terms = sorted(set(remaining_terms), key=lambda item: (-len(item), item))
                if remaining_terms:
                    raise ValueError(
                        "Persisted card still contains source terms that need role translation: "
                        + ", ".join(remaining_terms)
                    )
                finalized = validated
                attempts_log.append(
                    {"attempt": attempt, "status": "accepted", "raw_response": content}
                )
                break
            except Exception as exc:  # noqa: BLE001 - preserve draft on isolated author failure.
                attempts_log.append(
                    {"attempt": attempt, "status": "rejected", "error": str(exc), "raw_response": content}
                )
                messages = base_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Return a complete corrected Skill JSON. Keep case implementation "
                            "names out and state only the smallest portable responsibility or "
                            "diagnostic contract. "
                            f"Validation feedback: {exc}"
                        ),
                    },
                ]

        debug.append(
            {
                "update_id": update.get("update_id"),
                "skill_type": skill_type,
                "semantic_center": semantic_center,
                "semantic_center_debug": semantic_center_debug,
                "attempts": attempts_log,
                "finalized": finalized is not None,
            }
        )
        if finalized is None:
            _replace_finalization_failure_with_no_update(
                result=result,
                skill_type=skill_type,
                reason="Skill card finalization could not produce a portable role-level card.",
            )
            continue
        update["skill"] = finalized
        skill_update = (result.get("skill_updates") or {}).get(skill_type)
        if isinstance(skill_update, dict):
            skill_update["skill"] = finalized
        retained_updates.append(update)

    result["materialized_updates"] = retained_updates
    result["card_finalization"] = debug
    if output_path:
        write_json(output_path, {"card_finalization": debug})
    return result


def _derive_semantic_center(
    *,
    skill_type: str,
    draft_skill: dict[str, Any],
    llm_client: Any,
    prompt: str,
    attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the small portable center used as the only card-writer input."""

    base_messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"skill_type": skill_type, "draft_skill": draft_skill},
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    messages = list(base_messages)
    attempt_log: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        response = llm_client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
            # gpt-5-mini may spend a substantial part of the visible output
            # budget on reasoning before emitting the tiny JSON object.  Keep
            # this aligned with card finalization so a valid center is not
            # truncated into an empty assistant content field.
            max_tokens=2048,
        )
        content = response.get("content") or ""
        try:
            center = _validate_semantic_center(extract_json_object(content), skill_type=skill_type)
            attempt_log.append({"attempt": attempt, "status": "accepted", "raw_response": content})
            return center, {"attempts": attempt_log}
        except Exception as exc:  # noqa: BLE001 - same strict schema retry.
            last_error = exc
            attempt_log.append(
                {"attempt": attempt, "status": "rejected", "error": str(exc), "raw_response": content}
            )
            messages = base_messages + [
                {
                    "role": "user",
                    "content": (
                        "Return the exact JSON schema. Express one portable role relationship and "
                        "contract, without repository identifiers or implementation inventories. "
                        f"Validation feedback: {exc}"
                    ),
                }
            ]
    raise ValueError(f"Semantic-center generation failed: {last_error}")


def _validate_semantic_center(raw: Any, *, skill_type: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Semantic center must be an object.")
    center = str(raw.get("semantic_center") or "").strip()
    contract = str(raw.get("contract") or "").strip()
    roles = raw.get("roles")
    if not center or not contract or not isinstance(roles, list):
        raise ValueError("Semantic center requires semantic_center, roles, and contract.")
    normalized_roles = list(
        dict.fromkeys(str(role).strip() for role in roles if isinstance(role, str) and role.strip())
    )
    if len(normalized_roles) != 2:
        raise ValueError("Semantic center requires exactly two roles.")
    if skill_type == "project_skill":
        max_center_chars, max_contract_chars = 120, 100
    else:
        # A Strategy center retains one causal contrast plus the evidence
        # question that separates the two candidate roles.
        max_center_chars, max_contract_chars = 220, 160
    if (
        len(center) > max_center_chars
        or len(contract) > max_contract_chars
        or any(len(role) > 90 for role in normalized_roles)
    ):
        raise ValueError("Semantic center exceeds its compactness budget.")
    normalized = {"semantic_center": center, "roles": normalized_roles, "contract": contract}
    if skill_type == "strategy_skill":
        normalized = _canonicalize_handoff_strategy_center(normalized)
    return normalized


def _canonicalize_handoff_strategy_center(center: dict[str, Any]) -> dict[str, Any]:
    """Collapse equivalent before/after handoff decisions into one card center.

    The semantic-center model is still responsible for deciding whether a case
    is a handoff decision at all.  Once it explicitly identifies a first
    contract violation across that handoff, retaining local role labels (for
    example, producer versus dispatcher) only fragments an otherwise identical
    diagnostic method into many Strategy cards.
    """

    text = " ".join(
        [
            str(center.get("semantic_center") or "").lower(),
            str(center.get("contract") or "").lower(),
        ]
    )
    is_handoff = "handoff" in text or "hand-off" in text
    identifies_origin = "first" in text and any(
        token in text for token in ("violation", "invalid", "incorrect", "contradict")
    )
    compares_sides = "before" in text and any(
        token in text for token in ("after", "introduced", "downstream", "later")
    )
    if not (is_handoff and identifies_origin and compares_sides):
        return center
    return {
        "semantic_center": "Determine where the first contract violation occurs across a responsibility handoff.",
        "roles": ["upstream responsibility boundary", "later responsibility boundary"],
        "contract": (
            "Compare observable state before and after the handoff; the first contradictory "
            "state determines the higher-ranked boundary."
        ),
    }


_CAPITALIZED_SOURCE_EXCLUSIONS = {
    "A",
    "An",
    "And",
    "Artifact",
    "Boundary",
    "Compare",
    "Consumers",
    "Comparison",
    "Contract",
    "Evidence",
    "For",
    "If",
    "Initialization",
    "Input",
    "Knowledge",
    "Ranking",
    "Rank",
    "Responsibility",
    "Skill",
    "Source",
    "State",
    "Strategy",
    "The",
    "This",
    "Use",
    "When",
}


def _source_symbol_terms(skill: dict[str, Any]) -> list[str]:
    """Find code-like vocabulary that a card author must restate as roles."""

    text = "\n".join(
        [
            str(skill.get("title") or ""),
            str(skill.get("trigger") or ""),
            *[str(item) for item in skill.get("knowledge") or []],
        ]
    )
    terms = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w+)+\b", text))
    terms.update(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", text))
    terms.update(re.findall(r"(?<!\w)--[A-Za-z][A-Za-z0-9-]*\b|(?<!\w)-[A-Za-z]\b", text))
    # Plain capitalized words (for example, "Sequence") are normal prose and
    # must not be treated as source identifiers. PascalCase names such as
    # NamedApi are much more likely to be repository vocabulary.
    terms.update(re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9_]*)+\b", text))
    return sorted(terms, key=lambda item: (-len(item), item))[:24]


def _remaining_source_terms(skill: dict[str, Any], source_terms: list[str]) -> list[str]:
    text = "\n".join(
        [
            str(skill.get("value") or ""),
            str(skill.get("title") or ""),
            str(skill.get("trigger") or ""),
            *[str(item) for item in skill.get("knowledge") or []],
        ]
    )
    normalized_text = re.sub(r"[^a-z0-9]+", "", text.lower())
    remaining: list[str] = []
    for term in source_terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE):
            remaining.append(term)
            continue
        # A model can evade a literal ``hook_manager.py`` check by writing
        # ``hook-manager``. Compound source symbols remain source vocabulary
        # after punctuation changes, so reject that alias as well.
        source_stem = re.sub(r"\.[a-z0-9]{1,5}$", "", term.lower())
        normalized_term = re.sub(r"[^a-z0-9]+", "", source_stem)
        if len(normalized_term) >= 8 and normalized_term in normalized_text:
            remaining.append(term)
    return remaining


def _portable_card_identifier_violations(skill: dict[str, Any]) -> list[str]:
    """Reject implementation-style names introduced during card finalization.

    ``value`` deliberately remains a snake_case semantic identifier.  The
    reader-facing fields must instead use role-level prose; otherwise a model
    can replace one source identifier with another and still evade the source
    term check derived from the draft.
    """

    text = "\n".join(
        [
            str(skill.get("title") or ""),
            str(skill.get("trigger") or ""),
            *[str(item) for item in skill.get("knowledge") or []],
        ]
    )
    violations = set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", text))
    violations.update(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w+)+\b", text))
    violations.update(re.findall(r"(?<!\w)--[A-Za-z][A-Za-z0-9-]*\b|(?<!\w)-[A-Za-z]\b", text))
    return sorted(violations, key=lambda item: (-len(item), item))


def _replace_finalization_failure_with_no_update(
    *, result: dict[str, Any], skill_type: str, reason: str
) -> None:
    skill_update = (result.get("skill_updates") or {}).get(skill_type)
    if isinstance(skill_update, dict):
        skill_update.update(
            {
                "decision": "no_update",
                "target_skill_id": None,
                "skill": None,
                "no_update_reason": reason,
            }
        )


def validate_reflector_output(
    payload: dict[str, Any],
    *,
    trajectory_evidence: dict[str, Any],
    skill_search_context: dict[str, Any],
    skill_types: tuple[str, ...] | None = None,
    strict_final_cards: bool = False,
    isolate_skill_errors: bool = False,
) -> dict[str, Any]:
    raw_updates = payload.get("skill_updates")
    if not isinstance(raw_updates, dict):
        raise ValueError("Reflector output missing skill_updates object.")
    outcome_type = str((trajectory_evidence.get("outcome") or {}).get("label") or "failure")
    source_case = str((trajectory_evidence.get("case") or {}).get("instance_id") or "") or None
    candidates = _candidate_skills_by_id(skill_search_context)
    normalized: dict[str, dict[str, Any]] = {}
    materialized: list[dict[str, Any]] = []
    protocol_errors: list[dict[str, str]] = []
    # Existing V2 callers deliberately keep their two-card schema. V3 passes
    # its explicit Fault/Strategy pair.
    for skill_type in (skill_types or LEGACY_REFLECTOR_SKILL_TYPES):
        raw = raw_updates.get(skill_type)
        try:
            if not isinstance(raw, dict):
                raise ValueError(f"skill_updates.{skill_type} must be an object.")
            update, applied = _validate_skill_update(
                skill_type=skill_type,
                raw_update=raw,
                candidate_skills=candidates,
                outcome_type=outcome_type,
                source_case=source_case,
                update_index=len(materialized) + 1,
                strict_final_cards=strict_final_cards,
            )
        except Exception as exc:
            if not isolate_skill_errors:
                raise
            reason = f"{skill_type} protocol failed: {exc}"
            update = _isolated_no_update(reason)
            applied = None
            protocol_errors.append({"skill_type": skill_type, "error": str(exc)})
        normalized[skill_type] = update
        if applied:
            materialized.append(applied)
    return {
        "case_summary": str(payload.get("case_summary") or ""),
        "outcome_type": outcome_type,
        "skill_updates": normalized,
        "materialized_updates": materialized,
        "protocol_errors": protocol_errors,
        "no_update_reason": (
            None if materialized else str(payload.get("no_update_reason") or _combined_no_update_reason(normalized))
        ),
    }


def _isolated_no_update(reason: str) -> dict[str, Any]:
    """Represent one invalid Skill update without discarding sibling updates."""

    return {
        "decision": "no_update",
        "target_skill_id": None,
        "rationale": reason,
        "skill": None,
        "no_update_reason": reason,
    }


def _validate_skill_update(
    *,
    skill_type: str,
    raw_update: dict[str, Any],
    candidate_skills: dict[str, dict[str, Any]],
    outcome_type: str,
    source_case: str | None,
    update_index: int,
    strict_final_cards: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decision = str(raw_update.get("decision") or "no_update").strip()
    if decision not in UPDATE_DECISIONS:
        raise ValueError(f"skill_updates.{skill_type}.decision is invalid: {decision!r}")
    target = str(raw_update.get("target_skill_id") or "").strip() or None
    rationale = str(raw_update.get("rationale") or "").strip()
    reason = str(raw_update.get("no_update_reason") or "").strip()
    raw_skill = raw_update.get("skill")

    if decision == "no_update":
        if target or raw_skill not in (None, {}):
            raise ValueError(f"skill_updates.{skill_type} no_update requires null target and skill.")
        if not reason and not rationale:
            raise ValueError(f"skill_updates.{skill_type} no_update requires a reason.")
        return {
            "decision": decision,
            "target_skill_id": None,
            "rationale": rationale,
            "skill": None,
            "no_update_reason": reason or rationale,
        }, None

    if not rationale:
        raise ValueError(f"skill_updates.{skill_type} requires rationale.")
    if decision == "create_new":
        if target:
            raise ValueError(f"skill_updates.{skill_type} create_new requires null target_skill_id.")
        skill = _validate_complete_skill(
            raw_skill, skill_type, strict_final_cards=strict_final_cards
        )
        operation = "create"
    else:
        if not target:
            raise ValueError(f"skill_updates.{skill_type}.{decision} requires target_skill_id.")
        candidate = candidate_skills.get(target)
        if candidate is None:
            raise ValueError(f"target_skill_id={target!r} was not retrieved for this evolution query.")
        if candidate.get("skill_type") != skill_type:
            raise ValueError("Skill updates must target a candidate of the same type.")
        if decision == "preserve_existing":
            if raw_skill not in (None, {}):
                raise ValueError("preserve_existing requires skill=null.")
            return {
                "decision": decision,
                "target_skill_id": target,
                "rationale": rationale,
                "skill": None,
                "no_update_reason": None,
            }, _materialize_update(
                update_index, "preserve", skill_type, target, None, rationale, outcome_type, source_case
            )
        skill = _validate_complete_skill(raw_skill, skill_type, strict_final_cards=strict_final_cards)
        operation = "rewrite"

    normalized = {
        "decision": decision,
        "target_skill_id": target,
        "rationale": rationale,
        "skill": skill,
        "no_update_reason": None,
    }
    return normalized, _materialize_update(
        update_index, operation, skill_type, target, skill, rationale, outcome_type, source_case
    )


def _validate_complete_skill(
    raw: Any, skill_type: str, *, strict_final_cards: bool = False
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"skill_updates.{skill_type}.skill must be a complete skill object.")
    if skill_type == "fault_skill":
        skill = {
            "fault_family": validate_fault_family(raw.get("fault_family")),
            "fault_subtype": str(raw.get("fault_subtype") or "").strip(),
            "title": str(raw.get("title") or "").strip(),
            "trigger": str(raw.get("trigger") or "").strip(),
            "knowledge": validate_fault_knowledge(raw.get("knowledge")),
        }
        required = ("fault_family", "fault_subtype", "title", "trigger", "knowledge")
    else:
        skill = {
            "value": str(raw.get("value") or "").strip(),
            "title": str(raw.get("title") or "").strip(),
            "trigger": str(raw.get("trigger") or "").strip(),
            "knowledge": validate_atomic_knowledge(raw.get("knowledge")),
        }
        required = ("value", "title", "trigger", "knowledge")
    missing = [key for key in required if not skill[key]]
    if missing:
        raise ValueError(f"skill_updates.{skill_type}.skill missing: {', '.join(missing)}")
    if strict_final_cards:
        # V3 persists runtime cards directly. Enforce the embedding-card
        # contract before an update can reach the active bank.
        skill = validate_portable_skill_card(skill, skill_type=skill_type)
    return skill


def _materialize_update(
    index: int,
    operation: str,
    skill_type: str,
    target: str | None,
    skill: dict[str, Any] | None,
    rationale: str,
    outcome_type: str,
    source_case: str | None,
) -> dict[str, Any]:
    return {
        "update_id": f"update_{index:03d}_{skill_type}",
        "operation": operation,
        "skill_type": skill_type,
        "target_skill_id": target,
        "skill": skill,
        "rationale": rationale,
        "outcome_type": outcome_type,
        "source_cases": [source_case] if source_case else [],
    }


def _candidate_skills_by_id(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for skill in context.get("candidate_target_skills", []) or []:
        if isinstance(skill, dict) and skill.get("skill_id"):
            candidates[str(skill["skill_id"])] = skill
    for values in (context.get("candidates_by_skill_type") or {}).values():
        for skill in values or []:
            if isinstance(skill, dict) and skill.get("skill_id"):
                candidates[str(skill["skill_id"])] = skill
    for skill in context.get("runtime_loaded_skills", []) or []:
        if isinstance(skill, dict) and skill.get("skill_id"):
            candidates[str(skill["skill_id"])] = skill
    return candidates


def _combined_no_update_reason(updates: dict[str, dict[str, Any]]) -> str:
    return "; ".join(
        f"{skill_type}: {update.get('no_update_reason') or update.get('rationale')}"
        for skill_type, update in updates.items()
        if update.get("decision") == "no_update"
    ) or "No reusable update was selected."


def _no_update_fallback(
    trajectory_evidence: dict[str, Any],
    error: Exception | None,
    *,
    skill_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    reason = f"Reflector protocol failed: {error}" if error else "Reflector protocol failed."
    updates = {
        skill_type: {
            "decision": "no_update",
            "target_skill_id": None,
            "rationale": reason,
            "skill": None,
            "no_update_reason": reason,
        }
        for skill_type in (skill_types or SKILL_TYPES)
    }
    return {
        "case_summary": "No SkillBank update was applied.",
        "outcome_type": str((trajectory_evidence.get("outcome") or {}).get("label") or "failure"),
        "skill_updates": updates,
        "materialized_updates": [],
        "no_update_reason": reason,
    }
