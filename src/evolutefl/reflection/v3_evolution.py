"""V3 evolution separates static Project knowledge from case-derived lessons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.evaluation import evaluate_ranked_functions
from evolutefl.evaluation.patch_ground_truth import split_function_identity
from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from evolutefl.skills.fault_taxonomy import compact_fault_taxonomy, validate_fault_family
from evolutefl.skills.schema import (
    normalize_repo_id,
    validate_portable_skill_card,
    validate_project_add,
    validate_project_create,
)

from .reflector import run_reflector
from .trajectory_compactor import build_compacted_trajectory


V3_REQUIRED_EVENTS = {
    "project_skill_request", "project_skill_loaded", "issue_revealed",
    "fault_skill_request", "fault_skill_loaded", "strategy_skill_request", "strategy_skill_loaded",
}


def run_v3_case_evolution(**kwargs: Any) -> dict[str, Any]:
    """Run a static Project builder plus case-derived Fault/Strategy reflection.

    Project creation is deliberately isolated from issue, outcome, ranked
    candidates and patch material. Fault and Strategy are the only case
    feedback updates, so a failure cannot silently rewrite static knowledge.
    """
    from .case_evolution import _eligibility, _label_outcome, _read_trajectory

    case_dir = Path(kwargs["case_run_dir"])
    out_dir = Path(kwargs.get("output_dir") or case_dir / "case_evolution")
    out_dir.mkdir(parents=True, exist_ok=True)
    config, repo, issue = kwargs["config"], str(kwargs["repo"]), str(kwargs["issue"])
    result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    trajectory = _read_trajectory(case_dir / "trajectory.jsonl")
    events = {str(item.get("event") or "") for item in trajectory}
    missing = sorted(V3_REQUIRED_EVENTS - events)
    stage_state = {"compatible": not missing, "missing_events": missing,
                   **{kind: _stage_state(trajectory, kind) for kind in ("project_skill", "fault_skill", "strategy_skill")}}
    ground_truth = kwargs.get("ground_truth_functions") or []
    outcome = _label_outcome(result, result.get("ranked_functions") or [], ground_truth, bool(kwargs.get("force")))
    # V3 learns Fault guidance from every completed case: hits reinforce a
    # successful candidate-formation path, while misses capture an omitted
    # investigation direction. Non-completed runs remain ineligible.
    eligible, reason = _eligibility(
        result,
        outcome,
        ground_truth,
        bool(kwargs.get("force")),
        reflect_success=True,
        reflect_failure=True,
    )
    if not stage_state["compatible"]:
        eligible, reason = False, "incompatible V3 trajectory: " + ", ".join(missing)
    client = kwargs.get("llm_client") or OpenAICompatibleClient.from_config(config.get("llm", {}))
    bank = make_skill_bank(config)
    fault_only = bool(kwargs.get("fault_only"))
    project_result = (
        {
            "decision": "preserve_existing",
            "reason": "Skipped during Fault-only trajectory replay.",
            "applied_updates": [],
        }
        if fault_only
        else _run_project_builder(
            bank,
            client,
            repo,
            trajectory,
            out_dir,
            config,
            repository_manifest=_read_json_if_present(case_dir / "repository_manifest.json"),
            completed=str(result.get("status") or "") == "completed",
        )
    )
    project_applied_updates = list(project_result.get("applied_updates", []))
    summary: dict[str, Any] = {"eligible": eligible, "reason": reason, "repo": repo, "outcome": outcome,
                               "stage_state": stage_state, "project_knowledge": project_result,
                               "applied_updates": list(project_applied_updates), "updated_skill_ids": []}
    if not eligible:
        write_json(out_dir / "case_evolution_summary.json", summary)
        return summary
    evidence = {
        "case": {"instance_id": result.get("instance_id") or case_dir.name, "repo": repo},
        "problem_statement": issue, "outcome": outcome,
        "prediction": {"ranked_functions": result.get("ranked_functions", []), "final_summary": result.get("final_summary", "")},
        "ground_truth": {"functions": ground_truth, "locations": kwargs.get("ground_truth_locations") or [],
                           "patch_context": str(kwargs.get("ground_truth_patch") or "")[:12000]},
        "stage_state": stage_state, **build_compacted_trajectory(trajectory),
    }
    write_json(out_dir / "trajectory_evidence.json", evidence)
    queries = _generate_case_queries(client, evidence, bank.strategy_catalog(), out_dir, config)
    fault_catalog = bank.fault_catalog(queries["fault_family"])
    fault_target = _select_fault_evolution_target(
        client=client,
        fault_family=queries["fault_family"],
        fault_subtype_query=queries["fault_subtype_query"],
        catalog=fault_catalog.get("candidate_skills") or [],
        out_dir=out_dir,
        config=config,
    )
    context = bank.search_for_v3_evolution(
        fault_family=queries["fault_family"],
        selected_fault_skill_id=fault_target.get("selected_skill_id"),
        selected_strategy_skill_id=queries.get("selected_strategy_skill_id"),
        strategy_diagnostic_context=queries["strategy_diagnostic_context"],
        limit_per_type=int((config.get("reflection") or {}).get("reflector_candidate_top_k", 5)))
    context["fault_catalog_selection"] = fault_target
    write_json(out_dir / "reflector_skill_search_context.json", context)
    reflection_config = config.get("reflection") or {}
    learning_signals = _reflection_learning_signals(
        result.get("ranked_functions") or [], ground_truth
    )
    fault_evidence = _build_skill_reflection_view(
        evidence=evidence, trajectory=trajectory, skill_type="fault_skill",
        learning_signal=learning_signals["fault_skill"],
    )
    strategy_evidence = _build_skill_reflection_view(
        evidence=evidence, trajectory=trajectory, skill_type="strategy_skill",
        learning_signal=learning_signals["strategy_skill"],
    )
    fault_context = _skill_type_context(context, "fault_skill")
    strategy_context = _skill_type_context(context, "strategy_skill")
    write_json(out_dir / "fault_reflection_view.json", fault_evidence)
    write_json(out_dir / "strategy_reflection_view.json", strategy_evidence)

    fault_prompt_key = (
        "fault_success_reflector_prompt_path"
        if learning_signals["fault_skill"]["label"] == "success"
        else "fault_failure_reflector_prompt_path"
    )
    fault_prompt_default = (
        "prompt_records/reflection/fault_success_reflector_v1.txt"
        if learning_signals["fault_skill"]["label"] == "success"
        else "prompt_records/reflection/fault_failure_reflector_v1.txt"
    )
    fault_prompt = resolve_path(reflection_config.get(
        fault_prompt_key, fault_prompt_default
    )).read_text(encoding="utf-8")
    strategy_prompt = resolve_path(reflection_config.get(
        "strategy_reflector_prompt_path", "prompt_records/reflection/strategy_reflector_v3.txt"
    )).read_text(encoding="utf-8")
    reflector_attempts = int(reflection_config.get("reflector_attempts", 2))
    fault_output = run_reflector(
        trajectory_evidence=fault_evidence,
        evolution_queries={
            "fault_family": queries["fault_family"],
            "fault_subtype_query": queries["fault_subtype_query"],
        },
        skill_search_context=fault_context, llm_client=client, prompt=fault_prompt,
        attempts=reflector_attempts, output_path=out_dir / "fault_reflector_debug.json",
        skill_types=("fault_skill",), strict_final_cards=True,
        isolate_skill_errors_on_final_attempt=True,
    )
    write_json(out_dir / "fault_reflector_output.json", fault_output)
    strategy_output = (
        _skipped_reflector_output(
            "strategy_skill", "Skipped during Fault-only trajectory replay."
        )
        if fault_only
        else run_reflector(
            trajectory_evidence=strategy_evidence,
            evolution_queries={
                "strategy_diagnostic_context": queries["strategy_diagnostic_context"],
                "selected_strategy_skill_id": queries.get("selected_strategy_skill_id"),
                "strategy_selection_reason": queries.get("strategy_selection_reason", ""),
            },
            skill_search_context=strategy_context, llm_client=client, prompt=strategy_prompt,
            attempts=reflector_attempts, output_path=out_dir / "strategy_reflector_debug.json",
            skill_types=("strategy_skill",), strict_final_cards=True,
            isolate_skill_errors_on_final_attempt=True,
        )
    )
    write_json(out_dir / "strategy_reflector_output.json", strategy_output)
    output = _merge_reflector_outputs(fault_output, strategy_output, learning_signals)
    write_json(out_dir / "reflector_output.json", output)
    applied, applied_materialized, failed = [], [], []
    for update in output.get("materialized_updates", []):
        try:
            result = bank.apply_update(update)
            applied.append(result)
            applied_materialized.append({
                "operation": _report_operation(result.get("action")),
                "skill_type": update.get("skill_type"),
                "target_skill_id": update.get("target_skill_id"),
                "updated_skill_id": result.get("updated_skill_id"),
            })
        except Exception as exc:  # per-card isolation
            failed.append({"update_id": update.get("update_id"), "error": str(exc)})
    write_json(out_dir / "applied_updates.json", applied)
    write_json(out_dir / "failed_updates.json", failed)
    summary["applied_updates"].extend(applied)
    runtime_fault_family = str(
        ((stage_state.get("fault_skill") or {}).get("request") or {}).get("request", {}).get("fault_family")
        or ""
    )
    summary.update({"evolution_queries": queries, "learning_signals": learning_signals,
                    "fault_only": fault_only,
                    "runtime_fault_family": runtime_fault_family or None,
                    "reflector_fault_family": queries["fault_family"],
                    "fault_family_consistent": runtime_fault_family == queries["fault_family"],
                    "fault_target_selection": fault_target,
                    "skill_updates": output.get("skill_updates"),
                    "updated_skill_ids": [item.get("updated_skill_id") for item in summary["applied_updates"] if item.get("updated_skill_id")],
                    "failed_updates": failed})
    materialized_updates = [
        {
            "operation": _report_operation(item.get("action")),
            "skill_type": "project_skill",
            "target_skill_id": None,
            "updated_skill_id": item.get("updated_skill_id"),
        }
        for item in project_applied_updates
    ] + applied_materialized
    summary["materialized_updates"] = materialized_updates
    summary["updated_skill_types"] = sorted({
        str(update.get("skill_type")) for update in materialized_updates
        if update.get("skill_type")
    })
    write_json(out_dir / "case_evolution_summary.json", summary)
    return summary


def _skipped_reflector_output(skill_type: str, reason: str) -> dict[str, Any]:
    """Return a merge-compatible no-op result for an intentionally skipped call."""

    return {
        "case_summary": reason,
        "skill_updates": {skill_type: None},
        "materialized_updates": [],
        "protocol_errors": [],
        "no_update_reason": reason,
    }


def _reflection_learning_signals(
    ranked_functions: list[str], ground_truth_functions: list[str]
) -> dict[str, dict[str, Any]]:
    """Separate candidate-formation learning from final-ranking learning."""

    metrics = evaluate_ranked_functions(ranked_functions, ground_truth_functions)
    rank = metrics.get("rank")
    fault_signal = {
        "label": "success" if metrics.get("top5") else "failure",
        "target_rank": rank,
        "reason": (
            "The ground-truth function entered the final Top-5 candidate set."
            if metrics.get("top5")
            else "The ground-truth function never entered the final Top-5 candidate set."
        ),
        "learning_scope": "fault-guided candidate formation",
    }
    granularity_matches = _enclosing_symbol_matches(
        ranked_functions, ground_truth_functions
    )
    if rank == 1:
        strategy_label = "success"
        strategy_reason = "The ground-truth function was ranked first."
    elif isinstance(rank, int) and rank <= 5:
        strategy_label = "failure"
        strategy_reason = "The ground-truth function was found but ranked below another candidate."
    elif granularity_matches:
        strategy_label = "granularity_mismatch"
        strategy_reason = (
            "The final candidates identified an enclosing symbol in the correct file, but did not "
            "refine it to the ground-truth function."
        )
    else:
        strategy_label = "not_applicable"
        strategy_reason = (
            "The ground-truth function was absent from Top-5, so final re-ranking cannot repair "
            "candidate formation."
        )
    return {
        "fault_skill": fault_signal,
        "strategy_skill": {
            "label": strategy_label, "target_rank": rank, "reason": strategy_reason,
            "learning_scope": (
                "candidate refinement from an enclosing symbol to a responsible function"
                if strategy_label == "granularity_mismatch"
                else "candidate comparison and final ranking"
            ),
            "granularity_matches": granularity_matches if strategy_label == "granularity_mismatch" else [],
        },
    }


def _enclosing_symbol_matches(
    ranked_functions: list[str], ground_truth_functions: list[str],
) -> list[dict[str, str]]:
    """Find class-or-container candidates that enclose an omitted target function."""

    matches: list[dict[str, str]] = []
    for prediction in ranked_functions[:5]:
        prediction_file, prediction_symbol = split_function_identity(prediction)
        if not prediction_symbol:
            continue
        for ground_truth in ground_truth_functions:
            ground_truth_file, ground_truth_symbol = split_function_identity(ground_truth)
            if prediction_file != ground_truth_file:
                continue
            if ground_truth_symbol.startswith(f"{prediction_symbol}."):
                matches.append({
                    "enclosing_candidate": prediction,
                    "ground_truth_function": ground_truth,
                })
    return matches


def _build_skill_reflection_view(
    *, evidence: dict[str, Any], trajectory: list[dict[str, Any]],
    skill_type: str, learning_signal: dict[str, Any],
) -> dict[str, Any]:
    """Build a stage-bounded evidence view for one independent Reflector."""

    if skill_type == "fault_skill":
        phase_events = _events_between(
            trajectory, start_event="issue_revealed", stop_event="strategy_skill_request"
        )
        stage_state = {
            name: (evidence.get("stage_state") or {}).get(name)
            for name in ("project_skill", "fault_skill")
        }
        scope = "Fault-guided exploration and formation of the final candidate set."
    else:
        is_granularity_mismatch = str(learning_signal.get("label") or "") == "granularity_mismatch"
        phase_events = _events_between(
            trajectory,
            start_event="issue_revealed" if is_granularity_mismatch else "strategy_skill_request",
            stop_event=None,
        )
        stage_state = {
            "strategy_skill": (evidence.get("stage_state") or {}).get("strategy_skill")
        }
        scope = (
            "Refinement from an observed enclosing symbol to the responsible function."
            if is_granularity_mismatch
            else "Comparison of concrete candidates and their final ordering."
        )
    view = {
        "case": evidence.get("case") or {},
        "problem_statement": evidence.get("problem_statement", ""),
        "outcome": learning_signal,
        "reflection_scope": scope,
        "prediction": evidence.get("prediction") or {},
        "ground_truth": evidence.get("ground_truth") or {},
        "stage_state": stage_state,
    }
    view.update(build_compacted_trajectory(phase_events, head_events=10, tail_events=14))
    return view


def _events_between(
    trajectory: list[dict[str, Any]], *, start_event: str, stop_event: str | None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    active = False
    finish_events: list[dict[str, Any]] = []
    for event in trajectory:
        kind = str(event.get("event") or "")
        if kind == "finish":
            finish_events.append(event)
        if kind == start_event:
            active = True
        if active and stop_event and kind == stop_event:
            break
        if active:
            selected.append(event)
    if stop_event:
        selected.extend(event for event in finish_events if event not in selected)
    return selected


def _skill_type_context(context: dict[str, Any], skill_type: str) -> dict[str, Any]:
    """Expose only same-type candidates to one Reflector call."""

    candidates = list((context.get("candidates_by_skill_type") or {}).get(skill_type) or [])
    result = {
        "queries": {skill_type: (context.get("queries") or {}).get(skill_type)},
        "candidates_by_skill_type": {skill_type: candidates},
        "candidate_target_skills": candidates,
        "search_traces": {
            skill_type: (context.get("search_traces") or {}).get(skill_type) or {}
        },
    }
    if skill_type == "strategy_skill":
        result["selected_strategy_skill_id"] = context.get("selected_strategy_skill_id")
    return result


def _merge_reflector_outputs(
    fault_output: dict[str, Any], strategy_output: dict[str, Any],
    learning_signals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Keep the historical combined artifact while preserving call isolation."""

    materialized = [
        *(fault_output.get("materialized_updates") or []),
        *(strategy_output.get("materialized_updates") or []),
    ]
    reasons = [
        str(value.get("no_update_reason") or "").strip()
        for value in (fault_output, strategy_output) if value.get("no_update_reason")
    ]
    return {
        "case_summary": " | ".join(filter(None, [
            str(fault_output.get("case_summary") or "").strip(),
            str(strategy_output.get("case_summary") or "").strip(),
        ])),
        "outcome_type": str((learning_signals.get("fault_skill") or {}).get("label") or "failure"),
        "learning_signals": learning_signals,
        "skill_updates": {
            "fault_skill": (fault_output.get("skill_updates") or {}).get("fault_skill"),
            "strategy_skill": (strategy_output.get("skill_updates") or {}).get("strategy_skill"),
        },
        "materialized_updates": materialized,
        "protocol_errors": [
            *(fault_output.get("protocol_errors") or []),
            *(strategy_output.get("protocol_errors") or []),
        ],
        "no_update_reason": None if materialized else "; ".join(reasons) or "No reusable update was selected.",
        "reflection_runs": {"fault_skill": fault_output, "strategy_skill": strategy_output},
    }


def _report_operation(action: Any) -> str:
    """Normalize persisted SkillBank actions for experiment reporting."""

    return {
        "create_new": "create",
        "rewrite": "rewrite",
        "preserve_existing": "preserve",
        "no_update": "no_update",
    }.get(str(action or ""), str(action or "unknown"))


def _stage_state(trajectory: list[dict[str, Any]], skill_type: str) -> dict[str, Any]:
    request = next((event for event in trajectory if event.get("event") == f"{skill_type}_request"), None)
    loaded = next((event for event in trajectory if event.get("event") == f"{skill_type}_loaded"), None)
    return {"attempted": bool(request and loaded), "request": request, "loaded_skill_id": (loaded or {}).get("loaded_skill_id")}


def _run_project_builder(
    bank: Any,
    client: Any,
    repo: str,
    trajectory: list[dict[str, Any]],
    out_dir: Path,
    config: dict[str, Any],
    *,
    repository_manifest: dict[str, Any],
    completed: bool,
) -> dict[str, Any]:
    """Build static Project knowledge without exposing case outcome material."""
    if not completed:
        return {"decision": "no_update", "reason": "Explorer did not complete.", "applied_updates": []}
    # Static Project knowledge may use only events before the issue is exposed.
    # Later assistant/tool messages often quote the issue and would otherwise
    # leak failure-specific information into a supposedly repository-only card.
    orientation: list[dict[str, Any]] = []
    for event in trajectory:
        if event.get("event") == "issue_revealed":
            break
        if str(event.get("event")) in {
            "assistant_tool_calls", "tool_result", "project_skill_request", "project_skill_loaded"
        }:
            orientation.append(event)
    repo_id = normalize_repo_id(repo)
    current_project_skill = bank.get_project_skill(repo_id)
    evidence = {
        "repo_id": repo_id,
        "repository_manifest": repository_manifest,
        "repository_orientation": build_compacted_trajectory(orientation),
        "current_project_skill": current_project_skill,
    }
    write_json(out_dir / "project_knowledge_evidence.json", evidence)
    prompt_path = resolve_path((config.get("reflection") or {}).get("project_builder_prompt_path", "prompt_records/reflection/project_knowledge_builder_v3.txt"))
    base_messages = [
        {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
    ]
    messages = list(base_messages)
    attempts_log: list[dict[str, Any]] = []
    attempts = max(1, int((config.get("reflection") or {}).get("project_builder_attempts", 2)))
    for attempt in range(1, attempts + 1):
        response = client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1024,
        )
        content = response.get("content") or ""
        try:
            raw = extract_json_object(content)
            decision = str(raw.get("decision") or "no_update")
            if decision == "no_update":
                write_json(out_dir / "project_knowledge_output.json", {
                    "attempt": attempt, "project_builder_output": raw, "attempts": attempts_log,
                })
                return {
                    "decision": decision,
                    "reason": str(raw.get("reason") or ""),
                    "applied_updates": [],
                }
            if decision == "create_new":
                if current_project_skill is not None:
                    raise ValueError("create_new is invalid because this repository already has a Project Skill.")
                card = validate_project_create(raw.get("skill"), expected_repo_id=repo_id)
                update = {"decision": "create_new", "repo_id": repo_id, "skill": card}
            elif decision == "add":
                if current_project_skill is None:
                    raise ValueError("add is invalid because this repository has no Project Skill yet.")
                if normalize_repo_id(raw.get("repo_id") or repo_id) != repo_id:
                    raise ValueError("add repo_id must match the current repository.")
                update = {
                    "decision": "add",
                    "repo_id": repo_id,
                    "knowledge_to_add": validate_project_add(raw.get("knowledge_to_add")),
                }
            else:
                raise ValueError(f"Unsupported Project decision: {decision!r}.")
            applied = bank.apply_project_update(update)
            attempts_log.append({"attempt": attempt, "status": "accepted", "raw_response": content})
            write_json(out_dir / "project_knowledge_output.json", {
                "attempt": attempt, "project_builder_output": raw, "attempts": attempts_log,
            })
            return {"decision": decision, "applied_updates": [applied]}
        except Exception as exc:  # Retry the same static evidence and schema.
            attempts_log.append({"attempt": attempt, "status": "rejected", "error": str(exc), "raw_response": content})
            messages = base_messages + [{
                "role": "user",
                "content": (
                    "Return one valid Project decision using the independent create_new/add/no_update "
                    "schema. Keep repo_id equal to the input repo_id and use atomic static facts. "
                    f"Validation feedback: {exc}"
                ),
            }]
    write_json(out_dir / "project_knowledge_output.json", {"failed": True, "attempts": attempts_log})
    return {
        "decision": "no_update",
        "reason": "Project Builder could not produce a valid repository knowledge update.",
        "applied_updates": [],
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _generate_case_queries(client: Any, evidence: dict[str, Any], catalog: list[dict[str, Any]], out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    prompt_path = resolve_path((config.get("reflection") or {}).get("evolution_query_prompt_path", "prompt_records/reflection/evolution_query_v3.txt"))
    base_messages = [
        {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps({
            "trajectory_evidence": evidence,
            "fault_taxonomy": compact_fault_taxonomy(),
            "strategy_skill_catalog": catalog,
        }, ensure_ascii=False)},
    ]
    messages = list(base_messages)
    errors: list[dict[str, Any]] = []
    attempts = max(1, int((config.get("reflection") or {}).get("query_attempts", 2)))
    for attempt in range(1, attempts + 1):
        response = client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            result = extract_json_object(content)
            result["fault_family"] = validate_fault_family(result.get("fault_family"))
            for key in ("fault_subtype_query", "strategy_diagnostic_context"):
                if not str(result.get(key) or "").strip():
                    raise ValueError(f"V3 evolution query missing {key}.")
            write_json(out_dir / "evolution_queries.json", {
                "attempt": attempt, "evolution_queries": result, "raw_response": content,
            })
            return result
        except Exception as exc:  # Same evidence and schema; no input rewriting.
            errors.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = base_messages + [{
                "role": "user",
                "content": (
                    "Return the exact requested JSON object with one valid fault_family and "
                    "non-empty fault_subtype_query and strategy_diagnostic_context."
                ),
            }]
    write_json(out_dir / "evolution_queries.json", {"failed": True, "attempts": errors})
    raise ValueError(f"V3 evolution query generation failed: {errors[-1]['error']}")


def _select_fault_evolution_target(
    *,
    client: Any,
    fault_family: str,
    fault_subtype_query: str,
    catalog: list[dict[str, Any]],
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select one same-family update target without exposing catalog knowledge."""

    family = validate_fault_family(fault_family)
    catalog_ids = {str(item.get("skill_id") or "") for item in catalog}
    catalog_json = json.dumps(catalog, ensure_ascii=False)
    if not catalog:
        result = {
            "selected_skill_id": None,
            "reason": "No active Fault Skill exists in the classified family.",
            "fault_family": family,
            "catalog_count": 0,
            "catalog_char_count": 2,
        }
        write_json(out_dir / "fault_skill_target_selection.json", result)
        return result
    reflection_config = config.get("reflection") or {}
    prompt_path = resolve_path(reflection_config.get(
        "fault_target_selector_prompt_path",
        "prompt_records/reflection/fault_skill_target_selector_v1.txt",
    ))
    payload = {
        "fault_family": family,
        "fault_subtype_query": str(fault_subtype_query or "").strip(),
        "candidates": catalog,
    }
    base_messages = [
        {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    attempts = max(1, int(reflection_config.get("fault_target_selector_attempts", 2)))
    errors: list[dict[str, Any]] = []
    messages = list(base_messages)
    for attempt in range(1, attempts + 1):
        response = client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.get("content") or ""
        try:
            selected = extract_json_object(content)
            selected_id = str(selected.get("selected_skill_id") or "").strip() or None
            if selected_id is not None and selected_id not in catalog_ids:
                raise ValueError("selected_skill_id must be from the supplied family catalog or null.")
            result = {
                "selected_skill_id": selected_id,
                "reason": str(selected.get("reason") or "").strip(),
                "fault_family": family,
                "catalog_count": len(catalog),
                "catalog_char_count": len(catalog_json),
                "selector_input_char_count": len(json.dumps(payload, ensure_ascii=False)),
                "attempt": attempt,
            }
            write_json(out_dir / "fault_skill_target_selection.json", result)
            return result
        except Exception as exc:  # retry the same compact directory
            errors.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = base_messages + [{
                "role": "user",
                "content": "Return selected_skill_id from the supplied catalog or null, plus a short reason.",
            }]
    result = {
        "selected_skill_id": None,
        "reason": "Fault target selector failed; no existing Skill is safe to rewrite.",
        "fault_family": family,
        "catalog_count": len(catalog),
        "catalog_char_count": len(catalog_json),
        "attempts": errors,
    }
    write_json(out_dir / "fault_skill_target_selection.json", result)
    return result
