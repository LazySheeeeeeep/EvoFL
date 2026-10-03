from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.evaluation import evaluate_ranked_functions
from evolutefl.json_utils import read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank

from .reflector import finalize_skill_cards, generate_evolution_queries, run_reflector
from .trajectory_compactor import build_compacted_trajectory


REQUIRED_STAGE_EVENTS = {
    "project_skill_request",
    "project_skill_loaded",
    "strategy_skill_request",
    "strategy_skill_loaded",
}

_GENERIC_EVIDENCE_TERMS = {
    "A",
    "An",
    "And",
    "As",
    "Compare",
    "Consumer",
    "Consumers",
    "Data",
    "Expected",
    "For",
    "If",
    "Inspect",
    "Rank",
    "The",
    "This",
    "Use",
    "When",
}


def run_case_evolution(
    *,
    case_run_dir: str | Path,
    repo: str,
    issue: str,
    config: dict[str, Any],
    llm_client: Any | None = None,
    force: bool = False,
    ground_truth_patch: str = "",
    ground_truth_functions: list[str] | None = None,
    ground_truth_locations: list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    reflect_success: bool = False,
    reflect_failure: bool = True,
    fault_only: bool = False,
    repo_path: str | Path | None = None,
    patch_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if str((config.get("explorer") or {}).get("workflow_version") or "").lower() == "v5":
        from .v5_evolution import run_v5_case_evolution

        return run_v5_case_evolution(case_run_dir=case_run_dir, repo=repo, issue=issue, config=config,
            llm_client=llm_client, repo_path=repo_path, patch_metadata=patch_metadata,
            ground_truth_patch=ground_truth_patch, ground_truth_functions=ground_truth_functions,
            ground_truth_locations=ground_truth_locations, output_dir=output_dir)
    if str((config.get("explorer") or {}).get("workflow_version") or "v2").lower() == "v3":
        from .v3_evolution import run_v3_case_evolution

        return run_v3_case_evolution(
            case_run_dir=case_run_dir, repo=repo, issue=issue, config=config, llm_client=llm_client,
            force=force, ground_truth_patch=ground_truth_patch, ground_truth_functions=ground_truth_functions,
            ground_truth_locations=ground_truth_locations, output_dir=output_dir,
            reflect_success=reflect_success, reflect_failure=reflect_failure,
            fault_only=fault_only,
        )
    case_dir = Path(case_run_dir)
    out_dir = Path(output_dir) if output_dir else case_dir / "case_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    trajectory_path = case_dir / "trajectory.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Explorer result.json not found: {result_path}")

    result = read_json(result_path)
    trajectory = _read_trajectory(trajectory_path)
    ranked = result.get("ranked_functions", []) or []
    ground_truth_functions = ground_truth_functions or []
    outcome = _label_outcome(result, ranked, ground_truth_functions, force)
    eligible, reason = _eligibility(
        result,
        outcome,
        ground_truth_functions,
        force,
        reflect_success=reflect_success,
        reflect_failure=reflect_failure,
    )
    stage_state = _stage_state(trajectory)
    if eligible and not stage_state["compatible"]:
        eligible = False
        reason = "incompatible trajectory: missing staged skill request/load events"

    summary: dict[str, Any] = {
        "eligible": eligible,
        "reason": reason,
        "case_run_dir": str(case_dir),
        "repo": repo,
        "outcome": outcome,
        "stage_state": stage_state,
        "applied_updates": [],
        "updated_skill_ids": [],
    }
    if not eligible:
        write_json(out_dir / "case_evolution_summary.json", summary)
        return summary

    reflection_cfg = config.get("reflection", {}) or {}
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))
    bank = make_skill_bank(config)
    trajectory_evidence = _build_trajectory_evidence(
        case_dir=case_dir,
        repo=repo,
        issue=issue,
        result=result,
        trajectory=trajectory,
        outcome=outcome,
        ground_truth_functions=ground_truth_functions,
        ground_truth_locations=ground_truth_locations or [],
        ground_truth_patch=ground_truth_patch,
        stage_state=stage_state,
    )
    write_json(out_dir / "trajectory_evidence.json", trajectory_evidence)

    query_prompt = resolve_path(
        reflection_cfg.get(
            "evolution_query_prompt_path",
            "prompt_records/reflection/evolution_query_v0.txt",
        )
    ).read_text(encoding="utf-8")
    strategy_catalog = bank.strategy_catalog()
    evolution_queries = generate_evolution_queries(
        trajectory_evidence=trajectory_evidence,
        strategy_catalog=strategy_catalog,
        llm_client=client,
        prompt=query_prompt,
        attempts=int(reflection_cfg.get("query_attempts", 2)),
        output_path=out_dir / "evolution_queries_debug.json",
    )
    write_json(out_dir / "evolution_queries.json", evolution_queries)

    skill_search_context = bank.search_for_evolution(
        project_skill_query=evolution_queries["project_skill_query"],
        selected_strategy_skill_id=evolution_queries["selected_strategy_skill_id"],
        strategy_diagnostic_context=evolution_queries["strategy_diagnostic_context"],
        limit_per_type=int(reflection_cfg.get("reflector_candidate_top_k", 5)),
    )
    _add_runtime_loaded_candidates(skill_search_context, bank, stage_state)
    write_json(out_dir / "reflector_skill_search_context.json", skill_search_context)

    reflector_prompt = resolve_path(
        reflection_cfg.get(
            "reflector_prompt_path",
            "prompt_records/reflection/reflector_skill_v0.txt",
        )
    ).read_text(encoding="utf-8")
    reflector_output = run_reflector(
        trajectory_evidence=trajectory_evidence,
        evolution_queries=evolution_queries,
        skill_search_context=skill_search_context,
        llm_client=client,
        prompt=reflector_prompt,
        attempts=int(reflection_cfg.get("reflector_attempts", 2)),
        output_path=out_dir / "reflector_debug.json",
    )
    card_finalizer_prompt = resolve_path(
        reflection_cfg.get(
            "skill_card_finalizer_prompt_path",
            "prompt_records/reflection/skill_card_finalizer_v0.txt",
        )
    ).read_text(encoding="utf-8")
    semantic_center_prompt = resolve_path(
        reflection_cfg.get(
            "skill_semantic_center_prompt_path",
            "prompt_records/reflection/skill_semantic_center_v0.txt",
        )
    ).read_text(encoding="utf-8")
    reflector_output = finalize_skill_cards(
        reflector_output=reflector_output,
        llm_client=client,
        prompt=card_finalizer_prompt,
        semantic_center_prompt=semantic_center_prompt,
        semantic_center_attempts=int(reflection_cfg.get("skill_semantic_center_attempts", 3)),
        source_terms=_finalizer_source_terms(trajectory_evidence),
        # Keep the compact-card contract strict, but allow one additional
        # schema-feedback turn for small correctable formatting violations.
        attempts=int(reflection_cfg.get("skill_card_finalizer_attempts", 3)),
        output_path=out_dir / "skill_card_finalization_debug.json",
    )
    if outcome.get("label") == "success" and reflect_success:
        reflector_output = _retain_project_updates_from_success(reflector_output)
    write_json(out_dir / "reflector_output.json", reflector_output)

    applied_updates: list[dict[str, Any]] = []
    failed_updates: list[dict[str, Any]] = []
    for update in reflector_output.get("materialized_updates", []) or []:
        try:
            applied_updates.append(bank.apply_update(update))
        except Exception as exc:  # noqa: BLE001 - isolate atomic skill update failures.
            failed_updates.append(
                {
                    "update_id": update.get("update_id"),
                    "operation": update.get("operation"),
                    "target_skill_id": update.get("target_skill_id"),
                    "error": str(exc),
                }
            )
    write_json(out_dir / "applied_updates.json", applied_updates)
    write_json(out_dir / "failed_updates.json", failed_updates)
    write_json(out_dir / "skill_bank_delta.json", applied_updates)

    summary.update(
        {
            "trajectory_evidence_path": str(out_dir / "trajectory_evidence.json"),
            "evolution_queries": evolution_queries,
            "evolution_queries_path": str(out_dir / "evolution_queries.json"),
            "reflector_skill_search_context_path": str(out_dir / "reflector_skill_search_context.json"),
            "reflector_output_path": str(out_dir / "reflector_output.json"),
            "skill_updates": reflector_output.get("skill_updates"),
            "applied_updates": applied_updates,
            "failed_updates": failed_updates,
            "updated_skill_ids": [
                item.get("updated_skill_id") for item in applied_updates if item.get("updated_skill_id")
            ],
            "updated_skill_types": _updated_skill_types(
                reflector_output.get("materialized_updates", []) or []
            ),
            "no_update_reason": reflector_output.get("no_update_reason"),
        }
    )
    write_json(out_dir / "case_evolution_summary.json", summary)
    return summary


def _finalizer_source_terms(trajectory_evidence: dict[str, Any]) -> list[str]:
    """Extract concrete symbols from the case evidence for portable-card checks."""

    ground_truth = trajectory_evidence.get("ground_truth") or {}
    prediction = trajectory_evidence.get("prediction") or {}
    text = "\n".join(
        [
            str(trajectory_evidence.get("problem_statement") or ""),
            str(ground_truth.get("patch_context") or ""),
            *[str(item) for item in prediction.get("ranked_functions") or []],
            *[str(item) for item in ground_truth.get("functions") or []],
        ]
    )
    terms = set(re.findall(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b", text))
    # Class names in function identities are source vocabulary even when the
    # class has a single capitalized word (for example, ``Delta``). Limit this
    # extraction to identifier-shaped contexts rather than all prose words.
    terms.update(re.findall(r"::([A-Z][A-Za-z0-9_]*)\.", text))
    terms.update(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\.[a-z_][A-Za-z0-9_]*\b", text))
    return sorted(
        (term for term in terms if len(term) > 2 and term not in _GENERIC_EVIDENCE_TERMS),
        key=lambda item: (-len(item), item),
    )[:48]


def _retain_project_updates_from_success(reflector_output: dict[str, Any]) -> dict[str, Any]:
    """Keep successful trajectories as system-knowledge evidence, not ranking lessons.

    A completed Top-5 hit can establish a stable Project architecture, while a
    Strategy card requires a corrective contrast from a localization miss.
    ``reflect_success`` is opt-in; this guard makes its scope explicit even if
    the Reflector returns both card types.
    """

    output = dict(reflector_output)
    output["materialized_updates"] = [
        update
        for update in output.get("materialized_updates", []) or []
        if update.get("skill_type") == "project_skill"
    ]
    updates = dict(output.get("skill_updates") or {})
    updates["strategy_skill"] = {
        "decision": "no_update",
        "target_skill_id": None,
        "rationale": "Successful localization supplies no corrective Strategy lesson.",
        "skill": None,
        "no_update_reason": "Strategy Skills are learned only from localization misses.",
    }
    output["skill_updates"] = updates
    output["success_project_only"] = True
    return output


def _read_trajectory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _stage_state(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    events = [str(event.get("event") or "") for event in trajectory]
    missing = sorted(REQUIRED_STAGE_EVENTS - set(events))
    return {
        "compatible": not missing,
        "missing_events": missing,
        "project_skill": _one_stage_state(trajectory, "project_skill"),
        "strategy_skill": _one_stage_state(trajectory, "strategy_skill"),
    }


def _one_stage_state(trajectory: list[dict[str, Any]], skill_type: str) -> dict[str, Any]:
    request = next((event for event in trajectory if event.get("event") == f"{skill_type}_request"), None)
    loaded = next((event for event in trajectory if event.get("event") == f"{skill_type}_loaded"), None)
    return {
        "attempted": request is not None and loaded is not None,
        "request": request,
        "loaded_skill_id": (loaded or {}).get("loaded_skill_id"),
        "matched": bool((loaded or {}).get("loaded_skill_id")),
        "search_trace": (loaded or {}).get("search_trace", {}),
    }


def _build_trajectory_evidence(
    *,
    case_dir: Path,
    repo: str,
    issue: str,
    result: dict[str, Any],
    trajectory: list[dict[str, Any]],
    outcome: dict[str, Any],
    ground_truth_functions: list[str],
    ground_truth_locations: list[dict[str, Any]],
    ground_truth_patch: str,
    stage_state: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "case": {
            "instance_id": result.get("instance_id") or case_dir.name,
            "repo": repo,
            "case_run_dir": str(case_dir),
        },
        "problem_statement": issue,
        "outcome": outcome,
        "prediction": {
            "ranked_functions": result.get("ranked_functions", []),
            "top_5_results": (result.get("ranked_functions", []) or [])[:5],
            "final_summary": result.get("final_summary", ""),
        },
        "ground_truth": {
            "functions": ground_truth_functions,
            "locations": ground_truth_locations,
            "patch_context": ground_truth_patch[:12000],
        },
        "stage_state": stage_state,
    }
    evidence.update(build_compacted_trajectory(trajectory))
    return evidence


def _label_outcome(
    result: dict[str, Any], ranked: list[str], ground_truth: list[str], force: bool
) -> dict[str, Any]:
    status = result.get("status")
    if status == "completed" and ground_truth:
        # Use the same Python-module normalization as batch metrics. Explorer
        # models often emit dotted module paths, while patch ground truth uses
        # repository paths such as ``package/module.py::symbol``.
        hit = bool(evaluate_ranked_functions(ranked, ground_truth).get("top5"))
        return {
            "label": "success" if hit else "failure",
            "top5_hit": hit,
            "reason": "completed top-5 hit" if hit else "completed top-5 miss",
        }
    if status == "completed":
        return {"label": "failure", "top5_hit": False, "reason": "completed but ground truth missing"}
    return {
        "label": "failure",
        "top5_hit": False,
        "reason": f"system-level failure status={status!r}" + (" forced" if force else ""),
    }


def _eligibility(
    result: dict[str, Any],
    outcome: dict[str, Any],
    ground_truth: list[str],
    force: bool,
    *,
    reflect_success: bool,
    reflect_failure: bool,
) -> tuple[bool, str]:
    if force:
        return True, "force enabled"
    if result.get("status") != "completed":
        return False, "case is not completed"
    if not ground_truth:
        return False, "ground truth functions are missing"
    if outcome.get("label") == "success" and not reflect_success:
        return False, "completed top-5 hit but reflect_success is disabled"
    if outcome.get("label") == "failure" and not reflect_failure:
        return False, "completed top-5 miss but reflect_failure is disabled"
    return True, str(outcome.get("reason") or "completed case")


def _updated_skill_types(updates: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(update.get("skill_type"))
            for update in updates
            if isinstance(update, dict) and update.get("skill_type")
        )
    )


def _add_runtime_loaded_candidates(
    skill_search_context: dict[str, Any], bank: Any, stage_state: dict[str, Any]
) -> None:
    """Expose runtime context without weakening Strategy semantic matching.

    A loaded Project Skill remains a useful rewrite candidate because it was
    selected from concrete repository structure. Strategy cards instead encode
    a narrowly conditioned decision; only the independently selected catalog
    card may become a Strategy update target. Other runtime-loaded Strategy
    cards remain visible as trajectory evidence but cannot be preserved merely
    because Explorer happened to inspect them.
    """

    runtime_loaded: list[dict[str, Any]] = []
    known_ids = {
        str(skill.get("skill_id") or "")
        for skill in (skill_search_context.get("candidate_target_skills") or [])
        if isinstance(skill, dict)
    }
    by_type = skill_search_context.setdefault("candidates_by_skill_type", {})
    for skill_type in ("project_skill", "strategy_skill"):
        skill_id = str(
            ((stage_state.get(skill_type) or {}).get("loaded_skill_id") or "")
        ).strip()
        if not skill_id:
            continue
        skill = bank.get_active_skill(skill_id, skill_type=skill_type)
        if not skill:
            continue
        candidate = {**skill, "candidate_source": "runtime_loaded"}
        runtime_loaded.append(candidate)
        if skill_type == "strategy_skill":
            selected_id = str(skill_search_context.get("selected_strategy_skill_id") or "").strip()
            if skill_id != selected_id:
                continue
        if skill_id not in known_ids:
            by_type.setdefault(skill_type, []).append(candidate)
            skill_search_context.setdefault("candidate_target_skills", []).append(candidate)
            known_ids.add(skill_id)
    skill_search_context["runtime_loaded_skills"] = runtime_loaded
