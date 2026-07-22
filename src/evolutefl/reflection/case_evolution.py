from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.evaluation import hit_at_k
from evolutefl.issue_abstraction import abstract_issue, fallback_issue_abstraction
from evolutefl.json_utils import read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank

from .insight import build_insight
from .reflector import run_reflector
from .trajectory_compactor import build_compacted_trajectory


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
    reflect_success: bool = True,
    reflect_failure: bool = True,
    legacy_insight: bool = False,
    refresh_issue_abstraction: bool = False,
) -> dict[str, Any]:
    case_dir = Path(case_run_dir)
    out_dir = Path(output_dir) if output_dir else case_dir / "case_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    trajectory_path = case_dir / "trajectory.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Explorer result.json not found: {result_path}")

    result = read_json(result_path)
    trajectory = _read_trajectory(trajectory_path)
    ranked_functions = result.get("ranked_functions", [])
    ground_truth_functions = ground_truth_functions or []
    outcome = _label_outcome(result, ranked_functions, ground_truth_functions, force)
    eligible, reason = _eligibility(
        result,
        outcome,
        ground_truth_functions,
        force,
        reflect_success=reflect_success,
        reflect_failure=reflect_failure,
    )

    summary: dict[str, Any] = {
        "eligible": eligible,
        "reason": reason,
        "case_run_dir": str(case_dir),
        "repo": repo,
        "outcome": outcome,
        "legacy_insight": legacy_insight,
        "applied_updates": [],
        "updated_skill_ids": [],
    }
    if not eligible:
        write_json(out_dir / "case_evolution_summary.json", summary)
        return summary

    reflection_cfg = config.get("reflection", {})
    project_type_dedup_top_k = int(reflection_cfg.get("project_type_dedup_top_k", 5))
    strategy_dedup_top_k = int(reflection_cfg.get("strategy_dedup_top_k", 5))
    reflector_prompt = resolve_path(reflection_cfg.get("reflector_prompt_path", "prompt_records/reflection/reflector_skill_v0.txt")).read_text(encoding="utf-8")
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))
    bank = make_skill_bank(config)
    issue_abstraction = _load_issue_abstraction(
        case_dir,
        repo,
        issue,
        config,
        llm_client=client,
        output_dir=out_dir,
        refresh=refresh_issue_abstraction,
    )
    skill_search = bank.search_for_explorer(repo, issue, issue_abstraction=issue_abstraction)
    trajectory_evidence = _build_trajectory_evidence(
        case_dir=case_dir,
        repo=repo,
        issue=issue,
        issue_abstraction=issue_abstraction,
        result=result,
        trajectory=trajectory,
        outcome=outcome,
        ground_truth_functions=ground_truth_functions,
        ground_truth_locations=ground_truth_locations or [],
        ground_truth_patch=ground_truth_patch,
        skill_search=skill_search,
    )
    write_json(out_dir / "trajectory_evidence.json", trajectory_evidence)

    insight: dict[str, Any] | None = None
    if legacy_insight:
        insight_input = {
            "instance_id": result.get("instance_id") or case_dir.name,
            "repo": repo,
            "problem_statement": issue,
            "trajectory": trajectory,
            "top_5_results": ranked_functions[:5],
            "ranked_functions": ranked_functions,
            "ground_truth_functions": ground_truth_functions,
            "ground_truth_locations": ground_truth_locations or [],
            "ground_truth_patch": ground_truth_patch,
            "final_summary": result.get("final_summary", ""),
            "outcome": outcome,
        }
        write_json(out_dir / "insight_input.json", insight_input)
        insight_prompt = resolve_path(reflection_cfg.get("insight_prompt_path", "prompt_records/reflection/reflection_insight_v1.txt")).read_text(encoding="utf-8")
        insight = build_insight(
            case_input=insight_input,
            llm_client=client,
            prompt=insight_prompt,
            output_path=out_dir / "insight_debug.json",
        )
        write_json(out_dir / "insight.json", insight)
        skill_search_context = bank.search_for_reflector(
            repo,
            issue,
            insight,
            issue_abstraction=issue_abstraction,
            explorer_search=skill_search,
            project_candidate_limit=project_type_dedup_top_k,
            strategy_candidate_limit=strategy_dedup_top_k,
        )
    else:
        skill_search_context = bank.search_for_reflector(
            repo,
            issue,
            issue_abstraction=issue_abstraction,
            explorer_search=skill_search,
            project_candidate_limit=project_type_dedup_top_k,
            strategy_candidate_limit=strategy_dedup_top_k,
        )
    write_json(out_dir / "skill_search_context.json", skill_search_context)

    reflector_output = run_reflector(
        insight=insight,
        trajectory_evidence=None if legacy_insight else trajectory_evidence,
        issue_abstraction=issue_abstraction,
        skill_search_context=skill_search_context,
        llm_client=client,
        prompt=reflector_prompt,
        output_path=out_dir / "reflector_debug.json",
    )
    write_json(out_dir / "reflector_output.json", reflector_output)

    applied_updates: list[dict[str, Any]] = []
    failed_updates: list[dict[str, Any]] = []
    for update in reflector_output.get("materialized_updates", []):
        try:
            applied = bank.apply_update(update)
            applied_updates.append(applied)
        except Exception as exc:  # noqa: BLE001 - isolate an invalid atomic update.
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
            "insight": insight,
            "issue_abstraction": issue_abstraction,
            "trajectory_evidence": trajectory_evidence,
            "reflector_output": reflector_output,
            "skill_updates": reflector_output.get("skill_updates"),
            "applied_updates": applied_updates,
            "failed_updates": failed_updates,
            "outcome_type": outcome.get("label"),
            "updated_skill_ids": [item.get("updated_skill_id") for item in applied_updates if item.get("updated_skill_id")],
            "updated_skill_types": _updated_skill_types_from_updates(reflector_output.get("materialized_updates", [])),
            "no_update_reason": reflector_output.get("no_update_reason"),
        }
    )
    write_json(out_dir / "case_evolution_summary.json", summary)
    return summary


def _read_trajectory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_issue_abstraction(
    case_dir: Path,
    repo: str,
    issue: str,
    config: dict[str, Any],
    *,
    llm_client: Any | None = None,
    output_dir: Path | None = None,
    refresh: bool = False,
) -> dict[str, Any] | None:
    abstraction_cfg = config.get("issue_abstraction", {}) or {}
    if abstraction_cfg.get("enabled", True) is False:
        return None
    if refresh:
        if llm_client is None:
            raise ValueError("Refreshing issue abstraction requires an LLM client.")
        prompt_path = resolve_path(
            abstraction_cfg.get("prompt_path", "prompt_records/explorer/issue_abstraction_v0.txt")
        )
        abstraction = abstract_issue(
            repo=repo,
            issue=issue,
            llm_client=llm_client,
            prompt=prompt_path.read_text(encoding="utf-8"),
            output_path=(output_dir / "issue_abstraction_debug.json") if output_dir else None,
        )
        if output_dir:
            write_json(output_dir / "issue_abstraction.json", abstraction)
        return abstraction
    abstraction_path = case_dir / "issue_abstraction.json"
    if abstraction_path.exists():
        try:
            payload = read_json(abstraction_path)
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001 - old/corrupt run artifacts should not block evolution.
            pass
    return fallback_issue_abstraction(repo, issue)


def _label_outcome(
    result: dict[str, Any],
    ranked_functions: list[str],
    ground_truth_functions: list[str],
    force: bool,
) -> dict[str, Any]:
    status = result.get("status")
    if status == "completed" and ground_truth_functions:
        hit = hit_at_k(ranked_functions, ground_truth_functions, k=5)
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
    ground_truth_functions: list[str],
    force: bool,
    *,
    reflect_success: bool,
    reflect_failure: bool,
) -> tuple[bool, str]:
    if force:
        return True, "force enabled"
    if result.get("status") != "completed":
        return False, "case is not completed"
    if not ground_truth_functions:
        return False, "ground truth functions are missing"
    if outcome.get("label") == "success" and not reflect_success:
        return False, "completed top-5 hit but reflect_success is disabled"
    if outcome.get("label") == "failure" and not reflect_failure:
        return False, "completed top-5 miss but reflect_failure is disabled"
    return True, outcome.get("reason", "completed case")


def _build_trajectory_evidence(
    *,
    case_dir: Path,
    repo: str,
    issue: str,
    issue_abstraction: dict[str, Any] | None,
    result: dict[str, Any],
    trajectory: list[dict[str, Any]],
    outcome: dict[str, Any],
    ground_truth_functions: list[str],
    ground_truth_locations: list[dict[str, Any]],
    ground_truth_patch: str,
    skill_search: dict[str, Any],
) -> dict[str, Any]:
    compacted = build_compacted_trajectory(trajectory)
    evidence = {
        "case": {
            "instance_id": result.get("instance_id") or case_dir.name,
            "repo": repo,
            "case_run_dir": str(case_dir),
        },
        "problem_statement": issue,
        "issue_abstraction": issue_abstraction,
        "outcome": outcome,
        "prediction": {
            "ranked_functions": result.get("ranked_functions", []),
            "top_5_results": result.get("ranked_functions", [])[:5],
            "final_summary": result.get("final_summary", ""),
        },
        "ground_truth": {
            "functions": ground_truth_functions,
            "locations": ground_truth_locations,
            "patch_context": ground_truth_patch[:12000],
        },
        "retrieved_skill_context": {
            "matched_skills": skill_search.get("matched_skills", []),
            "skill_type_slots": skill_search.get("skill_type_slots", {}),
            "skill_search_trace": skill_search.get("skill_search_trace", {}),
        },
        "evolution_policy_hint": _policy_hint(outcome, bool(skill_search.get("matched_skills"))),
    }
    evidence.update(compacted)
    return evidence


def _policy_hint(outcome: dict[str, Any], has_relevant_skill: bool) -> str:
    if outcome.get("label") == "success" and has_relevant_skill:
        return "Success with a relevant retrieved skill: preserve it unless the case supports a coherent full-card rewrite."
    if outcome.get("label") == "success":
        return "Success without relevant retrieved skill: create only if the trajectory reveals transferable localization knowledge."
    if has_relevant_skill:
        return "Failure with a relevant retrieved skill: rewrite it only when semantic identity is unchanged; otherwise create or no_update."
    return "Failure without relevant retrieved skill: create only if the lesson is transferable; otherwise no_update."


def _updated_skill_types_from_updates(updates: list[dict[str, Any]]) -> list[str]:
    skill_types: list[str] = []
    for update in updates:
        skill_type = update.get("skill_type") if isinstance(update, dict) else None
        if skill_type and skill_type not in skill_types:
            skill_types.append(str(skill_type))
    return skill_types
