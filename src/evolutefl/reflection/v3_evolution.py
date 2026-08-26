"""V3 evolution separates static Project knowledge from case-derived lessons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from evolutefl.skills.schema import validate_portable_skill_card

from .reflector import run_reflector
from .trajectory_compactor import build_compacted_trajectory


V3_REQUIRED_EVENTS = {
    "project_skill_request", "project_skill_loaded", "issue_revealed",
    "issue_skill_request", "issue_skill_loaded", "strategy_skill_request", "strategy_skill_loaded",
}


def run_v3_case_evolution(**kwargs: Any) -> dict[str, Any]:
    """Run a static Project builder plus case-derived Issue/Strategy reflection.

    Project creation is deliberately isolated from issue, outcome, ranked
    candidates and patch material. Issue and Strategy are the only case
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
                   **{kind: _stage_state(trajectory, kind) for kind in ("project_skill", "issue_skill", "strategy_skill")}}
    ground_truth = kwargs.get("ground_truth_functions") or []
    outcome = _label_outcome(result, result.get("ranked_functions") or [], ground_truth, bool(kwargs.get("force")))
    eligible, reason = _eligibility(result, outcome, ground_truth, bool(kwargs.get("force")),
                                    reflect_success=bool(kwargs.get("reflect_success")),
                                    reflect_failure=bool(kwargs.get("reflect_failure", True)))
    if not stage_state["compatible"]:
        eligible, reason = False, "incompatible V3 trajectory: " + ", ".join(missing)
    client = kwargs.get("llm_client") or OpenAICompatibleClient.from_config(config.get("llm", {}))
    bank = make_skill_bank(config)
    project_result = _run_project_builder(
        bank,
        client,
        repo,
        trajectory,
        out_dir,
        config,
        repository_manifest=_read_json_if_present(case_dir / "repository_manifest.json"),
        completed=str(result.get("status") or "") == "completed",
    )
    summary: dict[str, Any] = {"eligible": eligible, "reason": reason, "repo": repo, "outcome": outcome,
                               "stage_state": stage_state, "project_knowledge": project_result,
                               "applied_updates": project_result.get("applied_updates", []), "updated_skill_ids": []}
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
    context = bank.search_for_v3_evolution(issue_skill_query=queries["issue_skill_query"],
        selected_strategy_skill_id=queries.get("selected_strategy_skill_id"),
        strategy_diagnostic_context=queries["strategy_diagnostic_context"],
        limit_per_type=int((config.get("reflection") or {}).get("reflector_candidate_top_k", 5)))
    write_json(out_dir / "reflector_skill_search_context.json", context)
    prompt = resolve_path((config.get("reflection") or {}).get("reflector_prompt_path", "prompt_records/reflection/reflector_skill_v0.txt")).read_text(encoding="utf-8")
    output = run_reflector(trajectory_evidence=evidence, evolution_queries=queries, skill_search_context=context,
                           llm_client=client, prompt=prompt, attempts=int((config.get("reflection") or {}).get("reflector_attempts", 2)),
                           output_path=out_dir / "reflector_debug.json", skill_types=("issue_skill", "strategy_skill"),
                           strict_final_cards=True, isolate_skill_errors_on_final_attempt=True)
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
    summary.update({"evolution_queries": queries, "skill_updates": output.get("skill_updates"),
                    "updated_skill_ids": [item.get("updated_skill_id") for item in summary["applied_updates"] if item.get("updated_skill_id")],
                    "failed_updates": failed})
    materialized_updates = [
        {
            "operation": _report_operation(item.get("action")),
            "skill_type": "project_skill",
            "target_skill_id": None,
            "updated_skill_id": item.get("updated_skill_id"),
        }
        for item in project_result.get("applied_updates", [])
    ] + applied_materialized
    summary["materialized_updates"] = materialized_updates
    summary["updated_skill_types"] = sorted({
        str(update.get("skill_type")) for update in materialized_updates
        if update.get("skill_type")
    })
    write_json(out_dir / "case_evolution_summary.json", summary)
    return summary


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
        return {"decision": "preserve_existing", "reason": "Explorer did not complete.", "applied_updates": []}
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
    request = next((event for event in trajectory if event.get("event") == "project_skill_request"), {})
    request_query = str(request.get("query") or request.get("system_summary") or repo).strip()
    candidate_context = bank.search_project_candidates(request_query, limit=5)
    evidence = {
        "repo": repo,
        "repository_manifest": repository_manifest,
        "repository_orientation": build_compacted_trajectory(orientation),
        "same_type_candidates": candidate_context.get("candidate_skills", []),
    }
    write_json(out_dir / "project_knowledge_evidence.json", evidence)
    loaded = next((event for event in trajectory if event.get("event") == "project_skill_loaded"), {})
    if loaded.get("loaded_skill_id"):
        return {"decision": "preserve_existing", "reason": "A Project Skill was already matched at runtime.", "applied_updates": []}
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
            decision = str(raw.get("decision") or "preserve_existing")
            if decision != "create_new":
                write_json(out_dir / "project_knowledge_output.json", {
                    "attempt": attempt, "project_builder_output": raw, "attempts": attempts_log,
                })
                return {"decision": decision, "reason": str(raw.get("reason") or ""), "applied_updates": []}
            card = validate_portable_skill_card(raw.get("skill") or {}, skill_type="project_skill")
            applied = bank.apply_update({"operation": "create", "skill_type": "project_skill", "target_skill_id": None,
                                         "skill": card, "rationale": str(raw.get("reason") or "Repository orientation."), "source_cases": []})
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
                    "Return the exact JSON schema again. For create_new, provide exactly three "
                    "portable knowledge statements, each no longer than 360 characters. "
                    f"Validation feedback: {exc}"
                ),
            }]
    write_json(out_dir / "project_knowledge_output.json", {"failed": True, "attempts": attempts_log})
    return {
        "decision": "preserve_existing",
        "reason": "Project Builder could not produce a valid compact static card.",
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
        {"role": "user", "content": json.dumps({"trajectory_evidence": evidence, "strategy_skill_catalog": catalog}, ensure_ascii=False)},
    ]
    messages = list(base_messages)
    errors: list[dict[str, Any]] = []
    attempts = max(1, int((config.get("reflection") or {}).get("query_attempts", 2)))
    for attempt in range(1, attempts + 1):
        response = client.chat(messages=messages, tool_choice="none", response_format={"type": "json_object"})
        content = response.get("content") or ""
        try:
            result = extract_json_object(content)
            for key in ("issue_skill_query", "strategy_diagnostic_context"):
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
                "content": "Return the exact requested JSON object with non-empty issue_skill_query and strategy_diagnostic_context.",
            }]
    write_json(out_dir / "evolution_queries.json", {"failed": True, "attempts": errors})
    raise ValueError(f"V3 evolution query generation failed: {errors[-1]['error']}")
