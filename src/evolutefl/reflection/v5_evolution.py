"""Fault evolution from reconstructed investigation and checked evidence gaps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.evaluation import evaluate_ranked_functions
from evolutefl.investigation import repository_identity
from evolutefl.json_utils import extract_json_object, read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from evolutefl.skills.fault_taxonomy import compact_fault_taxonomy, validate_fault_family
from .reflector import run_reflector
from .v3_evolution import _select_fault_evolution_target
from .v5_investigator import investigate


def _same_repository_identity(recorded: dict, current: dict) -> bool:
    """Compare source content, tolerating a missing .git directory on extraction."""
    if recorded.get("source_sha256") != current.get("source_sha256"):
        return False
    if recorded.get("text_file_count") != current.get("text_file_count"):
        return False
    recorded_head = recorded.get("git_head")
    current_head = current.get("git_head")
    if recorded_head and current_head and recorded_head != current_head:
        return False
    return True


def _compact_investigation_index(index: dict) -> dict:
    """Provide the Investigator a navigable timeline, not the raw Explorer transcript."""
    timeline = []
    for event in index.get("timeline", []):
        compact = {key: event.get(key) for key in (
            "event", "step", "tool", "observation_id", "purpose", "based_on", "candidate_updates",
            "unresolved_references"
        ) if key in event}
        for key in ("purpose",):
            if isinstance(compact.get(key), str) and len(compact[key]) > 400:
                compact[key] = compact[key][:400] + "..."
        timeline.append(compact)
    return {
        "trace_version": index.get("trace_version"),
        "timeline": timeline,
        "observations": index.get("observations", []),
        "usage_note": "Read an observation by ID to inspect exactly what Explorer received. Continuation pages are separate observations.",
    }


def run_v5_case_evolution(**kwargs: Any) -> dict:
    case_dir = Path(kwargs["case_run_dir"])
    out = Path(kwargs.get("output_dir") or case_dir / "case_evolution")
    out.mkdir(parents=True, exist_ok=True)
    config = kwargs["config"]
    summary = {"trace_version": "v5", "status": "skipped", "eligible": False, "reason": "",
               "failure_stage": None, "applied_updates": [], "updated_skill_ids": [], "updated_skill_types": [],
               "reflection_mode": "evidence_driven", "learning_foci": []}
    stage = "eligibility"
    try:
        for filename in ("run_context.json", "result.json"):
            if not (case_dir / filename).is_file():
                raise ValueError(f"V5 trajectory required; missing {filename}")
        result, context = read_json(case_dir / "result.json"), read_json(case_dir / "run_context.json")
        if result.get("trace_version") != "v5" or context.get("trace_version") != "v5":
            raise ValueError("Incompatible trajectory: v5 format required")
        issue = str(kwargs.get("issue") or "").strip()
        ground_truth = kwargs.get("ground_truth_functions") or []
        if not issue or not ground_truth:
            raise ValueError("Non-empty issue and ground truth functions are required")
        if result.get("status") != "completed":
            summary["reason"] = "Only completed Explorer cases are eligible"
            return summary
        for filename in ("trajectory.jsonl", "investigation_index.json", "observations.jsonl", "fault_skill_request.json", "fault_skill_search.json"):
            if not (case_dir / filename).is_file():
                raise ValueError(f"Incomplete v5 investigation; missing {filename}")
        events = [json.loads(line) for line in (case_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not {"run_started", "fault_skill_request", "fault_skill_loaded", "finish"} <= {e.get("event") for e in events}:
            raise ValueError("Incomplete v5 native-control trajectory")
        metrics = evaluate_ranked_functions(result.get("ranked_functions", []), ground_truth,
            strict=bool(config.get("reflection", {}).get("strict_function_matching", False)))
        hit = bool(metrics["top5"])
        outcome = {"label": "success" if hit else "failure", "top5_hit": hit,
                   "reason": "completed top-5 hit" if hit else "completed top-5 miss",
                   "function_metrics": metrics,
                   "interpretation": "Patch-function ranking outcome only; investigation quality is assessed from observations."}
        runtime_request = read_json(case_dir / "fault_skill_request.json")["request"]
        runtime = validate_fault_family(runtime_request["fault_family"])
        summary.update(eligible=True, function_metrics=metrics, outcome=outcome, runtime_fault_family=runtime)
        supplied = kwargs.get("repo_path")
        if not supplied:
            raise ValueError("V5 evolution requires explicit repo_path")
        root = Path(supplied).resolve(strict=True)
        stage = "repository_identity"
        identity = repository_identity(root)
        if not _same_repository_identity(context["repository_identity"], identity):
            raise ValueError("Repository differs from the Explorer snapshot")
        initial = read_json(case_dir / "initial_payload.json")
        if issue != initial.get("bug_report") or kwargs.get("repo") != context.get("repo"):
            raise ValueError("Issue/repo does not match the original Explorer run")
        index = read_json(case_dir / "investigation_index.json")
        patch = str(kwargs.get("ground_truth_patch") or "")
        patch_metadata = kwargs.get("patch_metadata") or {"source": "caller", "direction": "unknown"}
        evidence = {"repo": kwargs["repo"], "problem_statement": issue, "outcome": outcome,
            "prediction": {"ranked_functions": result.get("ranked_functions", []), "summary": result.get("final_summary", "")},
            "investigation_index": _compact_investigation_index(index),
            "ground_truth": {"functions": ground_truth, "locations": kwargs.get("ground_truth_locations") or [],
                             "patch": patch, "patch_metadata": patch_metadata}}
        write_json(out / "trajectory_evidence.json", evidence)
        client = kwargs.get("llm_client") or OpenAICompatibleClient.from_config(config.get("llm", {}))
        stage = "investigation"
        conclusion = investigate(client=client, config=config, root=root, case_dir=case_dir,
                                 directory=out / "supplementary", evidence=evidence)
        summary.update(investigation_status=conclusion["status"], investigation_tool_calls=conclusion["tool_call_count"])
        summary["learning_foci"] = sorted({f.get("learning_focus", "uncertain") for f in conclusion["findings"]})
        if repository_identity(root) != identity:
            raise ValueError("Repository changed during investigation")
        if conclusion["status"] != "resolved" or conclusion.get("failure_stage"):
            summary.update(status="failed" if conclusion.get("failure_stage") else "completed",
                           failure_stage=conclusion.get("failure_stage"), reason=conclusion["summary"], decision="no_update")
            write_json(out / "fault_reflector_output.json", {"skill_updates": {"fault_skill": {
                "decision": "no_update", "target_skill_id": None, "skill": None,
                "rationale": conclusion["summary"], "no_update_reason": conclusion["summary"]}}})
            write_json(out / "applied_updates.json", [])
            return summary
        stage = "evolution_query"
        cfg = config.get("reflection", {})
        query_prompt = resolve_path(cfg.get("evolution_query_prompt_path", "prompt_records/reflection/evolution_query_v5.txt")).read_text(encoding="utf-8")
        query_input = {"repo": kwargs["repo"], "issue": issue, "outcome": outcome,
                       "investigation_conclusion": conclusion, "fault_families": compact_fault_taxonomy(),
                       "runtime_fault_request": runtime_request}
        attempts = []
        queries = None
        for attempt in range(max(1, int(cfg.get("query_attempts", 2)))):
            response = client.chat(messages=[{"role": "system", "content": query_prompt},
                {"role": "user", "content": json.dumps(query_input, ensure_ascii=False)}],
                response_format={"type": "json_object"}, tool_choice="none")
            attempts.append(response)
            write_json(out / "evolution_query_debug.json", {"input": query_input, "responses": attempts})
            try:
                raw = extract_json_object(response.get("content") or "")
                queries = {"fault_family": validate_fault_family(raw.get("fault_family")),
                           "fault_subtype_query": str(raw.get("fault_subtype_query") or "").strip()}
                if not queries["fault_subtype_query"]:
                    raise ValueError("Empty fault_subtype_query")
                break
            except (ValueError, TypeError):
                queries = None
        if queries is None:
            raise ValueError("Evolution query protocol failed")
        write_json(out / "evolution_queries.json", queries)
        summary.update(reflector_fault_family=queries["fault_family"],
                       fault_family_consistent=runtime == queries["fault_family"])
        stage = "fault_candidate_selection"
        bank = make_skill_bank(config)
        catalog = bank.fault_evolution_catalog([queries["fault_family"], runtime])
        routing_context = {"runtime_request": runtime_request, "evolution_family": queries["fault_family"],
                           "queried_families": catalog["skill_search_trace"]["queried_families"]}
        target = _select_fault_evolution_target(client=client, fault_family=queries["fault_family"],
            fault_subtype_query=queries["fault_subtype_query"], catalog=catalog.get("candidate_skills", []),
            out_dir=out, config=config, routing_context=routing_context)
        selected = bank.get_active_skill(target["selected_skill_id"], skill_type="fault_skill") if target.get("selected_skill_id") else None
        search_context = {"candidate_target_skills": [selected] if selected else [], "fault_catalog_selection": target,
                          "routing_context": routing_context, "search_trace": catalog["skill_search_trace"]}
        write_json(out / "reflector_skill_search_context.json", search_context)
        stage = "fault_reflector"
        prompt_path = cfg.get("fault_reflector_prompt_path", "prompt_records/reflection/fault_reflector_v5.txt")
        prompt = resolve_path(prompt_path).read_text(encoding="utf-8")
        summary["reflector_prompt_path"] = str(prompt_path)
        # The patch stays in the investigation stage. Skill writing sees the checked process lesson.
        view = {"repo": kwargs["repo"], "problem_statement": issue, "outcome": outcome,
                "investigation_conclusion": conclusion, "runtime_fault_request": runtime_request}
        write_json(out / "fault_reflection_view.json", view)
        output = run_reflector(trajectory_evidence=view, evolution_queries=queries, skill_search_context=search_context,
            llm_client=client, prompt=prompt, attempts=int(cfg.get("reflector_attempts", 2)),
            output_path=out / "fault_reflector_debug.json", skill_types=("fault_skill",), strict_final_cards=True,
            max_tokens=int(cfg.get("reflector_max_tokens", 8192)),
            truncation_retries=int(cfg.get("reflector_truncation_retries", 1)),
            truncation_retry_max_tokens=int(cfg.get("reflector_truncation_retry_max_tokens", 8192)),
            disable_deepseek_thinking_on_truncation=bool(
                cfg.get("deepseek_disable_thinking_on_reflector_truncation", True)
            ))
        write_json(out / "fault_reflector_output.json", output)
        write_json(out / "reflector_output.json", output)
        if read_json(out / "fault_reflector_debug.json").get("failed"):
            raise ValueError("Fault Reflector output protocol failed")
        stage = "skill_update"
        applied = []
        for update in output.get("materialized_updates", []):
            card = update.get("skill")
            if update.get("operation") == "create" and card["fault_family"] != queries["fault_family"]:
                raise ValueError("New Skill primary family must match the independently classified evolution family")
            applied.append(bank.apply_update(update))
        write_json(out / "applied_updates.json", applied)
        routes = []
        for update, effect in zip(output.get("materialized_updates", []), applied):
            updated = bank.get_active_skill(effect.get("updated_skill_id"), skill_type="fault_skill")
            before = (selected or {}).get("retrieval_families", []) if update.get("target_skill_id") else []
            after = (updated or {}).get("retrieval_families", [])
            routes.append({"skill_id": effect.get("updated_skill_id"), "before": before, "after": after,
                           "added": [f for f in after if f not in before], "removed": [f for f in before if f not in after]})
        summary.update(status="completed", reason="Evidence-supported reflection completed", applied_updates=applied,
            skill_updates=output["skill_updates"], decision=output["skill_updates"]["fault_skill"]["decision"],
            updated_skill_ids=[a["updated_skill_id"] for a in applied if a.get("updated_skill_id")],
            updated_skill_types=["fault_skill"] if applied else [], runtime_fault_family=runtime,
            retrieval_entry_changes=routes, fault_catalog_search_trace=catalog["skill_search_trace"],
            reflector_fault_family=queries["fault_family"], fault_family_consistent=runtime == queries["fault_family"])
    except Exception as exc:
        summary.update(status="failed", failure_stage=stage, reason=str(exc))
    finally:
        write_json(out / "case_evolution_summary.json", summary)
    return summary
