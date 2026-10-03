"""Evaluate the FL-only SWE-Exp adaptation on frozen RQ1 public cases.

This arm keeps SWE-Exp's issue-type E5 recall, LLM selection, and separate
Instructor guidance, but replaces APR modification with function localization.
It is not a reproduction of the author's reported patch-repair score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from build_swe_exp_loc_memory import AUTHOR_PROMPTS, _response_object, literal_prompts
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from prepare_rq1_memory_baselines import EXPANDED, HISTORICAL, OUTPUT as PREPARATION, read_json, write_json
from prepare_swe_exp_loc import DESTINATION
from run_rq1_memory_baseline import materialize
import run_swe_explore_v5_stratified as bench
from swe_exp_loc_retrieval import E5Encoder, load_index, query_text, recall
from swe_exp_loc_selector import InstructorGuidedClient, select_one


OUT = DESTINATION / "evaluation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(*, expanded: Path, preparation: Path, destination: Path,
                    allow_partial_smoke: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    prepared = read_json(preparation / "preparation_summary.json")
    if prepared["status"] != "complete" or digest(preparation / "episodes.jsonl") != prepared["prepared_hashes"]["episodes.jsonl"]:
        raise ValueError("Acquisition episodes are incomplete or changed")
    for name, expected in prepared["manifest_hashes"].items():
        if digest(expanded / name) != expected:
            raise ValueError("Frozen RQ1 manifest changed: " + name)
    cases = read_json(expanded / "evaluation_cases.json")
    if len(cases) != 500 or len({row["instance_id"] for row in cases}) != 500:
        raise ValueError("Expected 500 unique frozen evaluation cases")
    if any(not str(row.get("problem_statement") or "").strip() for row in cases):
        raise ValueError("Frozen evaluation contains an empty issue")
    episodes = {row["instance_id"]: row for row in (
        json.loads(line) for line in (preparation / "episodes.jsonl").read_text(encoding="utf-8").splitlines())}
    plan = [json.loads(line) for line in (destination / "acquisition_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    eligible = {row["instance_id"] for row in plan
                if row["eligible"] and not row["skill_conditioned_acquisition"]}
    if len(episodes) != 400 or len(eligible) != 335 or eligible & {case["instance_id"] for case in cases}:
        raise ValueError("Acquisition/evaluation identities changed")
    index_path = destination / "e5_issue_types.jsonl"
    indexed_ids = ([json.loads(line)["instance_id"] for line in index_path.read_text(encoding="utf-8").splitlines()]
                   if index_path.is_file() else [])
    experiences = [read_json(destination / "experiences" / (cid + ".json")) for cid in indexed_ids]
    if not allow_partial_smoke and set(indexed_ids) != eligible:
        raise ValueError("Full SWE-Exp-Loc acquisition and E5 index are required")
    if set(indexed_ids) - eligible or len(set(indexed_ids)) != len(indexed_ids):
        raise ValueError("E5 index contains an ineligible acquisition case")
    for row in experiences:
        source = episodes[row["instance_id"]]
        expected = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if (row.get("source_episode_sha256") != expected or row.get("issue") != source["issue"]
                or row.get("source_skill_conditioned")):
            raise ValueError("Experience source changed: " + row["instance_id"])
    indexed = load_index(experiences, index_path)
    return cases, experiences, indexed


def classify_issue(client, issue: str, prompts: dict) -> dict:
    result = _response_object(client, [
        {"role": "system", "content": prompts["issue_type_system_prompt"]},
        {"role": "user", "content": prompts["issue_type_user_prompt"].format(issue)},
    ])
    if not all(isinstance(result.get(key), str) and result[key].strip()
               for key in ("issue_type", "description")):
        raise ValueError("SWE-Exp query issue type is incomplete")
    return {key: result[key].strip() for key in ("issue_type", "description")}


def evaluate_one(case: dict, *, client, encoder, experiences: list[dict], indexed: list[dict],
                 prompts: dict, expanded: Path, historical: Path, output: Path) -> dict:
    cid = case["instance_id"]
    directory = output / "cases" / cid
    result_path = directory / "result.json"
    if result_path.is_file():
        result = read_json(result_path)
        selection = read_json(directory / "experience_selection.json")
        return {"instance_id": cid, "status": result["status"],
                "predictions": (result.get("ranked_functions") or [])[:5],
                "metrics": bench.strict_metrics((result.get("ranked_functions") or [])[:5],
                                                case["function_ground_truth"]),
                "memory_source_id": selection["selection"]["selected_source_id"],
                "instructor_calls": len(read_json(directory / "instructor_guidance.json"))}
    if directory.exists() and any(directory.iterdir()):
        attempt = 1
        while directory.with_name(f"{cid}.interrupted_{attempt}").exists():
            attempt += 1
        directory.rename(directory.with_name(f"{cid}.interrupted_{attempt}"))
    issue = case["problem_statement"].strip()
    query = classify_issue(client, issue, prompts)
    vector = encoder.encode([query_text(query)])[0]
    candidates = recall(query=query, current_repo=case["repo"],
                        experiences=experiences, indexed=indexed, query_vector=vector)
    by_id = {row["instance_id"]: row for row in experiences}
    selection = select_one(client, issue=issue, candidates=candidates,
                           experiences=by_id, prompts=prompts)
    workspace, root = materialize(case, output, expanded, historical)
    config = copy.deepcopy(read_json(expanded / "config.json"))
    config["skill_bank"]["path"] = str(expanded / "empty_skills.jsonl")
    config["skill_bank"]["enabled_skill_types"] = []
    guided = (InstructorGuidedClient(client, issue=issue,
                                    selected_experience=by_id[selection["selected_source_id"]])
              if selection["selected_source_id"] else client)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "experience_selection.json", {"query": query,
        "candidates": candidates, "selection": selection, "index_model": "intfloat/multilingual-e5-large-instruct"})
    try:
        result = run_explorer(task={"instance_id": cid, "repo": case["repo"],
            "base_commit": case["base_commit"], "bug_report": issue,
            "repo_path": str(root), "run_dir": str(directory)},
            config=config, llm_client=guided, skill_bank=make_skill_bank(config))
    finally:
        bench.OUT = output
        bench.clean_workspace(workspace)
        write_json(directory / "instructor_guidance.json", getattr(guided, "guidance", []))
    predictions = (result.get("ranked_functions") or [])[:5]
    return {"instance_id": cid, "status": result["status"], "predictions": predictions,
            "metrics": bench.strict_metrics(predictions, case["function_ground_truth"]),
            "memory_source_id": selection["selected_source_id"],
            "instructor_calls": len(getattr(guided, "guidance", []))}


def run(*, limit: int = 500, dry_run: bool = False, allow_partial_smoke: bool = False,
        output: Path = OUT, expanded: Path = EXPANDED, historical: Path = HISTORICAL,
        preparation: Path = PREPARATION, destination: Path = DESTINATION,
        client=None, encoder=None) -> dict:
    if limit < 1 or limit > 500:
        raise ValueError("Evaluation limit must be 1..500")
    if allow_partial_smoke and limit != 1:
        raise ValueError("Partial acquisition is allowed only for a one-case smoke")
    cases, experiences, indexed = validate_inputs(expanded=expanded, preparation=preparation,
        destination=destination, allow_partial_smoke=allow_partial_smoke)
    if dry_run:
        return {"status": "dry_run", "evaluation_selected": limit,
                "acquisition_experiences": len(experiences), "model_calls": 0,
                "partial_acquisition_smoke": allow_partial_smoke}
    if client is None:
        client = OpenAICompatibleClient.from_config(read_json(expanded / "config.json")["llm"])
    encoder = encoder or E5Encoder()
    prompts = literal_prompts(AUTHOR_PROMPTS)
    output.mkdir(parents=True, exist_ok=True)
    audit = {"evaluation_manifest_sha256": digest(expanded / "evaluation_cases.json"),
        "experience_index_sha256": digest(destination / "e5_issue_types.jsonl"),
        "acquisition_count": len(experiences), "excluded_skill_conditioned_acquisitions": 62,
        "partial_acquisition_smoke": allow_partial_smoke,
        "method_label": "SWE-Exp-Loc (FL adaptation, not official APR reproduction)"}
    audit_path = output / "input_audit.json"
    if audit_path.is_file() and read_json(audit_path) != audit:
        raise ValueError("SWE-Exp-Loc evaluation input changed; use a separate output directory")
    write_json(audit_path, audit)
    rows = []
    for case in cases[:limit]:
        try:
            row = evaluate_one(case, client=client, encoder=encoder, experiences=experiences,
                indexed=indexed, prompts=prompts, expanded=expanded, historical=historical,
                output=output)
        except Exception as exc:
            row = {"instance_id": case["instance_id"], "status": "adapter_failed",
                   "error_type": type(exc).__name__, "error": str(exc)[:300],
                   "predictions": [], "metrics": {"top1": False, "top3": False,
                   "top5": False, "mrr": 0}, "memory_source_id": None,
                   "instructor_calls": 0}
        rows.append(row)
        count = len(rows)
        summary = {"method": "SWE-Exp-Loc (FL adaptation)", "processed": count,
                   "target": limit, "acquisition_experiences": len(experiences),
                   "statuses": dict(Counter(item["status"] for item in rows)),
                   "selected_memory_count": sum(item["memory_source_id"] is not None for item in rows),
                   "instructor_calls": sum(item["instructor_calls"] for item in rows),
                   **{metric: sum(item["metrics"][metric] for item in rows) / count
                      for metric in ("top1", "top3", "top5", "mrr")}, "cases": rows}
        write_json(output / "comparison_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(limit=args.limit, dry_run=args.dry_run,
                         allow_partial_smoke=args.allow_partial_smoke,
                         output=args.output), indent=2))
