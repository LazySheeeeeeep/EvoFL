"""Run the frozen RQ1 Explorer with one flat LLM selection over all Fault Skills."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer.v5_agent import V5ExplorerAgent
from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from evolutefl.skills.fault_taxonomy import validate_fault_family

OUTPUT = ROOT / "runs/rq1_global_fault_catalog_ablation_20260929"
SELECTOR_PROMPT = ROOT / "prompt_records/explorer/fault_skill_global_selector_ablation_v1.txt"
RETRY_STATUSES = {"llm_request_failed", "adapter_failed", "agent_interrupted"}


class GlobalCatalogExplorer(V5ExplorerAgent):
    """Keep Explorer timing but replace family routing with one global catalog call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fault_selector_prompt = SELECTOR_PROMPT.read_text(encoding="utf-8")

    def load_fault(self, args: dict, issue: str, log) -> dict:
        requested_family = validate_fault_family(args.get("fault_family"))
        for field in ("fault_signature", "project_context", "suspected_path"):
            if not isinstance(args.get(field), str) or not args[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        request = {
            "issue_report": issue,
            "request": args,
            "orientation": {
                "observed_context": args["project_context"],
                "prior_investigation": log.timeline[-4:],
            },
        }
        candidates = []
        if "fault_skill" in self.skill_bank.enabled_skill_types:
            candidates = sorted(
                ({"skill_id": skill.skill_id, "title": skill.title, "trigger": skill.trigger}
                 for skill in self.skill_bank.active_skills()
                 if skill.skill_type == "fault_skill"),
                key=lambda item: (item["title"].lower(), item["skill_id"]),
            )
        selected, attempts = {"selected_skill_id": None, "reason": "Empty or disabled catalog."}, []
        if candidates:
            selected, attempts = self._select_fault_skill(
                {**request, "candidates": candidates},
                candidate_ids={candidate["skill_id"] for candidate in candidates},
            )
        selected_id = selected.get("selected_skill_id")
        matched = self.skill_bank.get_active_skill(selected_id, skill_type="fault_skill") if selected_id else None
        validation = {"applicable": False, "reason": "No candidate selected."}
        if matched:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.fault_validator_prompt},
                    {"role": "user", "content": json.dumps({**request, "candidate": matched}, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"}, tool_choice="none", temperature=0,
            )
            validation = extract_json_object(response.get("content") or "{}")
            if validation.get("applicable") is not True:
                matched = None
        catalog_chars = len(json.dumps(candidates, ensure_ascii=False))
        trace = {
            "retrieval_mode": "llm_global_title_trigger_catalog_v1",
            "requested_fault_family": requested_family,
            "catalog_skill_ids": [candidate["skill_id"] for candidate in candidates],
            "catalog_count": len(candidates),
            "catalog_char_count": catalog_chars,
            "selector_input_char_count": len(json.dumps({**request, "candidates": candidates}, ensure_ascii=False)),
            "selector": selected,
            "selector_attempts": attempts,
            "validator": validation,
            "loaded_skill_primary_family": (matched or {}).get("fault_family"),
            "family_route_bypassed": True,
        }
        return {"skill_type": "fault_skill", "fault_family": requested_family,
                "matched_skill": matched, "search_trace": trace}


def prepare(output: Path, limit: int) -> list[dict]:
    rq.verify_protocol()
    cases = rq.read(rq.OUT / "evaluation_cases.json")
    if len(cases) != 500 or not 1 <= limit <= 500:
        raise ValueError("Expected the frozen 500-case evaluation set")
    bank = rq.OUT / "frozen_skills.jsonl"
    bank_sha = rq.sha(bank)
    if bank_sha != rq.read(rq.OUT / "training_complete.json")["bank_sha256"]:
        raise ValueError("Frozen SkillBank changed")
    manifest = {
        "method": "single LLM selection over the complete Fault Skill title+trigger catalog",
        "target": limit,
        "case_ids": [case["instance_id"] for case in cases[:limit]],
        "evaluation_sha256": rq.sha(rq.OUT / "evaluation_cases.json"),
        "bank_sha256": bank_sha,
        "selector_prompt_sha256": hashlib.sha256(SELECTOR_PROMPT.read_bytes()).hexdigest(),
        "knowledge_visibility": "selected card only",
        "validator": "frozen RQ1 Fault Skill validator",
        "other_explorer_settings": "frozen RQ1 configuration",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection_manifest.json"
    if path.exists() and rq.read(path) != manifest:
        raise ValueError("Ablation inputs changed after the run began")
    if not path.exists():
        write_json(path, manifest)
    return cases[:limit]


def evaluate_one(case: dict, output: Path) -> dict:
    cid = case["instance_id"]
    directory = output / "cases" / cid
    path = directory / "result.json"
    try:
        if path.exists():
            result = rq.read(path)
            if result.get("status") in RETRY_STATUSES:
                archived = directory.with_name(f"{cid}.retry_failed")
                if archived.exists():
                    shutil.rmtree(archived)
                directory.rename(archived)
                result = None
        else:
            result = None
        if result is None:
            workspace = None
            try:
                workspace, root = rq.materialize(case)
                cfg = copy.deepcopy(rq.read(rq.OUT / "config.json"))
                cfg["skill_bank"]["path"] = str(rq.OUT / "frozen_skills.jsonl")
                agent = GlobalCatalogExplorer(
                    llm_client=OpenAICompatibleClient.from_config(cfg["llm"]),
                    skill_bank=make_skill_bank(cfg), config=cfg,
                    system_prompt=Path(cfg["explorer"]["system_prompt_path"]).read_text(encoding="utf-8"),
                )
                result = agent.run({"instance_id": cid, "repo": case["repo"],
                    "base_commit": case["base_commit"], "bug_report": case["problem_statement"],
                    "repo_path": str(root), "run_dir": str(directory)})
            finally:
                if workspace is not None:
                    bench.clean_workspace(workspace)
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        search = rq.read(directory / "fault_skill_search.json", {})
        return {"instance_id": cid, "status": result.get("status"), "predictions": predictions,
                "metrics": bench.strict_metrics(predictions, case["function_ground_truth"]),
                "loaded_skill_id": (search.get("matched_skill") or {}).get("skill_id"),
                "catalog_count": (search.get("search_trace") or {}).get("catalog_count"),
                "steps": result.get("steps"), "forced_finish": result.get("forced_finish")}
    except Exception as exc:
        return {"instance_id": cid, "status": "adapter_failed",
                "error": f"{type(exc).__name__}: {str(exc)[:350]}", "predictions": [],
                "metrics": bench.strict_metrics([], case["function_ground_truth"]),
                "loaded_skill_id": None}


def summarize(rows: list[dict], limit: int) -> dict:
    return {"arm": "global_title_trigger_catalog", "processed": len(rows), "target": limit,
            "statuses": dict(Counter(row["status"] for row in rows)),
            "loaded_skill_count": sum(bool(row.get("loaded_skill_id")) for row in rows),
            **{key: sum(row["metrics"][key] for row in rows) / len(rows) if rows else None
               for key in ("top1", "top3", "top5", "mrr")}, "cases": rows}


def run(limit: int, workers: int, output: Path = OUTPUT, *, dry_run: bool = False,
        resume_index: int = 1) -> dict:
    cases = prepare(output, limit)
    if dry_run:
        return rq.read(output / "selection_manifest.json")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be set in this process")
    if not 1 <= resume_index <= len(cases):
        raise ValueError("resume_index must be within the selected cases")
    existing_all = {
        row["instance_id"]: row
        for row in rq.read(output / "comparison_summary.json", {}).get("cases", [])
    }
    rerun_ids = {
        case["instance_id"] for case in cases[resume_index - 1:]
        if case["instance_id"] not in existing_all
        or existing_all[case["instance_id"]].get("status") in RETRY_STATUSES
    }
    completed = {
        case_id: row for case_id, row in existing_all.items()
        if case_id not in rerun_ids
    }
    pending = [case for case in cases if case["instance_id"] in rerun_ids]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(evaluate_one, case, output): case["instance_id"] for case in pending}
        for future in as_completed(futures):
            cid = futures[future]
            completed[cid] = future.result()
            ordered = [completed[case["instance_id"]] for case in cases if case["instance_id"] in completed]
            write_json(output / "comparison_summary.json", summarize(ordered, limit))
            print(f"{len(ordered)}/{limit} {cid}: {completed[cid]['status']}", flush=True)
    return summarize([completed[case["instance_id"]] for case in cases], limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-index", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    result = run(args.limit, args.workers, args.output, dry_run=args.dry_run,
                 resume_index=args.resume_index)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
