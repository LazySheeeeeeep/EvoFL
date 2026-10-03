"""Compare upfront issue-only Fault Skill loading with frozen RQ1 localization."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer.v5_agent import V5ExplorerAgent
from evolutefl.json_utils import write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank

OUTPUT = ROOT / "runs/rq1_issue_only_fault_skill_ablation_20260929"
RETRY_STATUSES = {"llm_request_failed", "adapter_failed", "agent_interrupted"}


def ablation_prompt(original: str) -> str:
    marker = "The issue is the main anchor."
    if original.count(marker) != 1:
        raise ValueError("Frozen Explorer prompt has an unexpected format")
    return (
        "You localize faulty functions in an unfamiliar repository from an issue report.\n\n"
        "For this comparison, first request load_fault_skill from the issue alone, before reading any "
        "repository source. Select a fault family and summarize only issue-visible symptoms. Set "
        "project_context to 'repository not yet inspected' and suspected_path to 'undetermined'. "
        "After the Skill result, inspect the repository and rank functions from source evidence.\n\n"
        + marker + original.split(marker, 1)[1]
    )


class IssueOnlyExplorer(V5ExplorerAgent):
    def _chat_step(self, request: dict, *, step: int, traces: list, directory: Path,
                   deadline: float) -> dict:
        if step == 1:
            original = json.loads(request["messages"][1]["content"])
            self._repo_context = {key: original.get(key) for key in ("repo", "base_commit")}
            public = {key: original[key] for key in ("bug_report", "fault_families")}
            request["messages"][1]["content"] = json.dumps(public, ensure_ascii=False)
            write_json(directory / "first_request_payload.json", public)
        elif (getattr(self, "_repo_context", None) is not None
              and not getattr(self, "_repo_revealed", False)
              and (directory / "fault_skill_request.json").exists()):
            request["messages"].append({"role": "user", "content": json.dumps({
                **self._repo_context, "instruction": "Continue localization using repository evidence."
            }, ensure_ascii=False)})
            self._repo_revealed = True
        return super()._chat_step(request, step=step, traces=traces, directory=directory, deadline=deadline)

    def load_fault(self, args: dict, issue: str, log) -> dict:
        if log.observations:
            raise ValueError("Issue-only retrieval cannot use repository observations")
        actual = {**args, "project_context": "repository not yet inspected",
                  "suspected_path": "undetermined"}
        write_json(log.directory / "issue_only_retrieval_context.json", {
            "issue_report": issue, "request": actual, "prior_investigation": [],
        })
        return super().load_fault(actual, issue, SimpleNamespace(timeline=[]))


def prepare(output: Path, limit: int) -> list[dict]:
    rq.verify_protocol()
    cases = rq.read(rq.OUT / "evaluation_cases.json")
    if len(cases) != 500 or not 1 <= limit <= 500:
        raise ValueError("Expected the frozen 500-case evaluation set")
    bank = rq.OUT / "frozen_skills.jsonl"
    bank_sha = rq.sha(bank)
    if bank_sha != rq.read(rq.OUT / "training_complete.json")["bank_sha256"]:
        raise ValueError("Frozen SkillBank changed")
    source_prompt = Path(rq.read(rq.OUT / "config.json")["explorer"]["system_prompt_path"])
    prompt = ablation_prompt(source_prompt.read_text(encoding="utf-8"))
    manifest = {"method": "upfront issue-only Fault Skill loading", "target": limit,
                "case_ids": [case["instance_id"] for case in cases[:limit]],
                "evaluation_sha256": rq.sha(rq.OUT / "evaluation_cases.json"),
                "bank_sha256": bank_sha, "source_prompt_sha256": rq.sha(source_prompt),
                "issue_only_prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "fault_skill_attempt_step": 1, "other_explorer_settings": "frozen RQ1 configuration"}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "selection_manifest.json"
    prompt_path = output / "explorer_issue_only_prompt.txt"
    if manifest_path.exists():
        if rq.read(manifest_path) != manifest or prompt_path.read_text(encoding="utf-8") != prompt:
            raise ValueError("Ablation inputs changed after the run began")
    else:
        write_json(manifest_path, manifest)
        prompt_path.write_text(prompt, encoding="utf-8")
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
                cfg["explorer"]["fault_skill_attempt_step"] = 1
                cfg["explorer"]["system_prompt_path"] = str(output / "explorer_issue_only_prompt.txt")
                agent = IssueOnlyExplorer(llm_client=OpenAICompatibleClient.from_config(cfg["llm"]),
                    skill_bank=make_skill_bank(cfg), config=cfg,
                    system_prompt=(output / "explorer_issue_only_prompt.txt").read_text(encoding="utf-8"))
                result = agent.run({"instance_id": cid, "repo": case["repo"],
                    "base_commit": case["base_commit"], "bug_report": case["problem_statement"],
                    "repo_path": str(root), "run_dir": str(directory)})
            finally:
                if workspace is not None:
                    bench.clean_workspace(workspace)
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        search = rq.read(directory / "fault_skill_search.json", {})
        return {"instance_id": cid, "status": result.get("status"),
                "predictions": predictions, "metrics": bench.strict_metrics(
                    predictions, case["function_ground_truth"]),
                "loaded_skill_id": (search.get("matched_skill") or {}).get("skill_id"),
                "steps": result.get("steps"), "forced_finish": result.get("forced_finish")}
    except Exception as exc:
        return {"instance_id": cid, "status": "adapter_failed", "error":
                f"{type(exc).__name__}: {str(exc)[:350]}", "predictions": [],
                "metrics": bench.strict_metrics([], case["function_ground_truth"]),
                "loaded_skill_id": None}


def summarize(rows: list[dict], limit: int) -> dict:
    return {"arm": "issue_only_upfront_skill", "processed": len(rows), "target": limit,
            "statuses": dict(Counter(row["status"] for row in rows)),
            "loaded_skill_count": sum(bool(row["loaded_skill_id"]) for row in rows),
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
    report = run(args.limit, args.workers, args.output, dry_run=args.dry_run,
                 resume_index=args.resume_index)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
