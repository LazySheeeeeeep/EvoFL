"""Retry only unfinished with-skill RQ1 evaluations without changing frozen results."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench


def summarize(rows: list[dict]) -> dict:
    summary = {
        "processed": len(rows),
        "target": 500,
        "label_policy": rq.POLICY,
        "note": "Original paired results are preserved; only completed with-skill retries replace failed entries.",
        "arms": {},
        "cases": rows,
    }
    for arm in rq.ARMS:
        summary["arms"][arm] = {
            key: sum(row["arms"][arm]["metrics"][key] for row in rows) / len(rows)
            for key in ("top1", "top3", "top5", "mrr")
        }
        summary["arms"][arm]["statuses"] = dict(Counter(row["arms"][arm]["status"] for row in rows))
        summary["arms"][arm]["loaded_skill_count"] = sum(
            bool(row["arms"][arm].get("loaded_skill_id")) for row in rows
        )
        summary["arms"][arm]["forced_finish_count"] = sum(
            bool(row["arms"][arm].get("forced_finish")) for row in rows
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--preflight", action="store_true", help="Validate frozen inputs and list unfinished cases")
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 3:
        parser.error("--max-attempts must be between 1 and 3")

    rq.verify_protocol()
    original = rq.read(rq.OUT / "comparison_summary.json")
    if original["processed"] != 500 or original["label_policy"] != rq.POLICY:
        raise ValueError("The frozen 500-case evaluation is incomplete or incompatible")
    cases = {case["instance_id"]: case for case in rq.read(rq.OUT / "evaluation_cases.json")}
    if set(cases) != {row["instance_id"] for row in original["cases"]}:
        raise ValueError("Evaluation case IDs differ from the frozen summary")
    bank = rq.OUT / "frozen_skills.jsonl"
    bank_sha = rq.read(rq.OUT / "training_complete.json")["bank_sha256"]
    if rq.sha(bank) != bank_sha:
        raise ValueError("Frozen SkillBank changed")

    output = rq.OUT / "with_skill_failure_retries_20260929"
    output.mkdir(exist_ok=True)
    failed = [row for row in original["cases"] if row["arms"]["with_skill"]["status"] != "completed"]
    if args.preflight:
        print(f"Frozen bank verified; {len(failed)} unfinished with-skill cases:")
        for row in failed:
            print(f"{row['instance_id']}: {row['arms']['with_skill']['status']}")
        return
    pending = [row for row in failed if not (
        output / "cases" / row["instance_id"] / "attempt_1" / "result.json"
    ).exists()]
    if pending and not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Set DEEPSEEK_API_KEY in this process before running retries")
    config = rq.read(rq.OUT / "config.json")
    audit = []
    for row in failed:
        cid = row["instance_id"]
        attempts = []
        for number in range(1, args.max_attempts + 1):
            directory = output / "cases" / cid / f"attempt_{number}"
            result_path = directory / "result.json"
            if result_path.exists():
                result = rq.read(result_path)
            else:
                workspace = None
                try:
                    workspace, root = rq.materialize(cases[cid])
                    result = rq.explorer(cases[cid], config, root, directory, bank)
                except Exception as exc:
                    result = {"status": "retry_error", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                finally:
                    if workspace is not None:
                        bench.clean_workspace(workspace)
            attempts.append({
                "number": number,
                "status": result.get("status"),
                "result_path": str(result_path) if result_path.exists() else None,
                "error": result.get("error"),
            })
            rq.write(output / "retry_progress.json", {
                "time": time.time(), "bank_sha256": bank_sha, "target": len(failed),
                "completed_cases": len(audit), "current_case": cid, "attempts": attempts,
            })
            if result.get("status") == "completed":
                break
        audit.append({
            "instance_id": cid,
            "original_status": row["arms"]["with_skill"]["status"],
            "attempts": attempts,
            "final_status": result.get("status"),
        })
        rq.write(output / "retry_audit.json", audit)
        print(f"{len(audit)}/{len(failed)} {cid}: {result.get('status')}", flush=True)

    corrected = copy.deepcopy(original["cases"])
    audit_by_id = {item["instance_id"]: item for item in audit}
    for row in corrected:
        item = audit_by_id.get(row["instance_id"])
        if not item or item["final_status"] != "completed":
            continue
        result = rq.read(Path(item["attempts"][-1]["result_path"]))
        predictions = list(dict.fromkeys(result.get("ranked_functions", [])))[:5]
        row["arms"]["with_skill"] = {
            "status": "completed",
            "predictions": predictions,
            "metrics": bench.strict_metrics(predictions, row["functions"]),
            "loaded_skill_id": (rq.read(
                Path(item["attempts"][-1]["result_path"]).parent / "fault_skill_search.json", {}
            ).get("matched_skill") or {}).get("skill_id"),
            "forced_finish": result.get("forced_finish"),
            "steps": result.get("steps"),
            "runtime_seconds": result.get("runtime_seconds"),
            "retry_result_path": item["attempts"][-1]["result_path"],
        }
    rq.write(output / "comparison_summary_with_retries.json", summarize(corrected))
    rq.write(output / "retry_complete.json", {
        "original_summary": str(rq.OUT / "comparison_summary.json"),
        "bank_sha256": bank_sha,
        "retried_case_count": len(failed),
        "recovered_case_count": sum(item["final_status"] == "completed" for item in audit),
        "unrecovered_case_ids": [item["instance_id"] for item in audit if item["final_status"] != "completed"],
    })


if __name__ == "__main__":
    main()
