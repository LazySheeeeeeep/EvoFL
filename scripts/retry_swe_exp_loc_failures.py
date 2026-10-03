"""Retry failed SWE-Exp-Loc evaluation cases without overwriting the main run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from prepare_rq1_memory_baselines import EXPANDED, HISTORICAL, OUTPUT as PREPARATION, read_json, write_json
from prepare_swe_exp_loc import DESTINATION
from run_swe_exp_loc_baseline import (
    evaluate_one,
    validate_inputs,
    OUT as MAIN_OUT,
)
from swe_exp_loc_retrieval import E5Encoder
from swe_exp_loc_selector import InstructorGuidedClient
from evolutefl.llm.client import OpenAICompatibleClient
from build_swe_exp_loc_memory import AUTHOR_PROMPTS, literal_prompts
import run_swe_explore_v5_stratified as bench


MAIN_RUN_OUT = DESTINATION / "evaluation_full335"
RETRY_OUT = DESTINATION / "retry_failed_20260930"
FAILED_STATUSES = {"llm_request_failed", "adapter_failed", "agent_interrupted"}


def failed_cases(summary: dict) -> list[dict]:
    return [
        row for row in summary.get("cases", [])
        if row.get("status") in FAILED_STATUSES
    ]


def retry_one(
    case: dict,
    *,
    client,
    encoder,
    experiences: list[dict],
    indexed: list[dict],
    prompts: dict,
    expanded: Path,
    historical: Path,
    output: Path,
    attempts: int,
) -> dict:
    cid = case["instance_id"]
    attempts_report = []
    last = None
    for attempt in range(1, attempts + 1):
        directory = output / "cases" / cid
        if directory.exists():
            archived = directory.with_name(f"{cid}.attempt_{attempt - 1}")
            if archived.exists():
                shutil.rmtree(archived)
            directory.rename(archived)
        started = time.monotonic()
        try:
            last = evaluate_one(
                case,
                client=client,
                encoder=encoder,
                experiences=experiences,
                indexed=indexed,
                prompts=prompts,
                expanded=expanded,
                historical=historical,
                output=output,
            )
        except Exception as exc:
            last = {
                "instance_id": cid,
                "status": "adapter_failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "predictions": [],
                "metrics": {"top1": False, "top3": False, "top5": False, "mrr": 0},
                "memory_source_id": None,
                "instructor_calls": 0,
            }
        report = {
            "attempt": attempt,
            "seconds": round(time.monotonic() - started, 3),
            "status": last.get("status"),
            "error": last.get("error"),
        }
        attempts_report.append(report)
        if last.get("status") not in FAILED_STATUSES:
            break
        time.sleep(min(5 * attempt, 20))
    return {
        **last,
        "retry_attempts": attempts_report,
        "retry_resolved": last.get("status") not in FAILED_STATUSES,
    }


def merge_summary(source: dict, retry_rows: list[dict]) -> dict:
    replacements = {row["instance_id"]: row for row in retry_rows if row["retry_resolved"]}
    rows = [replacements.get(row["instance_id"], row) for row in source["cases"]]
    count = len(rows)
    statuses = dict(Counter(row["status"] for row in rows))
    merged = {
        **{key: value for key, value in source.items() if key != "cases"},
        "processed": count,
        "target": 500,
        "statuses": statuses,
        "retry_failed_input_count": len(retry_rows),
        "retry_resolved_count": sum(row["retry_resolved"] for row in retry_rows),
        "retry_still_failed_count": sum(not row["retry_resolved"] for row in retry_rows),
        **{
            metric: sum(bool(row["metrics"][metric]) for row in rows) / count
            for metric in ("top1", "top3", "top5")
        },
        "mrr": sum(row["metrics"]["mrr"] for row in rows) / count,
        "cases": rows,
    }
    return merged


def run(
    *,
    source_summary: Path,
    output: Path = RETRY_OUT,
    expanded: Path = EXPANDED,
    historical: Path = HISTORICAL,
    preparation: Path = PREPARATION,
    destination: Path = DESTINATION,
    attempts: int = 3,
) -> dict:
    source = read_json(source_summary)
    targets = failed_cases(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "work").mkdir(exist_ok=True)
    cases, experiences, indexed = validate_inputs(
        expanded=expanded,
        preparation=preparation,
        destination=destination,
    )
    by_id = {case["instance_id"]: case for case in cases}
    missing = [row["instance_id"] for row in targets if row["instance_id"] not in by_id]
    if missing:
        raise ValueError("Retry targets are absent from the frozen evaluation set: " + ", ".join(missing))
    client = OpenAICompatibleClient.from_config(read_json(expanded / "config.json")["llm"])
    encoder = E5Encoder()
    prompts = literal_prompts(AUTHOR_PROMPTS)
    retry_rows = []
    for index, row in enumerate(targets, 1):
        case = by_id[row["instance_id"]]
        result = retry_one(
            case,
            client=client,
            encoder=encoder,
            experiences=experiences,
            indexed=indexed,
            prompts=prompts,
            expanded=expanded,
            historical=historical,
            output=output,
            attempts=attempts,
        )
        retry_rows.append(result)
        write_json(output / "progress.json", {
            "completed": index,
            "target": len(targets),
            "instance_id": case["instance_id"],
            "status": result["status"],
            "resolved": result["retry_resolved"],
        })
        print(
            f"{index}/{len(targets)} {case['instance_id']}: "
            f"{result['status']} resolved={result['retry_resolved']}",
            flush=True,
        )
    summary = {
        "source_summary": str(source_summary),
        "failed_case_count": len(targets),
        "resolved_count": sum(row["retry_resolved"] for row in retry_rows),
        "still_failed_count": sum(not row["retry_resolved"] for row in retry_rows),
        "statuses": dict(Counter(row["status"] for row in retry_rows)),
        "rows": retry_rows,
    }
    write_json(output / "retry_summary.json", summary)
    merged = merge_summary(source, retry_rows)
    write_json(MAIN_RUN_OUT / "comparison_summary_with_retries.json", merged)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=MAIN_RUN_OUT / "comparison_summary.json",
    )
    parser.add_argument("--output", type=Path, default=RETRY_OUT)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    result = run(
        source_summary=args.source_summary,
        output=args.output,
        attempts=args.attempts,
    )
    print(json.dumps(result, indent=2))
