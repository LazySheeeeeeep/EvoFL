"""Run a fixed, resumable LocAgent FL pilot on the frozen RQ1 evaluation set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from normalize_locagent_functions import normalize_file
from run_locagent_live_smoke import run as run_author
import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench

OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/locagent_rq1_pilot10"
SEED = 20260929


class GraphBuildTimeout(Exception):
    pass


def build_graph_bounded(case_id: str, output: Path, timeout_seconds: int) -> float:
    started = time.monotonic()
    command = [
        "timeout", "--signal=INT", "--kill-after=10s", f"{timeout_seconds}s",
        sys.executable, str(ROOT / "scripts/prepare_locagent_graph.py"),
        "--case-id", case_id, "--source", str(rq.OUT),
        "--output", str(output),
    ]
    environment = {key: value for key, value in os.environ.items() if key != "DEEPSEEK_API_KEY"}
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=environment,
                                timeout=timeout_seconds + 30)
    except subprocess.TimeoutExpired as exc:
        raise GraphBuildTimeout(f"Author graph build exceeded {timeout_seconds}s") from exc
    elapsed = round(time.monotonic() - started, 3)
    if result.returncode == 124:
        raise GraphBuildTimeout(f"Author graph build exceeded {timeout_seconds}s")
    if result.returncode != 0:
        raise RuntimeError(f"Author graph build failed ({result.returncode}): {result.stderr[-500:]}")
    if not (output / f"{case_id}.pkl").is_file():
        raise RuntimeError("Author graph build reported success without an index")
    return elapsed


def write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_cases(limit: int) -> list[dict]:
    rq.verify_protocol()
    cases = rq.read(rq.OUT / "evaluation_cases.json")
    if len(cases) != 500 or len({case["instance_id"] for case in cases}) != 500:
        raise ValueError("Frozen evaluation case set is invalid")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    return random.Random(SEED).sample(cases, 500)[:limit]


def summarize(rows: list[dict], target: int) -> dict:
    return {
        "method": "LocAgent-Loc (DeepSeek FL adaptation)",
        "processed": len(rows), "target": target,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "top1": sum(row["metrics"]["top1"] for row in rows) / len(rows) if rows else None,
        "top3": sum(row["metrics"]["top3"] for row in rows) / len(rows) if rows else None,
        "top5": sum(row["metrics"]["top5"] for row in rows) / len(rows) if rows else None,
        "mrr": sum(row["metrics"]["mrr"] for row in rows) / len(rows) if rows else None,
        "model_calls": sum(row.get("model_calls", 0) for row in rows),
        "cases": rows,
    }


def run(limit: int, output: Path = OUTPUT, *, dry_run: bool = False,
        graph_timeout_seconds: int = 600) -> dict:
    if graph_timeout_seconds < 1:
        raise ValueError("graph_timeout_seconds must be positive")
    selected = select_cases(limit)
    bank_sha = rq.sha(rq.OUT / "frozen_skills.jsonl")
    if bank_sha != rq.read(rq.OUT / "training_complete.json")["bank_sha256"]:
        raise ValueError("Frozen SkillBank changed")
    manifest = {
        "seed": SEED, "target": limit, "graph_timeout_seconds": graph_timeout_seconds,
        "case_ids": [case["instance_id"] for case in selected],
        "evaluation_sha256": rq.sha(rq.OUT / "evaluation_cases.json"),
        "bank_sha256": bank_sha,
        "public_model_fields": ["instance_id", "repo", "base_commit", "problem_statement"],
        "note": "The author localization loop is adapted to DeepSeek; no author fine-tuning is reproduced.",
    }
    if dry_run:
        return manifest
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be set in this process")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "selection_manifest.json"
    if manifest_path.exists():
        if rq.read(manifest_path) != manifest:
            raise ValueError("Pilot selection or frozen input changed")
    else:
        write(manifest_path, manifest)

    rows = []
    for case in selected:
        cid = case["instance_id"]
        case_output = output / "cases" / cid
        record_path = case_output / "record.json"
        if record_path.exists():
            record = rq.read(record_path)
        else:
            graph = output / "graph_indexes" / f"{cid}.pkl"
            print(f"Starting {len(rows) + 1}/{limit} {cid} (graph)", flush=True)
            try:
                graph_elapsed = 0.0
                if not graph.exists():
                    graph_elapsed = build_graph_bounded(cid, output / "graph_indexes",
                                                        graph_timeout_seconds)
                report = run_author(cid, rq.OUT, output / "graph_indexes", case_output)
                archive = rq.OUT / "sources" / f"{cid}.tar.gz"
                source = rq.read(rq.OUT / "materialized" / f"{cid}.json")
                normalized = normalize_file(case_output / "smoke_result.json", graph,
                    case_output / "normalized_prediction.json", archive,
                    source["original_archive_sha256"])
                predictions = normalized["ranked_functions"][:5]
                status = report["status"] if predictions else "empty_prediction"
                record = {"instance_id": cid, "status": status,
                          "predictions": predictions, "model_calls": report["model_calls"],
                          "graph_build_seconds": graph_elapsed,
                          "graph_source": str(graph), "metrics": bench.strict_metrics(
                              predictions, case["function_ground_truth"])}
            except GraphBuildTimeout as exc:
                record = {"instance_id": cid, "status": "graph_timeout",
                          "error": str(exc), "model_calls": 0,
                          "predictions": [], "metrics": bench.strict_metrics([], case["function_ground_truth"])}
            except Exception as exc:
                record = {"instance_id": cid, "status": "adapter_failed", "error":
                          f"{type(exc).__name__}: {str(exc)[:350]}", "model_calls": 0,
                          "predictions": [], "metrics": bench.strict_metrics([], case["function_ground_truth"])}
            write(record_path, record)
        rows.append(record)
        write(output / "comparison_summary.json", summarize(rows, limit))
        print(f"{len(rows)}/{limit} {cid}: {record['status']}", flush=True)
    return summarize(rows, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--graph-timeout-seconds", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.limit, args.output, dry_run=args.dry_run,
                 graph_timeout_seconds=args.graph_timeout_seconds)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
