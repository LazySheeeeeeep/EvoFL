"""SWE-Exp-Loc with RAG-Success trajectories as the experience source."""
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_skill_count_ablation as bank_runner
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from swe_exp_loc_selector import InstructorGuidedClient

SOURCE = bank_runner.SOURCE
EPISODES = ROOT / "runs/rq1_memory_baseline_preparation_20260924/episodes.jsonl"
RAG_SUCCESS = ROOT / "runs/rag_success_qwen_embedding_500_20261003"
OUTPUT = ROOT / "runs/swe_exp_loc_rag_success_500_20261003"
WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_SWE_EXP_RAG_WORK_ROOT",
        "/home/lql/.cache/swe_exp_loc_rag_success",
    )
)
RETRY_STATUSES = {"llm_request_failed", "adapter_failed", "agent_interrupted"}


def read(path: Path, default=None):
    return bank_runner.read(path, default)


def write(path: Path, value) -> None:
    bank_runner.write(path, value)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rag_memories() -> dict[str, dict]:
    episodes = {row["instance_id"]: row for row in read_jsonl(EPISODES)}
    summary = read(RAG_SUCCESS / "comparison_summary.json", {})
    selection = {}
    for row in summary.get("cases") or []:
        source_id = row.get("memory_source_id")
        if not source_id or source_id not in episodes:
            raise ValueError(f"Missing RAG-Success memory for {row.get('instance_id')}")
        source = episodes[source_id]
        selection[row["instance_id"]] = {
            "issue": source["issue"],
            "past_investigation": source.get("investigation") or [],
            "past_predicted_functions": source.get("predicted_functions") or [],
            "past_final_summary": source.get("final_summary") or "",
            "source_instance_id": source_id,
            "source_repo": source.get("repo"),
            "retrieval_score": row.get("retrieval_score"),
            "source_skill_conditioned": bool(source.get("skill_conditioned")),
        }
    if len(selection) != 500:
        raise ValueError("Expected 500 RAG-Success retrieval mappings")
    return selection


def prepare(output: Path) -> dict[str, dict]:
    memories = rag_memories()
    manifest = {
        "method": "SWE-Exp-Loc + RAG-Success experience injection",
        "memory_source": str(RAG_SUCCESS),
        "memory_count": 356,
        "evaluation_count": 500,
        "selector": "disabled; reuse RAG-Success top-1",
        "instructor": "SWE-Exp-Loc Instructor",
        "skillbank_enabled": False,
    }
    path = output / "input_audit.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("RAG-Success SWE-Exp-Loc inputs changed")
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write(path, manifest)
    return memories


def evaluate_one(case: dict, memory: dict, output: Path) -> dict:
    case_id = case["instance_id"]
    directory = output / "cases" / case_id
    result_path = directory / "result.json"
    existing = read(result_path)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if directory.exists():
        attempt = 1
        while directory.with_name(f"{case_id}.interrupted_{attempt}").exists():
            attempt += 1
        directory.rename(directory.with_name(f"{case_id}.interrupted_{attempt}"))
    workspace = WORK_ROOT / "work" / case_id
    started = time.monotonic()
    try:
        root = bank_runner.materialize(case, workspace)
        config = bank_runner.build_config(SOURCE / "empty_skills.jsonl")
        config["skill_bank"]["enabled_skill_types"] = []
        client = OpenAICompatibleClient.from_config(config["llm"])
        guided = InstructorGuidedClient(
            client,
            issue=case["problem_statement"],
            selected_experience=memory,
        )
        result = run_explorer(
            task={
                "instance_id": case_id,
                "repo": case["repo"],
                "base_commit": case["base_commit"],
                "bug_report": case["problem_statement"],
                "repo_path": str(root),
                "run_dir": str(directory),
            },
            config=config,
            llm_client=guided,
            skill_bank=make_skill_bank(config),
        )
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        record = {
            "instance_id": case_id,
            "repo": case["repo"],
            "status": result.get("status"),
            "predictions": predictions,
            "metrics": bank_runner.bench.strict_metrics(
                predictions, case["function_ground_truth"]
            ),
            "memory_source_id": memory["source_instance_id"],
            "retrieval_score": memory["retrieval_score"],
            "memory_source_skill_conditioned": memory["source_skill_conditioned"],
            "instructor_calls": len(guided.guidance),
            "runtime_seconds": result.get("runtime_seconds")
            or round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        record = {
            "instance_id": case_id,
            "repo": case["repo"],
            "status": "adapter_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "predictions": [],
            "metrics": bank_runner.bench.strict_metrics(
                [], case["function_ground_truth"]
            ),
            "memory_source_id": memory["source_instance_id"],
            "retrieval_score": memory["retrieval_score"],
            "memory_source_skill_conditioned": memory["source_skill_conditioned"],
            "instructor_calls": 0,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    write(result_path, record)
    return record


def summarize(rows: list[dict], limit: int) -> dict:
    return {
        "method": "SWE-Exp-Loc + RAG-Success",
        "processed": len(rows),
        "target": limit,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "memory_retrieved": sum(row.get("memory_source_id") is not None for row in rows),
        "instructor_calls": sum(row.get("instructor_calls") or 0 for row in rows),
        **{
            metric: sum(row["metrics"][metric] for row in rows) / len(rows)
            if rows
            else None
            for metric in ("top1", "top3", "top5", "mrr")
        },
        "cases": rows,
    }


def run(*, limit: int = 500, workers: int = 12, output: Path = OUTPUT) -> dict:
    memories = prepare(output)
    cases = read(SOURCE / "evaluation_cases.json", [])[:limit]
    previous = read(output / "comparison_summary.json", {})
    existing = {row["instance_id"]: row for row in previous.get("cases", [])}
    pending = [
        case
        for case in cases
        if case["instance_id"] not in existing
        or existing[case["instance_id"]].get("status") in RETRY_STATUSES
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(evaluate_one, case, memories[case["instance_id"]], output): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            existing[case_id] = future.result()
            ordered = [
                existing[case["instance_id"]]
                for case in cases
                if case["instance_id"] in existing
            ]
            summary = summarize(ordered, len(cases))
            write(output / "comparison_summary.json", summary)
            print(
                f"SWE-Exp-Loc+RAG-Success {len(ordered)}/{len(cases)} "
                f"{case_id}: {existing[case_id]['status']}",
                flush=True,
            )
    return summarize([existing[case["instance_id"]] for case in cases], len(cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(limit=args.limit, workers=args.workers, output=args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
