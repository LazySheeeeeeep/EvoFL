"""RAG-Success: top-1 Qwen retrieval over successful acquisition trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_rag_all_qwen_embedding_500 as rag

SOURCE = rag.SOURCE
HISTORICAL = rag.HISTORICAL
ALL_RAG_OUTPUT = ROOT / "runs/rag_all_qwen_embedding_500_20261001"
TRAINING = SOURCE / "training"
HISTORICAL_TRAINING = HISTORICAL / "training"
OUTPUT = ROOT / "runs/rag_success_qwen_embedding_500_20261003"


def read(path: Path, default=None):
    return rag.read(path, default)


def write(path: Path, value) -> None:
    rag.write(path, value)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def acquisition_cases() -> list[dict]:
    cases = [
        *read(SOURCE / "original_training.json", []),
        *read(SOURCE / "training_additions.json", []),
    ]
    if len(cases) != 400:
        raise ValueError("Expected 400 acquisition cases")
    return cases


def success_ids() -> list[str]:
    result = []
    for case in acquisition_cases():
        case_id = case["instance_id"]
        record = read(TRAINING / case_id / "record.json")
        if record is None:
            record = read(HISTORICAL_TRAINING / case_id / "record.json")
        if not record or not isinstance(record.get("metrics"), dict):
            raise ValueError(f"Missing acquisition outcome for {case_id}")
        if record["metrics"].get("top5") is True:
            result.append(case_id)
    return result


def memory_rows(ids: list[str]) -> list[dict]:
    rows = rag.read_jsonl(ALL_RAG_OUTPUT / "memory_embeddings.jsonl")
    by_id = {row["instance_id"]: row for row in rows}
    missing = [case_id for case_id in ids if case_id not in by_id]
    if missing:
        raise ValueError(f"RAG-All memory embeddings missing {len(missing)} success cases")
    return [by_id[case_id] for case_id in ids]


def query_rows(limit: int) -> list[dict]:
    rows = rag.read_jsonl(ALL_RAG_OUTPUT / "query_embeddings.jsonl")
    if len(rows) < limit:
        raise ValueError("RAG-All query embeddings are incomplete")
    return rows[:limit]


def prepare(output: Path, limit: int) -> tuple[list[dict], list[dict], dict[str, dict]]:
    ids = success_ids()
    all_episodes = rag.read_jsonl(rag.EPISODES)
    by_id = {row["instance_id"]: row for row in all_episodes}
    episodes = [by_id[case_id] for case_id in ids]
    manifest = {
        "method": "RAG-Success",
        "memory_source": "main acquisition Explorer Top-5 hit",
        "memory_count": len(ids),
        "evaluation_count": limit,
        "embedding_model": rag.QWEN_MODEL,
        "top_k": 1,
        "success_ids_sha256": sha256_text("\n".join(ids)),
        "source_embeddings": str(ALL_RAG_OUTPUT),
        "skillbank_enabled": False,
        "ground_truth_in_memory": False,
    }
    path = output / "experiment_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("RAG-Success frozen inputs changed")
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write(path, manifest)
    memories = rag.select_memories(episodes, memory_rows(ids), query_rows(limit))
    return episodes, query_rows(limit), memories


def summarize(rows: list[dict], limit: int) -> dict:
    return {
        "method": "RAG-Success",
        "embedding_model": rag.QWEN_MODEL,
        "top_k": 1,
        "processed": len(rows),
        "target": limit,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "memory_retrieved": sum(row.get("memory_source_id") is not None for row in rows),
        "memory_from_skill_conditioned_source": sum(
            row.get("memory_source_skill_conditioned") is True for row in rows
        ),
        **{
            metric: sum(row["metrics"][metric] for row in rows) / len(rows)
            if rows
            else None
            for metric in ("top1", "top3", "top5", "mrr")
        },
        "cases": rows,
    }


def run(*, limit: int = 500, workers: int = 12, output: Path = OUTPUT) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be 1..500")
    _, _, memories = prepare(output, limit)
    evaluation = read(SOURCE / "evaluation_cases.json", [])[:limit]
    previous = read(output / "comparison_summary.json", {})
    existing = {row["instance_id"]: row for row in previous.get("cases", [])}
    pending = [
        case
        for case in evaluation
        if case["instance_id"] not in existing
        or existing[case["instance_id"]].get("status") in rag.RETRY_STATUSES
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                rag.evaluate_one,
                case,
                memories[case["instance_id"]],
                output=output,
                expanded=SOURCE,
                historical=HISTORICAL,
            ): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            existing[case_id] = future.result()
            ordered = [
                existing[case["instance_id"]]
                for case in evaluation
                if case["instance_id"] in existing
            ]
            summary = summarize(ordered, limit)
            write(output / "comparison_summary.json", summary)
            print(
                f"RAG-Success {len(ordered)}/{limit} {case_id}: "
                f"{existing[case_id]['status']}",
                flush=True,
            )
    return summarize(
        [existing[case["instance_id"]] for case in evaluation],
        limit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be supplied")
    result = run(limit=args.limit, workers=args.workers, output=args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
