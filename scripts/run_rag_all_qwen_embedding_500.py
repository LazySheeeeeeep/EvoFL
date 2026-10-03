"""Evaluate RAG-All: Qwen retrieval over all 400 acquisition trajectories.

The retrieved trajectory is injected directly into the Explorer prompt. No
Fault SkillBank, reflection output, patch, or evaluation ground truth is used.
"""
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
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from rq1_memory_runtime import MemoryInjectedClient
from run_rq1_memory_baseline import materialize
import run_swe_explore_v5_stratified as bench

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
HISTORICAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
PREPARATION = ROOT / "runs/rq1_memory_baseline_preparation_20260924"
OUTPUT = ROOT / "runs/rag_all_qwen_embedding_500_20261001"
EPISODES = PREPARATION / "episodes.jsonl"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.7-text-embedding-flash"
QWEN_BATCH_SIZE = 4
QWEN_MAX_BATCH_CHARS = 48000
RETRY_STATUSES = {
    "failed",
    "empty_prediction",
    "llm_request_failed",
    "adapter_failed",
    "agent_interrupted",
}
WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_RAG_ALL_WORK_ROOT",
        "/home/lql/.cache/rag_all_qwen_work",
    )
)


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qwen_embed(texts: list[str], *, timeout: int = 180) -> list[list[float]]:
    if not texts or len(texts) > QWEN_BATCH_SIZE:
        raise ValueError(f"Qwen embedding batch must contain 1..{QWEN_BATCH_SIZE} texts")
    key = os.environ.get("QWEN_API_KEY")
    if not key:
        raise ValueError("QWEN_API_KEY must be supplied")
    body = json.dumps(
        {"model": QWEN_MODEL, "input": texts, "encoding_format": "float"}
    ).encode("utf-8")
    delays = [2, 4, 8, 16, 32]
    last_error: Exception | None = None
    for attempt in range(len(delays) + 1):
        request = urllib.request.Request(
            f"{QWEN_BASE_URL}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = sorted(payload.get("data") or [], key=lambda row: row.get("index", 0))
            vectors = [row.get("embedding") for row in rows]
            if len(vectors) != len(texts) or any(
                not isinstance(vector, list) for vector in vectors
            ):
                raise ValueError("Qwen embedding response has an invalid vector count")
            return vectors
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Qwen embedding HTTP {exc.code}: {detail[:500]}"
            )
            transient_internal = exc.code == 400 and any(
                marker in detail
                for marker in ("batching backend", "InternalError", "timeout")
            )
            if (
                exc.code not in {429, 500, 502, 503, 504}
                and not transient_internal
            ) or attempt == len(delays):
                raise last_error from exc
        except Exception as exc:
            last_error = exc
            if attempt == len(delays):
                raise
        time.sleep(delays[attempt])
    assert last_error is not None
    raise last_error


def qwen_embed_resilient(texts: list[str]) -> list[list[float]]:
    try:
        return qwen_embed(texts)
    except Exception:
        if len(texts) == 1:
            raise
        middle = len(texts) // 2
        return [
            *qwen_embed_resilient(texts[:middle]),
            *qwen_embed_resilient(texts[middle:]),
        ]


def embedding_batches(texts: list[str]) -> list[list[str]]:
    batches, current, current_chars = [], [], 0
    for text in texts:
        size = len(text)
        if current and (
            len(current) >= QWEN_BATCH_SIZE
            or current_chars + size > QWEN_MAX_BATCH_CHARS
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(text)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def episode_text(episode: dict) -> str:
    payload = {
        "repo": episode["repo"],
        "issue": episode["issue"],
        "investigation": episode["investigation"],
        "predicted_functions": episode.get("predicted_functions") or [],
        "final_summary": episode.get("final_summary") or "",
    }
    return "Historical fault-localization trajectory:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def query_text(issue: str) -> str:
    return "Repository issue for fault localization:\n" + issue.strip()


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def prepare_embeddings(limit: int, output: Path) -> tuple[list[dict], list[dict]]:
    evaluation = read(SOURCE / "evaluation_cases.json", [])
    episodes = read_jsonl(EPISODES)
    if len(episodes) != 400 or len(evaluation) != 500:
        raise ValueError("Expected 400 acquisition episodes and 500 evaluation cases")
    if {row["instance_id"] for row in episodes} & {
        row["instance_id"] for row in evaluation
    }:
        raise ValueError("Acquisition/evaluation overlap")
    manifest = {
        "method": "RAG-All",
        "memory_count": len(episodes),
        "evaluation_count": limit,
        "embedding_model": QWEN_MODEL,
        "top_k": 1,
        "episodes_sha256": digest(EPISODES),
        "evaluation_sha256": digest(SOURCE / "evaluation_cases.json"),
        "skillbank_enabled": False,
        "ground_truth_in_memory": False,
    }
    path = output / "experiment_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("RAG-All frozen inputs changed")
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write(path, manifest)
    memory_path = output / "memory_embeddings.jsonl"
    if memory_path.is_file():
        memory_rows = read_jsonl(memory_path)
    else:
        memory_rows = []
        texts = [episode_text(row) for row in episodes]
        completed = 0
        for batch in embedding_batches(texts):
            vectors = qwen_embed_resilient(batch)
            for episode, text, vector in zip(
                episodes[completed : completed + len(batch)], batch, vectors
            ):
                memory_rows.append(
                    {
                        "instance_id": episode["instance_id"],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "embedding": vector,
                    }
                )
            completed += len(batch)
            write_jsonl(memory_path, memory_rows)
            print(f"memory embeddings {completed}/{len(texts)}", flush=True)
    query_path = output / "query_embeddings.jsonl"
    selected = evaluation[:limit]
    if query_path.is_file():
        query_rows = read_jsonl(query_path)
    else:
        query_rows = []
        texts = [query_text(case["problem_statement"]) for case in selected]
        completed = 0
        for batch in embedding_batches(texts):
            vectors = qwen_embed_resilient(batch)
            for case, text, vector in zip(
                selected[completed : completed + len(batch)], batch, vectors
            ):
                query_rows.append(
                    {
                        "instance_id": case["instance_id"],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "embedding": vector,
                    }
                )
            completed += len(batch)
            write_jsonl(query_path, query_rows)
            print(f"query embeddings {completed}/{len(texts)}", flush=True)
    return memory_rows, query_rows


def select_memories(
    episodes: list[dict], memory_rows: list[dict], query_rows: list[dict]
) -> dict[str, dict]:
    by_id = {row["instance_id"]: row for row in memory_rows}
    result = {}
    for query in query_rows:
        scored = sorted(
            (
                (
                    cosine(query["embedding"], by_id[episode["instance_id"]]["embedding"]),
                    episode,
                )
                for episode in episodes
                if episode["instance_id"] != query["instance_id"]
            ),
            key=lambda item: (-item[0], item[1]["instance_id"]),
        )
        score, episode = scored[0]
        result[query["instance_id"]] = {
            "source_instance_id": episode["instance_id"],
            "source_repo": episode["repo"],
            "retrieval_score": round(score, 8),
            "past_issue": episode["issue"],
            "past_investigation": episode["investigation"],
            "past_predicted_functions": episode.get("predicted_functions") or [],
            "past_final_summary": episode.get("final_summary") or "",
            "source_skill_conditioned": bool(episode.get("skill_conditioned")),
        }
    return result


def evaluate_one(
    case: dict,
    memory: dict,
    *,
    output: Path,
    expanded: Path,
    historical: Path,
) -> dict:
    case_id = case["instance_id"]
    directory = output / "cases" / case_id
    result_path = directory / "result.json"
    existing = read(result_path)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if directory.exists():
        attempt = 1
        while directory.with_name(
            f"{case_id}.interrupted_{attempt}"
        ).exists():
            attempt += 1
        archived = directory.with_name(f"{case_id}.interrupted_{attempt}")
        directory.rename(archived)
    workspace = None
    try:
        workspace, root = materialize(case, WORK_ROOT, expanded, historical)
        config = copy.deepcopy(read(expanded / "config.json"))
        config["skill_bank"]["path"] = str(expanded / "empty_skills.jsonl")
        config["skill_bank"]["enabled_skill_types"] = []
        client = OpenAICompatibleClient.from_config(config["llm"])
        client = MemoryInjectedClient(
            client,
            instance_id=case_id,
            issue=case["problem_statement"],
            memory=memory,
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
            llm_client=client,
            skill_bank=make_skill_bank(config),
        )
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        record = {
            "instance_id": case_id,
            "repo": case["repo"],
            "status": result.get("status"),
            "predictions": predictions,
            "metrics": bench.strict_metrics(predictions, case["function_ground_truth"]),
            "memory_source_id": memory["source_instance_id"],
            "retrieval_score": memory["retrieval_score"],
            "memory_source_skill_conditioned": memory["source_skill_conditioned"],
            "memory_injection_count": getattr(client, "injection_count", 0),
            "steps": result.get("steps"),
            "runtime_seconds": result.get("runtime_seconds"),
        }
    except Exception as exc:
        record = {
            "instance_id": case_id,
            "repo": case["repo"],
            "status": "adapter_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "predictions": [],
            "metrics": bench.strict_metrics([], case["function_ground_truth"]),
            "memory_source_id": memory["source_instance_id"],
            "retrieval_score": memory["retrieval_score"],
            "memory_source_skill_conditioned": memory["source_skill_conditioned"],
        }
    finally:
        if workspace is not None:
            bench.OUT = WORK_ROOT
            bench.clean_workspace(workspace)
    write(result_path, record)
    return record


def summarize(rows: list[dict], limit: int) -> dict:
    return {
        "method": "RAG-All",
        "embedding_model": QWEN_MODEL,
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


def run(
    *,
    limit: int = 500,
    workers: int = 4,
    output: Path = OUTPUT,
    dry_run: bool = False,
) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be 1..500")
    episodes = read_jsonl(EPISODES)
    if dry_run:
        evaluation = read(SOURCE / "evaluation_cases.json", [])
        return {
            "method": "RAG-All",
            "status": "dry_run",
            "memory_count": len(episodes),
            "evaluation_selected": min(limit, len(evaluation)),
            "top_k": 1,
            "model_calls": 0,
        }
    if not os.environ.get("QWEN_API_KEY"):
        raise ValueError("QWEN_API_KEY must be supplied")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be supplied")
    memory_rows, query_rows = prepare_embeddings(limit, output)
    memories = select_memories(episodes, memory_rows, query_rows)
    evaluation = read(SOURCE / "evaluation_cases.json", [])[:limit]
    previous = read(output / "comparison_summary.json", {})
    existing = {row["instance_id"]: row for row in previous.get("cases", [])}
    pending = [
        case
        for case in evaluation
        if case["instance_id"] not in existing
        or existing[case["instance_id"]].get("status") in RETRY_STATUSES
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                evaluate_one,
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
                f"RAG-All {len(ordered)}/{limit} {case_id}: "
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(
        limit=args.limit,
        workers=args.workers,
        output=args.output,
        dry_run=args.dry_run,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
