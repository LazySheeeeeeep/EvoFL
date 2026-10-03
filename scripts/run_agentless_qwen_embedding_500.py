"""Run Agentless-FL file retrieval and related-function localization on 500 cases.

Embeddings use the official Qwen compatible endpoint
``qwen3.7-text-embedding-flash``. LLM localization remains DeepSeek V4 Flash.
The runner is resumable at case granularity and never executes repair stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_agentless_heldout30 as agentless
from prepare_rgfl_reasoning import source_files
from run_swe_explore_v5_stratified import safe_archive_filter, strict_metrics

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUT = ROOT / "runs/agentless_full500_qwen_embedding_flash_20260930"
NATIVE_WORK_ROOT = Path(
    os.environ.get("EVOLUTEFL_AGENTLESS_WORK_ROOT", "/home/lql/.cache/agentless_qwen_work")
)
QWEN_API_KEY_ENV = "QWEN_API_KEY"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.7-text-embedding-flash"
QWEN_BATCH_SIZE = 16
QWEN_MAX_BATCH_CHARS = 12000
TOP_N_FILES = 3
RETRIEVAL_FILES = 100
RETRY_STATUSES = {
    "failed",
    "empty_prediction",
    "llm_request_failed",
    "adapter_failed",
    "agent_interrupted",
}


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks_for(path: str, source: str) -> list[dict]:
    rows = []
    for index, start in enumerate(range(0, len(source), 2000)):
        text = source[start:start + 2000]
        if text.strip():
            rows.append({
                "chunk_id": f"{path}#chunk-{index}",
                "path": path,
                "text": f"File: {path}\nCode chunk:\n{text}",
            })
    return rows


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def qwen_embed(texts: list[str], *, timeout: int = 120) -> list[list[float]]:
    if not texts or len(texts) > QWEN_BATCH_SIZE:
        raise ValueError(f"Qwen embedding batch must contain 1..{QWEN_BATCH_SIZE} texts")
    key = os.environ.get(QWEN_API_KEY_ENV)
    if not key:
        raise ValueError(f"{QWEN_API_KEY_ENV} must be supplied")
    body = json.dumps({
        "model": QWEN_MODEL,
        "input": texts,
        "encoding_format": "float",
    }).encode("utf-8")
    delays = [2, 4, 8, 16, 32]
    last_error: Exception | None = None
    for attempt in range(len(delays) + 1):
        request = urllib.request.Request(
            f"{QWEN_BASE_URL}/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = sorted(payload.get("data") or [], key=lambda row: row.get("index", 0))
            vectors = [row.get("embedding") for row in rows]
            if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
                raise ValueError("Qwen embedding response has an invalid vector count")
            return vectors
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Qwen embedding HTTP {exc.code}: {detail[:500]}")
            transient_internal = (
                exc.code == 400
                and any(
                    marker in detail
                    for marker in ("batching backend", "InternalError", "timeout")
                )
            )
            if (
                exc.code not in {429, 500, 502, 503, 504}
                and not transient_internal
            ) or attempt == len(delays):
                raise last_error from exc
        except Exception as exc:  # transport errors are retried with fixed text
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


def embedding_batches(chunks: list[dict]) -> list[list[dict]]:
    batches = []
    current = []
    current_chars = 0
    for row in chunks:
        size = len(row["text"])
        if current and (
            len(current) >= QWEN_BATCH_SIZE
            or current_chars + size > QWEN_MAX_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def retrieve_files(structure: dict, problem_statement: str) -> tuple[list[str], dict]:
    files = source_files(structure)
    chunks = []
    for path, source in sorted(files.items()):
        chunks.extend(chunks_for(path, source))
    if not chunks:
        raise ValueError("No Python chunks available for retrieval")
    started = time.monotonic()
    vectors = []
    batches = embedding_batches(chunks)
    for batch in batches:
        vectors.extend(qwen_embed_resilient([row["text"] for row in batch]))
    query_vector = qwen_embed([f"Repository issue:\n{problem_statement}"])[0]
    scored = sorted(
        ((cosine(query_vector, vector), row) for row, vector in zip(chunks, vectors)),
        key=lambda item: (-item[0], item[1]["chunk_id"]),
    )
    selected = []
    seen = set()
    for score, row in scored:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        selected.append({"path": row["path"], "score": score, "chunk_id": row["chunk_id"]})
        if len(selected) >= RETRIEVAL_FILES:
            break
    return [row["path"] for row in selected], {
        "source_files": len(files),
        "chunks": len(chunks),
        "batch_size": QWEN_BATCH_SIZE,
        "max_batch_chars": QWEN_MAX_BATCH_CHARS,
        "batch_count": len(batches),
        "embedding_seconds": round(time.monotonic() - started, 3),
        "chunks_per_second": round(len(chunks) / (time.monotonic() - started), 3),
        "selected": selected,
    }


def combine_files(model_files: list[str], retrieved_files: list[str], top_n: int) -> list[str]:
    counts = Counter()
    for path in model_files[:top_n]:
        counts[path] += 1
    for path in retrieved_files[:top_n]:
        counts[path] += 1
    return [path for path, _ in counts.most_common()]


def materialize(case: dict, workspace: Path) -> Path:
    archive = SOURCE / "sources" / f"{case['instance_id']}.tar.gz"
    expected = read(SOURCE / "materialized" / f"{case['instance_id']}.json", {})[
        "original_archive_sha256"
    ]
    if sha(archive) != expected:
        raise ValueError("Frozen source archive changed")
    archive_cache = NATIVE_WORK_ROOT / "archives" / archive.name
    archive_cache.parent.mkdir(parents=True, exist_ok=True)
    if not archive_cache.is_file() or sha(archive_cache) != expected:
        temporary = archive_cache.with_suffix(".tmp")
        shutil.copyfile(archive, temporary)
        if sha(temporary) != expected:
            raise ValueError("Copied source archive checksum changed")
        temporary.replace(archive_cache)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    with tarfile.open(archive_cache) as stream:
        stream.extractall(workspace, filter=safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one repository root")
    return roots[0]


def run_case(case: dict, cfg: dict, output: Path = OUT) -> dict:
    case_id = case["instance_id"]
    case_dir = output / "cases" / case_id
    result_path = case_dir / "result.json"
    existing = read(result_path)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if existing is not None:
        archived = case_dir.with_name(f"{case_id}.retry_failed")
        if archived.exists():
            shutil.rmtree(archived)
        case_dir.rename(archived)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        "instance_id": case_id,
        "repo": case["repo"],
        "source_sha256": sha(SOURCE / "sources" / f"{case_id}.tar.gz"),
        "status": "failed",
        "predictions": [],
        "stages": {},
    }
    workspace = NATIVE_WORK_ROOT / "work" / case_id
    try:
        root = materialize(case, workspace)
        with (case_dir / "parse.log").open("w", encoding="utf-8") as parse_log:
            with contextlib.redirect_stdout(parse_log):
                structure = agentless.build_structure(root)
        from agentless.util.preprocess_data import filter_none_python, filter_out_test_files, get_repo_files
        filter_none_python(structure)
        filter_out_test_files(structure)

        from agentless.fl.FL import LLMFL
        import agentless.util.model as upstream_model

        bridge = agentless.Bridge(cfg, case_dir)
        original_factory = upstream_model.make_model
        upstream_model.make_model = bridge.factory
        logger = logging.getLogger(case_id)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(case_dir / "agentless.log")
        logger.addHandler(handler)
        try:
            fl = LLMFL(case_id, structure, case["problem_statement"], cfg["model"], "openai", logger)
            fl.max_tokens = 4096

            model_path = case_dir / "file_stage.json"
            if model_path.exists():
                model_stage = read(model_path)
                model_files = model_stage["found_files"]
            else:
                calls_before = len(list((case_dir / "requests").glob("*.json")))
                stage_started = time.monotonic()
                model_files, artifact, trajectory = fl.localize()
                bridge.calls = len(list((case_dir / "requests").glob("*.json")))
                model_stage = {
                    "seconds": round(time.monotonic() - stage_started, 3),
                    "model_calls": bridge.calls - calls_before,
                    "found_files": model_files,
                    "artifact": artifact,
                    "trajectory": trajectory,
                }
                write(model_path, model_stage)
            result["stages"]["llm_file_localization"] = {
                key: model_stage[key] for key in ("seconds", "model_calls", "found_files")
            }

            retrieval_path = case_dir / "retrieval_stage.json"
            if retrieval_path.exists():
                retrieval_stage = read(retrieval_path)
                retrieved_files = retrieval_stage["found_files"]
            else:
                stage_started = time.monotonic()
                retrieved_files, retrieval_report = retrieve_files(structure, case["problem_statement"])
                retrieval_stage = {
                    "seconds": round(time.monotonic() - stage_started, 3),
                    "found_files": retrieved_files,
                    **retrieval_report,
                }
                write(retrieval_path, retrieval_stage)
            result["stages"]["embedding_retrieval"] = retrieval_stage

            combined = combine_files(model_files, retrieved_files, TOP_N_FILES)
            result["stages"]["combine"] = {
                "model_files": model_files[:TOP_N_FILES],
                "retrieval_files": retrieved_files[:TOP_N_FILES],
                "combined_files": combined,
            }

            related_path = case_dir / "related_stage.json"
            if related_path.exists():
                related_stage = read(related_path)
            else:
                calls_before = len(list((case_dir / "requests").glob("*.json")))
                stage_started = time.monotonic()
                bridge.stage = "related"
                locs, artifact, trajectory = fl.localize_function_from_compressed_files(
                    combined[:TOP_N_FILES],
                    temperature=0,
                    keep_old_order=False,
                    compress_assign=True,
                )
                predictions, ignored = agentless.rank_functions(
                    locs,
                    get_repo_files(structure, combined[:TOP_N_FILES]),
                )
                bridge.calls = len(list((case_dir / "requests").glob("*.json")))
                related_stage = {
                    "seconds": round(time.monotonic() - stage_started, 3),
                    "model_calls": bridge.calls - calls_before,
                    "predictions": predictions,
                    "ignored_locations": ignored,
                    "artifact": artifact,
                    "trajectory": trajectory,
                }
                write(related_path, related_stage)
            result["predictions"] = related_stage.get("predictions", [])
            result["ignored_locations"] = related_stage.get("ignored_locations", [])
            result["stages"]["related_function_localization"] = {
                key: related_stage[key] for key in ("seconds", "model_calls")
            }
            result["status"] = "completed" if result["predictions"] else "empty_prediction"
            result["metrics"] = strict_metrics(result["predictions"][:5], case["function_ground_truth"])
        finally:
            upstream_model.make_model = original_factory
            logger.removeHandler(handler)
            handler.close()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["metrics"] = strict_metrics([], case["function_ground_truth"])
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
    result["runtime_seconds"] = round(time.monotonic() - started, 3)
    result["llm_calls"] = len(list((case_dir / "requests").glob("*.json")))
    write(result_path, result)
    return result


def summarize(rows: list[dict], target: int) -> dict:
    return {
        "method": "Agentless-FL with Qwen Flash embedding retrieval and combine",
        "embedding_model": QWEN_MODEL,
        "processed": len(rows),
        "target": target,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "top1": sum(row["metrics"]["top1"] for row in rows) / len(rows) if rows else None,
        "top3": sum(row["metrics"]["top3"] for row in rows) / len(rows) if rows else None,
        "top5": sum(row["metrics"]["top5"] for row in rows) / len(rows) if rows else None,
        "mrr": sum(row["metrics"]["mrr"] for row in rows) / len(rows) if rows else None,
        "llm_calls": sum(row.get("llm_calls", 0) for row in rows),
        "cases": rows,
    }


def run(limit: int, workers: int, output: Path = OUT, resume_index: int = 1) -> dict:
    if not os.environ.get(QWEN_API_KEY_ENV):
        raise ValueError(f"{QWEN_API_KEY_ENV} must be supplied")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be supplied")
    cases = read(SOURCE / "evaluation_cases.json", [])
    if len(cases) != 500:
        raise ValueError("Expected the frozen 500-case evaluation set")
    cfg = read(SOURCE / "config.json")["llm"]
    output.mkdir(parents=True, exist_ok=True)
    NATIVE_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    selected = cases[:limit]
    if not 1 <= resume_index <= len(selected):
        raise ValueError("resume_index must be within the selected cases")
    existing_all = {
        row["instance_id"]: row
        for row in read(output / "comparison_summary.json", {}).get("cases", [])
    }
    rerun_ids = {
        case["instance_id"] for case in selected[resume_index - 1:]
        if case["instance_id"] not in existing_all
        or existing_all[case["instance_id"]].get("status") in RETRY_STATUSES
    }
    existing = {
        case_id: row for case_id, row in existing_all.items()
        if case_id not in rerun_ids
    }
    pending = [case for case in selected if case["instance_id"] in rerun_ids]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(run_case, case, cfg, output): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            existing[case_id] = future.result()
            ordered = [existing[case["instance_id"]] for case in selected if case["instance_id"] in existing]
            write(output / "comparison_summary.json", summarize(ordered, limit))
            write(output / "progress.json", {
                "completed": len(ordered),
                "target": limit,
                "instance_id": case_id,
                "status": existing[case_id]["status"],
                "runtime_seconds": existing[case_id].get("runtime_seconds"),
            })
            print(
                f"{len(ordered)}/{limit} {case_id}: {existing[case_id]['status']} "
                f"({existing[case_id].get('runtime_seconds')}s)",
                flush=True,
            )
    ordered = [existing[case["instance_id"]] for case in selected]
    summary = summarize(ordered, limit)
    write(output / "comparison_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume-index", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500 or not 1 <= args.workers <= 8:
        parser.error("limit must be 1..500 and workers must be 1..8")
    report = run(args.limit, args.workers, args.output, args.resume_index)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
