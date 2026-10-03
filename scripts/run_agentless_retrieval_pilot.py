"""Run a three-case Agentless-FL pilot with local Jina file retrieval.

This follows the pinned Agentless commit's localization path:
LLM file localization, embedding retrieval, file-list combination, and
related-function localization. Repair and patch validation are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_agentless_heldout30 as agentless
from prepare_rgfl_reasoning import source_files
from run_rgfl_jina_retrieval_smoke import chunks_for, cosine, embed
from run_swe_explore_v5_stratified import safe_archive_filter, strict_metrics

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUT = ROOT / "runs/external_fl_baseline_preflight_20260924/agentless_retrieval_pilot"
PILOT_IDS = (
    "pallets__flask-4045",
    "mwaskom__seaborn-2457",
    "pylint-dev__pylint-4330",
)
TOP_N_FILES = 3
EMBED_BATCH_SIZE = 32
RETRIEVAL_FILES = 100


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


def case_by_id(case_id: str) -> dict:
    cases = read(SOURCE / "evaluation_cases.json", [])
    matches = [case for case in cases if case["instance_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Frozen case not uniquely found: {case_id}")
    return matches[0]


def materialize(case: dict, workspace: Path) -> Path:
    archive = SOURCE / "sources" / f"{case['instance_id']}.tar.gz"
    expected = read(SOURCE / "materialized" / f"{case['instance_id']}.json", {})[
        "original_archive_sha256"
    ]
    if sha(archive) != expected:
        raise ValueError("Frozen source archive changed")
    if workspace.exists():
        raise ValueError("Workspace already exists")
    workspace.mkdir(parents=True)
    with tarfile.open(archive) as stream:
        stream.extractall(workspace, filter=safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one repository root")
    return roots[0]


def retrieval_files(
    structure: dict,
    problem_statement: str,
    *,
    case_dir: Path,
    batch_size: int,
    top_files: int,
) -> tuple[list[str], dict]:
    files = source_files(structure)
    chunks = []
    for path, source in sorted(files.items()):
        chunks.extend(chunks_for(path, source))
    if not chunks:
        raise ValueError("No Python source chunks available for retrieval")

    cache_path = case_dir / "jina_vectors.jsonl"
    cached = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["chunk_id"]] = row["vector"]

    vectors = []
    started = time.monotonic()
    with cache_path.open("w", encoding="utf-8") as cache:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            missing = [row for row in batch if row["chunk_id"] not in cached]
            if missing:
                encoded = embed([row["text"] for row in missing], "retrieval.passage")
                for row, vector in zip(missing, encoded):
                    cached[row["chunk_id"]] = vector
            for row in batch:
                cache.write(json.dumps({"chunk_id": row["chunk_id"], "vector": cached[row["chunk_id"]]}) + "\n")
            vectors.extend(cached[row["chunk_id"]] for row in batch)
            print(
                f"embedded {min(start + batch_size, len(chunks))}/{len(chunks)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )

    query_vector = embed(
        [f"Repository issue:\n{problem_statement}"],
        "retrieval.query",
    )[0]
    scored = sorted(
        (
            (cosine(query_vector, vector), row)
            for row, vector in zip(chunks, vectors)
        ),
        key=lambda item: (-item[0], item[1]["chunk_id"]),
    )
    selected = []
    seen = set()
    for score, row in scored:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        selected.append({"path": row["path"], "score": score, "chunk_id": row["chunk_id"]})
        if len(selected) >= top_files:
            break
    report = {
        "source_files": len(files),
        "chunks": len(chunks),
        "batch_size": batch_size,
        "retrieval_files": len(selected),
        "embedding_seconds": round(time.monotonic() - started, 3),
        "chunks_per_second": round(len(chunks) / (time.monotonic() - started), 3),
        "selected": selected,
    }
    return [row["path"] for row in selected], report


def combine_files(model_files: list[str], retrieved_files: list[str], top_n: int) -> list[str]:
    counts = Counter()
    for path in model_files[:top_n]:
        counts[path] += 1
    for path in retrieved_files[:top_n]:
        counts[path] += 1
    return [path for path, _ in counts.most_common()]


def run_case(case: dict, cfg: dict) -> dict:
    case_id = case["instance_id"]
    case_dir = OUT / "cases" / case_id
    result_path = case_dir / "result.json"
    if result_path.exists():
        return read(result_path)
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
    workspace = OUT / "work" / case_id
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

            stage_started = time.monotonic()
            model_files, model_artifact, model_trajectory = fl.localize()
            result["stages"]["llm_file_localization"] = {
                "seconds": round(time.monotonic() - stage_started, 3),
                "model_calls": bridge.calls,
                "found_files": model_files,
            }

            stage_started = time.monotonic()
            retrieved_files, retrieval_report = retrieval_files(
                structure,
                case["problem_statement"],
                case_dir=case_dir,
                batch_size=EMBED_BATCH_SIZE,
                top_files=RETRIEVAL_FILES,
            )
            combined = combine_files(model_files, retrieved_files, TOP_N_FILES)
            result["stages"]["embedding_retrieval"] = {
                "seconds": round(time.monotonic() - stage_started, 3),
                "found_files": retrieved_files,
                **retrieval_report,
            }
            result["stages"]["combine"] = {
                "model_files": model_files[:TOP_N_FILES],
                "retrieval_files": retrieved_files[:TOP_N_FILES],
                "combined_files": combined,
            }

            if combined:
                bridge.stage = "related"
                stage_started = time.monotonic()
                locs, related_artifact, related_trajectory = fl.localize_function_from_compressed_files(
                    combined[:TOP_N_FILES],
                    temperature=0,
                    keep_old_order=False,
                    compress_assign=True,
                )
                predictions, ignored = agentless.rank_functions(
                    locs,
                    get_repo_files(structure, combined[:TOP_N_FILES]),
                )
                result["predictions"] = predictions
                result["ignored_locations"] = ignored
                result["stages"]["related_function_localization"] = {
                    "seconds": round(time.monotonic() - stage_started, 3),
                    "model_calls": bridge.calls,
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
            import shutil
            shutil.rmtree(workspace)
    result["runtime_seconds"] = round(time.monotonic() - started, 3)
    result["llm_calls"] = len(list((case_dir / "requests").glob("*.json")))
    write(result_path, result)
    return result


def run(case_ids: tuple[str, ...]) -> dict:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be supplied")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(exist_ok=True)
    cfg = read(SOURCE / "config.json")["llm"]
    rows = []
    for case_id in case_ids:
        case = case_by_id(case_id)
        row = run_case(case, cfg)
        rows.append(row)
        write(OUT / "progress.json", {
            "completed": len(rows),
            "target": len(case_ids),
            "instance_id": case_id,
            "status": row["status"],
            "runtime_seconds": row["runtime_seconds"],
        })
        print(f"{len(rows)}/{len(case_ids)} {case_id}: {row['status']} ({row['runtime_seconds']}s)", flush=True)
    summary = {
        "method": "Agentless-FL with LLM file localization + local Jina retrieval + combine + related functions",
        "adaptation": "local Jina-v3 replaces text-embedding-3-small",
        "processed": len(rows),
        "target": len(case_ids),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "top1": sum(row["metrics"]["top1"] for row in rows) / len(rows) if rows else None,
        "top3": sum(row["metrics"]["top3"] for row in rows) / len(rows) if rows else None,
        "top5": sum(row["metrics"]["top5"] for row in rows) / len(rows) if rows else None,
        "mrr": sum(row["metrics"]["mrr"] for row in rows) / len(rows) if rows else None,
        "cases": rows,
    }
    write(OUT / "comparison_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+", default=list(PILOT_IDS))
    args = parser.parse_args()
    report = run(tuple(args.case_ids))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
