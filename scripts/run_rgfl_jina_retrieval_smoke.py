"""Measure RGFL-style file retrieval with the local Jina-v3 embedding server.

This is a retrieval-only pilot. It reproduces the author's default simple
index at file-chunk granularity, but uses local Jina embeddings instead of
OpenAI text-embedding-3-small.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_rgfl_reasoning import source_files

DEFAULT_PREPARED = ROOT / "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/astropy__astropy-13033"
DEFAULT_OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/rgfl_jina_pilot/astropy__astropy-13033"
CASE_ID = "astropy__astropy-13033"
EMBEDDING_URL = "http://127.0.0.1:8008/embed"
MODEL = "jinaai/jina-embeddings-v3"
CHUNK_CHARS = 2000
BATCH_SIZE = 32
TOP_CHUNKS = 100


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def embed(texts: list[str], task: str, timeout: int = 600) -> list[list[float]]:
    body = json.dumps({
        "texts": texts,
        "task": task,
        "truncate_dim": 1024,
        "max_length": 512,
    }).encode("utf-8")
    request = urllib.request.Request(
        EMBEDDING_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Jina embedding server returned an invalid vector count")
    if vectors and len(vectors[0]) != 1024:
        raise ValueError("Jina embedding dimension changed")
    return vectors


def chunks_for(path: str, source: str) -> list[dict]:
    rows = []
    for index, start in enumerate(range(0, len(source), CHUNK_CHARS)):
        text = source[start:start + CHUNK_CHARS]
        if not text.strip():
            continue
        rows.append({
            "chunk_id": f"{path}#chunk-{index}",
            "path": path,
            "text": f"File: {path}\nCode chunk:\n{text}",
        })
    return rows


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def run(
    prepared: Path,
    output: Path,
    *,
    batch_size: int,
    top_chunks: int,
    limit_chunks: int | None = None,
) -> dict:
    started = time.monotonic()
    case = json.loads((prepared / "case.jsonl").read_text(encoding="utf-8"))
    if case.get("instance_id") != CASE_ID:
        raise ValueError("Unexpected prepared case")
    structure = json.loads(
        (prepared / "repo_structures" / f"{CASE_ID}.json").read_text(encoding="utf-8")
    )["structure"]
    files = source_files(structure)
    chunks = []
    for path, source in sorted(files.items()):
        chunks.extend(chunks_for(path, source))
    if limit_chunks is not None:
        chunks = chunks[:limit_chunks]
    if not chunks:
        raise ValueError("No source chunks to embed")

    vectors = []
    embedding_started = time.monotonic()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors.extend(embed([row["text"] for row in batch], "retrieval.passage"))
        completed = min(start + batch_size, len(chunks))
        print(
            f"embedded {completed}/{len(chunks)} chunks "
            f"({time.monotonic() - embedding_started:.1f}s)",
            flush=True,
        )
    embedding_seconds = time.monotonic() - embedding_started

    query = (
        f"Repository: {case['repo']}\n"
        f"Issue:\n{case['problem_statement']}"
    )
    query_vector = embed([query], "retrieval.query")[0]
    scored = sorted(
        (
            (cosine(query_vector, vector), row)
            for row, vector in zip(chunks, vectors)
        ),
        key=lambda item: (-item[0], item[1]["chunk_id"]),
    )
    selected_rows = []
    seen_paths = set()
    for score, row in scored:
        if row["path"] in seen_paths:
            continue
        seen_paths.add(row["path"])
        selected_rows.append({"path": row["path"], "score": score, "chunk_id": row["chunk_id"]})
        if len(selected_rows) >= top_chunks:
            break

    output.mkdir(parents=True, exist_ok=True)
    locations = {
        "instance_id": CASE_ID,
        "found_files": [row["path"] for row in selected_rows],
        "file_scores": {row["path"]: row["score"] for row in selected_rows},
    }
    (output / "retrieve_locs.jsonl").write_text(
        json.dumps(locations, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "combined_locs.jsonl").write_text(
        json.dumps({"instance_id": CASE_ID, "found_files": locations["found_files"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "completed",
        "instance_id": CASE_ID,
        "embedding_model": MODEL,
        "embedding_task_document": "retrieval.passage",
        "embedding_task_query": "retrieval.query",
        "retrieval_variant": "RGFL simple-index file chunks; local Jina-v3 replaces text-embedding-3-small",
        "source_files": len(files),
        "source_chars": sum(len(value) for value in files.values()),
        "chunks": len(chunks),
        "batch_size": batch_size,
        "top_chunks": top_chunks,
        "selected_files": len(locations["found_files"]),
        "embedding_seconds": round(embedding_seconds, 3),
        "total_seconds": round(time.monotonic() - started, 3),
        "chunks_per_second": round(len(chunks) / embedding_seconds, 3) if embedding_seconds else None,
        "output": str(output / "retrieve_locs.jsonl"),
    }
    write_json(output / "retrieval_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--top-chunks", type=int, default=TOP_CHUNKS)
    parser.add_argument("--limit-chunks", type=int)
    args = parser.parse_args()
    result = run(
        args.prepared,
        args.output,
        batch_size=args.batch_size,
        top_chunks=args.top_chunks,
        limit_chunks=args.limit_chunks,
    )
    print(json.dumps(result, indent=2))
