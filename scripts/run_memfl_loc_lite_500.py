"""Run a test-free MemFL-Loc-lite baseline on the frozen SWE-bench cases.

The memory pipeline is intentionally lightweight:
  acquisition trajectory -> dynamic guidance
  acquisition repository -> static project/component memory
  evaluation issue -> lexical retrieval -> memory injection -> existing Explorer

It is a disclosed adaptation, not a reproduction of the Java/Defects4J MemFL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evolutefl.explorer import run_explorer
from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from rq1_memory_runtime import MemoryInjectedClient
from run_swe_explore_v5_stratified import safe_archive_filter, strict_metrics

EXPANDED = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
TEMPORAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
OUTPUT = ROOT / "runs/memfl_loc_lite_500_20261001"
STATIC_PATH = OUTPUT / "static_memory.json"
BANK_PATH = OUTPUT / "dynamic_memory.jsonl"
SUMMARY_PATH = OUTPUT / "comparison_summary.json"
NATIVE_WORK_ROOT = Path(
    os.environ.get("EVOLUTEFL_MEMFL_WORK_ROOT", "/home/lql/.cache/memfl_loc_lite_work")
)

TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
}


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN.findall(text.lower()) if token not in STOP}


def parse_json_response(content: str) -> dict:
    """Parse one JSON object even when a provider appends extra text."""
    try:
        value = extract_json_object(content)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            value, _ = json.JSONDecoder().raw_decode(content.lstrip())
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}


def training_cases() -> list[dict]:
    rows = []
    for name in ("original_training.json", "training_additions.json"):
        rows.extend(read(EXPANDED / name, []))
    if len(rows) != 400 or len({row["instance_id"] for row in rows}) != 400:
        raise ValueError("Expected 400 unique acquisition cases")
    return rows


def locate_training(case_id: str) -> tuple[Path, Path, Path]:
    candidates = (
        (EXPANDED / "training" / case_id, EXPANDED / "sources" / f"{case_id}.tar.gz"),
        (TEMPORAL / "training" / case_id, TEMPORAL / "sources" / f"{case_id}.tar.gz"),
    )
    for directory, archive in candidates:
        trajectory = directory / "explorer" / "trajectory.jsonl"
        result = directory / "explorer" / "result.json"
        if trajectory.is_file() and result.is_file() and archive.is_file():
            return trajectory, result, archive
    raise FileNotFoundError(f"Acquisition artifacts missing for {case_id}")


def trajectory_excerpt(path: Path, limit: int = 20) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") in {"assistant", "investigation_action", "finish"}:
            rows.append({
                key: row.get(key)
                for key in (
                    "event", "step", "content", "tool", "arguments",
                    "ranked_functions", "summary",
                )
                if row.get(key) is not None
            })
    return rows[-limit:]


def dynamic_memory(case: dict, trajectory_path: Path, result_path: Path) -> dict:
    result = read(result_path, {})
    prompt = (
        "Derive one reusable dynamic debugging-guidance memory from a previous "
        "fault-localization attempt. Do not use or mention a ground-truth patch. "
        "Return JSON with non-empty string fields issue_signature, symptom_pattern, "
        "investigation_guidance, ranking_implication, and retrieval_text.\n\n"
        + json.dumps({
            "issue": case["problem_statement"],
            "repo": case["repo"],
            "final_summary": result.get("final_summary", ""),
            "predicted_functions": result.get("ranked_functions", []),
            "investigation": trajectory_excerpt(trajectory_path),
        }, ensure_ascii=False)
    )
    client = OpenAICompatibleClient.from_config(read(EXPANDED / "config.json")["llm"])
    required = (
        "issue_signature", "symptom_pattern", "investigation_guidance",
        "ranking_implication", "retrieval_text",
    )
    last = {}
    for attempt in range(2):
        messages = [
            {"role": "system", "content": "You distill reusable debugging guidance."},
            {"role": "user", "content": prompt},
        ]
        if attempt:
            messages.append({
                "role": "user",
                "content": "Return one JSON object with the five required non-empty string fields.",
            })
        response = client.chat(
            messages=messages,
            response_format={"type": "json_object"},
            tool_choice="none",
            temperature=0,
            max_tokens=4096,
        )
        last = parse_json_response(response.get("content") or "{}")
        if isinstance(last, dict):
            for holder in (
                last,
                last.get("dynamic_memory") if isinstance(last.get("dynamic_memory"), dict) else {},
                last.get("guidance") if isinstance(last.get("guidance"), dict) else {},
            ):
                if all(isinstance(holder.get(key), str) and holder[key].strip() for key in required):
                    return {
                        "source_instance_id": case["instance_id"],
                        "repo": case["repo"],
                        **{key: holder[key].strip() for key in required},
                    }
    raise ValueError(f"Dynamic memory generation failed for {case['instance_id']}: {list(last)}")


def extract_archive(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter=safe_archive_filter)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one repository root")
    return roots[0]


def static_memory(repo: str, case: dict, archive: Path) -> dict:
    work = NATIVE_WORK_ROOT / "static" / case["instance_id"]
    root = extract_archive(archive, work)
    tree = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.is_file() and not path.is_symlink()
    )[:500]
    excerpts = []
    for path in (
        "README.rst", "README.md", "setup.py", "pyproject.toml",
        next((name for name in tree if name.endswith("__init__.py")), ""),
    ):
        source = root / path
        if path and source.is_file():
            excerpts.append(f"### {path}\n{source.read_text(encoding='utf-8', errors='replace')[:4000]}")
    client = OpenAICompatibleClient.from_config(read(EXPANDED / "config.json")["llm"])
    prompt = (
        "Construct reusable static project memory for fault localization. "
        "Describe the project purpose, major responsibility boundaries, and "
        "likely failure-propagation boundaries. Do not discuss any specific issue, "
        "target function, repair patch, or historical case. Return JSON with "
        "project_summary and component_summaries; component_summaries is a list "
        "of objects containing scope, purpose, and boundary.\n\n"
        + json.dumps({"repo": repo, "python_files": tree, "excerpts": excerpts}, ensure_ascii=False)
    )
    response = client.chat(
        messages=[
            {"role": "system", "content": "You build reusable project knowledge."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        tool_choice="none",
        temperature=0,
        max_tokens=4096,
    )
    parsed = parse_json_response(response.get("content") or "{}")
    write_json(OUTPUT / "static_responses" / f"{case['instance_id']}.json", response)
    if isinstance(parsed.get("project_summary"), str) and parsed["project_summary"].strip():
        project_summary = parsed["project_summary"].strip()
        components = parsed.get("component_summaries") or []
        generation_mode = "llm"
    else:
        top_levels = Counter(path.split("/", 1)[0] for path in tree)
        project_summary = (
            f"{repo} is a Python repository with {len(tree)} indexed Python files. "
            "Its reusable project memory should emphasize module responsibility "
            "boundaries and data/control flow between the most frequently connected components."
        )
        components = [
            {
                "scope": scope,
                "purpose": f"Contains {count} indexed Python files in this repository subtree.",
                "boundary": "Verify the exact responsibility and call relationships in the current snapshot.",
            }
            for scope, count in top_levels.most_common(20)
        ]
        generation_mode = "deterministic_fallback"
    shutil.rmtree(work, ignore_errors=True)
    return {
        "repo": repo,
        "source_instance_id": case["instance_id"],
        "project_summary": project_summary,
        "component_summaries": components,
        "generation_mode": generation_mode,
    }


def build_memory(workers: int) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    NATIVE_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    acquisition = training_cases()
    existing = {row["source_instance_id"]: row for row in read_jsonl(BANK_PATH)}
    pending = [row for row in acquisition if row["instance_id"] not in existing]
    rows = list(existing.values())
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {}
            for case in pending:
                trajectory, result, _ = locate_training(case["instance_id"])
                futures[pool.submit(dynamic_memory, case, trajectory, result)] = case["instance_id"]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                write_jsonl(BANK_PATH, sorted(rows, key=lambda item: item["source_instance_id"]))
                write_json(OUTPUT / "memory_progress.json", {
                    "dynamic_memories": len(rows),
                    "target": 400,
                    "last_instance_id": row["source_instance_id"],
                })
                print(f"dynamic memory {len(rows)}/400 {row['source_instance_id']}", flush=True)
    static = read(STATIC_PATH, {})
    repos = sorted({row["repo"] for row in acquisition})
    for repo in repos:
        if repo in static:
            continue
        case = next(row for row in acquisition if row["repo"] == repo)
        _, _, archive = locate_training(case["instance_id"])
        static[repo] = static_memory(repo, case, archive)
        write_json(STATIC_PATH, static)
        print(f"static memory {len(static)}/{len(repos)} {repo}", flush=True)
    manifest = {
        "method": "MemFL-Loc-lite (test-free SWE adaptation)",
        "dynamic_memories": len(rows),
        "static_repositories": len(static),
        "note": "Lexical retrieval plus existing Explorer; not an original MemFL reproduction.",
    }
    write_json(OUTPUT / "memory_manifest.json", manifest)
    return manifest


def retrieve_memory(issue: str, repo: str, bank: list[dict], static: dict, top_k: int = 2) -> dict:
    query = tokens(issue)
    scored = []
    for row in bank:
        document = tokens(row["retrieval_text"] + " " + row["issue_signature"] + " " + row["symptom_pattern"])
        overlap = query & document
        if not overlap:
            continue
        score = sum(1.0 for _ in overlap) / math.sqrt(len(document) or 1)
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["source_instance_id"]))
    selected = [row for _, row in scored[:top_k]]
    return {
        "static_memory": static.get(repo),
        "dynamic_memories": selected,
        "retrieval_trace": {
            "candidate_count": len(scored),
            "selected_instance_ids": [row["source_instance_id"] for row in selected],
        },
    }


def run_evaluation_case(case: dict, bank: list[dict], static: dict, output: Path, llm_cfg: dict) -> dict:
    case_id = case["instance_id"]
    directory = output / "cases" / case_id
    result_path = directory / "result.json"
    existing = read(result_path)
    if existing is not None and existing.get("status") not in {"failed", "agent_interrupted"}:
        return existing
    memory = retrieve_memory(case["problem_statement"], case["repo"], bank, static)
    write_json(directory / "memory_selection.json", memory)
    archive = EXPANDED / "sources" / f"{case_id}.tar.gz"
    root = extract_archive(archive, NATIVE_WORK_ROOT / "evaluation" / case_id)
    config = copy.deepcopy(read(EXPANDED / "config.json"))
    config["skill_bank"]["path"] = str(EXPANDED / "empty_skills.jsonl")
    config["skill_bank"]["enabled_skill_types"] = []
    client = OpenAICompatibleClient.from_config(llm_cfg)
    guided = MemoryInjectedClient(
        client,
        instance_id=case_id,
        issue=case["problem_statement"],
        memory=memory,
    )
    started = time.monotonic()
    try:
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
    finally:
        shutil.rmtree(NATIVE_WORK_ROOT / "evaluation" / case_id, ignore_errors=True)
    predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
    record = {
        "instance_id": case_id,
        "repo": case["repo"],
        "status": result.get("status"),
        "predictions": predictions,
        "metrics": strict_metrics(predictions, case["function_ground_truth"]),
        "memory_injection_count": guided.injection_count,
        "retrieved_source_ids": memory["retrieval_trace"]["selected_instance_ids"],
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    write_json(result_path, record)
    return record


def summarize(rows: list[dict], target: int) -> dict:
    return {
        "method": "MemFL-Loc-lite (test-free SWE adaptation)",
        "processed": len(rows),
        "target": target,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "top1": sum(row["metrics"]["top1"] for row in rows) / len(rows) if rows else None,
        "top3": sum(row["metrics"]["top3"] for row in rows) / len(rows) if rows else None,
        "top5": sum(row["metrics"]["top5"] for row in rows) / len(rows) if rows else None,
        "mrr": sum(row["metrics"]["mrr"] for row in rows) / len(rows) if rows else None,
        "cases": rows,
    }


def evaluate(limit: int, workers: int, wait_for_agentless: bool) -> dict:
    if wait_for_agentless:
        while True:
            result = subprocess.run(
                ["pgrep", "-f", "scripts/run_agentless_qwen_embedding_500.py"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                break
            print("waiting for Agentless runner to finish before starting MemFL-Loc-lite", flush=True)
            time.sleep(60)
    cases = read(EXPANDED / "evaluation_cases.json", [])
    if len(cases) != 500:
        raise ValueError("Expected 500 frozen evaluation cases")
    bank = read_jsonl(BANK_PATH)
    static = read(STATIC_PATH, {})
    if len(bank) != 400:
        raise ValueError("Full 400-case dynamic memory bank is required")
    llm_cfg = read(EXPANDED / "config.json")["llm"]
    existing = {row["instance_id"]: row for row in read(SUMMARY_PATH, {}).get("cases", [])}
    selected = cases[:limit]
    pending = [case for case in selected if case["instance_id"] not in existing]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(run_evaluation_case, case, bank, static, OUTPUT, llm_cfg): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            existing[case_id] = future.result()
            ordered = [existing[case["instance_id"]] for case in selected if case["instance_id"] in existing]
            write_json(SUMMARY_PATH, summarize(ordered, limit))
            print(f"{len(ordered)}/{limit} {case_id}: {existing[case_id]['status']}", flush=True)
    ordered = [existing[case["instance_id"]] for case in selected]
    summary = summarize(ordered, limit)
    write_json(SUMMARY_PATH, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-memory", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wait-for-agentless", action="store_true")
    args = parser.parse_args()
    if not args.build_memory and not args.evaluate:
        parser.error("select --build-memory and/or --evaluate")
    if args.build_memory:
        print(json.dumps(build_memory(args.workers), indent=2))
    if args.evaluate:
        print(json.dumps({key: value for key, value in evaluate(
            args.limit, args.workers, args.wait_for_agentless
        ).items() if key != "cases"}, indent=2))
