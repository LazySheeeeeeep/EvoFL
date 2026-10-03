"""Run the frozen RQ1 main method on 500 cases with qwen3.8-flash.

This runner reuses the frozen evaluation case set, Fault SkillBank, Explorer
prompts, and source archives. It performs localization only: no training,
reflection, evolution, or SkillBank writes.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUT = ROOT / "runs/rq1_main_qwen3_8_flash_eval500_20260929"
NATIVE_WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_NATIVE_WORK_ROOT",
        str(Path.home() / ".cache" / "evolutefl_rq1_qwen_work"),
    )
)
MODEL = "qwen3.8-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ARM = "with_skill"
POLICY = "changed_patch_functions_intersect_existing_base_functions_v1"
RETRY_STATUSES = {"llm_request_failed", "adapter_failed", "agent_interrupted"}


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_workspace(path: Path) -> None:
    root = (NATIVE_WORK_ROOT / "work").resolve()
    resolved = path.resolve()
    if path.is_symlink() or resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Unsafe workspace cleanup target")
    if path.exists():
        shutil.rmtree(path)


def materialize(case: dict, workspace: Path) -> tuple[Path, str]:
    cid = case["instance_id"]
    archive = SOURCE / "sources" / f"{cid}.tar.gz"
    manifest = read(SOURCE / "materialized" / f"{cid}.json", {})
    expected = manifest.get("original_archive_sha256")
    if not archive.is_file() or not expected or sha(archive) != expected:
        raise ValueError(f"Frozen source archive is missing or changed for {cid}")
    clean_workspace(workspace)
    workspace.mkdir(parents=True)
    archive_cache = NATIVE_WORK_ROOT / "archives" / archive.name
    archive_cache.parent.mkdir(parents=True, exist_ok=True)
    if not archive_cache.is_file() or sha(archive_cache) != expected:
        temporary = archive_cache.with_suffix(".tmp")
        shutil.copyfile(archive, temporary)
        if sha(temporary) != expected:
            raise ValueError(f"Copied source archive checksum mismatch for {cid}")
        temporary.replace(archive_cache)
    with tarfile.open(archive_cache) as stream:
        stream.extractall(workspace, filter=bench.safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError(f"Expected one repository root for {cid}")
    root = roots[0]
    if os.environ.get("EVOLUTEFL_VERIFY_SOURCE_HASHES") == "1":
        audit = read(SOURCE / "function_audit" / f"{cid}.json", {})
        for source in audit.get("sources", []):
            source_path = root / source["path"]
            if not source_path.is_file() or sha(source_path) != source["sha256"]:
                raise ValueError(f"Extracted base source differs from audit for {cid}: {source['path']}")
    return root, sha(archive)


def build_config() -> dict:
    config = copy.deepcopy(read(SOURCE / "config.json"))
    config["llm"] = {
        "base_url": BASE_URL,
        "base_url_env": "QWEN_BASE_URL",
        "api_key_env": "QWEN_API_KEY",
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 8192,
        "timeout": 180,
        "max_attempts": 5,
        # Qwen thinking mode rejects object/required tool_choice values.
        # The agent still exposes only the required native tool on forced turns.
        "supports_tool_choice": False,
    }
    config["skill_bank"]["path"] = str(SOURCE / "frozen_skills.jsonl")
    config["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
    config["skill_bank"]["retrieval_mode"] = "lexical"
    config["embedding"]["enabled"] = False
    config["explorer"]["truncation_retry_max_tokens"] = 16384
    config["explorer"]["finalization_max_tokens"] = 16384
    if config["llm"].get("api_key"):
        raise ValueError("Credentials must remain environment-only")
    return config


def prepare(output: Path, limit: int, workers: int) -> list[dict]:
    rq.verify_protocol()
    cases = read(SOURCE / "evaluation_cases.json", [])
    if len(cases) != 500 or not 1 <= limit <= 500:
        raise ValueError("Expected the frozen 500-case evaluation set")
    bank = SOURCE / "frozen_skills.jsonl"
    expected_bank = read(SOURCE / "training_complete.json", {}).get("bank_sha256")
    if not expected_bank or sha(bank) != expected_bank:
        raise ValueError("Frozen SkillBank changed")
    config = build_config()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "frozen_config.json"
    write(config_path, config)
    manifest = {
        "method": "EvoluteFL frozen main method, localization only",
        "arm": ARM,
        "model": MODEL,
        "base_url": BASE_URL,
        "limit": limit,
        "case_ids": [case["instance_id"] for case in cases[:limit]],
        "evaluation_sha256": sha(SOURCE / "evaluation_cases.json"),
        "bank_sha256": sha(bank),
        "config_sha256": sha(config_path),
        "workers": workers,
        "evolution": False,
        "skillbank_writes": False,
        "label_policy": POLICY,
        "benchmark": "frozen RQ1 500-case evaluation set",
    }
    path = output / "selection_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Cross-model evaluation inputs changed")
    if not path.exists():
        write(path, manifest)
    return cases[:limit]


def choose_run_dir(case_dir: Path) -> Path:
    attempt = 1
    while True:
        candidate = case_dir / f"explorer_{attempt}"
        if not candidate.exists():
            return candidate
        if not (candidate / "trajectory.jsonl").exists() and not (candidate / "result.json").exists():
            return candidate
        if (candidate / "result.json").exists():
            return candidate
        attempt += 1


def evaluate_one(case: dict, output: Path) -> dict:
    cid = case["instance_id"]
    case_dir = output / "cases" / cid
    record_path = case_dir / "record.json"
    existing = read(record_path)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if existing is not None:
        archived = case_dir.with_name(f"{cid}.retry_failed")
        if archived.exists():
            shutil.rmtree(archived)
        case_dir.rename(archived)
    workspace = NATIVE_WORK_ROOT / "work" / cid
    run_dir = choose_run_dir(case_dir)
    try:
        root, archive_sha = materialize(case, workspace)
        config = read(output / "frozen_config.json")
        result = run_explorer(
            task={
                "instance_id": cid,
                "repo": case["repo"],
                "base_commit": case["base_commit"],
                "bug_report": case["problem_statement"],
                "repo_path": str(root),
                "run_dir": str(run_dir),
            },
            config=config,
            llm_client=OpenAICompatibleClient.from_config(config["llm"]),
            skill_bank=make_skill_bank(config),
        )
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        search = read(run_dir / "fault_skill_search.json", {})
        trace = search.get("search_trace") or {}
        record = {
            "instance_id": cid,
            "repo": case["repo"],
            "archive_sha256": archive_sha,
            "arm": ARM,
            "status": result.get("status"),
            "error": result.get("error"),
            "predictions": predictions,
            "metrics": bench.strict_metrics(predictions, case["function_ground_truth"]),
            "loaded_skill_id": (search.get("matched_skill") or {}).get("skill_id"),
            "fault_family": search.get("fault_family"),
            "catalog_count": trace.get("catalog_count"),
            "steps": result.get("steps"),
            "forced_finish": result.get("forced_finish", False),
            "runtime_seconds": result.get("runtime_seconds"),
            "explorer_dir": str(run_dir),
        }
    except Exception as exc:
        record = {
            "instance_id": cid,
            "repo": case["repo"],
            "arm": ARM,
            "status": "adapter_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "predictions": [],
            "metrics": bench.strict_metrics([], case["function_ground_truth"]),
            "loaded_skill_id": None,
            "steps": None,
            "forced_finish": False,
            "runtime_seconds": None,
            "explorer_dir": str(run_dir),
        }
    finally:
        if workspace.exists():
            clean_workspace(workspace)
    write(record_path, record)
    return record


def summarize(rows: list[dict], limit: int) -> dict:
    return {
        "processed": len(rows),
        "target": limit,
        "model": MODEL,
        "base_url": BASE_URL,
        "arm": ARM,
        "label_policy": POLICY,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "loaded_skill_count": sum(bool(row.get("loaded_skill_id")) for row in rows),
        "forced_finish_count": sum(bool(row.get("forced_finish")) for row in rows),
        **{
            metric: sum(row["metrics"][metric] for row in rows) / len(rows) if rows else None
            for metric in ("top1", "top3", "top5", "mrr")
        },
        "cases": rows,
    }


def run(limit: int, workers: int, output: Path, resume_index: int = 1) -> dict:
    cases = prepare(output, limit, workers)
    if not os.environ.get("QWEN_API_KEY"):
        raise ValueError("QWEN_API_KEY must be supplied to this process")
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
            pool.submit(evaluate_one, case, output): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            cid = futures[future]
            existing[cid] = future.result()
            ordered = [
                existing[case["instance_id"]]
                for case in selected
                if case["instance_id"] in existing
            ]
            write(output / "comparison_summary.json", summarize(ordered, limit))
            write(output / "progress.json", {
                "completed": len(ordered),
                "target": limit,
                "instance_id": cid,
                "status": existing[cid]["status"],
                "updated_at": time.time(),
            })
            print(f"{len(ordered)}/{limit} {cid}: {existing[cid]['status']}", flush=True)
    ordered = [existing[case["instance_id"]] for case in selected]
    write(output / "comparison_summary.json", summarize(ordered, limit))
    return summarize(ordered, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume-index", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500 or not 1 <= args.workers <= 8:
        parser.error("limit must be 1..500 and workers must be 1..8")
    result = run(args.limit, args.workers, args.output, args.resume_index)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
