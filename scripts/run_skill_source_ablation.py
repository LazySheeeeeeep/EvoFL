"""Rebuild success-only and failure-only SkillBanks and evaluate both."""
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

from evolutefl.explorer import run_explorer
from evolutefl.json_utils import write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from run_swe_explore_v5_stratified import safe_archive_filter, strict_metrics

EXPANDED = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
TEMPORAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
OUTPUT = Path(
    os.environ.get(
        "EVOLUTEFL_SKILL_SOURCE_OUTPUT",
        str(ROOT / "runs/skill_source_ablation_20261001"),
    )
)
WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_SOURCE_ABLATION_WORK_ROOT",
        "/home/lql/.cache/skill_source_ablation",
    )
)
ARMS = ("success_only", "failure_only")


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquisition_cases() -> list[dict]:
    rows = []
    for name in ("original_training.json", "training_additions.json"):
        rows.extend(read(EXPANDED / name, []))
    if len(rows) != 400 or len({row["instance_id"] for row in rows}) != 400:
        raise ValueError("Expected 400 unique acquisition cases")
    return rows


def case_evolution_paths(case_id: str) -> tuple[Path, Path]:
    for directory in (
        EXPANDED / "training" / case_id / "evolution",
        TEMPORAL / "training" / case_id / "evolution",
    ):
        summary = directory / "case_evolution_summary.json"
        output = directory / "reflector_output.json"
        if summary.is_file():
            return summary, output
    raise FileNotFoundError(f"Missing evolution records for {case_id}")


def materialized_update(summary: dict, reflector_output: Path) -> tuple[str, dict | None]:
    output = read(reflector_output, {})
    updates = output.get("materialized_updates") or []
    if updates:
        return updates[0].get("operation"), updates[0]
    skill_update = ((summary.get("skill_updates") or {}).get("fault_skill") or {})
    decision = str(skill_update.get("decision") or "")
    operation = {
        "create_new": "create",
        "rewrite_existing": "rewrite",
        "preserve_existing": "preserve",
        "no_update": "preserve",
    }.get(decision)
    if operation is None:
        return "", None
    return operation, {
        "operation": operation,
        "skill_type": "fault_skill",
        "target_skill_id": skill_update.get("target_skill_id"),
        "skill": skill_update.get("skill"),
        "rationale": skill_update.get("rationale"),
    }


def build_bank(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    directory = OUTPUT / "banks" / arm
    directory.mkdir(parents=True, exist_ok=True)
    bank_path = directory / "skills.jsonl"
    bank_path.write_text("", encoding="utf-8")
    config = copy.deepcopy(read(EXPANDED / "config.json"))
    config["skill_bank"]["path"] = str(bank_path)
    config["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
    config["skill_bank"]["retrieval_mode"] = "lexical"
    config["embedding"]["enabled"] = False
    bank = make_skill_bank(config)
    rows = []
    selected_label = "success" if arm == "success_only" else "failure"
    for case in acquisition_cases():
        summary_path, output_path = case_evolution_paths(case["instance_id"])
        summary = read(summary_path, {})
        outcome = str((summary.get("outcome") or {}).get("label") or "")
        if outcome != selected_label:
            continue
        operation, update = materialized_update(summary, output_path)
        if update is None or operation == "preserve":
            rows.append({
                "case_id": case["instance_id"], "outcome": outcome,
                "operation": operation or "none", "status": "no_op",
            })
            continue
        target = update.get("target_skill_id")
        try:
            effect = bank.apply_update(update)
            rows.append({
                "case_id": case["instance_id"], "outcome": outcome,
                "operation": operation, "target_skill_id": target,
                "status": "applied", "effect": effect,
            })
        except Exception as exc:
            rows.append({
                "case_id": case["instance_id"], "outcome": outcome,
                "operation": operation, "target_skill_id": target,
                "status": "discarded_missing_target",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
    report = {
        "arm": arm,
        "selected_outcome": selected_label,
        "bank_path": str(bank_path),
        "bank_sha256": sha(bank_path),
        "updates": rows,
        "applied_count": sum(row["status"] == "applied" for row in rows),
        "discarded_count": sum(row["status"] == "discarded_missing_target" for row in rows),
        "no_op_count": sum(row["status"] == "no_op" for row in rows),
    }
    write_json(directory / "build_report.json", report)
    return report


def materialize_case(case: dict, workspace: Path) -> Path:
    archive = EXPANDED / "sources" / f"{case['instance_id']}.tar.gz"
    manifest = read(EXPANDED / "materialized" / f"{case['instance_id']}.json", {})
    expected = manifest.get("original_archive_sha256")
    if not archive.is_file() or not expected or sha(archive) != expected:
        raise ValueError(f"Frozen source archive changed for {case['instance_id']}")
    cache = WORK_ROOT / "archives" / archive.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or sha(cache) != expected:
        temporary = cache.with_suffix(".tmp")
        shutil.copyfile(archive, temporary)
        if sha(temporary) != expected:
            raise ValueError("Copied source archive checksum mismatch")
        temporary.replace(cache)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    with tarfile.open(cache) as stream:
        stream.extractall(workspace, filter=safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one repository root")
    return roots[0]


def evaluate_case(case: dict, *, arm: str, bank_path: Path, output: Path,
                  config_path: Path) -> dict:
    case_id = case["instance_id"]
    case_dir = output / "cases" / case_id
    result_path = case_dir / "result.json"
    existing = read(result_path)
    if existing is not None:
        return existing
    workspace = WORK_ROOT / "work" / arm / case_id
    root = materialize_case(case, workspace)
    config = read(config_path)
    config["skill_bank"]["path"] = str(bank_path)
    started = time.monotonic()
    try:
        result = run_explorer(
            task={
                "instance_id": case_id,
                "repo": case["repo"],
                "base_commit": case["base_commit"],
                "bug_report": case["problem_statement"],
                "repo_path": str(root),
                "run_dir": str(case_dir),
            },
            config=config,
            llm_client=OpenAICompatibleClient.from_config(config["llm"]),
            skill_bank=make_skill_bank(config),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
    search = read(case_dir / "fault_skill_search.json", {})
    matched = search.get("matched_skill") or {}
    record = {
        "instance_id": case_id,
        "repo": case["repo"],
        "status": result.get("status"),
        "predictions": predictions,
        "metrics": strict_metrics(predictions, case["function_ground_truth"]),
        "loaded_skill_id": matched.get("skill_id"),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "steps": result.get("steps"),
    }
    write_json(result_path, record)
    return record


def summarize(rows: list[dict], target: int) -> dict:
    loaded = [row for row in rows if row.get("loaded_skill_id")]

    def means(group):
        return {
            metric: sum(row["metrics"][metric] for row in group) / len(group)
            if group else None
            for metric in ("top1", "top3", "top5", "mrr")
        }

    return {
        "processed": len(rows),
        "target": target,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "all_cases": means(rows),
        "loaded_cases": means(loaded),
        "loaded_count": len(loaded),
        "cases": rows,
    }


def evaluate_bank(arm: str, limit: int, workers: int) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    bank_path = OUTPUT / "banks" / arm / "skills.jsonl"
    if not bank_path.is_file():
        build_bank(arm)
    output = OUTPUT / "evaluation" / arm
    output.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(read(EXPANDED / "config.json"))
    config["skill_bank"]["path"] = str(bank_path)
    config["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
    config["skill_bank"]["retrieval_mode"] = "lexical"
    config["embedding"]["enabled"] = False
    config_path = output / "frozen_config.json"
    write_json(config_path, config)
    cases = read(EXPANDED / "evaluation_cases.json", [])[:limit]
    existing = {
        row["instance_id"]: row
        for row in read(output / "comparison_summary.json", {}).get("cases", [])
    }
    pending = [case for case in cases if case["instance_id"] not in existing]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                evaluate_case,
                case,
                arm=arm,
                bank_path=bank_path,
                output=output,
                config_path=config_path,
            ): case["instance_id"]
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
            write_json(output / "comparison_summary.json", summarize(ordered, limit))
            print(
                f"{arm} {len(ordered)}/{limit} {case_id}: "
                f"{existing[case_id]['status']}",
                flush=True,
            )
    return summarize([existing[case["instance_id"]] for case in cases], limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.build_only:
        print(json.dumps([build_bank(arm) for arm in ARMS], indent=2))
        return
    arms = (args.arm,) if args.arm else ARMS
    for arm in arms:
        build_bank(arm)
        result = evaluate_bank(arm, args.limit, args.workers)
        print(json.dumps(
            {key: value for key, value in result.items() if key != "cases"},
            indent=2,
        ))


if __name__ == "__main__":
    main()
