"""Chronological online adaptation over evaluation cases by merged_at."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_online100_replay_adaptation as online
import run_skill_count_ablation as bank_runner
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution

SOURCE = online.SOURCE
STARTING_BANK = SOURCE / "frozen_skills.jsonl"
PR_METADATA = SOURCE / "pr_metadata"


def read(path: Path, default=None):
    return online.read(path, default)


def write(path: Path, value) -> None:
    online.write(path, value)


def digest(path: Path) -> str:
    return online.digest(path)


def temporal_split(count: int) -> tuple[list[dict], list[dict], dict]:
    cases = read(SOURCE / "evaluation_cases.json", [])
    if len(cases) != 500:
        raise ValueError("Expected 500 evaluation cases")
    rows = []
    for case in cases:
        case_id = case["instance_id"]
        metadata = read(PR_METADATA / f"{case_id}.json", {})
        merged_at = metadata.get("merged_at")
        if not merged_at:
            raise ValueError(f"Missing merged_at for {case_id}")
        rows.append({**case, "merged_at": merged_at})
    rows.sort(key=lambda case: (case["merged_at"], case["created_at"], case["instance_id"]))
    adaptation, test = rows[:count], rows[count:]
    if len(adaptation) != count or len(test) != 500 - count:
        raise ValueError("Temporal split size changed")
    cutoff = adaptation[-1]["merged_at"]
    audit = {
        "count": count,
        "adaptation_ids": [case["instance_id"] for case in adaptation],
        "test_ids": [case["instance_id"] for case in test],
        "cutoff_merged_at": cutoff,
        "adaptation_merged_at_min": adaptation[0]["merged_at"],
        "adaptation_merged_at_max": max(case["merged_at"] for case in adaptation),
        "test_merged_at_min": min(case["merged_at"] for case in test),
        "test_created_before_cutoff": sum(
            case["created_at"] < cutoff for case in test
        ),
        "adaptation_repos": dict(Counter(case["repo"] for case in adaptation)),
        "test_repos": dict(Counter(case["repo"] for case in test)),
        "selection_rule": "ascending merged_at; ties by created_at and instance_id",
    }
    return adaptation, test, audit


def prepare(
    count: int, output: Path, prefix_output: Path | None = None
) -> tuple[list[dict], list[dict]]:
    adaptation, test, audit = temporal_split(count)
    if prefix_output is not None:
        prefix_manifest = read(prefix_output / "selection_manifest.json", {})
        prefix_ids = prefix_manifest.get("adaptation_ids") or []
        if not prefix_ids:
            raise ValueError("Temporal prefix manifest is missing")
        if [case["instance_id"] for case in adaptation[: len(prefix_ids)]] != prefix_ids:
            raise ValueError("Online-200 temporal prefix is not nested with Online-100")
        audit["reused_prefix_count"] = len(prefix_ids)
        audit["prefix_output"] = str(prefix_output)
    manifest = {
        "method": f"Online-{count} Temporal",
        "adaptation_count": count,
        "evaluation_count": len(test),
        "base_bank_sha256": digest(STARTING_BANK),
        "evaluation_sha256": digest(SOURCE / "evaluation_cases.json"),
        **audit,
    }
    path = output / "selection_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Temporal online split changed")
    if not path.exists():
        write(path, manifest)
    return adaptation, test


def adapt(
    adaptation: list[dict],
    output: Path,
    *,
    starting_bank: Path,
    prefix_count: int = 0,
) -> Path:
    working = output / "temporal_working_skills.jsonl"
    if not working.exists():
        shutil.copyfile(starting_bank, working)
    config = copy.deepcopy(read(SOURCE / "config.json"))
    records = []
    remaining = adaptation[prefix_count:]
    for offset, case in enumerate(remaining, 1):
        index = prefix_count + offset
        case_id = case["instance_id"]
        directory = output / "adaptation" / case_id
        record_path = directory / "record.json"
        saved = read(record_path)
        if (
            saved is not None
            and saved.get("evolution_status") == "completed"
            and saved["input_sha256"] == digest(working)
            and digest(Path(saved["output_bank"])) == saved["output_sha256"]
        ):
            working = Path(saved["output_bank"])
            records.append(saved)
            continue
        if directory.exists():
            attempt = 1
            while directory.with_name(f"{case_id}.failed_retry_{attempt}").exists():
                attempt += 1
            directory.rename(directory.with_name(f"{case_id}.failed_retry_{attempt}"))
        directory.mkdir(parents=True, exist_ok=True)
        transaction = directory / "skills.jsonl"
        if not transaction.exists():
            shutil.copyfile(working, transaction)
        workspace = online.NATIVE_WORK_ROOT / "temporal" / str(len(adaptation)) / case_id
        root = bank_runner.materialize(case, workspace)
        local = copy.deepcopy(config)
        local["skill_bank"]["path"] = str(transaction)
        local["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
        summary_path = directory / "evolution" / "case_evolution_summary.json"
        summary = read(summary_path)
        if summary is None:
            source_run = online.adaptation_dir(case_id, output)
            summary = run_case_evolution(
                case_run_dir=source_run,
                repo=case["repo"],
                issue=case["problem_statement"],
                repo_path=root,
                config=local,
                llm_client=OpenAICompatibleClient.from_config(local["llm"]),
                ground_truth_functions=case["function_ground_truth"],
                ground_truth_patch=case["patch"],
                patch_metadata={
                    "source": f"online-{len(adaptation)} temporal adaptation",
                    "direction": "buggy_base_to_repaired",
                },
                output_dir=directory / "evolution",
            )
        next_bank = transaction if summary.get("status") == "completed" else working
        record = {
            "instance_id": case_id,
            "index": index,
            "input_bank": str(working),
            "output_bank": str(next_bank),
            "input_sha256": digest(working),
            "output_sha256": digest(next_bank),
            "evolution_status": summary.get("status"),
        }
        write(record_path, record)
        working = next_bank
        records.append(record)
        completed_total = prefix_count + len(records)
        write(
            output / "adaptation_progress.json",
            {
                "completed": completed_total,
                "target": len(adaptation),
                "instance_id": case_id,
                "output_bank_sha256": digest(working),
            },
        )
        shutil.rmtree(workspace, ignore_errors=True)
        print(
            f"Online-{len(adaptation)} temporal adaptation "
            f"{completed_total}/{len(adaptation)} {case_id}",
            flush=True,
        )
    final = output / f"online{len(adaptation)}_temporal_skills.jsonl"
    shutil.copyfile(working, final)
    write(
        output / "adaptation_complete.json",
        {"count": prefix_count + len(records), "bank_sha256": digest(final)},
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=[100, 200], required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix_output = args.prefix_output.resolve() if args.prefix_output else None
    adaptation, test = prepare(args.count, output, prefix_output)
    final_bank = output / f"online{args.count}_temporal_skills.jsonl"
    if not final_bank.is_file():
        if prefix_output is None:
            starting_bank, prefix_count = STARTING_BANK, 0
        else:
            prefix_manifest = read(prefix_output / "selection_manifest.json", {})
            prefix_count = len(prefix_manifest.get("adaptation_ids") or [])
            starting_bank = prefix_output / f"online{prefix_count}_temporal_skills.jsonl"
            if not starting_bank.is_file():
                raise FileNotFoundError(f"Temporal prefix bank is missing: {starting_bank}")
        final_bank = adapt(
            adaptation,
            output,
            starting_bank=starting_bank,
            prefix_count=prefix_count,
        )
    summary = online.evaluate(test, final_bank, output, args.workers)
    summary["method"] = f"Online-{args.count} Temporal"
    write(output / "evaluation" / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
