"""Online-200 nested extension of Online-100 replay adaptation.

The first 100 adaptation cases and their updated bank are reused exactly. The
script adds 100 more cases, then evaluates the remaining 300 cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_skill_count_ablation as bank_runner
import run_online100_replay_adaptation as online100
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution

SOURCE = online100.SOURCE
OUTPUT = ROOT / "runs/online200_nested_replay_adaptation_eval300_20261002"
SOURCE100 = ROOT / "runs/online100_replay_adaptation_eval400_20261002"
SOURCE100_SELECTION = SOURCE100 / "selection_manifest.json"
SOURCE100_BANK = SOURCE100 / "online100_skills.jsonl"
SEED = 20261003
COUNT = 200


def read(path: Path, default=None):
    return online100.read(path, default)


def write(path: Path, value) -> None:
    online100.write(path, value)


def digest(path: Path) -> str:
    return online100.digest(path)


def quotas(cases: list[dict], target: int) -> dict[str, int]:
    counts = Counter(case["repo"] for case in cases)
    exact = {
        repo: count * target / len(cases)
        for repo, count in counts.items()
    }
    result = {repo: int(value) for repo, value in exact.items()}
    remaining = target - sum(result.values())
    for repo, _ in sorted(
        exact.items(),
        key=lambda item: (-(item[1] - result[item[0]]), item[0]),
    )[:remaining]:
        result[repo] += 1
    return result


def selection() -> tuple[list[dict], list[dict]]:
    cases = read(SOURCE / "evaluation_cases.json", [])
    source_manifest = read(SOURCE100_SELECTION, {})
    if len(cases) != 500 or not source_manifest:
        raise ValueError("Online-100 selection manifest is missing")
    by_id = {case["instance_id"]: case for case in cases}
    first = [by_id[case_id] for case_id in source_manifest["adaptation_ids"]]
    first_ids = {case["instance_id"] for case in first}
    target_quotas = quotas(cases, COUNT)
    existing_quotas = Counter(case["repo"] for case in first)
    additional_quotas = {
        repo: target_quotas[repo] - existing_quotas[repo]
        for repo in target_quotas
    }
    if any(value < 0 for value in additional_quotas.values()):
        raise ValueError("Online-100 exceeds an Online-200 repository quota")
    rng = random.Random(SEED)
    additional = []
    for repo in sorted(target_quotas):
        rows = [
            case
            for case in cases
            if case["repo"] == repo and case["instance_id"] not in first_ids
        ]
        rng.shuffle(rows)
        additional.extend(rows[: additional_quotas[repo]])
    selected = [*first, *additional]
    selected_ids = {case["instance_id"] for case in selected}
    remainder = [case for case in cases if case["instance_id"] not in selected_ids]
    if len(selected) != 200 or len(remainder) != 300:
        raise ValueError("Online-200 nested split size changed")
    return selected, remainder


def prepare(output: Path) -> tuple[list[dict], list[dict]]:
    selected, remainder = selection()
    manifest = {
        "method": "Online-200 nested replay adaptation",
        "source_online100_selection": str(SOURCE100_SELECTION),
        "source_online100_bank_sha256": digest(SOURCE100_BANK),
        "adaptation_count": len(selected),
        "evaluation_count": len(remainder),
        "adaptation_ids": [case["instance_id"] for case in selected],
        "evaluation_ids": [case["instance_id"] for case in remainder],
        "seed": SEED,
        "note": "First 100 adaptation cases are exactly Online-100; 100 are added.",
    }
    path = output / "selection_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Online-200 split changed")
    if not path.exists():
        write(path, manifest)
    return selected, remainder


def adapt_additional(selected: list[dict], output: Path) -> Path:
    working = output / "online200_working_skills.jsonl"
    if not working.exists():
        shutil.copyfile(SOURCE100_BANK, working)
    records = []
    additional = selected[100:]
    config = copy.deepcopy(read(SOURCE / "config.json"))
    for index, case in enumerate(additional, 101):
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
        workspace = online100.NATIVE_WORK_ROOT / "online200" / case_id
        root = bank_runner.materialize(case, workspace)
        local = copy.deepcopy(config)
        local["skill_bank"]["path"] = str(transaction)
        local["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
        summary_path = directory / "evolution" / "case_evolution_summary.json"
        summary = read(summary_path)
        if summary is None:
            source_run = online100.adaptation_dir(case_id, output)
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
                    "source": "online200 nested adaptation",
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
        write(
            output / "adaptation_progress.json",
            {
                "reused_first_100": True,
                "completed_additional": len(records),
                "additional_target": len(additional),
                "completed_total": 100 + len(records),
                "target": COUNT,
                "instance_id": case_id,
            },
        )
        shutil.rmtree(workspace, ignore_errors=True)
        print(
            f"Online-200 adaptation {100 + len(records)}/{COUNT} {case_id}",
            flush=True,
        )
    final = output / "online200_skills.jsonl"
    shutil.copyfile(working, final)
    write(
        output / "adaptation_complete.json",
        {"count": COUNT, "bank_sha256": digest(final)},
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected, remainder = prepare(output)
    final_bank = output / "online200_skills.jsonl"
    if not final_bank.is_file():
        final_bank = adapt_additional(selected, output)
    summary = online100.evaluate(remainder, final_bank, output, args.workers)
    summary["method"] = "Online-200 nested replay adaptation"
    write(output / "evaluation" / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
