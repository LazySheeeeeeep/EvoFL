"""Online-100: replay 100 evaluation trajectories into the frozen SkillBank.

The 100 adaptation cases are selected deterministically and proportionally by
repository. Their existing main-experiment trajectories are used for skill
updates, so this is a replay adaptation rather than a second Explorer pass.
The remaining 400 cases are evaluated with the resulting frozen bank.
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
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_skill_count_ablation as bank_runner
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import make_skill_bank

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUTPUT = ROOT / "runs/online100_replay_adaptation_eval400_20261002"
SEED = "20261002"
ADAPTATION_COUNT = 100
RETRY_ROOT = SOURCE / "with_skill_failure_retries_20260929"
FINAL_SUMMARY = RETRY_ROOT / "comparison_summary_with_retries.json"
STARTING_BANK = SOURCE / "frozen_skills.jsonl"
ONLINE_BANK = OUTPUT / "online100_skills.jsonl"
NATIVE_WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_ONLINE100_WORK_ROOT",
        "/home/lql/.cache/online100_replay_work",
    )
)
RETRY_STATUSES = {
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
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection() -> tuple[list[dict], list[dict]]:
    cases = read(SOURCE / "evaluation_cases.json", [])
    if len(cases) != 500:
        raise ValueError("Expected 500 frozen evaluation cases")
    groups = {}
    for case in cases:
        groups.setdefault(case["repo"], []).append(case)
    exact = {
        repo: len(rows) * ADAPTATION_COUNT / len(cases)
        for repo, rows in groups.items()
    }
    quotas = {repo: int(value) for repo, value in exact.items()}
    remaining = ADAPTATION_COUNT - sum(quotas.values())
    for repo, _ in sorted(
        exact.items(),
        key=lambda item: (-(item[1] - quotas[item[0]]), item[0]),
    )[:remaining]:
        quotas[repo] += 1
    rng = random.Random(SEED)
    selected = []
    for repo in sorted(groups):
        rows = list(groups[repo])
        rng.shuffle(rows)
        selected.extend(rows[: quotas[repo]])
    selected_ids = {case["instance_id"] for case in selected}
    remainder = [case for case in cases if case["instance_id"] not in selected_ids]
    if len(selected) != 100 or len(remainder) != 400:
        raise ValueError("Online-100 split size changed")
    return selected, remainder


def prepare(output: Path) -> tuple[list[dict], list[dict]]:
    selected, remainder = selection()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "Online-100 replay adaptation",
        "seed": SEED,
        "adaptation_count": len(selected),
        "evaluation_count": len(remainder),
        "adaptation_ids": [case["instance_id"] for case in selected],
        "evaluation_ids": [case["instance_id"] for case in remainder],
        "evaluation_sha256": digest(SOURCE / "evaluation_cases.json"),
        "base_bank_sha256": digest(STARTING_BANK),
        "trajectory_source": str(RETRY_ROOT),
        "note": (
            "The 100 adaptation cases are removed from the 400-case test set; "
            "their trajectories update the frozen bank."
        ),
    }
    path = output / "selection_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Online-100 split changed")
    if not path.exists():
        write(path, manifest)
    return selected, remainder


def adaptation_dir(case_id: str, output: Path) -> Path:
    retry = RETRY_ROOT / "cases" / case_id / "attempt_1"
    return retry if (retry / "result.json").is_file() else SOURCE / "with_skill" / "cases" / case_id


def adapt(selected: list[dict], output: Path) -> Path:
    config = copy.deepcopy(read(SOURCE / "config.json"))
    bank = STARTING_BANK
    rows = []
    for index, case in enumerate(selected, 1):
        case_id = case["instance_id"]
        directory = output / "adaptation" / case_id
        transaction = directory / "skills.jsonl"
        summary_path = directory / "evolution" / "case_evolution_summary.json"
        record_path = directory / "record.json"
        saved = read(record_path)
        if (
            saved is not None
            and saved.get("evolution_status") == "completed"
            and saved["input_sha256"] == digest(bank)
            and digest(Path(saved["output_bank"])) == saved["output_sha256"]
        ):
            bank = Path(saved["output_bank"])
            rows.append(saved)
            continue
        if directory.exists():
            attempt = 1
            while directory.with_name(f"{case_id}.failed_retry_{attempt}").exists():
                attempt += 1
            directory.rename(directory.with_name(f"{case_id}.failed_retry_{attempt}"))
        directory.mkdir(parents=True, exist_ok=True)
        if not transaction.exists():
            shutil.copyfile(bank, transaction)
        workspace = NATIVE_WORK_ROOT / "adaptation" / case_id
        root = bank_runner.materialize(case, workspace)
        local = copy.deepcopy(config)
        local["skill_bank"]["path"] = str(transaction)
        local["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
        write(directory / "frozen_config.json", local)
        summary = read(summary_path)
        if summary is None:
            source_run = adaptation_dir(case_id, output)
            if not (source_run / "trajectory.jsonl").is_file():
                raise FileNotFoundError(f"Trajectory missing for {case_id}")
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
                    "source": "online100 adaptation from frozen evaluation set",
                    "direction": "buggy_base_to_repaired",
                },
                output_dir=directory / "evolution",
            )
        next_bank = transaction if summary.get("status") == "completed" else bank
        record = {
            "instance_id": case_id,
            "index": index,
            "input_bank": str(bank),
            "output_bank": str(next_bank),
            "input_sha256": digest(bank),
            "output_sha256": digest(next_bank),
            "evolution_status": summary.get("status"),
        }
        write(record_path, record)
        bank = next_bank
        rows.append(record)
        write(
            output / "adaptation_progress.json",
            {
                "completed": len(rows),
                "target": len(selected),
                "instance_id": case_id,
                "output_bank_sha256": digest(bank),
            },
        )
        shutil.rmtree(workspace, ignore_errors=True)
        print(f"Online-100 adaptation {len(rows)}/{len(selected)} {case_id}", flush=True)
    shutil.copyfile(bank, output / "online100_skills.jsonl")
    write(
        output / "adaptation_complete.json",
        {"count": len(rows), "bank_sha256": digest(output / "online100_skills.jsonl")},
    )
    return output / "online100_skills.jsonl"


def evaluate_one(case: dict, bank: Path, output: Path) -> dict:
    case_id = case["instance_id"]
    destination = output / "evaluation" / "cases" / case_id / "result.json"
    existing = read(destination)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if destination.parent.exists():
        attempt = 1
        while destination.parent.with_name(
            f"{case_id}.interrupted_{attempt}"
        ).exists():
            attempt += 1
        destination.parent.rename(
            destination.parent.with_name(f"{case_id}.interrupted_{attempt}")
        )
    workspace = NATIVE_WORK_ROOT / "evaluation" / case_id
    run_dir = output / "evaluation" / "explorer" / case_id
    try:
        root = bank_runner.materialize(case, workspace)
        config = bank_runner.build_config(bank)
        result = bank_runner.rq.explorer(case, config, root, run_dir, bank)
        search = read(run_dir / "fault_skill_search.json", {})
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        record = bank_runner.result_record(
            case,
            "online100",
            result.get("status"),
            predictions,
            (search.get("matched_skill") or {}).get("skill_id"),
            "online100_post_adaptation",
            search=search,
            runtime_seconds=result.get("runtime_seconds"),
            error=result.get("error"),
        )
    except Exception as exc:
        record = bank_runner.result_record(
            case,
            "online100",
            "adapter_failed",
            [],
            None,
            "online100_post_adaptation",
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    write(destination, record)
    return record


def evaluate(remainder: list[dict], bank: Path, output: Path, workers: int) -> dict:
    previous = read(output / "evaluation" / "summary.json", {})
    existing = {row["instance_id"]: row for row in previous.get("cases", [])}
    pending = [
        case
        for case in remainder
        if case["instance_id"] not in existing
        or existing[case["instance_id"]].get("status") != "completed"
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(evaluate_one, case, bank, output): case["instance_id"]
            for case in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            existing[case_id] = future.result()
            ordered = [
                existing[case["instance_id"]]
                for case in remainder
                if case["instance_id"] in existing
            ]
            summary = bank_runner.summarize(ordered, len(remainder), "online100")
            write(output / "evaluation" / "summary.json", summary)
            print(
                f"Online-100 evaluation {len(ordered)}/{len(remainder)} "
                f"{case_id}: {existing[case_id]['status']}",
                flush=True,
            )
    return bank_runner.summarize(
        [existing[case["instance_id"]] for case in remainder],
        len(remainder),
        "online100",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    selected, remainder = prepare(output)
    online_bank = output / "online100_skills.jsonl"
    if not online_bank.is_file():
        online_bank = adapt(selected, output)
    summary = evaluate(remainder, online_bank, output, args.workers)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
