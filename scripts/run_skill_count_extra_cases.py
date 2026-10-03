"""Evaluate acquisition cases excluded from the 200/300-case SkillBanks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_skill_count_ablation as bank_runner
from run_skill_count_ablation import SOURCE, WORK_ROOT, summarize

OUTPUT = ROOT / "runs/skill_count_extra_cases_20261002"
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


def extra_cases(bank_name: str) -> list[dict]:
    additions = read(SOURCE / "training_additions.json", [])
    if len(additions) != 200:
        raise ValueError("Expected exactly 200 training additions")
    if bank_name == "200":
        return additions
    if bank_name == "300":
        return additions[100:]
    raise ValueError("Extra-case evaluation is defined only for banks 200 and 300")


def bank_path(bank_name: str) -> Path:
    return {
        "200": SOURCE / "bootstrap_skills.jsonl",
        "300": SOURCE / "skills_after300.jsonl",
    }[bank_name]


def evaluate_one(case: dict, bank_name: str, output: Path) -> dict:
    case_id = case["instance_id"]
    case_dir = output / f"bank_{bank_name}" / "cases" / case_id
    result_path = case_dir / "result.json"
    existing = read(result_path)
    if existing is not None and existing.get("status") not in RETRY_STATUSES:
        return existing
    if case_dir.exists():
        attempt = 1
        while case_dir.with_name(f"{case_id}.interrupted_{attempt}").exists():
            attempt += 1
        case_dir.rename(case_dir.with_name(f"{case_id}.interrupted_{attempt}"))
    workspace = None
    started = time.monotonic()
    bank = bank_path(bank_name)
    try:
        workspace = WORK_ROOT / "extra" / f"bank_{bank_name}" / case_id
        root = bank_runner.materialize(case, workspace)
        config = bank_runner.build_config(bank)
        result = bank_runner.rq.explorer(
            case,
            config,
            root,
            case_dir,
            bank,
        )
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        search = read(case_dir / "fault_skill_search.json", {})
        record = bank_runner.result_record(
            case,
            bank_name,
            result.get("status"),
            predictions,
            (search.get("matched_skill") or {}).get("skill_id"),
            "extra_holdout_rerun",
            search=search,
            runtime_seconds=result.get("runtime_seconds")
            or round(time.monotonic() - started, 3),
            error=result.get("error"),
        )
    except Exception as exc:
        record = bank_runner.result_record(
            case,
            bank_name,
            "adapter_failed",
            [],
            None,
            "extra_holdout_rerun",
            runtime_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
    write(result_path, record)
    return record


def run_bank(bank_name: str, workers: int, output: Path) -> dict:
    cases = extra_cases(bank_name)
    bank_output = output / f"bank_{bank_name}"
    previous = read(bank_output / "summary.json", {})
    existing = {row["instance_id"]: row for row in previous.get("cases", [])}
    pending = [
        case
        for case in cases
        if case["instance_id"] not in existing
        or existing[case["instance_id"]].get("status") != "completed"
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(evaluate_one, case, bank_name, output): case["instance_id"]
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
            summary = summarize(ordered, len(cases), bank_name)
            write(bank_output / "summary.json", summary)
            print(
                f"bank_{bank_name} extra {len(ordered)}/{len(cases)} "
                f"{case_id}: {existing[case_id]['status']}",
                flush=True,
            )
    return summarize(
        [existing[case["instance_id"]] for case in cases],
        len(cases),
        bank_name,
    )


def combine(bank_name: str, base_path: Path, extra_path: Path, destination: Path) -> dict:
    base = read(base_path, {})
    extra = read(extra_path, {})
    rows = [*base.get("cases", []), *extra.get("cases", [])]
    combined = summarize(rows, len(rows), bank_name)
    combined["base_cases"] = len(base.get("cases", []))
    combined["extra_cases"] = len(extra.get("cases", []))
    write(destination, combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", choices=["200", "300"], default=["200", "300"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for bank_name in args.banks:
        results[bank_name] = run_bank(bank_name, args.workers, output)
    combined = {}
    if "200" in results:
        combined["200"] = combine(
            "200",
            ROOT
            / "runs/skill_count_ablation_200_300_400_fixed500_20261001"
            / "evaluation/bank_200/summary.json",
            output / "bank_200/summary.json",
            output / "bank_200/combined_summary.json",
        )
    if "300" in results:
        combined["300"] = combine(
            "300",
            ROOT
            / "runs/skill_count_bank300_parallel_20261001"
            / "evaluation/bank_300/summary.json",
            output / "bank_300/summary.json",
            output / "bank_300/combined_summary.json",
        )
    write(
        output / "combined_summary.json",
        {
            bank: {key: value for key, value in summary.items() if key != "cases"}
            for bank, summary in combined.items()
        },
    )
    print(
        json.dumps(
            {
                bank: {key: value for key, value in summary.items() if key != "cases"}
                for bank, summary in combined.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
