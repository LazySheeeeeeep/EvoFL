"""Evaluate fixed 500 cases with SkillBanks grown from 200, 300, and 400 cases.

The expensive Explorer trajectory is reused when changing the SkillBank cannot
change the injected knowledge:

* no Skill selected in the smaller bank -> reuse the frozen no-Skill arm;
* the exact same Skill card selected -> reuse the frozen with-Skill arm;
* any other selection -> rerun the complete Explorer with the smaller bank.

The selector and validator are always rerun from the saved load_fault_skill
request, because a smaller catalog can change a no-Skill decision.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_rq1_expanded as rq
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer.v5_agent import V5ExplorerAgent
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank

SOURCE = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
DEFAULT_OUTPUT = ROOT / "runs/skill_count_ablation_200_300_400_fixed500_20261001"
WORK_ROOT = Path(
    os.environ.get(
        "EVOLUTEFL_SKILL_COUNT_WORK_ROOT",
        str(Path.home() / ".cache" / "skill_count_ablation"),
    )
)
BANK_FILES = {
    "200": SOURCE / "bootstrap_skills.jsonl",
    "300": SOURCE / "skills_after300.jsonl",
    "400": SOURCE / "frozen_skills.jsonl",
}
RETRY_STATUSES = {
    "llm_request_failed",
    "adapter_failed",
    "agent_interrupted",
    "preflight_failed",
}
_AGENTS: dict[tuple[str, str], V5ExplorerAgent] = {}


def read(path: Path, default=None):
    return rq.read(path, default)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha(path: Path) -> str:
    return rq.sha(path)


def canonical_card(card: dict | None) -> dict | None:
    if not card:
        return None
    return {
        key: card.get(key)
        for key in (
            "skill_id",
            "status",
            "version",
            "skill_type",
            "fault_family",
            "fault_subtype",
            "retrieval_families",
            "title",
            "trigger",
            "knowledge",
        )
        if key in card
    }


def card_hash(card: dict | None) -> str | None:
    normalized = canonical_card(card)
    if normalized is None:
        return None
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def expected_bank_hashes() -> dict[str, str]:
    bootstrap = SOURCE / "bootstrap_skills.jsonl"
    after300 = SOURCE / "skills_after300.jsonl"
    frozen = SOURCE / "frozen_skills.jsonl"
    training = read(SOURCE / "training_summary.json", {})
    records = training.get("cases") or []
    if len(records) < 100:
        raise ValueError("Training chain does not contain the first 100 additions")
    expected = {
        "200": records[0]["input_sha256"],
        "300": records[99]["output_sha256"],
        "400": read(SOURCE / "training_complete.json", {}).get("bank_sha256"),
    }
    actual = {
        "200": sha(bootstrap),
        "300": sha(after300),
        "400": sha(frozen),
    }
    if actual != expected:
        raise ValueError(
            "SkillBank prefix chain changed: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    return actual


def build_config(bank: Path) -> dict:
    config = copy.deepcopy(read(SOURCE / "config.json"))
    config["skill_bank"]["path"] = str(bank)
    config["skill_bank"]["enabled_skill_types"] = ["fault_skill"]
    config["skill_bank"]["retrieval_mode"] = "lexical"
    config["embedding"]["enabled"] = False
    config["llm"]["temperature"] = 0
    if config["llm"].get("api_key"):
        raise ValueError("Credentials must remain environment-only")
    return config


def prepare(output: Path, limit: int, workers: int, banks: list[str]) -> list[dict]:
    rq.verify_protocol()
    cases = read(SOURCE / "evaluation_cases.json", [])
    if len(cases) != 500 or not 1 <= limit <= 500:
        raise ValueError("Expected the frozen 500-case evaluation set")
    bank_hashes = expected_bank_hashes()
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "method": "Fault SkillBank size ablation on one fixed evaluation cohort",
        "source_run": str(SOURCE),
        "limit": limit,
        "workers": workers,
        "banks": banks,
        "case_ids": [case["instance_id"] for case in cases[:limit]],
        "evaluation_sha256": sha(SOURCE / "evaluation_cases.json"),
        "bank_sha256": {name: bank_hashes[name] for name in banks},
        "bank_paths": {name: str(BANK_FILES[name]) for name in banks},
        "reuse_policy": {
            "no_skill": "reuse frozen no_skill arm after selector+validator return null",
            "same_card": "reuse frozen with_skill arm when the complete selected card matches",
            "otherwise": "rerun full Explorer with the smaller SkillBank",
        },
        "skillbank_writes": False,
        "model": read(SOURCE / "config.json")["llm"]["model"],
    }
    path = output / "experiment_manifest.json"
    if path.exists():
        saved = read(path)
        for key in (
            "method",
            "source_run",
            "limit",
            "case_ids",
            "evaluation_sha256",
            "reuse_policy",
            "skillbank_writes",
            "model",
        ):
            if saved.get(key) != frozen.get(key):
                raise ValueError(
                    f"Skill-count ablation frozen input changed: {key}"
                )
        if not set(banks).issubset(set(saved.get("banks") or [])):
            raise ValueError("Requested bank is not part of the frozen experiment")
        for bank_name in banks:
            if saved.get("bank_sha256", {}).get(bank_name) != bank_hashes[bank_name]:
                raise ValueError(f"SkillBank changed: {bank_name}")
    else:
        write(path, frozen)
    return cases[:limit]


def agent_for(bank_name: str, output: Path) -> V5ExplorerAgent:
    key = (bank_name, str(output.resolve()))
    if key in _AGENTS:
        return _AGENTS[key]
    config = build_config(BANK_FILES[bank_name])
    config_path = output / "banks" / bank_name / "frozen_config.json"
    write(config_path, config)
    agent = V5ExplorerAgent(
        llm_client=OpenAICompatibleClient.from_config(config["llm"]),
        skill_bank=make_skill_bank(config),
        config=config,
        system_prompt=Path(config["explorer"]["system_prompt_path"]).read_text(
            encoding="utf-8"
        ),
    )
    _AGENTS[key] = agent
    return agent


def load_saved_request(case_id: str) -> tuple[dict, list[dict]]:
    candidates = [SOURCE / "with_skill" / "cases" / case_id]
    retry_root = SOURCE / "with_skill_failure_retries_20260929" / "cases" / case_id
    if retry_root.is_dir():
        candidates.extend(
            sorted(
                (
                    path
                    for path in retry_root.glob("attempt_*")
                    if path.is_dir()
                ),
                key=lambda path: path.name,
                reverse=True,
            )
        )
    request = index = None
    for directory in candidates:
        candidate_request = read(directory / "fault_skill_request.json")
        candidate_index = read(directory / "investigation_index.json", {})
        if candidate_request and candidate_index:
            request, index = candidate_request, candidate_index
            break
    if not request or not index:
        raise FileNotFoundError(f"Main/retry trajectory request is missing for {case_id}")
    timeline = list(index.get("timeline") or [])
    step = request.get("step")
    positions = [
        position
        for position, event in enumerate(timeline)
        if event.get("step") == step and event.get("tool") == "load_fault_skill"
    ]
    if not positions:
        raise ValueError(f"load_fault_skill event is missing for {case_id}")
    position = positions[-1]
    return request["request"], timeline[max(0, position - 3): position + 1]


def main_arms(case_id: str) -> dict:
    summary = read(SOURCE / "comparison_summary.json", {})
    for row in summary.get("cases") or []:
        if row.get("instance_id") == case_id:
            return row.get("arms") or {}
    raise KeyError(f"Main result is missing for {case_id}")


def preflight(case: dict, bank_name: str, output: Path) -> dict:
    case_id = case["instance_id"]
    destination = output / "preflight" / f"bank_{bank_name}" / f"{case_id}.json"
    existing = read(destination)
    if existing is not None and existing.get("status") == "completed":
        return existing
    request_args, prior = load_saved_request(case_id)
    agent = agent_for(bank_name, output)
    try:
        search = agent.load_fault(
            request_args,
            case["problem_statement"],
            SimpleNamespace(timeline=prior),
        )
        matched = search.get("matched_skill")
        record = {
            "instance_id": case_id,
            "bank": bank_name,
            "status": "completed",
            "fault_family": search.get("fault_family"),
            "loaded_skill_id": (matched or {}).get("skill_id"),
            "loaded_card_sha256": card_hash(matched),
            "matched_skill": matched,
            "search_trace": search.get("search_trace"),
        }
    except Exception as exc:
        record = {
            "instance_id": case_id,
            "bank": bank_name,
            "status": "preflight_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    write(destination, record)
    return record


def safe_archive_filter(member, path):
    return bench.safe_archive_filter(member, path)


def materialize(case: dict, workspace: Path) -> Path:
    case_id = case["instance_id"]
    archive = SOURCE / "sources" / f"{case_id}.tar.gz"
    manifest = read(SOURCE / "materialized" / f"{case_id}.json", {})
    expected = manifest.get("original_archive_sha256")
    if not archive.is_file() or not expected or sha(archive) != expected:
        raise ValueError(f"Frozen source archive changed for {case_id}")
    cache = WORK_ROOT / "archives" / archive.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or sha(cache) != expected:
        temporary = cache.with_suffix(".tmp")
        shutil.copyfile(archive, temporary)
        if sha(temporary) != expected:
            raise ValueError(f"Copied archive checksum mismatch for {case_id}")
        temporary.replace(cache)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    with tarfile.open(cache) as stream:
        stream.extractall(workspace, filter=safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError(f"Expected one repository root for {case_id}")
    return roots[0]


def result_record(case: dict, bank_name: str, status: str, predictions: list[str],
                  loaded_skill_id: str | None, reuse_source: str,
                  search: dict | None = None, runtime_seconds: float | None = None,
                  error: str | None = None) -> dict:
    return {
        "instance_id": case["instance_id"],
        "repo": case["repo"],
        "bank": bank_name,
        "status": status,
        "error": error,
        "predictions": predictions,
        "metrics": bench.strict_metrics(predictions, case["function_ground_truth"]),
        "loaded_skill_id": loaded_skill_id,
        "reuse_source": reuse_source,
        "fault_family": (search or {}).get("fault_family"),
        "runtime_seconds": runtime_seconds,
    }


def evaluate_one(case: dict, bank_name: str, output: Path) -> dict:
    case_id = case["instance_id"]
    destination = output / "evaluation" / f"bank_{bank_name}" / f"{case_id}.json"
    existing = read(destination)
    if existing is not None and existing.get("status") == "completed":
        return existing

    flight = preflight(case, bank_name, output)
    if flight.get("status") != "completed":
        return result_record(
            case,
            bank_name,
            "preflight_failed",
            [],
            None,
            "none",
            error=flight.get("error"),
        )

    arms = main_arms(case_id)
    main_search = read(SOURCE / "with_skill" / "cases" / case_id / "fault_skill_search.json", {})
    main_card = main_search.get("matched_skill")
    alt_card = flight.get("matched_skill")

    if alt_card is None:
        arm = arms.get("no_skill") or {}
        if arm.get("status") not in RETRY_STATUSES:
            record = result_record(
                case,
                bank_name,
                arm.get("status"),
                list(dict.fromkeys(arm.get("predictions") or []))[:5],
                None,
                "reuse_main_no_skill",
                runtime_seconds=arm.get("runtime_seconds"),
            )
            write(destination, record)
            return record

    if alt_card is not None and card_hash(alt_card) == card_hash(main_card):
        arm = arms.get("with_skill") or {}
        if arm.get("status") not in RETRY_STATUSES:
            record = result_record(
                case,
                bank_name,
                arm.get("status"),
                list(dict.fromkeys(arm.get("predictions") or []))[:5],
                (alt_card or {}).get("skill_id"),
                "reuse_main_same_skill",
                search=flight,
                runtime_seconds=arm.get("runtime_seconds"),
            )
            write(destination, record)
            return record

    workspace = WORK_ROOT / "work" / f"bank_{bank_name}" / case_id
    run_dir = output / "explorer" / f"bank_{bank_name}" / case_id
    attempt = 1
    while (run_dir / "result.json").exists():
        attempt += 1
        run_dir = output / "explorer" / f"bank_{bank_name}" / f"{case_id}_attempt{attempt}"
    started = time.monotonic()
    try:
        root = materialize(case, workspace)
        config = build_config(BANK_FILES[bank_name])
        result = rq.explorer(
            case,
            config,
            root,
            run_dir,
            BANK_FILES[bank_name],
        )
        predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
        search = read(run_dir / "fault_skill_search.json", {})
        record = result_record(
            case,
            bank_name,
            result.get("status"),
            predictions,
            (search.get("matched_skill") or {}).get("skill_id"),
            "rerun_selection_changed",
            search=search,
            runtime_seconds=result.get("runtime_seconds") or round(time.monotonic() - started, 3),
            error=result.get("error"),
        )
    except Exception as exc:
        record = result_record(
            case,
            bank_name,
            "adapter_failed",
            [],
            None,
            "rerun_selection_changed",
            runtime_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    write(destination, record)
    return record


def summarize(rows: list[dict], limit: int, bank_name: str) -> dict:
    def metrics(group: list[dict]) -> dict:
        return {
            metric: sum(row["metrics"][metric] for row in group) / len(group)
            if group
            else None
            for metric in ("top1", "top3", "top5", "mrr")
        }

    loaded = [row for row in rows if row.get("loaded_skill_id")]
    return {
        "bank": bank_name,
        "processed": len(rows),
        "target": limit,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "reuse_sources": dict(Counter(row["reuse_source"] for row in rows)),
        "loaded_skill_count": len(loaded),
        **metrics(rows),
        "loaded_cases": metrics(loaded),
        "cases": rows,
    }


def run_bank(cases: list[dict], bank_name: str, output: Path, workers: int) -> dict:
    bank_output = output / "evaluation" / f"bank_{bank_name}"
    bank_output.mkdir(parents=True, exist_ok=True)
    previous = read(bank_output / "summary.json", {})
    existing = {
        row["instance_id"]: row for row in previous.get("cases", [])
    }
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
                f"bank_{bank_name} {len(ordered)}/{len(cases)} {case_id}: "
                f"{existing[case_id]['status']} [{existing[case_id]['reuse_source']}]",
                flush=True,
            )
    return summarize(
        [existing[case["instance_id"]] for case in cases],
        len(cases),
        bank_name,
    )


def run(limit: int, workers: int, output: Path, banks: list[str]) -> dict:
    cases = prepare(output, limit, workers, banks)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY must be supplied to this process")
    results = {}
    for bank_name in banks:
        if bank_name == "400":
            summary = read(
                SOURCE
                / "with_skill_failure_retries_20260929"
                / "comparison_summary_with_retries.json",
                {},
            )
            rows = []
            for case in cases:
                source = next(
                    row for row in summary["cases"] if row["instance_id"] == case["instance_id"]
                )
                arm = source["arms"]["with_skill"]
                rows.append(
                    result_record(
                        case,
                        bank_name,
                        arm.get("status"),
                        list(dict.fromkeys(arm.get("predictions") or []))[:5],
                        arm.get("loaded_skill_id"),
                        "frozen_main_400",
                        runtime_seconds=arm.get("runtime_seconds"),
                    )
                )
            results[bank_name] = summarize(rows, len(cases), bank_name)
            write(
                output / "evaluation" / "bank_400" / "summary.json",
                results[bank_name],
            )
            continue
        results[bank_name] = run_bank(cases, bank_name, output, workers)
    combined = {
        "target": len(cases),
        "banks": {
            bank: {
                key: value
                for key, value in summary.items()
                if key != "cases"
            }
            for bank, summary in results.items()
        },
    }
    write(output / "combined_summary.json", combined)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--banks", nargs="+", choices=sorted(BANK_FILES), default=["200", "300", "400"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("limit must be between 1 and 500")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if args.dry_run:
        prepare(args.output, args.limit, args.workers, args.banks)
        print(json.dumps(read(args.output / "experiment_manifest.json", {}), indent=2))
    else:
        print(
            json.dumps(
                run(args.limit, args.workers, args.output, args.banks),
                ensure_ascii=False,
                indent=2,
            )
        )
