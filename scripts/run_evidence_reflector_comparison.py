"""Six-case audit, matched-history retraining, then held-out three-arm evaluation.

The audit is a manual gate, not an automatic semantic-quality validator.
Reuse the transactional replay runner without modifying historical experiments.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import run_experience_expansion99 as replay
import run_swe_explore_v5_stratified as benchmark
from evolutefl.config import load_config

ROOT = replay.ROOT
OLD = ROOT / "runs/experience_expansion99_deepseek_20260913"
OUT = ROOT / "runs/evidence_reflector_heldout30_deepseek_20260914"
read, write, sha = replay.read, replay.write, replay.sha


def audit_sample(cases):
    groups = {"top5_miss": [], "ranking_gap": [], "top1_hit": []}
    for case in cases:
        if not case["function_ground_truth"]:
            continue
        metric = case["original_no_skill_metrics"]
        group = "top1_hit" if metric["top1"] else "ranking_gap" if metric["top5"] else "top5_miss"
        groups[group].append(case)
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Need two available traces in each audit stratum")
    return [{**case, "audit_stratum": name} for name, group in groups.items() for case in group[:2]]


def exposed_ids(directory):
    """Exclude identifiable cases in retained manifests, results and training traces."""
    ids, files = set(), []
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("instance_id"), str):
                ids.add(value["instance_id"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for path in directory.rglob("*.json"):
        if OUT in path.parents or path.is_symlink() or "work" in path.parts:
            continue
        if path.name not in {"selected_cases.json", "task.json", "training_record.json", "initial_payload.json"}:
            continue
        try:
            collect(read(path))
            files.append(str(path))
        except (ValueError, OSError):
            raise ValueError(f"Cannot audit historical exposure: {path}")
    return ids, files


def prepare():
    if (OUT / "protocol.json").exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "config.json").exists():
        raise ValueError("Partial preparation requires inspection")
    cases = read(OLD / "selected_cases.json")
    if len(cases) != 99:
        raise ValueError("Expected the exact previous 99 training cases")
    if sha(OLD / "expanded_skills.jsonl") != read(OLD / "training_complete.json")["expanded_bank_sha256"]:
        raise ValueError("Old trained bank differs from its checkpoint")
    cfg = load_config()
    cfg["llm"] = read(OLD / "config.json")["llm"]
    if cfg["llm"].get("api_key"):
        raise ValueError("Environment credentials only")
    cfg["explorer"].update(workflow_version="v5", max_steps=30, max_runtime_seconds=1200)
    cfg["reflection"]["strict_function_matching"] = True
    cfg["embedding"]["enabled"] = False
    cfg["skill_bank"].update(enabled_skill_types=["fault_skill"], retrieval_mode="lexical")
    hashes = {}
    for section in ("explorer", "reflection"):
        for key, value in list(cfg[section].items()):
            if key.endswith("prompt_path"):
                dest = OUT / "prompts" / Path(value).name
                replay.copy_atomic(Path(value), dest)
                cfg[section][key] = str(dest)
                hashes[str(dest)] = sha(dest)
    replay.copy_atomic(OLD / "old_skills.jsonl", OUT / "initial_skills.jsonl")
    replay.copy_atomic(OLD / "expanded_skills.jsonl", OUT / "old_skills.jsonl")
    (OUT / "empty_skills.jsonl").write_text("")
    cfg["skill_bank"]["path"] = str(OUT / "initial_skills.jsonl")
    write(OUT / "config.json", cfg)
    write(OUT / "training_cases.json", cases)
    write(OUT / "audit_cases.json", audit_sample(cases))
    excluded, files = exposed_ids(ROOT / "runs")
    excluded.update(c["instance_id"] for c in cases)
    write(OUT / "exposure_audit.json", {"instance_ids": sorted(excluded), "source_files": files,
        "limitation": "Unseen relative to retained records; deleted historical exposures cannot be certified."})
    for path in [OUT / "config.json", OUT / "training_cases.json", OUT / "audit_cases.json",
                 OUT / "initial_skills.jsonl", OUT / "old_skills.jsonl", OUT / "empty_skills.jsonl",
                 OUT / "exposure_audit.json", *sorted((ROOT / "src").rglob("*.py")),
                 Path(__file__), Path(replay.__file__), Path(benchmark.__file__)]:
        hashes[str(path)] = sha(path)
    for case in cases:
        for name in replay.TRACE_FILES:
            path = replay.SOURCE / "no_skill/cases" / case["instance_id"] / name
            hashes[str(path)] = sha(path)
    write(OUT / "protocol.json", {"training_count": 99, "audit_count": 6, "evaluation_count": 30,
        "model": cfg["llm"]["model"], "seed": 20260914,
        "arms": {"no_skill": "empty bank", "old_bank": "old reflector trained bank (181)",
                 "expanded_bank": "new evidence-driven reflector, same initial bank and training order"},
        "note": "Prompt-package comparison; old bank was generated earlier, so provider-time drift remains a limitation.",
        "hashes": hashes})


def run_training(cases, cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "old_skills.jsonl").exists():
        replay.copy_atomic(OUT / "initial_skills.jsonl", directory / "old_skills.jsonl")
    replay.OUT = directory
    replay.train(cases, cfg)
    report = read(directory / "training_summary.json")
    report["planned"] = len(cases)
    write(directory / "training_summary.json", report)


def select_test_cases():
    if (OUT / "selected_cases.json").exists():
        selection = read(OUT / "selection_protocol.json")
        if sha(OUT / "selected_cases.json") != selection["sha256"]:
            raise ValueError("Evaluation selection changed")
        return read(OUT / "selected_cases.json")
    from datasets import load_dataset
    excluded = set(read(OUT / "exposure_audit.json")["instance_ids"])
    explore = load_dataset("SWE-Explore-Bench/SWE-Explore-Bench", split="train")
    verified = {c["instance_id"]: c for c in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    pool = [benchmark._merge_case(c, verified[c["instance_id"]]) for c in explore
            if c.get("dataset") == "verified" and c["instance_id"] in verified
            and c["instance_id"] not in excluded and (c.get("ground_truth") or {}).get("read_core_regions")]
    pool = [c for c in pool if all(c.get(k) for k in ("repo", "problem_statement", "base_commit", "patch"))]
    cases, quotas = benchmark.balanced_sample(pool, 30, 20260914)
    write(OUT / "selected_cases.json", cases)
    write(OUT / "selection_protocol.json", {"sha256": sha(OUT / "selected_cases.json"),
        "pool_size": len(pool), "project_quotas": quotas, "excluded_count": len(excluded)})
    return cases


def prepare_test_sources(cases):
    benchmark.OUT = OUT
    mapped = []
    for case in cases:
        path = OUT / "materialized" / (case["instance_id"] + ".json")
        saved = read(path)
        if saved is None:
            if __import__("shutil").disk_usage(OUT).free < 8 * 1024**3:
                raise RuntimeError("Less than 8 GiB free")
            work = OUT / "work" / case["instance_id"]
            try:
                root, digest = benchmark.unpack(case, work)
                symbols = benchmark.symbols_from_patch(case["patch"], root, source_side="old")
                saved = {**case, "function_ground_truth": symbols["functions"],
                         "original_archive_sha256": digest, "classes": symbols["classes"]}
                write(path, saved)
            except Exception as exc:
                write(OUT / "materialization_error.json", {"instance_id": case["instance_id"], "error": str(exc)})
                raise
            finally:
                benchmark.clean_workspace(work)
        mapped.append(saved)
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "audit", "full"), default="audit")
    args = parser.parse_args()
    prepare()
    replay.verify(read(OUT / "protocol.json"))
    if args.phase == "prepare":
        print("Prepared frozen experiment inputs", flush=True)
        return
    cfg = read(OUT / "config.json")
    if not os.environ.get(cfg["llm"]["api_key_env"]):
        raise ValueError("Credential environment variable missing")
    if args.phase == "audit":
        run_training(read(OUT / "audit_cases.json"), cfg, OUT / "audit")
        write(OUT / "current_case.json", {"phase": "awaiting_manual_audit", "count": 6})
        print("Six-case audit finished; inspect conclusions and cards before full training.", flush=True)
        return
    review = read(OUT / "audit_review.json", {})
    if review.get("approved") is not True or len(review.get("cases", [])) != 6:
        raise ValueError("Six-case manual review required before full training")
    run_training(read(OUT / "training_cases.json"), cfg, OUT / "retraining")
    replay.copy_atomic(OUT / "retraining/expanded_skills.jsonl", OUT / "expanded_skills.jsonl")
    replay.copy_atomic(OUT / "retraining/training_complete.json", OUT / "training_complete.json")
    replay.verify(read(OUT / "protocol.json"))
    cases = prepare_test_sources(select_test_cases())
    replay.OUT = OUT
    replay.SOURCE = OUT
    original_report = replay.evaluate_report
    def report(rows):
        original_report(rows)
        result = read(OUT / "comparison_summary.json")
        result.update(evaluation_kind="held-out relative to retained records", planned=30)
        write(OUT / "comparison_summary.json", result)
    replay.evaluate_report = report
    replay.evaluate(cases, cfg)
    replay.verify(read(OUT / "protocol.json"))
    write(OUT / "current_case.json", {"phase": "finished", "training": 99, "evaluation": 30})


if __name__ == "__main__":
    main()
