"""Frozen old-bank -> transactional reflection replay -> three-arm resubstitution.

Uses the 99 completed cases of the earlier SWE-Explore run. These are learning
and resubstitution cases, not an independent test set. No embedding or tests run.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolutefl.config import load_config
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import make_skill_bank
from run_swe_explore_v5_stratified import read, write, sha, safe_archive_filter, strict_metrics

SOURCE = ROOT / "runs/swe_explore_v5_stratified100_deepseek_20260910"
OUT = ROOT / "runs/experience_expansion99_deepseek_20260913"
ARMS = ("no_skill", "old_bank", "expanded_bank")
TRACE_FILES = ("result.json", "run_context.json", "initial_payload.json", "trajectory.jsonl",
               "observations.jsonl", "investigation_index.json", "fault_skill_request.json",
               "fault_skill_search.json")


def copy_atomic(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(dest)


def bank_stats(path, config):
    cfg = copy.deepcopy(config)
    cfg["skill_bank"]["path"] = str(path)
    cards = make_skill_bank(cfg).active_skills()
    return {"active_count": len(cards), "types": dict(Counter(c.skill_type for c in cards)),
            "bank_sha256": sha(path)}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        return
    if (OUT / "old_skills.jsonl").exists():
        raise ValueError("Partial preparation; inspect before replacing inputs")
    original = read(SOURCE / "protocol.json")
    for name, key in (("frozen_skills.jsonl", "bank_sha256"), ("selected_cases.json", "selection_sha256")):
        if sha(SOURCE / name) != original[key]:
            raise ValueError(f"Original frozen input changed: {name}")
    cases, excluded, hashes = [], [], {}
    for case in read(SOURCE / "selected_cases.json"):
        cid = case["instance_id"]
        pair_path = SOURCE / "paired" / f"{cid}.json"
        pair = read(pair_path)
        if pair.get("skip_reason"):
            excluded.append({"instance_id": cid, "reason": pair["skip_reason"]})
            continue
        if pair["arms"]["no_skill"]["status"] != "completed":
            raise ValueError(f"Expected completed no_skill trajectory: {cid}")
        prefix = SOURCE / "no_skill/cases" / cid
        hashes[str(pair_path)] = sha(pair_path)
        for name in TRACE_FILES:
            path = prefix / name
            hashes[str(path)] = sha(path)
        cases.append({**case, "function_ground_truth": pair["functions"],
                      "original_archive_sha256": pair["archive_sha256"],
                      "original_no_skill_metrics": pair["arms"]["no_skill"]["metrics"]})
    if len(cases) != 99:
        raise ValueError(f"Expected 99 completed cases, found {len(cases)}")
    cfg = load_config()
    cfg["llm"] = read(SOURCE / "frozen_config.json")["llm"]
    if cfg["llm"].get("api_key"):
        raise ValueError("Use environment credentials only")
    cfg["explorer"].update(workflow_version="v5", max_steps=30, max_runtime_seconds=1200)
    cfg["reflection"]["strict_function_matching"] = True
    cfg["embedding"]["enabled"] = False
    cfg["skill_bank"].update(enabled_skill_types=["fault_skill"], retrieval_mode="lexical")
    for section in ("explorer", "reflection"):
        for key, value in list(cfg[section].items()):
            if key.endswith("prompt_path"):
                dest = OUT / "prompts" / Path(value).name
                copy_atomic(Path(value), dest)
                cfg[section][key] = str(dest)
                hashes[str(dest)] = sha(dest)
    copy_atomic(SOURCE / "frozen_skills.jsonl", OUT / "old_skills.jsonl")
    (OUT / "empty_skills.jsonl").write_text("")
    cfg["skill_bank"]["path"] = str(OUT / "old_skills.jsonl")
    write(OUT / "config.json", cfg)
    write(OUT / "selected_cases.json", cases)
    for path in [OUT / "config.json", OUT / "selected_cases.json", OUT / "old_skills.jsonl",
                 OUT / "empty_skills.jsonl", *sorted((ROOT / "src").rglob("*.py")), Path(__file__)]:
        hashes[str(path)] = sha(path)
    write(OUT / "protocol.json", {
        "purpose": "experience expansion and same-case resubstitution, NOT held-out generalization",
        "selected_count": len(cases), "excluded": excluded,
        "training_source_arm": "no_skill", "initial_bank": bank_stats(OUT / "old_skills.jsonl", cfg),
        "training_function_eligible": sum(bool(c["function_ground_truth"]) for c in cases),
        "training_order": "original frozen sample order", "model": cfg["llm"]["model"],
        "arms": list(ARMS), "evaluation": "fresh Explorer conversations, frozen banks, no evolution",
        "scoring": "strict function identity; class-only/unmapped cases excluded from metric denominator",
        "patch_direction": "buggy_base_to_repaired; never applied to repository",
        "hashes": hashes,
    })


def verify(protocol):
    for name, digest in protocol["hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Frozen input/code changed: {name}")


def clean(work):
    boundary = (OUT / "work").resolve()
    if work.is_symlink() or work.resolve() == boundary or not work.resolve().is_relative_to(boundary):
        raise ValueError("Unsafe workspace cleanup")
    if work.exists():
        shutil.rmtree(work)


def materialize(case):
    if shutil.disk_usage(OUT).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; stopping without deleting experimental results")
    work = OUT / "work" / case["instance_id"]
    archive = SOURCE / "sources" / f"{case['instance_id']}.tar.gz"
    if sha(archive) != case["original_archive_sha256"]:
        raise ValueError("Source archive changed")
    clean(work)
    work.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter=safe_archive_filter)
    roots = list(work.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Unexpected repository archive layout")
    return work, roots[0]


def training_report(rows, bank, config):
    write(OUT / "training_summary.json", {
        "processed": len(rows), "planned": 99,
        "statuses": dict(Counter(r["status"] for r in rows)),
        "decisions": dict(Counter(r.get("decision", "none") for r in rows)),
        "outcomes": dict(Counter(r.get("outcome", "ineligible") for r in rows)),
        "failure_stages": dict(Counter(r.get("failure_stage") for r in rows if r["status"] == "failed")),
        "current_bank": str(bank), "bank": bank_stats(bank, config), "cases": rows})


def train(cases, cfg):
    bank, rows = OUT / "old_skills.jsonl", []
    for index, case in enumerate(cases, 1):
        cid = case["instance_id"]
        directory = OUT / "training" / cid
        record_path = directory / "training_record.json"
        saved = read(record_path)
        if saved:
            if sha(bank) != saved["input_bank_sha256"] or sha(Path(saved["output_bank"])) != saved["output_bank_sha256"]:
                raise ValueError("Training checkpoint chain differs")
            bank = Path(saved["output_bank"])
            rows.append(saved)
            continue
        write(OUT / "current_case.json", {"phase": "training", "index": index, "total": len(cases), "instance_id": cid})
        print(f"TRAIN {index}/{len(cases)} {cid}", flush=True)
        directory.mkdir(parents=True, exist_ok=True)
        initial_hash = sha(bank)
        transaction = directory / "candidate_skills.jsonl"
        summary_path = directory / "evolution/case_evolution_summary.json"
        evolution = read(summary_path)
        if not case["function_ground_truth"]:
            evolution = {"status": "skipped", "reason": "No function-level ground truth; class/module-only or unmapped"}
        elif evolution is None:
            if transaction.exists():
                raise RuntimeError(f"Unfinished training transaction needs review: {directory}")
            work, root = materialize(case)
            copy_atomic(bank, transaction)
            arm_cfg = copy.deepcopy(cfg)
            arm_cfg["skill_bank"]["path"] = str(transaction)
            evolution = run_case_evolution(
                case_run_dir=SOURCE / "no_skill/cases" / cid,
                repo=case["repo"], issue=case["problem_statement"], repo_path=root,
                config=arm_cfg, llm_client=OpenAICompatibleClient.from_config(cfg["llm"]),
                ground_truth_functions=case["function_ground_truth"], ground_truth_patch=case["patch"],
                patch_metadata={"source": "SWE-bench_Verified repair patch", "direction": "buggy_base_to_repaired"},
                output_dir=directory / "evolution")
            clean(work)
        if evolution["status"] == "completed" and transaction.exists():
            bank = transaction
        record = {"instance_id": cid, "status": evolution["status"],
                  "reason": evolution.get("reason"), "failure_stage": evolution.get("failure_stage"),
                  "outcome": evolution.get("outcome", {}).get("label"), "decision": evolution.get("decision", "none"),
                  "updated_skill_ids": evolution.get("updated_skill_ids", []),
                  "investigation_status": evolution.get("investigation_status"),
                  "investigation_tool_calls": evolution.get("investigation_tool_calls"),
                  "input_bank_sha256": initial_hash, "output_bank": str(bank), "output_bank_sha256": sha(bank)}
        write(record_path, record)
        rows.append(record)
        training_report(rows, bank, cfg)
    training_report(rows, bank, cfg)
    frozen = OUT / "expanded_skills.jsonl"
    if frozen.exists() and sha(frozen) != sha(bank):
        raise ValueError("Expanded frozen bank differs from completed training")
    if not frozen.exists():
        copy_atomic(bank, frozen)
    write(OUT / "training_complete.json", {"processed": len(rows), "expanded_bank_sha256": sha(frozen),
                                           "bank": bank_stats(frozen, cfg)})


def evaluate_report(rows):
    eligible = [r for r in rows if r["function_ground_truth"]]
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "completed": sum(r["arms"][arm]["status"] == "completed" for r in rows),
            "function_evaluable": len(eligible),
            **{k: sum(r["arms"][arm]["metrics"][k] for r in eligible) / len(eligible)
               if eligible else None for k in ("top1", "top3", "top5", "mrr")},
            "loaded": sum(bool(r["arms"][arm]["loaded_skill_id"]) for r in rows),
            "selector_selected": sum(bool(r["arms"][arm]["selector_selected_id"]) for r in rows),
            "validator_passed": sum(r["arms"][arm]["validator_applicable"] is True for r in rows),
            "forced_finish": sum(r["arms"][arm]["forced_finish"] for r in rows)}
    comparisons = {}
    for baseline in ("no_skill", "old_bank"):
        comparisons[f"expanded_vs_{baseline}"] = {
            "delta": {k: arms["expanded_bank"][k] - arms[baseline][k] if eligible else None
                      for k in ("top1", "top3", "top5", "mrr")},
            "top1_better": [r["instance_id"] for r in eligible if r["arms"]["expanded_bank"]["metrics"]["top1"]
                            and not r["arms"][baseline]["metrics"]["top1"]],
            "top1_worse": [r["instance_id"] for r in eligible if not r["arms"]["expanded_bank"]["metrics"]["top1"]
                           and r["arms"][baseline]["metrics"]["top1"]]}
    write(OUT / "comparison_summary.json", {"evaluation_kind": "training-set resubstitution",
          "completed_triplets": len(rows), "planned": 99, "arms": arms, "comparisons": comparisons, "cases": rows})


def evaluate(cases, cfg):
    paths = {"no_skill": OUT / "empty_skills.jsonl", "old_bank": OUT / "old_skills.jsonl",
             "expanded_bank": OUT / "expanded_skills.jsonl"}
    hashes = {arm: sha(path) for arm, path in paths.items()}
    if hashes["expanded_bank"] != read(OUT / "training_complete.json")["expanded_bank_sha256"]:
        raise ValueError("Expanded bank changed")
    rows, orders = [], list(itertools.permutations(ARMS))
    for index, case in enumerate(cases):
        cid = case["instance_id"]
        pairfile = OUT / "evaluated" / f"{cid}.json"
        if pairfile.exists():
            rows.append(read(pairfile))
            continue
        row = {"instance_id": cid, "function_ground_truth": case["function_ground_truth"], "arms": {}}
        for arm in orders[index % len(orders)]:
            for name, path in paths.items():
                if sha(path) != hashes[name]:
                    raise ValueError("Frozen evaluation bank changed")
            directory = OUT / arm / "cases" / cid
            write(OUT / "current_case.json", {"phase": "evaluation", "index": index + 1, "total": len(cases),
                                              "instance_id": cid, "arm": arm})
            result = read(directory / "result.json")
            if result is None:
                if (directory / "trajectory.jsonl").exists():
                    raise RuntimeError(f"Incomplete Explorer trajectory needs review: {directory}")
                print(f"EVAL {index + 1}/{len(cases)} {cid} {arm}", flush=True)
                work, root = materialize(case)
                arm_cfg = copy.deepcopy(cfg)
                arm_cfg["skill_bank"]["path"] = str(paths[arm])
                write(directory / "task.json", case)
                result = run_explorer(task={"instance_id": cid, "repo": case["repo"], "base_commit": case["base_commit"],
                    "bug_report": case["problem_statement"], "repo_path": str(root), "run_dir": str(directory)},
                    config=arm_cfg, llm_client=OpenAICompatibleClient.from_config(cfg["llm"]), skill_bank=make_skill_bank(arm_cfg))
                clean(work)
            search = read(directory / "fault_skill_search.json", {})
            trace = search.get("search_trace", {})
            loaded = (search.get("matched_skill") or {}).get("skill_id")
            if arm == "no_skill" and loaded:
                raise ValueError("No-skill arm loaded a Skill")
            row["arms"][arm] = {"status": result["status"], "error": result.get("error"),
                "predictions": result.get("ranked_functions", []),
                "metrics": strict_metrics(result.get("ranked_functions", []), case["function_ground_truth"]),
                "loaded_skill_id": loaded, "fault_family": search.get("fault_family"),
                "selector_selected_id": (trace.get("selector") or {}).get("selected_skill_id"),
                "validator_applicable": (trace.get("validator") or {}).get("applicable"),
                "steps": result.get("steps"), "forced_finish": result.get("forced_finish", False)}
        write(pairfile, row)
        rows.append(row)
        evaluate_report(rows)
    evaluate_report(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare()
    protocol = read(OUT / "protocol.json")
    verify(protocol)
    if args.prepare_only:
        print({k: v for k, v in protocol.items() if k != "hashes"}, flush=True)
        return
    cfg = read(OUT / "config.json")
    if not os.environ.get(cfg["llm"]["api_key_env"]):
        raise ValueError("Credential environment variable missing")
    cases = read(OUT / "selected_cases.json")
    train(cases, cfg)
    verify(protocol)
    evaluate(cases, cfg)
    verify(protocol)
    write(OUT / "current_case.json", {"phase": "finished", "training_cases": len(cases), "evaluation_cases": len(cases)})


if __name__ == "__main__":
    main()
