"""Evaluate episodic or flat-reflection memory on the frozen RQ1 cases.

This runner never writes into the three-arm RQ1 experiment. Paid evaluation is
gated on complete acquisition and an audited, frozen memory preparation set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from prepare_rq1_memory_baselines import EXPANDED, HISTORICAL, OUTPUT as PREPARATION, read_json, write_json
from rq1_memory_runtime import EpisodeRetriever, MemoryInjectedClient, episodic_memory_item
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/rq1_memory_baselines_20260924"
ARMS = ("episodic", "flat_reflection", "oracle_function_reflection")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_inputs(expanded: Path, preparation: Path, arm: str) -> tuple[list[dict], list[dict]]:
    protocol = read_json(expanded / "protocol.json")
    complete = read_json(expanded / "training_complete.json") if (expanded / "training_complete.json").exists() else {}
    prepared = read_json(preparation / "preparation_summary.json")
    if complete.get("count") != 400 or prepared.get("status") != "complete":
        raise ValueError("Finish 400 acquisition cases and refresh memory preparation before evaluation")
    if sha(expanded / "frozen_skills.jsonl") != complete["bank_sha256"]:
        raise ValueError("Frozen RQ1 bank changed")
    for path, expected in prepared["manifest_hashes"].items():
        if sha(expanded / path) != expected:
            raise ValueError("RQ1 input manifest changed: " + path)
    for path, expected in prepared["prepared_hashes"].items():
        if sha(preparation / path) != expected:
            raise ValueError("Prepared memory input changed: " + path)
    evaluation = read_json(expanded / "evaluation_cases.json")
    episodes = load_jsonl(preparation / "episodes.jsonl")
    if len(evaluation) != 500 or len(episodes) != 400:
        raise ValueError("Expected exactly 500 evaluation cases and 400 acquisition episodes")
    if len({row["instance_id"] for row in episodes}) != 400:
        raise ValueError("Duplicate acquisition episode")
    if {row["instance_id"] for row in episodes} & {case["instance_id"] for case in evaluation}:
        raise ValueError("Acquisition/evaluation identity overlap")
    if arm in {"flat_reflection", "oracle_function_reflection"}:
        lessons = []
        folder = ("flat_reflections" if arm == "flat_reflection"
                  else "oracle_function_reflections")
        for episode in episodes:
            path = preparation / folder / (episode["instance_id"] + ".json")
            if not path.is_file():
                raise ValueError("Flat reflection missing: " + episode["instance_id"])
            item = read_json(path)
            if item.get("instance_id") != episode["instance_id"] or not isinstance(item.get("lesson"), str):
                raise ValueError("Invalid flat reflection: " + episode["instance_id"])
            if arm == "oracle_function_reflection" and (
                not isinstance(item.get("source_oracle_sha256"), str)
                or item.get("ground_truth_source") not in {
                    "evolution_evidence", "frozen_training_manifest", "audited_patch_function_mapping"
                }
            ):
                raise ValueError("Invalid oracle reflection provenance: " + episode["instance_id"])
            expected_episode_hash = hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest()
            if item.get("source_episode_sha256") != expected_episode_hash:
                raise ValueError("Flat reflection source episode changed: " + episode["instance_id"])
            if item["lesson"].strip():
                lessons.append({**item, "issue": episode["issue"]})
        episodes = lessons
    return evaluation, episodes


def choose_memory(retriever: EpisodeRetriever, case: dict, arm: str) -> dict:
    selected = retriever.retrieve(issue=case["problem_statement"], instance_id=case["instance_id"], limit=1)
    if not selected:
        return {"arm": arm, "selected_source_id": None, "score": None, "memory": None}
    score, item = selected[0]["score"], selected[0]["episode"]
    if arm == "episodic":
        memory = episodic_memory_item(item)
    else:
        memory = {"source_instance_id": item["instance_id"], "source_repo": item["repo"],
                  "applicability": item["applicability"], "lesson": item["lesson"],
                  "source_skill_conditioned": item["source_skill_conditioned"]}
    return {"arm": arm, "selected_source_id": item["instance_id"], "score": score, "memory": memory}


def materialize(case: dict, output: Path, expanded: Path, historical: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; no automatic artifact deletion")
    archive = output / "sources" / (case["instance_id"] + ".tar.gz")
    if not archive.exists():
        for source in (expanded / "sources" / archive.name, historical / "sources" / archive.name):
            if source.is_file():
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, archive)
                break
    bench.OUT = output
    workspace = output / "work" / case["instance_id"]
    root, digest = bench.unpack(case, workspace)
    audit = read_json(expanded / "function_audit" / (case["instance_id"] + ".json"))
    if audit:
        for source in audit["sources"]:
            if sha(root / source["path"]) != source["sha256"]:
                raise ValueError("Materialized source differs from audited buggy base")
    return workspace, root


def evaluate_one(case: dict, arm: str, selected: dict, *, expanded: Path, historical: Path,
                 output: Path) -> dict:
    cid = case["instance_id"]
    directory = output / arm / "cases" / cid
    selection_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    checkpoint = directory / "memory_selection.json"
    result_path = directory / "result.json"
    if checkpoint.exists() and not result_path.exists():
        if read_json(checkpoint).get("selection_sha256") != selection_hash:
            raise ValueError("Mismatched interrupted memory evaluation checkpoint: " + cid)
        attempt = 1
        while directory.with_name(f"{cid}.interrupted_{attempt}").exists():
            attempt += 1
        directory.rename(directory.with_name(f"{cid}.interrupted_{attempt}"))
    if checkpoint.exists() or result_path.exists():
        if not checkpoint.exists() or not result_path.exists() or read_json(checkpoint)["selection_sha256"] != selection_hash:
            raise ValueError("Incomplete or mismatched memory evaluation checkpoint: " + cid)
        result = read_json(result_path)
    else:
        arm_output = output / arm
        workspace, root = materialize(case, arm_output, expanded, historical)
        config = copy.deepcopy(read_json(expanded / "config.json"))
        config["skill_bank"]["path"] = str(expanded / "empty_skills.jsonl")
        config["skill_bank"]["enabled_skill_types"] = []
        client = OpenAICompatibleClient.from_config(config["llm"])
        if selected["memory"]:
            client = MemoryInjectedClient(client, instance_id=cid,
                                          issue=case["problem_statement"], memory=selected["memory"])
        directory.mkdir(parents=True, exist_ok=True)
        write_json(checkpoint, {**selected, "selection_sha256": selection_hash})
        result = run_explorer(task={"instance_id": cid, "repo": case["repo"],
            "base_commit": case["base_commit"], "bug_report": case["problem_statement"],
            "repo_path": str(root), "run_dir": str(directory)}, config=config,
            llm_client=client, skill_bank=make_skill_bank(config))
        bench.OUT = arm_output
        bench.clean_workspace(workspace)
    return {"instance_id": cid, "status": result.get("status"),
            "predictions": (result.get("ranked_functions") or [])[:5],
            "metrics": bench.strict_metrics((result.get("ranked_functions") or [])[:5],
                                           case["function_ground_truth"]),
            "memory_source_id": selected["selected_source_id"], "memory_score": selected["score"]}


def run(*, arm: str, expanded: Path = EXPANDED, historical: Path = HISTORICAL,
        preparation: Path = PREPARATION, output: Path = OUT, limit: int = 500,
        dry_run: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown memory baseline arm")
    evaluation, memories = validate_inputs(expanded, preparation, arm)
    retriever = EpisodeRetriever(memories)
    selected_cases = evaluation[:limit]
    selections = [choose_memory(retriever, case, arm) for case in selected_cases]
    if dry_run:
        return {"arm": arm, "status": "dry_run", "acquisition_memories": len(memories),
                "evaluation_selected": len(selected_cases),
                "retrieval_coverage": sum(s["memory"] is not None for s in selections),
                "model_calls": 0}
    rows = []
    for case, selected in zip(selected_cases, selections):
        rows.append(evaluate_one(case, arm, selected, expanded=expanded,
                                 historical=historical, output=output))
        summary = {"arm": arm, "processed": len(rows), "target": len(selected_cases),
                   "statuses": dict(Counter(row["status"] for row in rows)),
                   "retrieved_count": sum(row["memory_source_id"] is not None for row in rows),
                   "top1": sum(row["metrics"]["top1"] for row in rows) / len(rows),
                   "top3": sum(row["metrics"]["top3"] for row in rows) / len(rows),
                   "top5": sum(row["metrics"]["top5"] for row in rows) / len(rows),
                   "mrr": sum(row["metrics"]["mrr"] for row in rows) / len(rows),
                   "cases": rows}
        write_json(output / arm / "comparison_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(arm=args.arm, limit=args.limit, dry_run=args.dry_run), indent=2))
