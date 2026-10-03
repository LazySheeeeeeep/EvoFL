"""Prepare auditable acquisition episodes for independent memory baselines.

This script is read-only with respect to the active RQ1 experiment. It does not
call an LLM or claim that skill-conditioned acquisition traces are no-skill data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
HISTORICAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
OUTPUT = ROOT / "runs/rq1_memory_baseline_preparation_20260924"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def compact_timeline(index: dict) -> list[dict]:
    timeline = []
    for event in index.get("timeline", []):
        if event.get("event") != "investigation_action":
            continue
        arguments = event.get("arguments") or {}
        timeline.append({
            "step": event.get("step"),
            "tool": event.get("tool"),
            "purpose": event.get("purpose"),
            "based_on": event.get("based_on") or [],
            "observation_id": event.get("observation_id"),
            "path": arguments.get("path"),
            "pattern": arguments.get("pattern"),
            "candidate_updates": event.get("candidate_updates") or [],
        })
    return timeline


def prepare(expanded: Path = EXPANDED, historical: Path = HISTORICAL,
            output: Path = OUTPUT) -> dict:
    protocol = read_json(expanded / "protocol.json")
    if protocol.get("training_target") != 400 or protocol.get("evaluation_target") != 500:
        raise ValueError("Unexpected RQ1 protocol; refusing to combine manifests")
    original = read_json(expanded / "original_training.json")
    additions = read_json(expanded / "training_additions.json")
    evaluation = read_json(expanded / "evaluation_cases.json")
    if len(original) != 200 or len(additions) != 200 or len(evaluation) != 500:
        raise ValueError("RQ1 acquisition/evaluation manifest size changed")
    sources = [(case, historical) for case in original] + [(case, expanded) for case in additions]
    train_ids = [case["instance_id"] for case, _ in sources]
    eval_ids = [case["instance_id"] for case in evaluation]
    if len(set(train_ids)) != 400 or len(set(eval_ids)) != 500 or set(train_ids) & set(eval_ids):
        raise ValueError("Duplicate or overlapping acquisition/evaluation case")
    if any(case["created_at"] >= protocol["train_before"] for case, _ in sources):
        raise ValueError("Acquisition case violates temporal cutoff")
    if any(case["created_at"] < protocol["test_created_from"] for case in evaluation):
        raise ValueError("Evaluation case violates temporal cutoff")

    episodes, reflection_inputs, missing = [], [], []
    for case, source in sources:
        cid = case["instance_id"]
        directory = source / "training" / cid
        paths = {name: directory / name for name in (
            "record.json", "explorer/result.json", "explorer/investigation_index.json",
            "explorer/initial_payload.json", "explorer/loaded_skills.json")}
        required = ["record.json", "explorer/result.json", "explorer/investigation_index.json",
                    "explorer/initial_payload.json"]
        absent = [name for name in required if not paths[name].is_file()]
        if absent:
            missing.append({"instance_id": cid, "missing": absent})
            continue
        record = read_json(paths["record.json"])
        result = read_json(paths["explorer/result.json"])
        index = read_json(paths["explorer/investigation_index.json"])
        payload = read_json(paths["explorer/initial_payload.json"])
        if record.get("instance_id") != cid or result.get("instance_id") != cid:
            raise ValueError(f"Case artifact identity mismatch: {cid}")
        if index.get("trace_version") != "v5" or result.get("trace_version") != "v5":
            raise ValueError(f"Non-v5 investigation: {cid}")
        if str(payload.get("bug_report") or "").strip() != case["problem_statement"].strip():
            raise ValueError(f"Issue text mismatch: {cid}")
        loaded = read_json(paths["explorer/loaded_skills.json"]) if paths["explorer/loaded_skills.json"].is_file() else {}
        loaded_ids = [skill["skill_id"] for skill in loaded.values()
                      if isinstance(skill, dict) and skill.get("skill_id")]
        episode = {
            "instance_id": cid,
            "repo": case["repo"],
            "created_at": case["created_at"],
            "issue": payload["bug_report"],
            "trace_origin": "sequential_evolutefl_training",
            "skill_conditioned": bool(loaded_ids),
            "loaded_skill_ids": loaded_ids,
            "status": result.get("status"),
            "investigation": compact_timeline(index),
            "predicted_functions": result.get("ranked_functions") or [],
            "final_summary": result.get("final_summary") or "",
            "artifact_hashes": {name: digest(paths[name]) for name in required},
        }
        episodes.append(episode)
        reflection_inputs.append({
            "instance_id": cid,
            "episode_index": len(episodes) - 1,
            "historical_patch_sha256": hashlib.sha256(case["patch"].encode()).hexdigest(),
            "patch_source": str(expanded / ("original_training.json" if source == historical
                                            else "training_additions.json")),
            "explorer_status": record.get("explorer_status"),
            "label_policy": "recompute_before_reflection",
            "note": "Patch and outcome are available only to the acquisition-side reflection builder, never retrieval.",
        })

    historical_ids = set(train_ids[:200])
    summary = {
        "status": "complete" if len(episodes) == 400 else "provisional",
        "acquisition_target": 400,
        "available_episode_count": len(episodes),
        "historical_episode_count": sum(row["instance_id"] in historical_ids for row in episodes),
        "skill_conditioned_episode_count": sum(row["skill_conditioned"] for row in episodes),
        "missing_count": len(missing),
        "missing": missing,
        "evaluation_count": 500,
        "evaluation_overlap_count": 0,
        "temporal_cutoffs": {"train_before": protocol["train_before"],
                             "test_created_from": protocol["test_created_from"]},
        "manifest_hashes": {name: digest(expanded / name) for name in (
            "protocol.json", "original_training.json", "training_additions.json", "evaluation_cases.json")},
        "data_lineage_warning": "Acquisition trajectories may already contain EvoluteFL skills. "
                                "Episode and flat-reflection arms built from them are memory-format ablations, "
                                "not independent no-skill acquisition baselines.",
        "next_steps": ["freeze 400 episodes after training", "build distinct episodic and flat-reflection memories",
                       "smoke retrieval without evaluation cases", "run paid evaluation on frozen 500-case manifest"],
    }
    episodes_path = output / "episodes.jsonl"
    inputs_path = output / "flat_reflection_inputs.jsonl"
    write_jsonl(episodes_path, episodes)
    write_jsonl(inputs_path, reflection_inputs)
    summary["prepared_hashes"] = {path.name: digest(path) for path in (episodes_path, inputs_path)}
    write_json(output / "preparation_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = prepare(args.expanded, args.historical, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "missing"}, indent=2))
