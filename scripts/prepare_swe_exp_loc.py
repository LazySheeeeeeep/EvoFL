"""Prepare an explicit SWE-Exp-style FL memory adaptation, without API calls."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_rq1_memory_baselines import EXPANDED, HISTORICAL, OUTPUT, read_json, write_json, write_jsonl
from run_swe_explore_v5_stratified import strict_metrics
from build_rq1_oracle_function_reflections import load_oracle_evidence, oracle_input

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "runs/external_fl_baseline_preflight_20260924/swe_exp_loc"


def literal_prompts(path: Path) -> dict:
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value.value
    return result


def run() -> dict:
    preparation = read_json(OUTPUT / "preparation_summary.json")
    if preparation["prepared_hashes"]["episodes.jsonl"] != hashlib.sha256((OUTPUT / "episodes.jsonl").read_bytes()).hexdigest():
        raise ValueError("Acquisition episodes changed")
    for name, expected in preparation["manifest_hashes"].items():
        if hashlib.sha256((EXPANDED / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen manifest changed: {name}")
    cases = read_json(EXPANDED / "original_training.json") + read_json(EXPANDED / "training_additions.json")
    by_id = {row["instance_id"]: row for row in cases}
    original_ids = {row["instance_id"] for row in read_json(EXPANDED / "original_training.json")}
    eval_ids = {row["instance_id"] for row in read_json(EXPANDED / "evaluation_cases.json")}
    prompts_path = ROOT / "SWE-Exp/moatless/experience/prompts/exp_prompts.py"
    prompts = literal_prompts(prompts_path)
    plans = []
    for line in (OUTPUT / "episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        case_id = episode["instance_id"]
        if case_id not in by_id or case_id in eval_ids:
            raise ValueError("Acquisition/evaluation overlap")
        case = by_id[case_id]
        source = HISTORICAL if case_id in original_ids else EXPANDED
        explorer = source / "training" / case_id / "explorer"
        trajectory = explorer / "trajectory.jsonl"
        if not trajectory.is_file():
            raise ValueError(f"Missing complete trajectory: {case_id}")
        truth = oracle_input(episode, case, load_oracle_evidence(episode, case, source))
        outcome = strict_metrics(episode["predicted_functions"][:5], truth["oracle_repaired_functions"])
        plans.append({"instance_id": case_id, "repo": episode["repo"],
                      "branch": "success" if outcome["top5"] else "failure",
                      "eligible": episode["status"] == "completed",
                      "skill_conditioned_acquisition": episode["skill_conditioned"],
                      "trajectory_path": str(trajectory), "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                      "outcome": outcome,
                      "source_case_manifest": str(EXPANDED / ("original_training.json" if case_id in original_ids else "training_additions.json"))})
    write_jsonl(DESTINATION / "acquisition_plan.jsonl", plans)
    write_json(DESTINATION / "author_prompt_snapshot.json", prompts)
    report = {"status": "acquisition_inputs_prepared", "model_calls": 0,
              "acquisition_count": len(plans), "eligible_count": sum(row["eligible"] for row in plans),
              "branches": dict(Counter(row["branch"] for row in plans if row["eligible"])),
              "skill_conditioned_count": sum(row["skill_conditioned_acquisition"] for row in plans),
              "author_prompt_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
              "adaptation": "SWE-Exp-Loc: shared acquisition trajectories, FL outcome, Instructor search/view/finish",
              "original_components": ["issue-type extraction", "success perspective / failure reflection",
                                      "multilingual-e5-large-instruct top-10 recall", "same-repo exclusion",
                                      "LLM selection and task-specific generalization", "Instructor"],
              "remaining": ["FL extraction and Instructor/action adapter", "E5 model setup and cache",
                            "memory generation", "author component smoke", "frozen evaluation"],
              "data_lineage_warning": preparation["data_lineage_warning"]}
    write_json(DESTINATION / "preparation_summary.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
