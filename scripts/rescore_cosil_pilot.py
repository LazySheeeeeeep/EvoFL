"""Rescore existing CoSIL author outputs without rerunning inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

if __package__:
    from .normalize_cosil_functions import normalize
    from .run_cosil_pilot import PILOT, EVALUATION, read_json, read_jsonl, score, selected_cases
else:
    from normalize_cosil_functions import normalize
    from run_cosil_pilot import PILOT, EVALUATION, read_json, read_jsonl, score, selected_cases


DEFAULT_RESULTS = PILOT.parent / "cosil_pilot_results_v2"


def rescore(pilot: Path = PILOT, results: Path = DEFAULT_RESULTS,
            evaluation: Path = EVALUATION, limit: int = 10,
            reuse_results: Path | None = None) -> dict:
    pilot = pilot.resolve(strict=True)
    results = results.resolve(strict=True)
    evaluation = evaluation.resolve(strict=True)
    if reuse_results is not None:
        reuse_results = reuse_results.resolve(strict=True)
    original_path = results / "comparison_summary.json"
    original = read_json(original_path)
    selected = selected_cases(pilot, limit)
    if [row["instance_id"] for row in selected] != [row["instance_id"] for row in original["cases"]]:
        raise ValueError("Pilot case IDs or ordering differ from the original run")
    rescored = []
    for case in original["cases"]:
        case_id = case["instance_id"]
        result = {key: value for key, value in case.items() if key != "metrics"}
        interrupted_reuse = (result["status"] == "invalid_author_output"
                             and "reused_author_output_sha256" in result
                             and "normalized.json" in result.get("error", ""))
        if result["status"] in {"completed", "empty_normalized_prediction"} or interrupted_reuse:
            author_path = results / "cases" / case_id / "function" / "loc_outputs_func.jsonl"
            if "reused_author_output_sha256" in result:
                if reuse_results is None:
                    raise ValueError("Explicit reuse-results path required for reused author output")
                author_path = reuse_results / "cases" / case_id / "function" / "loc_outputs_func.jsonl"
                if hashlib.sha256(author_path.read_bytes()).hexdigest() != result["reused_author_output_sha256"]:
                    raise ValueError(f"Reused author output changed: {case_id}")
            rows = read_jsonl(author_path)
            if len(rows) != 1 or rows[0].get("instance_id") != case_id:
                raise ValueError(f"Missing or ambiguous author output for {case_id}")
            structure = read_json(pilot / "repo_structures" / f"{case_id}.json")
            mapped = normalize(rows[0], structure["structure"])
            result["ranked_functions"] = mapped["ranked_functions"]
            result["status"] = "completed" if mapped["ranked_functions"] else "empty_normalized_prediction"
            result["recovered_mapping_count"] = sum(
                item["status"].startswith("raw_xml_recovered_") for item in mapped["mappings"]
            )
        rescored.append(result)
    scored = score(rescored, evaluation)
    count = len(scored)
    summary = {
        "method": "CoSIL",
        "adapter": "deepseek_non_thinking_v1+strict_raw_xml_path_recovery_v1",
        "source_summary_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "processed": count,
        "target": len(selected),
        "statuses": dict(Counter(item["status"] for item in scored)),
        "recovered_mapping_count": sum(item.get("recovered_mapping_count", 0) for item in scored),
        **{metric: sum(item["metrics"][metric] for item in scored) / count
           for metric in ("top1", "top3", "top5", "mrr")},
        "cases": scored,
    }
    (results / "comparison_summary_recovered.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--reuse-results", type=Path)
    args = parser.parse_args()
    result = rescore(args.pilot, args.results, args.evaluation, args.limit,
                     args.reuse_results)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
