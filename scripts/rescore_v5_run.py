"""Re-score saved predictions without modifying original runs or repositories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolutefl.evaluation import evaluate_ranked_functions, symbols_from_patch


def aggregate(rows, field):
    return {"count": len(rows), **{
        key: sum(row[field][key] for row in rows) / len(rows) if rows else None
        for key in ("top1", "top3", "top5", "reciprocal_rank")}}


def rescore(directory):
    original = json.loads((directory / "summary.json").read_text())
    selected = {c["instance_id"]: c for c in json.loads((directory / "selected_cases.json").read_text())}
    rows = []
    for case in original["cases"]:
        cid = case["instance_id"]
        predictions = case.get("ranked_functions", [])
        row = {"instance_id": cid, "predictions": predictions,
               "old_ground_truth": case.get("ground_truth_functions", []),
               "old_metrics": evaluate_ranked_functions(predictions, case.get("ground_truth_functions", []))}
        try:
            context = json.loads((directory / "cases" / cid / "run_context.json").read_text())
            scopes = symbols_from_patch(selected[cid]["patch"], context["repo_path"], source_side="new")
            row.update(scopes=scopes, function_eligible=bool(scopes["functions"]),
                       function_metrics=evaluate_ranked_functions(predictions, scopes["functions"]),
                       class_metrics=evaluate_ranked_functions(predictions, scopes["classes"]))
            row["changed"] = (row["old_ground_truth"] != scopes["functions"]
                              or row["old_metrics"] != row["function_metrics"])
        except (ValueError, OSError) as exc:
            row.update(mapping_error=str(exc), function_eligible=False)
        rows.append(row)
    valid = [r for r in rows if "mapping_error" not in r]
    eligible = [r for r in valid if r["function_eligible"]]
    report = {"prediction_source": "unchanged summary.json ranked_functions",
              "policy": "Nonblank, non-comment changed lines; old/new deepest AST scope; classes separate; no source writes.",
              "original_mixed_metrics": aggregate(rows, "old_metrics"),
              "function_metrics_eligible_only": aggregate(eligible, "function_metrics"),
              "function_metrics_all_mapped_cases_including_non_function_as_zero": aggregate(valid, "function_metrics"),
              "class_metrics_cases_with_class_changes": aggregate([r for r in valid if r["scopes"]["classes"]], "class_metrics"),
              "mapping_error_count": len(rows) - len(valid),
              "non_function_case_ids": [r["instance_id"] for r in valid if not r["function_eligible"]],
              "cases": rows}
    output = directory / "rescored_function_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    rescore(parser.parse_args().run_dir)
