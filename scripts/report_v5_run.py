"""Summarize a v5 case-by-case run without exposing provider credentials."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_report(directory: Path) -> dict:
    rows = []
    for case in sorted((directory / "cases").iterdir()):
        def read(name):
            path = case / name
            return json.loads(path.read_text()) if path.exists() else {}
        result = read("result.json")
        evolution = read("case_evolution/case_evolution_summary.json")
        conclusion = read("case_evolution/investigation_conclusion.json")
        search = read("fault_skill_search.json")
        index = read("investigation_index.json")
        actions = index.get("timeline", [])
        rows.append({"instance_id": case.name, "explorer_status": result.get("status"),
            "steps": result.get("steps"), "forced_finish": result.get("forced_finish", False),
            "fault_load_step": read("fault_skill_request.json").get("step"),
            "ranked_functions": result.get("ranked_functions", []),
            "function_metrics": evolution.get("function_metrics", {}),
            "loaded_skill_id": (search.get("matched_skill") or {}).get("skill_id"),
            "tool_counts": dict(Counter(r["tool"] for r in actions)),
            "trace_coverage": {"action_count": len(actions),
                "with_purpose": sum(bool(r.get("purpose")) for r in actions),
                "with_observation_references": sum(bool(r.get("based_on")) for r in actions),
                "with_candidate_updates": sum(bool(r.get("candidate_updates")) for r in actions)},
            "evolution_status": evolution.get("status"), "failure_stage": evolution.get("failure_stage"),
            "investigation_status": conclusion.get("status"), "supplementary_calls": conclusion.get("tool_call_count", 0),
            "findings": conclusion.get("findings", []), "decision": evolution.get("decision"),
            "updated_skill_ids": evolution.get("updated_skill_ids", [])})
    n = len(rows)
    metrics = {key: sum(float(row["function_metrics"].get(source, 0)) for row in rows) / n if n else 0
               for key, source in (("top1", "top1"), ("top3", "top3"), ("top5", "top5"), ("mrr", "reciprocal_rank"))}
    return {"case_count": n, "completed_count": sum(r["explorer_status"] == "completed" for r in rows),
            "evolution_completed_count": sum(r["evolution_status"] == "completed" for r in rows),
            "function_metrics": metrics, "cases": rows,
            "interpretation": "Smoke only: evidence-traceability and protocol acceptance, not an efficacy comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    path = args.run_dir / "v5_acceptance_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "case_count": report["case_count"], "function_metrics": report["function_metrics"]}))
