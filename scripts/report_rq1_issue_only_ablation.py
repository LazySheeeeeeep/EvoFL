"""Pair the issue-only Fault Skill ablation with frozen RQ1 case results."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import run_rq1_expanded as rq


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "runs/rq1_issue_only_fault_skill_ablation_20260929"
MAIN = rq.OUT / "with_skill_failure_retries_20260929/comparison_summary_with_retries.json"


def report(ablation: Path = ABLATION, main: Path = MAIN) -> dict:
    issue = rq.read(ablation / "comparison_summary.json")
    manifest = rq.read(ablation / "selection_manifest.json")
    frozen = rq.read(main)
    original = rq.read(rq.OUT / "evaluation_cases.json")
    expected_ids = [case["instance_id"] for case in original]
    if expected_ids != manifest["case_ids"] or manifest["target"] != 500:
        raise ValueError("Issue-only ablation does not use the frozen 500-case set")
    if manifest["bank_sha256"] != rq.sha(rq.OUT / "frozen_skills.jsonl"):
        raise ValueError("Frozen SkillBank changed")
    main_cases = {case["instance_id"]: case for case in frozen["cases"]}
    if set(main_cases) != set(expected_ids):
        raise ValueError("Main comparison does not cover the same frozen cases")
    issue_cases = issue["cases"]
    ids = [case["instance_id"] for case in issue_cases]
    if len(ids) != len(set(ids)) or any(cid not in main_cases for cid in ids):
        raise ValueError("Issue-only results contain unknown or duplicate cases")
    paired = {}
    for arm in ("issue_only", "with_skill", "no_skill"):
        paired[arm] = {}
        for metric in ("top1", "top3", "top5", "mrr"):
            values = [
                case["metrics"][metric] if arm == "issue_only" else
                main_cases[case["instance_id"]]["arms"][arm]["metrics"][metric]
                for case in issue_cases
            ]
            paired[arm][metric] = sum(values) / len(values) if values else None
    changes = {}
    for baseline in ("with_skill", "no_skill"):
        before = Counter()
        for case in issue_cases:
            old = main_cases[case["instance_id"]]["arms"][baseline]["metrics"]["top1"]
            new = case["metrics"]["top1"]
            before["improved" if new and not old else "regressed" if old and not new else "unchanged"] += 1
        changes[baseline] = dict(before)
    return {
        "processed": len(issue_cases), "target": 500,
        "complete": len(issue_cases) == 500,
        "paired_scores": paired,
        "top1_changes_vs": changes,
        "issue_only_statuses": dict(Counter(case["status"] for case in issue_cases)),
        "issue_only_loaded": sum(bool(case.get("loaded_skill_id")) for case in issue_cases),
        "note": "Incomplete runs show a paired prefix only, not a 500-case conclusion.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=ABLATION)
    parser.add_argument("--main", type=Path, default=MAIN)
    args = parser.parse_args()
    print(json.dumps(report(args.ablation, args.main), indent=2, ensure_ascii=False))
