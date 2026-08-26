"""Paired no-skill vs V3-Skill Explorer comparison for SWE-smith cases.

Run this script inside WSL after selecting a held-out case manifest.  It never
updates the supplied trained SkillBank: both arms pass ``--skip-evolution``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = ("no_issue_skill", "with_issue_skill")
SKILL_TYPES = ("project_skill", "issue_skill", "strategy_skill")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_v3_workflow(Path(args.config))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _read_json(Path(args.selected_cases_file))
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected_cases_file must contain a non-empty JSON list.")
    _require_non_empty_issues(selected)
    _write_json(out_dir / "selected_cases.json", selected)

    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_dir = out_dir / arm
        if not args.report_only:
            _run_arm_with_recovery(args, arm_dir, arm, len(selected))
        arm_summaries[arm] = _read_json(arm_dir / "summary.json")

    report = _build_report(out_dir, selected, arm_summaries)
    _write_json(out_dir / "comparison_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-cases-file", required=True)
    parser.add_argument("--trained-skill-bank", required=True)
    parser.add_argument("--trained-embedding-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/evolutefl.global.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--llm-seed", type=int, default=20260821)
    parser.add_argument(
        "--case-run-attempts",
        type=int,
        default=1,
        help="Maximum whole-arm passes. Later passes resume and retry only failed cases.",
    )
    parser.add_argument("--report-only", action="store_true")
    return parser


def _require_non_empty_issues(selected: list[dict[str, Any]]) -> None:
    invalid = [
        str(case.get("instance_id") or "<missing-instance-id>")
        for case in selected
        if not str(case.get("problem_statement") or "").strip()
    ]
    if invalid:
        raise ValueError(
            "The comparison manifest contains empty problem_statement values; "
            "replace these cases before running either arm: " + ", ".join(invalid)
        )


def _runner_command(args: argparse.Namespace, arm_dir: Path, arm: str, *, resume: bool = False) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_swesmith_case_by_case.py",
        "--selected-cases-file", str(Path(args.selected_cases_file).resolve()),
        "--output-dir", str(arm_dir),
        "--config", args.config,
        "--provider", args.provider,
        "--model", args.model,
        "--max-steps", str(args.max_steps),
        "--case-timeout-seconds", str(args.case_timeout_seconds),
        "--llm-seed", str(args.llm_seed),
        "--skip-evolution",
    ]
    command.extend([
        "--skill-bank-path", str(Path(args.trained_skill_bank).resolve()),
        "--embedding-cache-path", str(Path(args.trained_embedding_cache).resolve()),
        "--enable-embedding", "--retrieval-mode", "embedding",
        "--embedding-base-url", "http://127.0.0.1:8008",
    ])
    if arm == "no_issue_skill":
        command.extend(["--disable-skill-type", "issue_skill"])
    if resume:
        command.append("--resume")
    return command


def _run_arm_with_recovery(args: argparse.Namespace, arm_dir: Path, arm: str, expected_cases: int) -> None:
    attempts = max(1, int(args.case_run_attempts))
    for attempt in range(attempts):
        # A manually restarted comparison must recover already materialized
        # cases as well as retries inside this invocation.  Case-level result
        # artifacts remain authoritative, so resume does not rerun successes.
        resume = attempt > 0 or (arm_dir / "progress_summary.json").exists()
        subprocess.run(_runner_command(args, arm_dir, arm, resume=resume), cwd=ROOT, check=True)
        summary = _read_json(arm_dir / "summary.json")
        completed = sum(
            1 for case in summary.get("cases", []) if case.get("explorer_status") == "completed"
        )
        if completed == expected_cases:
            return
    raise RuntimeError(
        f"{arm} completed only {completed}/{expected_cases} cases after {attempts} run attempts."
    )


def _require_v3_workflow(config_path: Path) -> None:
    """Reject legacy configs before an experiment silently skips Issue Skill."""
    path = config_path if config_path.is_absolute() else ROOT / config_path
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read comparison config: {path}") from exc
    workflow = str((config.get("explorer") or {}).get("workflow_version") or "").lower()
    if workflow != "v3":
        raise ValueError(
            "run_swesmith_v3_compare requires explorer.workflow_version='v3'; "
            f"got {workflow or 'unset'} from {path}."
        )


def _build_report(
    out_dir: Path,
    selected: list[dict[str, Any]],
    arm_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arm_metrics = {
        arm: _arm_metrics(out_dir / arm, summary)
        for arm, summary in arm_summaries.items()
    }
    baseline = arm_metrics["no_issue_skill"]
    full = arm_metrics["with_issue_skill"]
    paired = _paired_rank_changes(
        arm_summaries["no_issue_skill"].get("cases", []),
        arm_summaries["with_issue_skill"].get("cases", []),
        baseline_dir=out_dir / "no_issue_skill",
        with_issue_dir=out_dir / "with_issue_skill",
    )
    return {
        "selected_instance_ids": [str(case.get("instance_id")) for case in selected],
        "same_case_manifest": _case_ids(arm_summaries["no_issue_skill"]) == _case_ids(arm_summaries["with_issue_skill"]),
        "arms": arm_metrics,
        "delta_with_issue_skill_minus_no_issue_skill": {
            metric: full["function_metrics"].get(metric, 0.0) - baseline["function_metrics"].get(metric, 0.0)
            for metric in ("top1", "top3", "top5", "mrr")
        } | {
            "completed_count": full["completed_count"] - baseline["completed_count"],
            "forced_finish_count": full["forced_finish_count"] - baseline["forced_finish_count"],
        },
        "tool_call_delta_with_issue_skill_minus_no_issue_skill": _counter_delta(
            baseline["tool_calls"], full["tool_calls"],
        ),
        "issue_skill_diagnostics": {
            "no_issue_skill": baseline["issue_skill_diagnostics"],
            "with_issue_skill": full["issue_skill_diagnostics"],
        },
        "candidate_and_rank_changes": paired,
    }


def _arm_metrics(arm_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    report = summary.get("report") or {}
    cases = summary.get("cases") or []
    stage = {kind: Counter() for kind in SKILL_TYPES}
    tool_calls: Counter[str] = Counter()
    issue_rows: list[dict[str, Any]] = []
    forced = 0
    for case in cases:
        case_dir = arm_dir / "cases" / str(case.get("instance_id") or "")
        events = _read_jsonl(case_dir / "trajectory.jsonl")
        if any(event.get("event") in {"forced_finish", "deterministic_finish"} for event in events):
            forced += 1
        for event in events:
            if event.get("event") == "assistant_tool_calls":
                for call in event.get("tool_calls") or []:
                    tool_calls[str((call.get("function") or {}).get("name") or "unknown")] += 1
            for kind in SKILL_TYPES:
                if event.get("event") == f"{kind}_request":
                    stage[kind]["attempted"] += 1
                elif event.get("event") == f"{kind}_loaded":
                    stage[kind]["loaded"] += int(bool(event.get("loaded_skill_id")))
                    trace = event.get("search_trace") or {}
                    if trace.get("candidate_selected_skill_ids") or trace.get("candidate_skill_ids"):
                        stage[kind]["recalled"] += 1
                    selector = trace.get("selector") or {}
                    selected = selector.get("selected_skill_id") not in {None, "", "none", "null"}
                    if selected:
                        stage[kind]["selected"] += 1
                    validator = trace.get("validator") or {}
                    if validator.get("applicable") or validator.get("approved"):
                        stage[kind]["validated"] += 1
        issue_rows.append(_issue_diagnostic_row(str(case.get("instance_id") or ""), events))
    return {
        "completed_count": int(report.get("completed_count") or 0),
        "forced_finish_count": forced,
        "function_metrics": report.get("function_metrics") or {},
        "skill_stages": {kind: dict(counts) for kind, counts in stage.items()},
        "issue_skill_diagnostics": {
            "counts": dict(stage["issue_skill"]),
            "primary_blocker": _primary_issue_blocker(issue_rows),
            "per_case": issue_rows,
        },
        "tool_calls": dict(tool_calls),
        "active_skill_count": report.get("active_skill_count"),
    }


def _paired_rank_changes(
    baseline_cases: list[dict[str, Any]],
    full_cases: list[dict[str, Any]],
    *,
    baseline_dir: Path,
    with_issue_dir: Path,
) -> list[dict[str, Any]]:
    baseline = {str(case.get("instance_id")): case for case in baseline_cases}
    rows: list[dict[str, Any]] = []
    for case in full_cases:
        instance_id = str(case.get("instance_id"))
        before = baseline.get(instance_id, {})
        before_ranked = before.get("ranked_functions") or []
        after_ranked = case.get("ranked_functions") or []
        before_metrics = before.get("function_metrics") or {}
        after_metrics = case.get("function_metrics") or {}
        before_trace = _read_jsonl(baseline_dir / "cases" / instance_id / "trajectory.jsonl")
        after_trace = _read_jsonl(with_issue_dir / "cases" / instance_id / "trajectory.jsonl")
        rows.append({
            "instance_id": instance_id,
            "ranked_functions_changed": before_ranked != after_ranked,
            "baseline_ranked_functions": before_ranked,
            "with_issue_skill_ranked_functions": after_ranked,
            "issue_skill_loaded_id": _loaded_skill_id(after_trace, "issue_skill"),
            "baseline_tool_calls": _tool_calls_for_events(before_trace),
            "with_issue_skill_tool_calls": _tool_calls_for_events(after_trace),
            "tool_call_delta_with_issue_skill_minus_baseline": _counter_delta(
                _tool_calls_for_events(before_trace), _tool_calls_for_events(after_trace),
            ),
            "top1_effect": _metric_effect(before_metrics, after_metrics, "top1"),
            "top3_effect": _metric_effect(before_metrics, after_metrics, "top3"),
            "top5_effect": _top5_effect(before_metrics, after_metrics),
            "mrr_delta": float(after_metrics.get("reciprocal_rank") or 0.0)
            - float(before_metrics.get("reciprocal_rank") or 0.0),
        })
    return rows


def _issue_diagnostic_row(instance_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    request = next((event for event in events if event.get("event") == "issue_skill_request"), {})
    loaded = next((event for event in events if event.get("event") == "issue_skill_loaded"), {})
    trace = loaded.get("search_trace") or {}
    selector = trace.get("selector") or {}
    validator = trace.get("validator") or {}
    recalled_ids = list(trace.get("candidate_selected_skill_ids") or trace.get("candidate_skill_ids") or [])
    return {
        "instance_id": instance_id,
        "attempted": bool(request),
        "recalled": bool(recalled_ids),
        "candidate_skill_ids": recalled_ids,
        "candidate_scores": trace.get("scores") or {},
        "selector_selected_skill_id": selector.get("selected_skill_id"),
        "selector_reason": selector.get("reason"),
        "validator_applicable": bool(validator.get("applicable") or validator.get("approved")),
        "validator_reason": validator.get("reason"),
        "loaded_skill_id": loaded.get("loaded_skill_id"),
        "notes": trace.get("notes") or [],
    }


def _primary_issue_blocker(rows: list[dict[str, Any]]) -> str:
    attempted = [row for row in rows if row["attempted"]]
    if not attempted:
        return "Issue stage was never attempted."
    if all("Issue Skill loading disabled." in row.get("notes", []) for row in attempted):
        return "Issue Skill loading was intentionally disabled for this baseline arm."
    if not any(row["recalled"] for row in attempted):
        return "Embedding recall produced no threshold-qualified Issue Skill candidates."
    if not any(row.get("selector_selected_skill_id") for row in attempted):
        return "The selector rejected all recalled Issue Skill candidates."
    if not any(row["validator_applicable"] for row in attempted):
        return "The validator rejected every selector-approved Issue Skill candidate."
    if not any(row.get("loaded_skill_id") for row in attempted):
        return "No Issue Skill was injected after validation."
    return "At least one Issue Skill was injected; inspect paired rank effects for usefulness."


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def _top5_effect(before: dict[str, Any], after: dict[str, Any]) -> str:
    return _metric_effect(before, after, "top5")


def _metric_effect(before: dict[str, Any], after: dict[str, Any], metric: str) -> str:
    before_hit, after_hit = bool(before.get(metric)), bool(after.get(metric))
    if not before_hit and after_hit:
        return "positive"
    if before_hit and not after_hit:
        return "negative"
    return "no_effect"


def _tool_calls_for_events(events: list[dict[str, Any]]) -> dict[str, int]:
    calls: Counter[str] = Counter()
    for event in events:
        if event.get("event") != "assistant_tool_calls":
            continue
        for call in event.get("tool_calls") or []:
            calls[str((call.get("function") or {}).get("name") or "unknown")] += 1
    return dict(calls)


def _loaded_skill_id(events: list[dict[str, Any]], skill_type: str) -> str | None:
    event = next(
        (
            item
            for item in events
            if item.get("event") == f"{skill_type}_loaded" and item.get("loaded_skill_id")
        ),
        None,
    )
    return str(event.get("loaded_skill_id")) if event else None


def _case_ids(summary: dict[str, Any]) -> list[str]:
    return [str(case.get("instance_id")) for case in summary.get("cases") or []]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
