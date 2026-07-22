from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import llm_config, load_config  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.reflection import run_case_evolution  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    config["llm"]["max_tokens"] = args.max_tokens
    configure_embedding(config, args)

    ground_truth = load_ground_truth_index(Path(args.runs_root))
    cases, scan = discover_cases(Path(args.runs_root), output_dir, ground_truth)
    if args.source_path_contains:
        marker = args.source_path_contains.lower()
        cases = [case for case in cases if marker in case["case_run_dir"].lower()]
        scan["source_path_filter"] = args.source_path_contains
        scan["source_filtered_case_count"] = len(cases)
    if args.require_ground_truth:
        cases = [case for case in cases if case["ground_truth_functions"]]
        scan["ground_truth_filtered_case_count"] = len(cases)
    excluded_instance_ids: set[str] = set()
    for selected_cases_file in args.exclude_selected_cases_file:
        payload = load_json(Path(selected_cases_file), default=[])
        entries = payload
        if isinstance(payload, dict):
            entries = payload.get("selected_cases") or payload.get("cases") or []
        if not isinstance(entries, list):
            continue
        excluded_instance_ids.update(
            str(item.get("instance_id"))
            for item in entries
            if isinstance(item, dict) and item.get("instance_id")
        )
    if excluded_instance_ids:
        before_exclusion = len(cases)
        cases = [case for case in cases if case["instance_id"] not in excluded_instance_ids]
        scan["excluded_instance_id_count"] = len(excluded_instance_ids)
        scan["excluded_case_count"] = before_exclusion - len(cases)
    if args.seed is not None:
        random.Random(args.seed).shuffle(cases)
        scan["sample_seed"] = args.seed
    if args.max_cases:
        cases = cases[: args.max_cases]
    write_json(output_dir / "selected_cases.json", cases)
    write_json(output_dir / "scan_summary.json", {**scan, "selected_count": len(cases)})
    if args.scan_only:
        print(json.dumps({**scan, "selected_count": len(cases)}, ensure_ascii=False, indent=2))
        return 0

    check_llm_config(config["llm"])
    client = OpenAICompatibleClient.from_config(config["llm"])
    progress_path = output_dir / "progress_summary.json"
    previous = load_json(progress_path, default={}) if args.resume else {}
    previous_by_id = {
        item["instance_id"]: item
        for item in previous.get("cases", [])
        if isinstance(item, dict) and item.get("instance_id")
    }
    summaries: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        instance_id = case["instance_id"]
        old = previous_by_id.get(instance_id)
        current_gt_count = len(case["ground_truth_functions"])
        old_gt_count = int((old or {}).get("ground_truth_function_count") or 0)
        previous_protocol_failure = reflector_protocol_failure(
            output_dir / "cases" / instance_id / "reflector_output.json"
        )
        if (
            old
            and old.get("status") == "completed"
            and not previous_protocol_failure
            and (old_gt_count > 0 or current_gt_count == 0)
        ):
            summaries.append(old)
            write_progress(progress_path, cases, summaries, config, scan)
            continue

        case_output = output_dir / "cases" / instance_id
        summary: dict[str, Any] = {
            "index": index,
            "instance_id": instance_id,
            "source_case_dir": case["case_run_dir"],
            "status": "running",
            "ground_truth_function_count": len(case["ground_truth_functions"]),
        }
        try:
            evolution = run_case_evolution(
                case_run_dir=case["case_run_dir"],
                repo=case["repo"],
                issue=case["problem_statement"],
                config=config,
                llm_client=client,
                force=True,
                ground_truth_patch=case["patch"],
                ground_truth_functions=case["ground_truth_functions"],
                ground_truth_locations=case["ground_truth_locations"],
                output_dir=case_output,
                reflect_success=True,
                reflect_failure=True,
                legacy_insight=False,
                refresh_issue_abstraction=True,
            )
            protocol_failure = reflector_protocol_failure(evolution)
            if protocol_failure:
                raise RuntimeError(protocol_failure)
            decisions = {
                skill_type: update.get("decision")
                for skill_type, update in (evolution.get("skill_updates") or {}).items()
                if isinstance(update, dict)
            }
            summary.update(
                {
                    "status": "completed",
                    "outcome_type": evolution.get("outcome_type"),
                    "top5_hit": (evolution.get("outcome") or {}).get("top5_hit"),
                    "decisions": decisions,
                    "updated_skill_ids": evolution.get("updated_skill_ids", []),
                    "updated_skill_types": evolution.get("updated_skill_types", []),
                    "failed_updates": evolution.get("failed_updates", []),
                }
            )
            if args.rebuild_embeddings_after_case:
                summary["embedding_rebuild"] = make_skill_bank(config).rebuild_embeddings()
        except Exception as exc:  # noqa: BLE001 - isolate each historical case.
            summary.update({"status": "failed", "error": str(exc)})

        summaries.append(summary)
        write_progress(progress_path, cases, summaries, config, scan)

    final = build_summary(cases, summaries, config, scan)
    write_json(output_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def reflector_protocol_failure(value: Any) -> str:
    if isinstance(value, Path):
        value = load_json(value, default={})
    if not isinstance(value, (dict, list)):
        return ""
    serialized = json.dumps(value, ensure_ascii=False)
    marker = "Reflector protocol failed:"
    if marker not in serialized:
        return ""
    match = re.search(r"Reflector protocol failed:[^\"\\n]*", serialized)
    return match.group(0).strip() if match else "Reflector protocol failed."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay existing Explorer trajectories through Abstractor and Reflector.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-path-contains", default="")
    parser.add_argument("--require-ground-truth", action="store_true")
    parser.add_argument(
        "--exclude-selected-cases-file",
        action="append",
        default=[],
        help="Exclude instance IDs listed in a previous selected_cases.json file.",
    )
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-embedding", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("lexical", "embedding", "hybrid"), default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument("--rebuild-embeddings-after-case", action="store_true")
    return parser


def configure_embedding(config: dict[str, Any], args: argparse.Namespace) -> None:
    embedding = config.setdefault("embedding", {})
    skill_bank = config.setdefault("skill_bank", {})
    embedding["enabled"] = bool(args.enable_embedding)
    embedding["base_url"] = args.embedding_base_url
    embedding.setdefault("endpoint", "/embed")
    embedding.setdefault("model", "jina-embeddings-v3")
    embedding.setdefault("query_task", "retrieval.query")
    embedding.setdefault("document_task", "retrieval.passage")
    embedding.setdefault("cache_path", "skill_pools/skill_bank_v0/embeddings/jina_v3/skill_embeddings.jsonl")
    embedding.setdefault("batch_size", 16)
    embedding.setdefault("timeout", 120)
    skill_bank["path"] = "skill_pools/skill_bank_v0/skills.jsonl"
    skill_bank["retrieval_mode"] = args.retrieval_mode
    skill_bank["embedding_min_score"] = args.embedding_min_score
    skill_bank["embedding_fallback_to_lexical"] = False
    skill_bank["max_matched_skills"] = 2
    skill_bank["max_per_skill_type"] = {"project_skill": 1, "strategy_skill": 1}


def check_llm_config(llm: dict[str, Any]) -> None:
    if not llm.get("base_url"):
        raise RuntimeError("LLM base_url is missing.")
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Required environment variable {api_key_env} is missing.")


def discover_cases(
    runs_root: Path,
    output_dir: Path,
    ground_truth: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories: set[Path] = set()
    patterns = (
        "*/cases/*/trajectory.jsonl",
        "*/*/cases/*/trajectory.jsonl",
        "*/cases/*/explorer/trajectory.jsonl",
        "*/*/cases/*/explorer/trajectory.jsonl",
    )
    for pattern in patterns:
        trajectories.update(path.resolve() for path in runs_root.glob(pattern))

    candidates: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, int] = {}
    for trajectory_path in sorted(trajectories):
        case_dir = trajectory_path.parent
        if not (case_dir / "result.json").exists() and (case_dir.parent / "result.json").exists():
            case_dir = case_dir.parent
        try:
            case_dir.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass
        result = load_json(case_dir / "result.json", default=None)
        task = load_json(case_dir / "task.json", default=None)
        if not isinstance(result, dict):
            rejected["missing_or_invalid_result"] = rejected.get("missing_or_invalid_result", 0) + 1
            continue
        if result.get("status") != "completed":
            rejected["not_completed"] = rejected.get("not_completed", 0) + 1
            continue
        if not isinstance(task, dict):
            rejected["missing_or_invalid_task"] = rejected.get("missing_or_invalid_task", 0) + 1
            continue
        issue = str(task.get("problem_statement") or task.get("bug_report") or task.get("issue") or "").strip()
        patch = str(task.get("patch") or "")
        if not issue:
            rejected["empty_issue"] = rejected.get("empty_issue", 0) + 1
            continue
        if not patch:
            rejected["empty_patch"] = rejected.get("empty_patch", 0) + 1
            continue
        instance_id = str(result.get("instance_id") or task.get("instance_id") or case_dir.name)
        gt = ground_truth.get(instance_id, {})
        task_ground_truth = list(task.get("ground_truth_functions") or [])
        inferred_ground_truth = infer_ground_truth_functions_from_patch(patch)
        ground_truth_functions = list(gt.get("functions") or task_ground_truth or inferred_ground_truth)
        candidate = {
            "instance_id": instance_id,
            "repo": str(task.get("repo") or ""),
            "problem_statement": issue,
            "patch": patch,
            "case_run_dir": str(case_dir),
            "trajectory_path": str(trajectory_path),
            "trajectory_bytes": trajectory_path.stat().st_size,
            "result_ranked_count": len(result.get("ranked_functions") or []),
            "ground_truth_functions": ground_truth_functions,
            "ground_truth_locations": list(gt.get("locations") or task.get("ground_truth_locations") or []),
            "ground_truth_source": (
                "historical_evaluation"
                if gt.get("functions")
                else "task"
                if task_ground_truth
                else "patch_hunk"
                if inferred_ground_truth
                else "unavailable"
            ),
        }
        candidates.setdefault(instance_id, []).append(candidate)

    selected = [max(items, key=_candidate_quality) for _, items in sorted(candidates.items())]
    return selected, {
        "trajectory_file_count": len(trajectories),
        "unique_eligible_case_count": len(selected),
        "duplicate_trajectory_count": sum(max(0, len(items) - 1) for items in candidates.values()),
        "ground_truth_recovered_count": sum(bool(item["ground_truth_functions"]) for item in selected),
        "rejected": rejected,
    }


def _candidate_quality(case: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(case["ground_truth_functions"])),
        int(case["result_ranked_count"]),
        int(case["trajectory_bytes"]),
    )


DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
HUNK_HEADER_RE = re.compile(r"^@@[^@]*@@\s*(.*)$")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)")
FUNCTION_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")


def infer_ground_truth_functions_from_patch(patch: str) -> list[str]:
    found: list[str] = []
    file_path = ""
    current_class: str | None = None
    current_function: str | None = None
    in_hunk = False

    for raw_line in patch.splitlines():
        diff_match = DIFF_HEADER_RE.match(raw_line)
        if diff_match:
            file_path = diff_match.group(2)
            current_class = None
            current_function = None
            in_hunk = False
            continue

        hunk_match = HUNK_HEADER_RE.match(raw_line)
        if hunk_match:
            in_hunk = True
            context = hunk_match.group(1)
            class_match = CLASS_RE.match(context)
            function_match = FUNCTION_RE.match(context)
            if class_match:
                current_class = class_match.group(1)
                current_function = None
            if function_match:
                current_function = function_match.group(1)
            continue

        if not in_hunk or not file_path.endswith(".py") or not raw_line:
            continue
        prefix = raw_line[0]
        if prefix not in {" ", "+", "-"} or raw_line.startswith(("+++", "---")):
            continue
        source_line = raw_line[1:]
        class_match = CLASS_RE.match(source_line)
        function_match = FUNCTION_RE.match(source_line)
        if class_match:
            current_class = class_match.group(1)
            current_function = None
        if function_match:
            current_function = function_match.group(1)
        if prefix in {"+", "-"} and current_function:
            qualified = f"{current_class}.{current_function}" if current_class else current_function
            found.append(f"{file_path}::{qualified}")

    return list(dict.fromkeys(found))


def load_ground_truth_index(runs_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    paths: set[Path] = set()
    for pattern in ("*/*evaluation*.json", "*/*summary*.json", "*/*/*evaluation*.json", "*/*/*summary*.json"):
        paths.update(runs_root.glob(pattern))
    for path in sorted(paths):
        payload = load_json(path, default=None)
        _collect_ground_truth(payload, index)
    return index


def _collect_ground_truth(value: Any, index: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        instance_id = value.get("instance_id")
        functions = value.get("ground_truth_functions")
        locations = value.get("ground_truth_locations")
        ground_truth = value.get("ground_truth")
        if isinstance(ground_truth, dict):
            functions = functions or ground_truth.get("functions")
            locations = locations or ground_truth.get("locations")
        if instance_id and isinstance(functions, list) and functions:
            current = index.setdefault(str(instance_id), {"functions": [], "locations": []})
            current["functions"] = list(dict.fromkeys([*current["functions"], *map(str, functions)]))
            if isinstance(locations, list):
                current["locations"] = locations
        for child in value.values():
            _collect_ground_truth(child, index)
    elif isinstance(value, list):
        for child in value:
            _collect_ground_truth(child, index)


def write_progress(
    path: Path,
    selected: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    scan: dict[str, Any],
) -> None:
    write_json(path, build_summary(selected, summaries, config, scan))


def build_summary(
    selected: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    scan: dict[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    for item in summaries:
        for skill_type, decision in (item.get("decisions") or {}).items():
            key = f"{skill_type}:{decision}"
            decisions[key] = decisions.get(key, 0) + 1
    return {
        "status": "completed" if len(summaries) == len(selected) else "running",
        "selected_count": len(selected),
        "processed_count": len(summaries),
        "completed_count": sum(item.get("status") == "completed" for item in summaries),
        "failed_count": sum(item.get("status") == "failed" for item in summaries),
        "updated_skill_count": sum(len(item.get("updated_skill_ids") or []) for item in summaries),
        "decision_counts": decisions,
        "provider": config.get("llm", {}).get("base_url"),
        "model": config.get("llm", {}).get("model"),
        "scan": scan,
        "cases": summaries,
    }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
