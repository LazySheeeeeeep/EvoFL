from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolutefl.config import llm_config, load_config  # noqa: E402
from evolutefl.issue_abstraction import abstract_issue  # noqa: E402
from evolutefl.json_utils import write_json  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402
SKILL_TYPES = ("project_skill", "strategy_skill")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    config.setdefault("embedding", {})
    config["embedding"]["enabled"] = True
    config["embedding"]["base_url"] = args.embedding_base_url
    config.setdefault("skill_bank", {})
    config["skill_bank"]["retrieval_mode"] = "embedding"
    config["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    config["skill_bank"]["embedding_fallback_to_lexical"] = False

    if not args.reuse_existing_abstractions:
        check_llm_config(config["llm"], args.provider)
    if args.reuse_existing_abstractions:
        selected = json.loads((out_dir / "selected_cases.json").read_text(encoding="utf-8-sig"))
    else:
        excluded_ids: set[str] = set()
        for excluded_path in args.exclude_selected_cases_file:
            excluded_rows = json.loads(Path(excluded_path).read_text(encoding="utf-8-sig"))
            excluded_ids.update(
                str(item.get("instance_id")) for item in excluded_rows if item.get("instance_id")
            )
        if args.exclude_all_runs_selected_cases:
            excluded_ids.update(load_run_selected_case_ids(ROOT / "runs", exclude_dir=out_dir))
        selected = select_unseen_cases(args, excluded_ids)
        write_json(out_dir / "selected_cases.json", selected)
        write_json(
            out_dir / "selection_summary.json",
            {"excluded_instance_count": len(excluded_ids), "selected_instance_ids": [case["instance_id"] for case in selected]},
        )
    previous_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and (out_dir / "progress_summary.json").exists():
        previous = json.loads((out_dir / "progress_summary.json").read_text(encoding="utf-8"))
        previous_by_id = {
            item["instance_id"]: item
            for item in previous.get("cases", [])
            if isinstance(item, dict) and item.get("instance_id")
        }

    client = None if args.reuse_existing_abstractions else OpenAICompatibleClient.from_config(config["llm"])
    skill_bank = make_skill_bank(config)
    abstraction_config = config.get("issue_abstraction", {})
    prompt_path = abstraction_config.get(
        "prompt_path", "prompt_records/explorer/issue_abstraction_v0.txt"
    )
    prompt = (ROOT / prompt_path).read_text(encoding="utf-8")
    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        if args.resume and case["instance_id"] in previous_by_id:
            case_results.append(previous_by_id[case["instance_id"]])
            continue
        case_dir = out_dir / "cases" / case["instance_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.reuse_existing_abstractions:
                abstraction_payload = json.loads(
                    (case_dir / "issue_abstraction.json").read_text(encoding="utf-8-sig")
                )
                abstraction = abstraction_payload.get("issue_abstraction", abstraction_payload)
            else:
                abstraction = abstract_issue(
                    repo=case["repo"],
                    issue=case["problem_statement"],
                    llm_client=client,
                    prompt=prompt,
                    attempts=int(abstraction_config.get("attempts", 2)),
                    output_path=case_dir / "issue_abstraction.json",
                )
            retrieval = skill_bank.search_for_explorer(
                case["repo"],
                case["problem_statement"],
                issue_abstraction=abstraction,
            )
            write_json(case_dir / "skill_search.json", retrieval)
            matched = retrieval.get("matched_skills", [])
            error = None
        except Exception as exc:  # Keep a single provider failure from aborting the probe.
            abstraction = {}
            retrieval = {}
            matched = []
            error = str(exc)
        by_skill_type = {skill_type: 0 for skill_type in SKILL_TYPES}
        for skill in matched:
            if skill.get("skill_type") in by_skill_type:
                by_skill_type[skill["skill_type"]] += 1
        search_trace = retrieval.get("skill_search_trace", {})
        skill_type_diagnostics = build_skill_type_diagnostics(search_trace, matched)
        selected_score_by_id = {
            skill_id: score
            for diagnostics in skill_type_diagnostics.values()
            for skill_id, score in zip(
                diagnostics.get("selected_skill_ids", []),
                diagnostics.get("selected_scores", []),
            )
        }
        matched_details = [
            {
                "skill_id": skill.get("skill_id"),
                "skill_type": skill.get("skill_type"),
                "value": skill.get("value"),
                "title": skill.get("title"),
                "trigger": skill.get("trigger"),
                "retrieval_text": skill.get("retrieval_text"),
                "knowledge": skill.get("knowledge"),
                "score": (
                    skill.get("score")
                    if skill.get("score") is not None
                    else selected_score_by_id.get(skill.get("skill_id"))
                ),
            }
            for skill in matched
        ]
        result = {
            "index": index,
            "instance_id": case["instance_id"],
            "repo": case["repo"],
            "problem_statement": case["problem_statement"],
            "matched_skill_count": len(matched),
            "matched_skill_ids": [skill.get("skill_id") for skill in matched],
            "matched_skill_types": by_skill_type,
            "matched_skill_details": matched_details,
            "skill_type_diagnostics": skill_type_diagnostics,
            "issue_abstraction": abstraction,
            "skill_search_trace": search_trace,
            "error": error,
        }
        write_json(case_dir / "result.json", result)
        case_results.append(result)
        write_json(out_dir / "progress_summary.json", {"cases": case_results})
        print(json.dumps({"index": index, **{k: result[k] for k in ("instance_id", "matched_skill_count", "matched_skill_types")}}, ensure_ascii=False))

    summary = summarize(case_results, args, skill_bank)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manual_relevance_review_queue.json", build_manual_review_queue(case_results))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize(results: list[dict[str, Any]], args: argparse.Namespace, skill_bank: Any) -> dict[str, Any]:
    total = len(results)
    top_scores = {
        skill_type: [
            item["skill_type_diagnostics"][skill_type].get("top_score")
            for item in results
            if item.get("skill_type_diagnostics", {}).get(skill_type, {}).get("top_score") is not None
        ]
        for skill_type in SKILL_TYPES
    }
    selected_scores = {
        skill_type: [
            score
            for item in results
            for score in item.get("skill_type_diagnostics", {}).get(skill_type, {}).get("selected_scores", [])
        ]
        for skill_type in SKILL_TYPES
    }
    return {
        "output_dir": str((ROOT / args.output_dir).resolve()),
        "dataset": args.swe_explore_dataset,
        "split": args.swe_explore_split,
        "sample_size": total,
        "seed": args.seed,
        "provider": args.provider,
        "model": args.model,
        "retrieval_mode": "embedding",
        "embedding_min_score": args.embedding_min_score,
        "skill_bank_active_count": len(skill_bank.active_skills()),
        "retrieval_hit_count": sum(bool(x["matched_skill_count"]) for x in results),
        "retrieval_hit_rate": sum(bool(x["matched_skill_count"]) for x in results) / total if total else 0.0,
        "skill_type_hit_count": {
            skill_type: sum(x["matched_skill_types"].get(skill_type, 0) > 0 for x in results)
            for skill_type in SKILL_TYPES
        },
        "skill_type_hit_rate": {
            skill_type: sum(x["matched_skill_types"].get(skill_type, 0) > 0 for x in results) / total if total else 0.0
            for skill_type in SKILL_TYPES
        },
        "matched_skill_count_distribution": dict(Counter(x["matched_skill_count"] for x in results)),
        "similarity_distribution": {
            skill_type: {
                "top_1_candidate": distribution(top_scores[skill_type]),
                "selected": distribution(selected_scores[skill_type]),
            }
            for skill_type in SKILL_TYPES
        },
        "abstractor_success_count": sum(not item.get("error") for item in results),
        "abstractor_failure_count": sum(bool(item.get("error")) for item in results),
        "cases": results,
    }


def select_unseen_cases(args: argparse.Namespace, excluded_ids: set[str]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required. Run this script in WSL.") from exc

    explore_rows = list(load_dataset(args.swe_explore_dataset, split=args.swe_explore_split))
    verified_rows = list(load_dataset(args.swe_verified_dataset, split=args.swe_verified_split))
    verified_by_id = {str(row["instance_id"]): row for row in verified_rows}
    candidates = [
        row
        for row in explore_rows
        if (not args.dataset_filter or row.get("dataset") == args.dataset_filter)
        and str(row.get("instance_id")) in verified_by_id
        and str(row.get("instance_id")) not in excluded_ids
        and (row.get("ground_truth") or {}).get("read_core_regions")
    ]
    random.Random(args.seed).shuffle(candidates)
    selected = []
    for row in candidates[: args.sample_size]:
        verified = verified_by_id[str(row["instance_id"])]
        selected.append(
            {
                "instance_id": str(row["instance_id"]),
                "dataset": str(row.get("dataset") or ""),
                "repo": str(verified.get("repo") or ""),
                "base_commit": str(verified.get("base_commit") or ""),
                "problem_statement": str(verified.get("problem_statement") or ""),
                "patch": str(verified.get("patch") or ""),
            }
        )
    if len(selected) < args.sample_size:
        raise RuntimeError(f"Only found {len(selected)} unseen SWE-Explore cases after applying exclusions.")
    return selected


def load_run_selected_case_ids(runs_root: Path, *, exclude_dir: Path | None = None) -> set[str]:
    ids: set[str] = set()
    for pattern in ("*/selected_cases.json", "*/*/selected_cases.json"):
        for path in runs_root.glob(pattern):
            if exclude_dir is not None:
                try:
                    path.resolve().relative_to(exclude_dir.resolve())
                    continue
                except ValueError:
                    pass
            try:
                rows = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(rows, list):
                ids.update(str(item.get("instance_id")) for item in rows if isinstance(item, dict) and item.get("instance_id"))
    return ids


def build_skill_type_diagnostics(trace: dict[str, Any], matched: list[dict[str, Any]]) -> dict[str, Any]:
    traces = trace.get("skill_type_traces", {}) or {}
    diagnostics: dict[str, Any] = {}
    for skill_type in SKILL_TYPES:
        type_trace = traces.get(skill_type, {}) or {}
        top_scores = type_trace.get("top_scores", []) or []
        selected_ids = set(type_trace.get("selected_skill_ids", []) or [])
        scores = type_trace.get("scores", {}) or {}
        selected = [skill for skill in matched if skill.get("skill_type") == skill_type]
        diagnostics[skill_type] = {
            "top_score": top_scores[0].get("score") if top_scores else None,
            "top_skill_id": top_scores[0].get("skill_id") if top_scores else None,
            "selected_skill_ids": [skill.get("skill_id") for skill in selected],
            "selected_scores": [
                float(scores[skill_id])
                for skill_id in selected_ids
                if skill_id in scores
            ],
            "threshold": type_trace.get("notes", []),
        }
    return diagnostics


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}

    def percentile(ratio: float) -> float:
        index = (len(ordered) - 1) * ratio
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def build_manual_review_queue(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for result in results:
        for skill in result.get("matched_skill_details", []):
            queue.append(
                {
                    "instance_id": result["instance_id"],
                    "repo": result["repo"],
                    "problem_statement": result["problem_statement"],
                    "issue_abstraction": result.get("issue_abstraction", {}),
                    "skill": skill,
                    "manual_label": "pending",
                    "manual_reason": "",
                }
            )
    return queue


def check_llm_config(llm: dict[str, Any], provider: str) -> None:
    if not llm.get("base_url"):
        raise RuntimeError(f"Provider {provider!r} has no base_url.")
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/swe_explore_abstractor_skill_probe_10_gpt5mini_jina")
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--swe-explore-dataset", default="SWE-Explore-Bench/SWE-Explore-Bench")
    parser.add_argument("--swe-verified-dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--swe-explore-split", default="train")
    parser.add_argument("--swe-verified-split", default="test")
    parser.add_argument("--dataset-filter", default="verified")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-selected-cases-file", action="append", default=[])
    parser.add_argument("--exclude-all-runs-selected-cases", action="store_true")
    parser.add_argument("--reuse-existing-abstractions", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
