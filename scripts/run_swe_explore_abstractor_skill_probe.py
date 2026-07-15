from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
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
from run_swe_explore_explorer_compare import select_cases  # noqa: E402


DIMENSIONS = ("general", "project_type", "fault_mode", "strategy_type")


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

    check_llm_config(config["llm"], args.provider)
    excluded_ids: set[str] = set()
    for excluded_path in args.exclude_selected_cases_file:
        excluded_rows = json.loads(Path(excluded_path).read_text(encoding="utf-8-sig"))
        excluded_ids.update(
            str(item.get("instance_id")) for item in excluded_rows if item.get("instance_id")
        )
    selection_size = args.sample_size + len(excluded_ids)
    selected = select_cases(
        SimpleNamespace(
            hf_endpoint=args.hf_endpoint,
            swe_explore_dataset=args.swe_explore_dataset,
            swe_verified_dataset=args.swe_verified_dataset,
            swe_explore_split=args.swe_explore_split,
            swe_verified_split=args.swe_verified_split,
            dataset_filter=args.dataset_filter,
            sample_size=selection_size,
            seed=args.seed,
        )
    )
    selected = [case for case in selected if case["instance_id"] not in excluded_ids][: args.sample_size]
    if len(selected) < args.sample_size:
        raise RuntimeError("Not enough non-overlapping SWE-Explore cases after applying exclusions.")
    write_json(out_dir / "selected_cases.json", selected)
    previous_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and (out_dir / "progress_summary.json").exists():
        previous = json.loads((out_dir / "progress_summary.json").read_text(encoding="utf-8"))
        previous_by_id = {
            item["instance_id"]: item
            for item in previous.get("cases", [])
            if isinstance(item, dict) and item.get("instance_id")
        }

    client = OpenAICompatibleClient.from_config(config["llm"])
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
        by_dimension = {dimension: 0 for dimension in DIMENSIONS}
        for skill in matched:
            if skill.get("dimension") in by_dimension:
                by_dimension[skill["dimension"]] += 1
        result = {
            "index": index,
            "instance_id": case["instance_id"],
            "repo": case["repo"],
            "matched_skill_count": len(matched),
            "matched_skill_ids": [skill.get("skill_id") for skill in matched],
            "matched_dimensions": by_dimension,
            "issue_abstraction": abstraction,
            "skill_search_trace": retrieval.get("skill_search_trace", {}),
            "error": error,
        }
        write_json(case_dir / "result.json", result)
        case_results.append(result)
        write_json(out_dir / "progress_summary.json", {"cases": case_results})
        print(json.dumps({"index": index, **{k: result[k] for k in ("instance_id", "matched_skill_count", "matched_dimensions")}}, ensure_ascii=False))

    summary = summarize(case_results, args, skill_bank)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize(results: list[dict[str, Any]], args: argparse.Namespace, skill_bank: Any) -> dict[str, Any]:
    total = len(results)
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
        "dimension_hit_count": {
            dimension: sum(x["matched_dimensions"].get(dimension, 0) > 0 for x in results)
            for dimension in DIMENSIONS
        },
        "dimension_hit_rate": {
            dimension: sum(x["matched_dimensions"].get(dimension, 0) > 0 for x in results) / total if total else 0.0
            for dimension in DIMENSIONS
        },
        "matched_skill_count_distribution": dict(Counter(x["matched_skill_count"] for x in results)),
        "cases": results,
    }


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
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
