"""Audit provider-recorded token usage for the frozen RQ1 experiments.

This script deliberately separates exact provider usage from proxies. Some
baselines log only the Explorer calls or round context lengths, so treating all
methods as having equally complete accounting would overstate what is known.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

MAIN = RUNS / "rq1_expanded400_eval500_v4flash_existingfunc_20260922"
TEMPORAL = RUNS / "rq1_temporal_deepseek_20260915"
MEMORY = RUNS / "rq1_memory_baselines_20260924"
MEMORY_PREP = RUNS / "rq1_memory_baseline_preparation_20260924"
EXTERNAL = RUNS / "external_fl_baseline_preflight_20260924"
SWE_EXP = EXTERNAL / "swe_exp_loc"
COSIL = EXTERNAL / "cosil_full500_results"
AGENTLESS = RUNS / "agentless_full500_qwen_embedding_flash_20260930"
MEMFL = RUNS / "memfl_loc_lite_500_20261001"
QWEN = RUNS / "rq1_main_qwen3_8_flash_eval500_20260929"

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
DETAIL_FIELDS = {
    "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens"),
    "cached_tokens": ("prompt_tokens_details", "cached_tokens"),
    "cache_hit_tokens": ("prompt_cache_hit_tokens",),
    "cache_miss_tokens": ("prompt_cache_miss_tokens",),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def new_counter() -> dict:
    return {
        "usage_records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
    }


def add_usage(counter: dict, usage: dict) -> None:
    if not isinstance(usage, dict):
        return
    if not any(key in usage for key in USAGE_FIELDS):
        return
    counter["usage_records"] += 1
    for key in USAGE_FIELDS:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            counter[key] += value
    for output_key, path in DETAIL_FIELDS.items():
        value = usage
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            counter[output_key] += value


def collect_usage(value, counter: dict) -> None:
    if isinstance(value, dict):
        add_usage(counter, value)
        for child in value.values():
            collect_usage(child, counter)
    elif isinstance(value, list):
        for child in value:
            collect_usage(child, counter)


def scan_file(path: Path, counter: dict, characters: dict) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    characters["files"] += 1
    characters["characters"] += len(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                collect_usage(json.loads(line), counter)
            except json.JSONDecodeError:
                continue
        return
    collect_usage(value, counter)


def unique_paths(patterns: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in patterns if path.is_file()})


def scan_paths(paths: Iterable[Path]) -> dict:
    counter = new_counter()
    characters = {"files": 0, "characters": 0}
    for path in unique_paths(paths):
        scan_file(path, counter, characters)
    return {**counter, **characters}


def glob(root: Path, pattern: str) -> list[Path]:
    return list(root.glob(pattern)) if root.exists() else []


def summarize_file_group(label: str, paths: Iterable[Path]) -> dict:
    return {"label": label, **scan_paths(paths)}


def main_eval() -> dict:
    explorer = (
        glob(MAIN, "with_skill/cases/*/llm_trace.json")
        + glob(MAIN, "no_skill/cases/*/llm_trace.json")
        + glob(MAIN, "with_skill_failure_retries_20260929/cases/**/llm_trace.json")
    )
    selector = (
        glob(MAIN, "with_skill/cases/*/fault_skill_search.json")
        + glob(MAIN, "with_skill_failure_retries_20260929/cases/**/fault_skill_search.json")
    )
    return {
        "explorer": summarize_file_group("with_skill + no_skill Explorer", explorer),
        "selector": summarize_file_group("fault selector", selector),
    }


def main_build() -> dict:
    groups = {}
    for label, root in (("historical_200", TEMPORAL), ("extension_200", MAIN)):
        explorer = (
            glob(root, "training/*/explorer/llm_trace.json")
            + glob(root, "training/*/explorer/fault_skill_search.json")
        )
        evolution = (
            glob(root, "training/*/evolution/evolution_query_debug.json")
            + glob(root, "training/*/evolution/fault_reflector_debug.json")
            + glob(root, "training/*/evolution/fault_skill_target_selector_debug.json")
            + glob(root, "training/*/evolution/supplementary/llm_trace.jsonl")
        )
        groups[label] = {
            "explorer_and_selector": summarize_file_group(
                f"{label} Explorer/Skill selection", explorer
            ),
            "reflection_and_investigator": summarize_file_group(
                f"{label} reflection/investigator", evolution
            ),
        }
    return groups


def memory_eval() -> dict:
    result = {}
    for arm in ("episodic", "flat_reflection", "oracle_function_reflection"):
        root = MEMORY / arm
        result[arm] = summarize_file_group(
            arm,
            glob(root, "cases/*/llm_trace.json")
            + glob(root, "cases/*/fault_skill_search.json"),
        )
    for label, root in (
        ("memfl_lite", MEMFL),
        ("swe_exp_loc", SWE_EXP / "evaluation_full335"),
        ("swe_exp_loc_retries", SWE_EXP / "retry_failed_20260930"),
        ("qwen_main", QWEN),
    ):
        result[label] = summarize_file_group(
            label,
            glob(root, "cases/*/llm_trace.json")
            + glob(root, "cases/*/fault_skill_search.json")
            + glob(root, "cases/*/explorer_*/llm_trace.json")
            + glob(root, "cases/*/explorer_*/fault_skill_search.json")
            + glob(root, "cases/**/llm_trace.json")
            + glob(root, "cases/**/fault_skill_search.json"),
        )
    return result


def agentless_usage() -> dict:
    return summarize_file_group(
        "agentless request responses",
        glob(AGENTLESS, "cases/*/requests/*.json"),
    )


ROUND_RE = re.compile(r"\[round \d+\] tokens=(\d+)")


def cosil_usage() -> dict:
    summary_paths = (
        glob(COSIL, "cases/*/file/loc_outputs.jsonl")
        + glob(COSIL, "cases/*/function/loc_outputs_func.jsonl")
    )
    summary = summarize_file_group("CoSIL final-summary API usage", summary_paths)
    rounds = 0
    context_tokens = 0
    for path in glob(COSIL, "cases/*/*/localization_logs/*.log"):
        for match in ROUND_RE.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            rounds += 1
            context_tokens += int(match.group(1))
    return {
        **summary,
        "recorded_rounds": rounds,
        "round_context_tokens_sum_proxy": context_tokens,
        "note": "Context length is not completion usage; final-summary usage alone is exact.",
    }


def unlogged_build_calls() -> dict:
    flat = list((MEMORY_PREP / "flat_reflections").glob("*.json"))
    oracle = list((MEMORY_PREP / "oracle_function_reflections").glob("*.json"))
    swe_exp = list((SWE_EXP / "experiences").glob("*.json"))
    memfl_rows = 0
    dynamic = MEMFL / "dynamic_memory.jsonl"
    if dynamic.is_file():
        memfl_rows = sum(
            bool(line.strip()) for line in dynamic.read_text(encoding="utf-8").splitlines()
        )
    return {
        "flat_reflection": {
            "model_calls": len(flat),
            "output_characters": sum(path.stat().st_size for path in flat),
            "usage_logged": False,
        },
        "oracle_function_reflection": {
            "model_calls": len(oracle),
            "output_characters": sum(path.stat().st_size for path in oracle),
            "usage_logged": False,
        },
        "swe_exp_loc_build": {
            "experiences": len(swe_exp),
            "minimum_model_calls": 2 * len(swe_exp),
            "usage_logged": False,
        },
        "memfl_lite_build": {
            "dynamic_memory_rows": memfl_rows,
            "minimum_model_calls": memfl_rows,
            "usage_logged": False,
        },
    }


def build_report() -> dict:
    return {
        "main_evaluation": main_eval(),
        "main_build": main_build(),
        "internal_memory_evaluation": memory_eval(),
        "agentless_evaluation": agentless_usage(),
        "cosil_evaluation": cosil_usage(),
        "unlogged_build_calls": unlogged_build_calls(),
    }


def flatten(report: dict):
    rows = []

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict) and "usage_records" in value:
            rows.append({"section": prefix, **value})
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else key, child)

    walk("", report)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/token_usage_audit_20261001.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    headers = (
        "section",
        "usage_records",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
    )
    print("\t".join(headers))
    for row in flatten(report):
        print("\t".join(str(row.get(header, "")) for header in headers))
    print(json.dumps(report["unlogged_build_calls"], ensure_ascii=False, indent=2))
