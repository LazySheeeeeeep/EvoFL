from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.request
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
from evolutefl.explorer import run_explorer  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402

from run_swebench_verified_explorer_compare import (  # noqa: E402
    classify_explorer_error,
    finish_empty,
    image_exists,
    materialize_repo_from_image,
    materialize_repo_from_git,
    prepare_arm_repo,
    run,
    safe_name,
    summarize_retrieval,
    summarize_tool_usage,
    write_json,
)


DEFAULT_OUTPUT_DIR = "runs/swe_explore_verified_10_compare_gpt5mini_jina_20260704"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
SWE_EXPLORE_DATASET = "SWE-Explore-Bench/SWE-Explore-Bench"
SWE_VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
DIMENSIONS = ("project_type", "fault_mode", "strategy_type", "general")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    config.setdefault("explorer", {})
    config["explorer"]["max_steps"] = args.max_steps
    config["explorer"]["max_runtime_seconds"] = args.case_timeout_seconds
    check_llm_config(config["llm"], args.provider)

    selected = load_selected_cases(args.selected_cases_file) if args.selected_cases_file else select_cases(args)
    write_json(out_dir / "selected_cases.json", selected)

    materialization = materialize_selected_cases(
        selected,
        out_dir,
        force_recopy=args.force_recopy_repos,
        materialization_mode=args.materialization_mode,
        pull_images=not args.no_pull_images,
        github_archive_mirror=args.github_archive_mirror or None,
    )
    write_json(out_dir / "materialization_report.json", materialization)
    if materialization["failed"]:
        print(json.dumps({"status": "materialization_failed", **materialization}, ensure_ascii=False, indent=2))
        return 2

    skill_config = make_skill_config(config, args)
    if args.rebuild_embeddings:
        rebuild = make_skill_bank(skill_config).rebuild_embeddings()
        write_json(out_dir / "embedding_rebuild.json", rebuild)

    arms = [
        ("baseline_no_skill", make_no_skill_config(config, out_dir)),
        ("with_skill", skill_config),
    ]
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_name, arm_config in arms:
        arm_summaries[arm_name] = run_arm(
            arm_name=arm_name,
            selected=selected,
            base_repo_root=out_dir / "repos_base",
            out_dir=out_dir,
            config=arm_config,
        )

    comparison = build_comparison(out_dir, selected, arm_summaries)
    write_json(out_dir / "comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--selected-cases-file")
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--swe-explore-dataset", default=SWE_EXPLORE_DATASET)
    parser.add_argument("--swe-verified-dataset", default=SWE_VERIFIED_DATASET)
    parser.add_argument("--swe-explore-split", default="train")
    parser.add_argument("--swe-verified-split", default="test")
    parser.add_argument("--dataset-filter", default="verified")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--case-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"], default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-timeout", type=float)
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument("--rebuild-embeddings", action="store_true", default=True)
    parser.add_argument("--force-recopy-repos", action="store_true")
    parser.add_argument(
        "--materialization-mode",
        choices=["docker_then_archive_then_git", "archive_then_git", "git"],
        default="docker_then_archive_then_git",
    )
    parser.add_argument("--no-pull-images", action="store_true")
    parser.add_argument("--github-archive-mirror", default="")
    return parser


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required. Run this script in WSL.") from exc

    explore_rows = list(load_dataset(args.swe_explore_dataset, split=args.swe_explore_split))
    verified_rows = list(load_dataset(args.swe_verified_dataset, split=args.swe_verified_split))
    verified_by_id = {row["instance_id"]: row for row in verified_rows}

    candidates = [
        row
        for row in explore_rows
        if (not args.dataset_filter or row.get("dataset") == args.dataset_filter)
        and row.get("instance_id") in verified_by_id
        and (row.get("ground_truth") or {}).get("read_core_regions")
    ]
    random.Random(args.seed).shuffle(candidates)
    selected = [_merge_case(row, verified_by_id[row["instance_id"]]) for row in candidates[: args.sample_size]]
    if len(selected) < args.sample_size:
        raise RuntimeError(f"Only found {len(selected)} eligible SWE-Explore cases.")
    return selected


def _merge_case(explore_row: dict[str, Any], verified_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": str(explore_row["instance_id"]),
        "dataset": str(explore_row.get("dataset") or ""),
        "repo_dir": str(explore_row.get("repo_dir") or ""),
        "repo_path_placeholder": str(explore_row.get("repo_path") or ""),
        "ground_truth": explore_row.get("ground_truth") or {},
        "read_step_info": explore_row.get("read_step_info") or {},
        "meta": explore_row.get("meta") or {},
        "repo": str(verified_row.get("repo") or ""),
        "base_commit": str(verified_row.get("base_commit") or ""),
        "problem_statement": str(verified_row.get("problem_statement") or ""),
        "patch": str(verified_row.get("patch") or ""),
    }


def load_selected_cases(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--selected-cases-file must contain a JSON list.")
    return data


def materialize_selected_cases(
    selected: list[dict[str, Any]],
    out_dir: Path,
    *,
    force_recopy: bool,
    materialization_mode: str,
    pull_images: bool,
    github_archive_mirror: str | None,
) -> dict[str, Any]:
    report = {"prepared": [], "failed": []}
    base_root = out_dir / "repos_base"
    base_root.mkdir(parents=True, exist_ok=True)
    for case in selected:
        repo_dir = base_root / safe_name(case["instance_id"])
        try:
            if repo_dir.exists() and any(repo_dir.iterdir()) and not force_recopy:
                source = "cache"
            else:
                source = materialize_one_case(
                    case,
                    repo_dir=repo_dir,
                    materialization_mode=materialization_mode,
                    pull_images=pull_images,
                    github_archive_mirror=github_archive_mirror,
                )
            report["prepared"].append(
                {
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    "base_commit": case["base_commit"],
                    "repo_dir": str(repo_dir),
                    "source": source,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep the full materialization report.
            report["failed"].append({"instance_id": case["instance_id"], "repo": case.get("repo"), "error": str(exc)})
    return report


def materialize_one_case(
    case: dict[str, Any],
    *,
    repo_dir: Path,
    materialization_mode: str,
    pull_images: bool,
    github_archive_mirror: str | None,
) -> str:
    errors: list[str] = []
    if materialization_mode == "docker_then_archive_then_git":
        image = f"swebench/sweb.eval.x86_64.{case['instance_id']}:latest"
        try:
            if not image_exists(image) and pull_images:
                run(["docker", "pull", image], timeout=1800)
            if image_exists(image):
                materialize_repo_from_image(image, repo_dir)
                return "docker_testbed"
            errors.append(f"docker image unavailable: {image}")
        except Exception as exc:  # noqa: BLE001 - continue to archive/git fallback.
            errors.append(f"docker: {exc}")
    if materialization_mode in {"docker_then_archive_then_git", "archive_then_git"}:
        try:
            materialize_repo_from_archive(case, repo_dir, mirror=github_archive_mirror)
            return "github_archive" if not errors else f"github_archive_after_{errors[0][:100]}"
        except Exception as exc:  # noqa: BLE001 - continue to git fallback.
            errors.append(f"archive: {exc}")
    try:
        materialize_repo_from_git(case, repo_dir)
        return "git_checkout" if not errors else f"git_checkout_after_{errors[0][:100]}"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"git: {exc}")
    raise RuntimeError(" ; ".join(errors))


def materialize_repo_from_archive(case: dict[str, Any], repo_dir: Path, *, mirror: str | None = None) -> None:
    repo = str(case.get("repo") or "")
    commit = str(case.get("base_commit") or "")
    if "/" not in repo or not commit:
        raise RuntimeError(f"Missing repo/base_commit for {case.get('instance_id')}")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    base = mirror.rstrip("/") if mirror else "https://github.com"
    url = f"{base}/{repo}/archive/{commit}.tar.gz"
    tmp = repo_dir.parent / f".tmp_{safe_name(case['instance_id'])}.tar.gz"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                tmp.write_bytes(response.read())
            with tarfile.open(tmp, mode="r:gz") as tar:
                members = tar.getmembers()
                if not members:
                    raise RuntimeError(f"Empty archive: {url}")
                top_dir = members[0].name.split("/")[0]
                tar.extractall(repo_dir.parent)
            extracted = repo_dir.parent / top_dir
            if not extracted.exists():
                raise RuntimeError(f"Archive did not produce expected directory: {top_dir}")
            shutil.move(str(extracted), str(repo_dir))
            return
        except Exception as exc:  # noqa: BLE001 - retry transient archive failures.
            last_error = exc
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            if attempt < 3:
                time.sleep(5 * attempt)
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(str(last_error) if last_error else f"Failed to download archive: {url}")


def make_no_skill_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    clone = json.loads(json.dumps(config))
    empty_skill_path = out_dir / "empty_skill_bank" / "skills.jsonl"
    empty_skill_path.parent.mkdir(parents=True, exist_ok=True)
    empty_skill_path.write_text("", encoding="utf-8")
    clone.setdefault("skill_bank", {})
    clone["skill_bank"]["path"] = str(empty_skill_path)
    clone["skill_bank"]["retrieval_mode"] = "lexical"
    clone.setdefault("embedding", {})
    clone["embedding"]["enabled"] = False
    return clone


def make_skill_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    clone = json.loads(json.dumps(config))
    clone.setdefault("skill_bank", {})
    clone["skill_bank"]["retrieval_mode"] = args.retrieval_mode
    clone["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    clone.setdefault("embedding", {})
    clone["embedding"]["enabled"] = True
    clone["embedding"]["base_url"] = args.embedding_base_url
    if args.embedding_timeout is not None:
        clone["embedding"]["timeout"] = args.embedding_timeout
    return clone


def run_arm(
    *,
    arm_name: str,
    selected: list[dict[str, Any]],
    base_repo_root: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_json(arm_dir / "selected_cases.json", selected)
    skill_bank = make_skill_bank(config)
    client = OpenAICompatibleClient.from_config(config["llm"])
    summaries: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        case_dir = arm_dir / "cases" / case["instance_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "task.json", case)
        base_repo = base_repo_root / safe_name(case["instance_id"])
        arm_repo = arm_dir / "repos" / safe_name(case["instance_id"])
        summary = {
            "index": index,
            "instance_id": case["instance_id"],
            "repo": case["repo"],
            "explorer_status": "not_started",
        }
        try:
            prepare_arm_repo(base_repo, arm_repo)
            result = run_explorer(
                task={
                    "instance_id": case["instance_id"],
                    "repo_path": str(arm_repo),
                    "repo": case["repo"],
                    "base_commit": case["base_commit"],
                    "bug_report": case["problem_statement"],
                    "run_dir": str(case_dir),
                },
                config=config,
                llm_client=client,
                skill_bank=skill_bank,
            )
            summary["explorer_status"] = result.get("status")
            summary["ranked_functions"] = result.get("ranked_functions", [])
        except Exception as exc:  # noqa: BLE001 - isolate case failures.
            summary["explorer_status"] = classify_explorer_error(exc)
            summary["error"] = str(exc)
        summaries.append(summary)
        write_json(arm_dir / "progress_summary.json", {"cases": summaries})
    arm_summary = {"arm": arm_name, "case_count": len(selected), "cases": summaries}
    write_json(arm_dir / "summary.json", arm_summary)
    return arm_summary


def build_comparison(out_dir: Path, selected: list[dict[str, Any]], arm_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arm_metrics: dict[str, Any] = {}
    for arm_name, arm_summary in arm_summaries.items():
        arm_dir = out_dir / arm_name
        final_rows = []
        read_rows = []
        for case in selected:
            case_dir = arm_dir / "cases" / case["instance_id"]
            repo_dir = arm_dir / "repos" / safe_name(case["instance_id"])
            final_regions = final_regions_from_result(case_dir / "result.json", repo_dir)
            read_regions = read_regions_from_trajectory(case_dir / "trajectory.jsonl")
            final_rows.append(score_case(case, final_regions, repo_dir, prediction_kind="final_function_regions"))
            read_rows.append(score_case(case, read_regions, repo_dir, prediction_kind="trajectory_read_regions"))
        final_summary = summarize_metric_rows(final_rows)
        read_summary = summarize_metric_rows(read_rows)
        retrieval_rows = build_retrieval_rows(arm_dir, final_rows)
        arm_metrics[arm_name] = {
            "summary": summarize_arm(arm_dir, arm_summary, final_summary),
            "swe_explore_final_metrics": without_cases(final_summary),
            "swe_explore_read_trajectory_metrics": without_cases(read_summary),
            "tool_usage": summarize_tool_usage(arm_dir),
            "retrieval_summary": summarize_retrieval(retrieval_rows),
            "cases": retrieval_rows,
        }
        write_json(arm_dir / "swe_explore_final_region_metrics.json", final_summary)
        write_json(arm_dir / "swe_explore_read_trajectory_metrics.json", read_summary)
        write_json(arm_dir / "skill_retrieval_accuracy_summary.json", arm_metrics[arm_name])
    no_skill = arm_metrics["baseline_no_skill"]["swe_explore_final_metrics"]
    with_skill = arm_metrics["with_skill"]["swe_explore_final_metrics"]
    return {
        "output_dir": str(out_dir),
        "dataset": SWE_EXPLORE_DATASET,
        "selected_instance_ids": [case["instance_id"] for case in selected],
        "same_cases": True,
        "primary_metric_view": "swe_explore_final_metrics",
        "arms": arm_metrics,
        "delta_final": {
            key: with_skill.get(key, 0.0) - no_skill.get(key, 0.0)
            for key in [
                "hit_file_rate",
                "hit_region_rate",
                "line_precision",
                "line_recall",
                "line_f1",
                "mrr_region",
                "top1_file",
                "top3_file",
                "top5_file",
            ]
        },
    }


def final_regions_from_result(result_path: Path, repo_dir: Path) -> list[tuple[str, int, int]]:
    if not result_path.exists():
        return []
    data = json.loads(result_path.read_text(encoding="utf-8"))
    ranked = data.get("ranked_functions") or data.get("finish", {}).get("action", {}).get("ranked_functions") or []
    regions: list[tuple[str, int, int]] = []
    for value in ranked[:10]:
        region = function_id_to_region(str(value), repo_dir)
        if region:
            regions.append(region)
    return dedupe_regions(regions)


def function_id_to_region(value: str, repo_dir: Path) -> tuple[str, int, int] | None:
    text = value.strip().replace("\\", "/")
    if "::" not in text:
        path = text
        qual = ""
    else:
        path, qual = text.split("::", 1)
    if not path:
        return None
    file_path = repo_dir / path
    if not file_path.exists():
        return None
    if qual and path.endswith(".py"):
        span = find_python_symbol_span(file_path, qual)
        if span:
            return (path, span[0], span[1])
    return (path, 1, count_lines(file_path))


def find_python_symbol_span(path: Path, qual: str) -> tuple[int, int] | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None
    spans: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join([*parents, child.name])
                start = int(getattr(child, "lineno", 1))
                end = int(getattr(child, "end_lineno", start) or start)
                spans.append((name, start, end))
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    candidates = [span for span in spans if span[0] == qual or span[0].endswith("." + qual) or span[0].split(".")[-1] == qual]
    if not candidates:
        return None
    candidates.sort(key=lambda span: (len(span[0]), span[2] - span[1]))
    return (candidates[0][1], candidates[0][2])


def read_regions_from_trajectory(path: Path) -> list[tuple[str, int, int]]:
    if not path.exists():
        return []
    regions: list[tuple[str, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "tool_result" or event.get("name") != "read_file":
            continue
        try:
            content = json.loads(event.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if not content.get("ok"):
            continue
        result = content.get("result") or {}
        path_value = result.get("path")
        if not path_value:
            continue
        start = int(result.get("start_line") or 1)
        end = int(result.get("end_line") or start)
        regions.append((str(path_value).replace("\\", "/"), start, end))
    return dedupe_regions(regions)


def score_case(case: dict[str, Any], preds: list[tuple[str, int, int]], repo_dir: Path, *, prediction_kind: str) -> dict[str, Any]:
    gt = case.get("ground_truth") or {}
    path_to_lines = build_path_to_lines(repo_dir, gt, preds)
    core_regions = normalize_regions(gt.get("read_core_regions") or [])
    optional_regions = normalize_regions_from_map(gt.get("read_optional_regions_map") or {})
    core_lines = regions_to_lines(core_regions, path_to_lines)
    optional_lines = regions_to_lines(optional_regions, path_to_lines)
    pred_lines = regions_to_lines(preds, path_to_lines)
    core_files = set(gt.get("read_core_files") or [region[0] for region in core_regions])
    pred_files = [region[0] for region in preds]
    pred_file_set = set(pred_files)
    first_region_rank = first_overlap_rank(preds, core_regions, path_to_lines)
    row = {
        "case": case["instance_id"],
        "prediction_kind": prediction_kind,
        "pred_regions": [{"path": p, "start": s, "end": e} for p, s, e in preds],
        "pred_region_count": len(preds),
        "core_file_count": len(core_files),
        "core_region_count": len(core_regions),
        "hit_file_rate": len(pred_file_set & core_files) / len(core_files) if core_files else 0.0,
        "hit_region_rate": hit_region_rate(preds, core_regions, path_to_lines),
        "noise_file_rate": noise_file_rate(pred_file_set, core_files, set(_optional_files(gt))),
        "line_precision": len(pred_lines & core_lines) / len(pred_lines) if pred_lines else 0.0,
        "line_recall": len(pred_lines & core_lines) / len(core_lines) if core_lines else 0.0,
        "context_efficiency": len(pred_lines & (core_lines | optional_lines)) / len(pred_lines) if pred_lines else 0.0,
        "first_useful_rank": first_region_rank,
        "mrr_region": 1.0 / first_region_rank if first_region_rank else 0.0,
        "top1_file": any(path in core_files for path in pred_files[:1]),
        "top3_file": any(path in core_files for path in pred_files[:3]),
        "top5_file": any(path in core_files for path in pred_files[:5]),
    }
    p = row["line_precision"]
    r = row["line_recall"]
    row["line_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return row


def build_path_to_lines(repo_dir: Path, gt: dict[str, Any], preds: list[tuple[str, int, int]]) -> dict[str, int]:
    paths = {region[0] for region in normalize_regions(gt.get("read_core_regions") or [])}
    paths.update(_optional_files(gt))
    paths.update(region[0] for region in preds)
    counts: dict[str, int] = {}
    for path in paths:
        target = repo_dir / path
        if target.is_file():
            counts[path] = count_lines(target)
    return counts


def count_lines(path: Path) -> int:
    try:
        return max(1, len(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    except OSError:
        return 1


def normalize_regions(regions: list[Any]) -> list[tuple[str, int, int]]:
    out = []
    for region in regions:
        if isinstance(region, dict):
            out.append((str(region["path"]), int(region["start"]), int(region["end"])))
        elif isinstance(region, (list, tuple)) and len(region) >= 3:
            out.append((str(region[0]), int(region[1]), int(region[2])))
    return out


def normalize_regions_from_map(region_map: dict[str, Any]) -> list[tuple[str, int, int]]:
    regions: list[tuple[str, int, int]] = []
    for value in region_map.values():
        if isinstance(value, list):
            regions.extend(normalize_regions(value))
    return regions


def regions_to_lines(regions: list[tuple[str, int, int]], path_to_lines: dict[str, int]) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for path, start, end in regions:
        resolved = resolve_interval(path, start, end, path_to_lines)
        if not resolved:
            continue
        left, right = resolved
        for line_no in range(left, right + 1):
            lines.add((path, line_no))
    return lines


def resolve_interval(path: str, start: int, end: int, path_to_lines: dict[str, int]) -> tuple[int, int] | None:
    line_count = path_to_lines.get(path)
    if end == -1 or start < 0:
        if not line_count:
            return None
        resolved_end = line_count if end == -1 else end
        resolved_start = line_count + start + 1 if start < 0 else start
        return (max(1, resolved_start), max(1, min(resolved_end, line_count)))
    if end < 1:
        return None
    return (max(1, start), max(1, end))


def hit_region_rate(preds: list[tuple[str, int, int]], core_regions: list[tuple[str, int, int]], path_to_lines: dict[str, int]) -> float:
    if not core_regions:
        return 0.0
    hit = 0
    for core in core_regions:
        if any(region_overlap(core, pred, path_to_lines) for pred in preds):
            hit += 1
    return hit / len(core_regions)


def first_overlap_rank(preds: list[tuple[str, int, int]], core_regions: list[tuple[str, int, int]], path_to_lines: dict[str, int]) -> int | None:
    for index, pred in enumerate(preds, start=1):
        if any(region_overlap(pred, core, path_to_lines) for core in core_regions):
            return index
    return None


def region_overlap(left: tuple[str, int, int], right: tuple[str, int, int], path_to_lines: dict[str, int]) -> bool:
    if left[0] != right[0]:
        return False
    left_interval = resolve_interval(left[0], left[1], left[2], path_to_lines)
    right_interval = resolve_interval(right[0], right[1], right[2], path_to_lines)
    if not left_interval or not right_interval:
        return False
    return left_interval[0] <= right_interval[1] and right_interval[0] <= left_interval[1]


def noise_file_rate(pred_files: set[str], core_files: set[str], optional_files: set[str]) -> float:
    if not pred_files:
        return 0.0
    return len(pred_files - core_files - optional_files) / len(pred_files)


def _optional_files(gt: dict[str, Any]) -> list[str]:
    files = []
    for value in (gt.get("read_optional_files_map") or {}).values():
        if isinstance(value, list):
            files.extend(str(item) for item in value)
    return files


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    numeric_keys = [
        "hit_file_rate",
        "hit_region_rate",
        "noise_file_rate",
        "line_precision",
        "line_recall",
        "line_f1",
        "context_efficiency",
        "mrr_region",
    ]
    bool_keys = ["top1_file", "top3_file", "top5_file"]
    summary = {"case_count": count, "cases": rows}
    for key in numeric_keys:
        summary[key] = sum(float(row.get(key, 0.0)) for row in rows) / count if count else 0.0
    for key in bool_keys:
        summary[key] = sum(1 for row in rows if row.get(key)) / count if count else 0.0
    summary["predicted_count"] = sum(1 for row in rows if row.get("pred_region_count", 0) > 0)
    return summary


def summarize_arm(arm_dir: Path, arm_summary: dict[str, Any], final_summary: dict[str, Any]) -> dict[str, Any]:
    cases = arm_summary["cases"]
    return {
        "completed_count": sum(1 for case in cases if case.get("explorer_status") == "completed"),
        "llm_request_failed_count": sum(1 for case in cases if case.get("explorer_status") == "llm_request_failed"),
        "agent_interrupted_count": sum(1 for case in cases if case.get("explorer_status") == "agent_interrupted"),
        "finish_empty_prediction_count": sum(1 for case_dir in (arm_dir / "cases").iterdir() if finish_empty(case_dir)),
        "predicted_count": final_summary["predicted_count"],
        "hit_file_rate": final_summary["hit_file_rate"],
        "hit_region_rate": final_summary["hit_region_rate"],
        "line_f1": final_summary["line_f1"],
        "mrr_region": final_summary["mrr_region"],
    }


def build_retrieval_rows(arm_dir: Path, metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_by_case = {row["case"]: row for row in metric_rows}
    rows = []
    for case_dir in sorted((arm_dir / "cases").iterdir()):
        matched_path = case_dir / "matched_skills.json"
        matched = json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.exists() else []
        counts = Counter(skill.get("dimension", "unknown") for skill in matched)
        metric = metric_by_case.get(case_dir.name, {})
        rows.append(
            {
                "case": case_dir.name,
                "matched_skill_count": len(matched),
                "matched_dimensions": sorted(counts),
                "matched_dimension_counts": dict(counts),
                "has_any_skill": bool(matched),
                "has_project_type": "project_type" in counts,
                "has_fault_mode": "fault_mode" in counts,
                "has_strategy_type": "strategy_type" in counts,
                "has_general": "general" in counts,
                "top1": bool(metric.get("top1_file")),
                "top3": bool(metric.get("top3_file")),
                "top5": bool(metric.get("top5_file")),
                "rank": metric.get("first_useful_rank"),
                "mrr": metric.get("mrr_region", 0.0),
            }
        )
    return rows


def dedupe_regions(regions: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    seen = set()
    out = []
    for region in regions:
        if region in seen:
            continue
        seen.add(region)
        out.append(region)
    return out


def without_cases(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "cases"}


def check_llm_config(llm: dict[str, Any], provider: str) -> None:
    if not llm.get("base_url"):
        raise RuntimeError(f"Provider {provider!r} has no base_url.")
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


if __name__ == "__main__":
    raise SystemExit(main())
