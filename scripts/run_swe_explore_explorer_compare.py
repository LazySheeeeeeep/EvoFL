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
from evolutefl.evaluation import evaluate_ranked_functions, functions_from_patch  # noqa: E402
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
SKILL_TYPES = ("project_skill", "fault_skill", "strategy_skill")
ARM_NAMES = ("baseline_no_skill", "project_only", "strategy_only", "with_skill")


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
    config["llm"]["temperature"] = args.temperature
    if args.llm_seed is not None:
        config["llm"]["seed"] = args.llm_seed
    if args.llm_max_attempts is not None:
        config["llm"]["max_attempts"] = args.llm_max_attempts
    if args.llm_timeout is not None:
        config["llm"]["timeout"] = args.llm_timeout
    config["_rerun_existing"] = args.rerun_existing
    if not (args.report_only or args.prepare_only):
        check_llm_config(config["llm"], args.provider)

    existing_manifest = out_dir / "selected_cases.json"
    if args.selected_cases_file:
        selected = load_selected_cases(args.selected_cases_file)
    elif args.report_only and existing_manifest.exists():
        # A report must describe the cases that produced the existing arm
        # artifacts. Re-sampling here corrupts the comparison denominator.
        selected = load_selected_cases(str(existing_manifest))
    else:
        selected = select_cases(args)
    selected = selected[: args.sample_size]
    if not args.report_only:
        write_json(existing_manifest, selected)
    if args.report_only:
        arm_summaries = {
            arm_name: read_required_json(out_dir / arm_name / "summary.json")
            for arm_name in ARM_NAMES
            if (out_dir / arm_name / "summary.json").exists()
        }
        comparison = build_comparison(out_dir, selected, arm_summaries, base_repo_root=_report_base_repo_root(out_dir))
        write_json(out_dir / "comparison_summary.json", comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0

    if args.repo_cache_root:
        base_repo_root = Path(args.repo_cache_root).resolve()
        materialization = validate_repo_cache(selected, base_repo_root)
    else:
        base_repo_root = out_dir / "repos_base"
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
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **materialization}, ensure_ascii=False, indent=2))
        return 0

    all_arms = [
        ("baseline_no_skill", make_no_skill_config(config, out_dir)),
        ("project_only", make_type_limited_skill_config(skill_config, "project_skill")),
        ("strategy_only", make_type_limited_skill_config(skill_config, "strategy_skill")),
        ("with_skill", skill_config),
    ]
    runtime_repo_root = (
        Path(args.runtime_repo_root).expanduser().resolve()
        if args.runtime_repo_root
        else None
    )
    selected_arm_names = (
        {"baseline_no_skill", "with_skill"}
        if args.arm == "both"
        else set(ARM_NAMES)
        if args.arm == "all"
        else {args.arm}
    )
    arms = [(name, arm_config) for name, arm_config in all_arms if name in selected_arm_names]
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_name, arm_config in arms:
        arm_summaries[arm_name] = run_arm(
            arm_name=arm_name,
            selected=selected,
            base_repo_root=base_repo_root,
            out_dir=out_dir,
            config=arm_config,
            repo_mode=args.repo_mode,
            runtime_repo_root=runtime_repo_root,
        )

    if args.arm not in {"both", "all"}:
        print(json.dumps({"status": "arm_completed", "arm": args.arm, "summary": arm_summaries[args.arm]}, ensure_ascii=False, indent=2))
        return 0

    comparison = build_comparison(out_dir, selected, arm_summaries, base_repo_root=base_repo_root)
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
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=20260804,
        help="OpenAI-compatible request seed shared by every arm for paired comparisons.",
    )
    parser.add_argument("--selected-cases-file")
    parser.add_argument(
        "--arm",
        choices=["both", "all", "baseline_no_skill", "project_only", "strategy_only", "with_skill"],
        default="both",
        help="Run a recovery arm, the historical two-arm comparison, or all four ablation arms.",
    )
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--swe-explore-dataset", default=SWE_EXPLORE_DATASET)
    parser.add_argument("--swe-verified-dataset", default=SWE_VERIFIED_DATASET)
    parser.add_argument("--swe-explore-split", default="train")
    parser.add_argument("--swe-verified-split", default="test")
    parser.add_argument("--dataset-filter", default="verified")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--case-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--llm-max-attempts",
        type=int,
        default=None,
        help="Optional transport retry limit for transient provider failures during a long evaluation.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="Optional per-request transport timeout in seconds for a recoverable evaluation run.",
    )
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"], default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-timeout", type=float)
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument(
        "--skill-bank-path",
        default="skill_pools/skill_bank_v0/skills.jsonl",
        help="SkillBank used exclusively by the with_skill arm.",
    )
    parser.add_argument(
        "--embedding-cache-path",
        default="",
        help="Optional embedding cache paired with --skill-bank-path.",
    )
    parser.add_argument("--rebuild-embeddings", action="store_true", default=True)
    parser.add_argument("--force-recopy-repos", action="store_true")
    parser.add_argument(
        "--materialization-mode",
        choices=["docker_then_archive_then_git", "archive_then_git", "git"],
        default="docker_then_archive_then_git",
    )
    parser.add_argument("--no-pull-images", action="store_true")
    parser.add_argument("--github-archive-mirror", default="")
    parser.add_argument("--repo-cache-root")
    parser.add_argument(
        "--repo-mode",
        choices=["shared_readonly", "copy"],
        default="copy",
        help="Create an arm-local repository copy by default; shared reuse is only safe when tools cannot mutate files.",
    )
    parser.add_argument(
        "--runtime-repo-root",
        default="",
        help=(
            "Optional local filesystem root for mutable arm repository copies. "
            "Keeping Explorer workspaces off /mnt drives avoids WSL 9P I/O during tool calls."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Select/materialize cases and rebuild embeddings without running Explorer.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun cases that already have a result.json instead of resuming them.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild metrics from existing arm outputs without materialization or LLM calls.",
    )
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
    # PowerShell's UTF-8 writer may emit a BOM; selected manifests are shared
    # between the Windows workspace and WSL evaluation runner.
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
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


def validate_repo_cache(selected: list[dict[str, Any]], base_root: Path) -> dict[str, Any]:
    report = {"prepared": [], "failed": []}
    for case in selected:
        repo_dir = base_root / safe_name(case["instance_id"])
        if repo_dir.is_dir() and any(repo_dir.iterdir()):
            report["prepared"].append(
                {
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    "base_commit": case["base_commit"],
                    "repo_dir": str(repo_dir),
                    "source": "external_repo_cache",
                }
            )
        else:
            report["failed"].append(
                {
                    "instance_id": case["instance_id"],
                    "repo": case.get("repo"),
                    "error": f"Missing cached repository: {repo_dir}",
                }
            )
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
            _download_github_archive(url, tmp)
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


def _download_github_archive(url: str, destination: Path) -> None:
    """Download a source archive without inheriting WSL's unreliable IPv6 path.

    GitHub archive requests can stall in WSL even when normal TCP connectivity
    is available.  Curl's IPv4/HTTP1.1 path is reliable in that environment;
    urllib remains a portable fallback where curl is unavailable.
    """

    curl = shutil.which("curl")
    if curl:
        run(
            [
                curl,
                "--fail",
                "--location",
                "--http1.1",
                "--ipv4",
                "--connect-timeout",
                "15",
                "--max-time",
                "600",
                "--retry",
                "2",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--output",
                str(destination),
                url,
            ],
            timeout=1830,
        )
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        destination.write_bytes(response.read())


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
    clone["skill_bank"]["path"] = args.skill_bank_path
    clone["skill_bank"]["retrieval_mode"] = args.retrieval_mode
    clone["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    clone.setdefault("embedding", {})
    clone["embedding"]["enabled"] = True
    clone["embedding"]["base_url"] = args.embedding_base_url
    if args.embedding_cache_path:
        clone["embedding"]["cache_path"] = args.embedding_cache_path
    if args.embedding_timeout is not None:
        clone["embedding"]["timeout"] = args.embedding_timeout
    return clone


def make_type_limited_skill_config(config: dict[str, Any], skill_type: str) -> dict[str, Any]:
    clone = json.loads(json.dumps(config))
    clone.setdefault("skill_bank", {})["enabled_skill_types"] = [skill_type]
    return clone


def run_arm(
    *,
    arm_name: str,
    selected: list[dict[str, Any]],
    base_repo_root: Path,
    out_dir: Path,
    config: dict[str, Any],
    repo_mode: str,
    runtime_repo_root: Path | None = None,
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
        arm_repo = (
            base_repo
            if repo_mode == "shared_readonly"
            else (runtime_repo_root or arm_dir / "repos") / arm_name / safe_name(case["instance_id"])
        )
        summary = {
            "index": index,
            "instance_id": case["instance_id"],
            "repo": case["repo"],
            "explorer_status": "not_started",
        }
        try:
            if repo_mode == "copy":
                prepare_arm_repo(base_repo, arm_repo)
            result_path = case_dir / "result.json"
            if result_path.exists() and not config.get("_rerun_existing", False):
                result = json.loads(result_path.read_text(encoding="utf-8"))
                summary["resumed"] = True
            else:
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
            write_json(
                case_dir / "error.json",
                {
                    "instance_id": case["instance_id"],
                    "status": summary["explorer_status"],
                    "error": str(exc),
                },
            )
        summaries.append(summary)
        write_json(arm_dir / "progress_summary.json", {"cases": summaries})
    arm_summary = {
        "arm": arm_name,
        "case_count": len(selected),
        "repo_mode": repo_mode,
        "runtime_repo_root": str(runtime_repo_root) if runtime_repo_root else "",
        "cases": summaries,
    }
    write_json(arm_dir / "summary.json", arm_summary)
    return arm_summary


def build_comparison(
    out_dir: Path,
    selected: list[dict[str, Any]],
    arm_summaries: dict[str, dict[str, Any]],
    *,
    base_repo_root: Path | None = None,
) -> dict[str, Any]:
    arm_metrics: dict[str, Any] = {}
    for arm_name, arm_summary in arm_summaries.items():
        arm_dir = out_dir / arm_name
        final_rows = []
        read_rows = []
        for case in selected:
            case_dir = arm_dir / "cases" / case["instance_id"]
            repo_dir = (
                base_repo_root / safe_name(case["instance_id"])
                if base_repo_root is not None
                else arm_dir / "repos" / safe_name(case["instance_id"])
            )
            final_regions = final_regions_from_result(case_dir / "result.json", repo_dir)
            read_regions = read_regions_from_trajectory(case_dir / "trajectory.jsonl")
            final_rows.append(score_case(case, final_regions, repo_dir, prediction_kind="final_function_regions"))
            read_rows.append(score_case(case, read_regions, repo_dir, prediction_kind="trajectory_read_regions"))
        function_rows = [
            score_function_case(
                case,
                arm_dir / "cases" / case["instance_id"] / "result.json",
                (
                    base_repo_root / safe_name(case["instance_id"])
                    if base_repo_root is not None
                    else arm_dir / "repos" / safe_name(case["instance_id"])
                ),
            )
            for case in selected
        ]
        final_summary = summarize_metric_rows(final_rows)
        read_summary = summarize_metric_rows(read_rows)
        function_summary = summarize_function_rows(function_rows)
        retrieval_rows = build_retrieval_rows(arm_dir, function_rows)
        arm_metrics[arm_name] = {
            "summary": summarize_arm(arm_dir, arm_summary, final_summary),
            "function_metrics": without_cases(function_summary),
            "swe_explore_final_metrics": without_cases(final_summary),
            "swe_explore_read_trajectory_metrics": without_cases(read_summary),
            "tool_usage": summarize_tool_usage(arm_dir),
            "retrieval_summary": summarize_function_retrieval(retrieval_rows),
            "cases": retrieval_rows,
        }
        write_json(arm_dir / "swe_explore_final_region_metrics.json", final_summary)
        write_json(arm_dir / "swe_explore_read_trajectory_metrics.json", read_summary)
        write_json(arm_dir / "patch_function_metrics.json", function_summary)
        write_json(arm_dir / "skill_retrieval_accuracy_summary.json", arm_metrics[arm_name])
    no_skill = arm_metrics.get("baseline_no_skill", {}).get("function_metrics", {})
    ablation_deltas = {
        arm_name: {
            key: metrics["function_metrics"].get(key, 0.0) - no_skill.get(key, 0.0)
            for key in ["top1", "top3", "top5", "mrr"]
        }
        for arm_name, metrics in arm_metrics.items()
        if arm_name != "baseline_no_skill"
    }
    with_skill = arm_metrics.get("with_skill", {}).get("function_metrics", {})
    return {
        "output_dir": str(out_dir),
        "dataset": SWE_EXPLORE_DATASET,
        "selected_instance_ids": [case["instance_id"] for case in selected],
        "same_cases": True,
        "primary_metric_view": "patch_function_metrics",
        "arms": arm_metrics,
        "delta_function": {
            key: with_skill.get(key, 0.0) - no_skill.get(key, 0.0)
            for key in ["top1", "top3", "top5", "mrr"]
        },
        "ablation_delta_function": ablation_deltas,
    }


def _report_base_repo_root(out_dir: Path) -> Path | None:
    report_path = out_dir / "materialization_report.json"
    if not report_path.exists():
        return None
    try:
        prepared = json.loads(report_path.read_text(encoding="utf-8")).get("prepared") or []
        first_repo = prepared[0].get("repo_dir") if prepared else None
        return Path(first_repo).parent if first_repo else None
    except (OSError, ValueError, TypeError):
        return None


def score_function_case(case: dict[str, Any], result_path: Path, repo_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    ranked = [
        str(value)
        for value in (
            result.get("ranked_functions")
            or result.get("finish", {}).get("action", {}).get("ranked_functions")
            or []
        )
    ]
    ground_truth = functions_from_patch(str(case.get("patch") or ""), repo_dir)
    metrics = evaluate_ranked_functions(ranked, ground_truth) if ground_truth else {
        "rank": None,
        "matched_ground_truth": None,
        "top1": False,
        "top3": False,
        "top5": False,
        "reciprocal_rank": 0.0,
    }
    return {
        "case": case["instance_id"],
        "evaluable": bool(ground_truth),
        "ranked_functions": ranked,
        "ground_truth_functions": ground_truth,
        **metrics,
    }


def summarize_function_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("evaluable")]
    count = len(evaluable)
    return {
        "case_count": len(rows),
        "evaluable_case_count": count,
        "top1": sum(1 for row in evaluable if row.get("top1")) / count if count else 0.0,
        "top3": sum(1 for row in evaluable if row.get("top3")) / count if count else 0.0,
        "top5": sum(1 for row in evaluable if row.get("top5")) / count if count else 0.0,
        "mrr": sum(float(row.get("reciprocal_rank", 0.0)) for row in evaluable) / count if count else 0.0,
        "cases": rows,
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
        loaded = load_staged_skills(case_dir)
        matched = [skill for skill in loaded.values() if isinstance(skill, dict)]
        counts = Counter(skill.get("skill_type", skill_type) for skill_type, skill in loaded.items() if skill)
        metric = metric_by_case.get(case_dir.name, {})
        project_search = read_optional_json(case_dir / "project_skill_search.json")
        strategy_search = read_optional_json(case_dir / "strategy_skill_search.json")
        rows.append(
            {
                "case": case_dir.name,
                "matched_skill_count": len(matched),
                "matched_skill_types": sorted(counts),
                "matched_skill_type_counts": dict(counts),
                "loaded_skill_ids": {
                    skill_type: skill.get("skill_id") if isinstance(skill, dict) else None
                    for skill_type, skill in loaded.items()
                },
                "has_any_skill": bool(matched),
                "has_project_skill": "project_skill" in counts,
                "has_strategy_skill": "strategy_skill" in counts,
                "project_skill_attempted": bool(project_search),
                "strategy_skill_attempted": bool(strategy_search),
                "project_top_score": first_top_score(project_search),
                "strategy_selection_mode": (strategy_search.get("search_trace") or {}).get("retrieval_mode"),
                "evaluable": bool(metric.get("evaluable")),
                "top1": bool(metric.get("top1")),
                "top3": bool(metric.get("top3")),
                "top5": bool(metric.get("top5")),
                "rank": metric.get("rank"),
                "mrr": metric.get("reciprocal_rank", 0.0),
            }
        )
    return rows


def load_staged_skills(case_dir: Path) -> dict[str, Any]:
    loaded = read_optional_json(case_dir / "loaded_skills.json")
    return {
        "project_skill": loaded.get("project_skill"),
        "strategy_skill": loaded.get("strategy_skill"),
    }


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def first_top_score(search: dict[str, Any]) -> float | None:
    top_scores = (search.get("search_trace") or {}).get("top_scores") or []
    if not top_scores:
        return None
    return float(top_scores[0]["score"])


def summarize_function_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("evaluable")]
    project_loaded = [row for row in evaluable if row["has_project_skill"]]
    strategy_loaded = [row for row in evaluable if row["has_strategy_skill"]]
    return {
        "all_evaluable": summarize_function_group(evaluable),
        "any_skill": summarize_function_group([row for row in evaluable if row["has_any_skill"]]),
        "no_skill": summarize_function_group([row for row in evaluable if not row["has_any_skill"]]),
        "project_skill": summarize_function_group(project_loaded),
        "strategy_skill": summarize_function_group(strategy_loaded),
        "attempt_rates": {
            "project_skill": (
                sum(1 for row in rows if row["project_skill_attempted"]) / len(rows) if rows else 0.0
            ),
            "strategy_skill": (
                sum(1 for row in rows if row["strategy_skill_attempted"]) / len(rows) if rows else 0.0
            ),
        },
        "load_rates": {
            "project_skill": sum(1 for row in rows if row["has_project_skill"]) / len(rows) if rows else 0.0,
            "strategy_skill": sum(1 for row in rows if row["has_strategy_skill"]) / len(rows) if rows else 0.0,
            "any_skill": sum(1 for row in rows if row["has_any_skill"]) / len(rows) if rows else 0.0,
        },
        "matched_skill_count_distribution": dict(Counter(row["matched_skill_count"] for row in rows)),
    }


def summarize_function_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "case_count": count,
        "top1": sum(1 for row in rows if row["top1"]) / count if count else 0.0,
        "top3": sum(1 for row in rows if row["top3"]) / count if count else 0.0,
        "top5": sum(1 for row in rows if row["top5"]) / count if count else 0.0,
        "mrr": sum(float(row["mrr"]) for row in rows) / count if count else 0.0,
        "cases": [row["case"] for row in rows],
    }


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
