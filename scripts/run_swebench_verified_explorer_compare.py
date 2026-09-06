from __future__ import annotations

import argparse
import ast
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import llm_config, load_config  # noqa: E402
from evolutefl.explorer import run_explorer  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402


DEFAULT_DATASETS = ("SWE-bench/SWE-bench", "princeton-nlp/SWE-bench", "SWE-bench/SWE-bench_Lite")
DEFAULT_OUTPUT_DIR = "runs/swe_bench_15_compare_gpt5mini_20260622"
SKILL_TYPES = ("project_skill", "fault_skill", "strategy_skill")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
DIFF_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")


def _windows_host_path(path: Path) -> str:
    """Translate a WSL-mounted path for legacy Docker Desktop call sites.

    Current materialization streams ``docker cp`` into the active filesystem,
    so the helper is not used by the normal WSL path.  Keeping this narrow
    conversion preserves compatibility for callers that still need a Windows
    host path when Docker Desktop is invoked directly.
    """

    parts = path.parts
    if len(parts) >= 3 and parts[1].lower() == "mnt" and len(parts[2]) == 1:
        suffix = "\\".join(parts[3:])
        return f"{parts[2].upper()}:\\{suffix}" if suffix else f"{parts[2].upper()}:\\"
    return str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    config.setdefault("explorer", {})
    config["explorer"]["max_steps"] = args.max_steps
    config["explorer"]["max_runtime_seconds"] = args.case_timeout_seconds
    check_llm_config(config["llm"], args.provider)

    selected = load_selected_cases(args.selected_cases_file) if args.selected_cases_file else select_verified_cases(args)
    for case in selected:
        case["image_name"] = choose_image_name(case)
    write_json(out_dir / "selected_cases.json", selected)

    materialization = materialize_selected_cases(
        selected,
        out_dir,
        pull_images=not args.no_pull_images,
        materialization_mode=args.materialization_mode,
        force_recopy=args.force_recopy_repos,
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
    parser.add_argument("--dataset-name", default="", help="Dataset name. Empty means try known Verified dataset names.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--selected-cases-file", help="Optional JSON file with preselected Verified cases.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--case-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--enable-embedding", action="store_true", default=True)
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"], default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-timeout", type=float)
    parser.add_argument("--embedding-min-score", type=float)
    parser.add_argument("--rebuild-embeddings", action="store_true", default=True)
    parser.add_argument(
        "--materialization-mode",
        choices=["docker", "git", "docker_then_git"],
        default="docker_then_git",
        help="How to materialize repositories. Git mode is enough for Explorer-only localization.",
    )
    parser.add_argument("--no-pull-images", action="store_true")
    parser.add_argument("--force-recopy-repos", action="store_true")
    return parser


def load_selected_cases(path: str) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--selected-cases-file must contain a JSON list.")
    selected: list[dict[str, Any]] = []
    for row in rows:
        selected.append(
            {
                "dataset_name": str(row.get("dataset_name") or "preselected"),
                "split": str(row.get("split") or ""),
                "instance_id": str(row.get("instance_id") or ""),
                "repo": str(row.get("repo") or ""),
                "base_commit": str(row.get("base_commit") or ""),
                "problem_statement": str(row.get("problem_statement") or row.get("issue") or ""),
                "patch": str(row.get("patch") or ""),
                "test_patch": str(row.get("test_patch") or ""),
                "image_name": str(row.get("image_name") or ""),
            }
        )
    return selected


def select_verified_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required. Run this script in WSL.") from exc

    dataset_names = [args.dataset_name] if args.dataset_name else list(DEFAULT_DATASETS)
    last_error: Exception | None = None
    rows = None
    dataset_name = ""
    for name in dataset_names:
        try:
            rows = list(load_dataset(name, split=args.split))
            dataset_name = name
            break
        except Exception as exc:  # noqa: BLE001 - try the next known dataset id.
            last_error = exc
    if rows is None:
        raise RuntimeError(f"Could not load SWE-bench Verified dataset: {last_error}")

    random.Random(args.seed).shuffle(rows)
    selected: list[dict[str, Any]] = []
    for row in rows[: args.sample_size]:
        selected.append(
            {
                "dataset_name": dataset_name,
                "split": args.split,
                "instance_id": str(row.get("instance_id") or ""),
                "repo": str(row.get("repo") or ""),
                "base_commit": str(row.get("base_commit") or ""),
                "problem_statement": str(row.get("problem_statement") or ""),
                "patch": str(row.get("patch") or ""),
                "test_patch": str(row.get("test_patch") or ""),
            }
        )
    if len(selected) < args.sample_size:
        raise RuntimeError(f"Only found {len(selected)} cases in {dataset_name}/{args.split}.")
    return selected


def choose_image_name(case: dict[str, Any]) -> str:
    explicit = case.get("image_name")
    if explicit:
        return str(explicit)
    instance_id = str(case["instance_id"])
    return f"swebench/sweb.eval.x86_64.{instance_id}:latest"


def materialize_selected_cases(
    selected: list[dict[str, Any]],
    out_dir: Path,
    *,
    pull_images: bool,
    materialization_mode: str,
    force_recopy: bool,
) -> dict[str, Any]:
    report = {"prepared": [], "failed": []}
    base_root = out_dir / "repos_base"
    base_root.mkdir(parents=True, exist_ok=True)
    for case in selected:
        image = case["image_name"]
        repo_dir = base_root / safe_name(case["instance_id"])
        try:
            source = materialize_one_case(
                case,
                image=image,
                repo_dir=repo_dir,
                pull_images=pull_images,
                materialization_mode=materialization_mode,
                force=force_recopy,
            )
            report["prepared"].append(
                {
                    "instance_id": case["instance_id"],
                    "image_name": image,
                    "repo_dir": str(repo_dir),
                    "source": source,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report all materialization failures.
            report["failed"].append(
                {
                    "instance_id": case["instance_id"],
                    "image_name": image,
                    "error": str(exc),
                }
            )
    return report


def materialize_one_case(
    case: dict[str, Any],
    *,
    image: str,
    repo_dir: Path,
    pull_images: bool,
    materialization_mode: str,
    force: bool,
) -> str:
    if repo_dir.exists() and any(repo_dir.iterdir()) and not force:
        return "cache"
    errors: list[str] = []
    if materialization_mode in {"docker", "docker_then_git"}:
        try:
            if not image_exists(image):
                if pull_images:
                    run(["docker", "pull", image], timeout=1800)
                if not image_exists(image):
                    raise RuntimeError(f"Docker image is not available after pull attempt: {image}")
            materialize_repo_from_image(image, repo_dir)
            return "docker_testbed"
        except Exception as exc:  # noqa: BLE001 - optionally fall back to git.
            errors.append(f"docker: {exc}")
            if materialization_mode == "docker":
                raise
    if materialization_mode in {"git", "docker_then_git"}:
        try:
            materialize_repo_from_git(case, repo_dir)
            return "git_checkout" if not errors else f"git_checkout_after_{errors[0][:120]}"
        except Exception as exc:  # noqa: BLE001 - include both failure modes.
            errors.append(f"git: {exc}")
    raise RuntimeError(" ; ".join(errors) if errors else f"Unknown materialization mode: {materialization_mode}")


def image_exists(image: str) -> bool:
    cp = subprocess.run(["docker", "image", "inspect", image], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cp.returncode == 0


def materialize_repo_from_image(image: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    container = f"evolutefl_verified_materialize_{safe_name(image)[:80]}"
    run(["docker", "rm", "-f", container], check=False)
    run(["docker", "create", "--name", container, image, "bash", "-lc", "sleep 1"], timeout=180)
    try:
        materialize_container_testbed(container, repo_dir)
    finally:
        run(["docker", "rm", "-f", container], check=False)


def materialize_container_testbed(container: str, repo_dir: Path) -> None:
    """Extract Docker's testbed tar stream in the active filesystem.

    This avoids Docker Desktop interpreting a WSL destination as a Windows
    path, which otherwise rejects valid repository symlinks during ``cp``.
    """
    cp = subprocess.run(
        ["docker", "cp", f"{container}:/testbed", "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"docker cp {container}:/testbed - failed: "
            f"{cp.stderr.decode('utf-8', errors='replace').strip()}"
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(cp.stdout), mode="r:") as archive:
        root = repo_dir.parent.resolve()
        for member in archive.getmembers():
            destination = (repo_dir.parent / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Unsafe archive member from Docker: {member.name}")
        archive.extractall(repo_dir.parent)
    extracted = repo_dir.parent / "testbed"
    if not extracted.is_dir():
        raise RuntimeError("Docker archive did not contain /testbed")
    if extracted != repo_dir:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        extracted.rename(repo_dir)


def materialize_repo_from_git(case: dict[str, Any], repo_dir: Path) -> None:
    repo = str(case.get("repo") or "")
    base_commit = str(case.get("base_commit") or "")
    if "/" not in repo:
        raise RuntimeError(f"Cannot clone invalid repo value: {repo!r}")
    if not base_commit:
        raise RuntimeError(f"Missing base_commit for {case.get('instance_id')}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_at_commit(repo_dir, base_commit):
        return
    url = f"https://github.com/{repo}.git"
    # Explorer-only localization needs the exact base working tree, not history.
    # Fetching the target commit directly avoids partial-clone lazy blob fetches
    # that can hang on large repos such as matplotlib.
    last_error: Exception | None = None
    for attempt in range(1, 4):
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        try:
            run(["git", "init", str(repo_dir)], timeout=120)
            run(["git", "-C", str(repo_dir), "remote", "add", "origin", url], timeout=120)
            run(
                [
                    "git",
                    "-C",
                    str(repo_dir),
                    "-c",
                    "http.version=HTTP/1.1",
                    "fetch",
                    "--depth",
                    "1",
                    "--no-tags",
                    "origin",
                    base_commit,
                ],
                timeout=1800,
            )
            run(["git", "-C", str(repo_dir), "checkout", "--detach", "FETCH_HEAD"], timeout=900)
            return
        except Exception as exc:  # noqa: BLE001 - retry transient GitHub/TLS failures.
            last_error = exc
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(str(last_error) if last_error else f"Failed to materialize {case.get('instance_id')}")


def repo_at_commit(repo_dir: Path, base_commit: str) -> bool:
    if not (repo_dir / ".git").exists():
        return False
    try:
        head = run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], timeout=30).strip()
    except Exception:
        return False
    return head == base_commit


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
    if args.embedding_min_score is not None:
        clone["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    clone.setdefault("embedding", {})
    if args.enable_embedding:
        clone["embedding"]["enabled"] = True
    if args.embedding_base_url:
        clone["embedding"]["base_url"] = args.embedding_base_url
        clone["embedding"]["enabled"] = True
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
            "image_name": case["image_name"],
            "explorer_status": "not_started",
        }
        try:
            prepare_arm_repo(base_repo, arm_repo)
            result = run_explorer(
                task={
                    "instance_id": case["instance_id"],
                    "repo_path": str(arm_repo),
                    "repo": case["repo"],
                    "base_commit": case.get("base_commit", ""),
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
    arm_summary = {
        "arm": arm_name,
        "case_count": len(selected),
        "cases": summaries,
    }
    write_json(arm_dir / "summary.json", arm_summary)
    return arm_summary


def prepare_arm_repo(base_repo: Path, arm_repo: Path) -> None:
    if arm_repo.exists():
        shutil.rmtree(arm_repo)
    arm_repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_repo, arm_repo, symlinks=True)


def build_comparison(out_dir: Path, selected: list[dict[str, Any]], arm_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ground_truth = build_ground_truth(selected, out_dir / "repos_base")
    arm_metrics: dict[str, Any] = {}
    for arm_name in arm_summaries:
        arm_dir = out_dir / arm_name
        preds = load_predictions(arm_dir)
        metrics = evaluate(ground_truth, preds)
        retrieval_rows = build_retrieval_rows(arm_dir, metrics["cases"])
        arm_metrics[arm_name] = {
            "summary": summarize_arm(arm_dir, arm_summaries[arm_name], metrics),
            "patch_metrics": without_cases(metrics),
            "tool_usage": summarize_tool_usage(arm_dir),
            "retrieval_summary": summarize_retrieval(retrieval_rows),
            "cases": retrieval_rows,
        }
        write_json(arm_dir / "patch_based_fl_metrics.json", metrics)
        write_json(arm_dir / "skill_retrieval_accuracy_summary.json", arm_metrics[arm_name])
    no_skill = arm_metrics["baseline_no_skill"]["patch_metrics"]
    with_skill = arm_metrics["with_skill"]["patch_metrics"]
    return {
        "output_dir": str(out_dir),
        "selected_instance_ids": [case["instance_id"] for case in selected],
        "same_cases": True,
        "arms": arm_metrics,
        "delta": {
            "top1": with_skill["top1"] - no_skill["top1"],
            "top3": with_skill["top3"] - no_skill["top3"],
            "top5": with_skill["top5"] - no_skill["top5"],
            "mrr": with_skill["mrr"] - no_skill["mrr"],
            "completed_count": arm_metrics["with_skill"]["summary"]["completed_count"]
            - arm_metrics["baseline_no_skill"]["summary"]["completed_count"],
        },
    }


def summarize_arm(arm_dir: Path, arm_summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    cases = arm_summary["cases"]
    return {
        "completed_count": sum(1 for case in cases if case.get("explorer_status") == "completed"),
        "materialization_failed_count": 0,
        "llm_request_failed_count": sum(1 for case in cases if case.get("explorer_status") == "llm_request_failed"),
        "agent_interrupted_count": sum(1 for case in cases if case.get("explorer_status") == "agent_interrupted"),
        "finish_empty_prediction_count": sum(1 for case_dir in (arm_dir / "cases").iterdir() if finish_empty(case_dir)),
        "top1": metrics["top1"],
        "top3": metrics["top3"],
        "top5": metrics["top5"],
        "mrr": metrics["mrr"],
    }


def classify_explorer_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "llm" in text or "api" in text or "http" in text or "timeout" in text:
        return "llm_request_failed"
    return "agent_interrupted"


def finish_empty(case_dir: Path) -> bool:
    result_path = case_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return not bool(result.get("ranked_functions") or result.get("finish", {}).get("action", {}).get("ranked_functions"))


def load_predictions(arm_dir: Path) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    for case_dir in sorted((arm_dir / "cases").iterdir()):
        result_path = case_dir / "result.json"
        if not result_path.exists():
            preds[case_dir.name] = []
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        preds[case_dir.name] = data.get("ranked_functions") or data.get("finish", {}).get("action", {}).get("ranked_functions") or []
    return preds


def build_ground_truth(selected: list[dict[str, Any]], repo_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for case in selected:
        instance_id = case["instance_id"]
        repo_dir = repo_root / safe_name(instance_id)
        files = parse_patch_files(case.get("patch") or "")
        gt: set[str] = set()
        for file_info in files:
            path = file_info["new_path"]
            if not path.endswith(".py"):
                continue
            spans = python_spans(repo_dir / path)
            for hunk in file_info["hunks"]:
                for line in changed_lines_for_hunk(hunk):
                    qual = outer_symbol_qual(spans, line)
                    if qual:
                        gt.add(f"{path}::{qual}")
        out[instance_id] = gt
    return out


def parse_patch_files(patch: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in patch.splitlines():
        match = DIFF_RE.match(line)
        if match:
            current = {"old_path": match.group(1), "new_path": match.group(2), "hunks": []}
            files.append(current)
            continue
        if current is None:
            continue
        hunk_match = HUNK_RE.match(line)
        if hunk_match:
            current["hunks"].append(
                {
                    "old_start": int(hunk_match.group(1)),
                    "new_start": int(hunk_match.group(3)),
                    "lines": [],
                }
            )
            continue
        if current["hunks"]:
            current["hunks"][-1]["lines"].append(line)
    return files


def changed_lines_for_hunk(hunk: dict[str, Any]) -> list[int]:
    old_line = int(hunk["old_start"])
    new_line = int(hunk["new_start"])
    changed: list[int] = []
    for raw in hunk["lines"]:
        if raw.startswith("\\ No newline"):
            continue
        prefix = raw[:1]
        if prefix == "-":
            changed.append(old_line)
            old_line += 1
        elif prefix == "+":
            changed.append(new_line)
            new_line += 1
        else:
            old_line += 1
            new_line += 1
    return sorted(set(changed))


def python_spans(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    spans: list[dict[str, Any]] = []

    def end_lineno(node: ast.AST) -> int:
        return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or getattr(node, "lineno", 0))

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join([*parents, child.name])
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                spans.append(
                    {
                        "kind": kind,
                        "qual": qual,
                        "start": int(getattr(child, "lineno", 1)),
                        "end": end_lineno(child),
                        "depth": len(parents),
                    }
                )
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return spans


def outer_symbol_qual(spans: list[dict[str, Any]], line: int) -> str | None:
    functions = [span for span in spans if span["kind"] == "function" and span["start"] <= line <= span["end"]]
    if functions:
        functions.sort(key=lambda span: (span["depth"], span["end"] - span["start"]))
        return str(functions[0]["qual"])
    classes = [span for span in spans if span["kind"] == "class" and span["start"] <= line <= span["end"]]
    if classes:
        classes.sort(key=lambda span: (span["depth"], span["end"] - span["start"]))
        return str(classes[0]["qual"])
    return None


def evaluate(ground_truth: dict[str, set[str]], preds_by_case: dict[str, list[str]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top1 = top3 = top5 = 0
    rr_sum = 0.0
    for instance_id, gts in sorted(ground_truth.items()):
        preds = preds_by_case.get(instance_id, [])
        rank = None
        hit_gt = None
        for index, pred in enumerate(preds[:5], start=1):
            for gt in gts:
                if function_match(pred, gt):
                    rank = index
                    hit_gt = gt
                    break
            if rank is not None:
                break
        row = {
            "case": instance_id,
            "rank": rank,
            "hit_gt": hit_gt,
            "top1": rank == 1,
            "top3": rank is not None and rank <= 3,
            "top5": rank is not None and rank <= 5,
            "mrr": 1.0 / rank if rank else 0.0,
            "gts": sorted(gts),
            "preds": preds,
        }
        top1 += int(row["top1"])
        top3 += int(row["top3"])
        top5 += int(row["top5"])
        rr_sum += float(row["mrr"])
        rows.append(row)
    count = len(rows)
    return {
        "case_count": count,
        "mapped_gt_cases": sum(1 for gts in ground_truth.values() if gts),
        "top1_count": top1,
        "top3_count": top3,
        "top5_count": top5,
        "top1": top1 / count if count else 0.0,
        "top3": top3 / count if count else 0.0,
        "top5": top5 / count if count else 0.0,
        "mrr": rr_sum / count if count else 0.0,
        "cases": rows,
    }


def function_match(pred: str, gt: str) -> bool:
    pred_file, pred_qual = split_function_id(pred)
    gt_file, gt_qual = split_function_id(gt)
    if pred_file != gt_file:
        return False
    return pred_qual == gt_qual or bool(pred_qual and gt_qual.endswith("." + pred_qual)) or (
        bool(gt_qual) and pred_qual == gt_qual.split(".")[-1]
    )


def split_function_id(value: str) -> tuple[str, str]:
    text = str(value).strip().replace("\\", "/")
    if "::" not in text:
        return text, ""
    file_path, qual = text.split("::", 1)
    return file_path, qual


def summarize_tool_usage(arm_dir: Path) -> dict[str, Any]:
    total: Counter[str] = Counter()
    forced_finish = 0
    per_case: dict[str, dict[str, int]] = {}
    for case_dir in sorted((arm_dir / "cases").iterdir()):
        counts: Counter[str] = Counter()
        trajectory = case_dir / "trajectory.jsonl"
        if trajectory.exists():
            for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "assistant_tool_calls":
                    for call in event.get("tool_calls") or []:
                        name = ((call.get("function") or {}).get("name")) or "unknown"
                        counts[name] += 1
                if event.get("event") == "forced_finish":
                    forced_finish += 1
        total.update(counts)
        per_case[case_dir.name] = dict(counts)
    return {"total": dict(total), "forced_finish_count": forced_finish, "per_case": per_case}


def build_retrieval_rows(arm_dir: Path, metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_by_case = {row["case"]: row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for case_dir in sorted((arm_dir / "cases").iterdir()):
        matched_path = case_dir / "matched_skills.json"
        matched = json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.exists() else []
        skill_types = sorted({skill.get("skill_type", "unknown") for skill in matched})
        counts = Counter(skill.get("skill_type", "unknown") for skill in matched)
        metric = metric_by_case.get(case_dir.name, {})
        rows.append(
            {
                "case": case_dir.name,
                "matched_skill_count": len(matched),
                "matched_skill_types": skill_types,
                "matched_skill_type_counts": dict(counts),
                "has_any_skill": bool(matched),
                "has_project_skill": "project_skill" in counts,
                "has_strategy_skill": "strategy_skill" in counts,
                "top1": bool(metric.get("top1")),
                "top3": bool(metric.get("top3")),
                "top5": bool(metric.get("top5")),
                "rank": metric.get("rank"),
                "mrr": metric.get("mrr", 0.0),
            }
        )
    return rows


def summarize_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    skill_type_success = {}
    for skill_type in SKILL_TYPES:
        key = f"has_{skill_type}"
        count = sum(1 for row in rows if row.get(key))
        skill_type_success[skill_type] = {"count": count, "rate": count / total if total else 0.0}
    return {
        "any_skill": summarize_group([row for row in rows if row["has_any_skill"]]),
        "no_skill": summarize_group([row for row in rows if not row["has_any_skill"]]),
        "skill_type_retrieval_success": skill_type_success,
        "by_skill_type_accuracy": {
            skill_type: summarize_group([row for row in rows if row.get(f"has_{skill_type}")])
            for skill_type in SKILL_TYPES
        },
        "matched_skill_count_distribution": dict(Counter(row["matched_skill_count"] for row in rows)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"case_count": 0, "top1": 0, "top3": 0, "top5": 0, "mrr": 0, "cases": []}
    return {
        "case_count": count,
        "top1": sum(1 for row in rows if row["top1"]) / count,
        "top3": sum(1 for row in rows if row["top3"]) / count,
        "top5": sum(1 for row in rows if row["top5"]) / count,
        "mrr": sum(float(row["mrr"]) for row in rows) / count,
        "cases": [row["case"] for row in rows],
    }


def without_cases(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "cases"}


def check_llm_config(llm: dict[str, Any], provider: str) -> None:
    if not llm.get("base_url"):
        raise RuntimeError(f"Provider {provider!r} has no base_url.")
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


def run(command: list[str], *, check: bool = True, timeout: float | None = None) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stderr.strip()}")
    return completed.stdout


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
