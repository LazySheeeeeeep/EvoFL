from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import llm_config, load_config  # noqa: E402
from evolutefl.evaluation import evaluate_ranked_functions, functions_from_patch  # noqa: E402
from evolutefl.explorer import run_explorer  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.reflection import run_case_evolution  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.setdefault("explorer", {})
    config["explorer"]["max_steps"] = _parse_max_steps_arg(args.max_steps)
    config["explorer"]["max_runtime_seconds"] = args.case_timeout_seconds

    saved_cases_path = out_dir / "selected_cases.json"
    if args.selected_cases_file:
        cases = json.loads(Path(args.selected_cases_file).read_text(encoding="utf-8-sig"))
    elif args.resume and saved_cases_path.exists():
        # A resumed run must retain its original sample even if Docker image
        # availability has changed since selection.  Re-sampling here could
        # silently mix a different training distribution into one SkillBank.
        cases = json.loads(saved_cases_path.read_text(encoding="utf-8-sig"))
    else:
        cases = select_cases(args.sample_size, args.seed, args.dataset_name, args.split, args.prefer_local_images)
    validate_case_issues(cases)
    write_json(out_dir / "selected_cases.json", cases)
    previous_by_id: dict[str, dict[str, Any]] = {}
    if args.resume:
        previous_by_id = load_resume_summaries(out_dir, cases)
    if args.prepare_only:
        prepared = []
        for case in cases:
            repo_dir = out_dir / "repos" / _safe_name(case["image_name"])
            repo_dir = materialize_repo(case["image_name"], repo_dir, force=args.force_recopy_repo)
            prepared.append({"instance_id": case["instance_id"], "repo_dir": str(repo_dir), "image_name": case["image_name"]})
        write_json(out_dir / "prepare_summary.json", {"cases": prepared})
        print(json.dumps({"status": "prepared", "output_dir": str(out_dir), "cases": prepared}, ensure_ascii=False, indent=2))
        return 0

    config["llm"] = llm_config(config, args.provider, args.model)
    if args.llm_seed is not None:
        config["llm"]["seed"] = args.llm_seed
    apply_embedding_overrides(config, args)
    apply_skill_bank_overrides(config, args)
    if args.disable_skills:
        configure_empty_skill_bank(config, out_dir)
    if args.disable_skill_type:
        enabled = set(config.setdefault("skill_bank", {}).get("enabled_skill_types") or (
            "project_skill", "issue_skill", "strategy_skill"
        ))
        enabled.difference_update(args.disable_skill_type)
        config["skill_bank"]["enabled_skill_types"] = sorted(enabled)
    check_llm_config(config["llm"], args.provider)

    bank_cfg = config.get("skill_bank", {})
    skill_bank_path = ROOT / bank_cfg.get("path", "skill_pools/skill_bank_v0/skills.jsonl")
    if args.reset_skill_bank:
        reset_skill_bank(skill_bank_path, out_dir)
        reset_embedding_cache(config, out_dir)
    skill_bank = make_skill_bank(config)
    client = OpenAICompatibleClient.from_config(config["llm"])
    summaries: list[dict[str, Any]] = []
    stopped_after_issue_skill_target = False
    for index, case in enumerate(cases, start=1):
        instance_id = case["instance_id"]
        previous_summary = previous_by_id.get(instance_id)
        if (
            args.resume
            and previous_summary
            and previous_summary.get("explorer_status") == "completed"
            and (
                previous_summary.get("evolution_status") == "completed"
                or args.skip_evolution
            )
        ):
            summaries.append(previous_summary)
            write_json(out_dir / "progress_summary.json", {"cases": summaries})
            continue
        case_dir = out_dir / "cases" / instance_id
        repo_dir = out_dir / "repos" / _safe_name(case["image_name"])
        if not args.resume and case_dir.exists() and any(case_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite existing case artifacts: {case_dir}. "
                "Use a new --output-dir or --resume."
            )
        case_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "index": index,
            "instance_id": instance_id,
            "repo": case["repo"],
            "image_name": case["image_name"],
            "explorer_status": "not_started",
            "evolution_status": "not_started",
        }
        try:
            repo_dir = materialize_repo(case["image_name"], repo_dir, force=args.force_recopy_repo)
            write_json(case_dir / "task.json", case)
            ground_truth_functions = functions_from_patch(case.get("patch", ""), repo_dir)
            summary["ground_truth_functions"] = ground_truth_functions
            explorer_result = run_explorer(
                task={
                    "instance_id": instance_id,
                    "repo_path": str(repo_dir),
                    "repo": case["repo"],
                    "base_commit": case.get("base_commit", ""),
                    "bug_report": case["problem_statement"],
                    "run_dir": str(case_dir),
                },
                config=config,
                llm_client=client,
                skill_bank=skill_bank,
            )
            summary["explorer_status"] = explorer_result.get("status")
            summary["ranked_functions"] = explorer_result.get("ranked_functions", [])
            summary["function_metrics"] = evaluate_ranked_functions(
                summary["ranked_functions"],
                ground_truth_functions,
            )
            summary["skill_usage"] = read_skill_usage(case_dir / "trajectory.jsonl")
            if args.skip_evolution:
                summary["evolution_status"] = "skipped"
            else:
                evolution = run_case_evolution(
                    case_run_dir=case_dir,
                    repo=case["repo"],
                    issue=case["problem_statement"],
                    config=config,
                    llm_client=client,
                    force=args.force_evolution,
                    ground_truth_patch=case.get("patch", ""),
                    ground_truth_functions=ground_truth_functions,
                    output_dir=case_dir / "case_evolution",
                    reflect_success=args.reflect_success,
                )
                summary["evolution_status"] = "completed"
                summary["evolution"] = {
                    "eligible": evolution.get("eligible"),
                    "reason": evolution.get("reason"),
                    "outcome": evolution.get("outcome", {}),
                    "updated_skill_ids": evolution.get("updated_skill_ids", []),
                    "updated_skill_types": evolution.get("updated_skill_types") or sorted({
                        str(update.get("skill_type")) for update in evolution.get("applied_updates", [])
                        if update.get("skill_type")
                    }),
                    "skill_updates": evolution.get("skill_updates", {}),
                    "project_skill_decision": (evolution.get("project_knowledge") or {}).get("decision"),
                    "applied_updates": [
                        {
                            "operation": update.get("operation"),
                            "skill_type": update.get("skill_type"),
                            "updated_skill_id": update.get("updated_skill_id"),
                        }
                        for update in evolution.get("materialized_updates", [])
                    ],
                }
                if args.rebuild_embeddings_after_case:
                    if _skill_content_changed(evolution.get("applied_updates", [])):
                        summary["embedding_rebuild"] = rebuild_embeddings_after_case(config)
                    else:
                        summary["embedding_rebuild"] = {
                            "skipped": True,
                            "reason": "no_skill_content_changed",
                        }
        except Exception as exc:  # noqa: BLE001 - batch isolation.
            summary["error"] = str(exc)
            if summary["explorer_status"] == "not_started":
                summary["explorer_status"] = "failed"
            elif summary["evolution_status"] == "not_started":
                summary["evolution_status"] = "failed"
        summaries.append(summary)
        write_json(out_dir / "progress_summary.json", {"cases": summaries})
        if args.stop_after_issue_skills:
            issue_skill_count = sum(
                1 for skill in skill_bank.active_skills() if skill.skill_type == "issue_skill"
            )
            if issue_skill_count >= args.stop_after_issue_skills:
                stopped_after_issue_skill_target = True
                break

    final = {
        "output_dir": str(out_dir),
        "sample_size": len(cases),
        "provider": args.provider,
        "model": args.model,
        "skill_bank_path": str(skill_bank_path),
        "explorer": {
            "max_steps": config["explorer"].get("max_steps"),
            "max_runtime_seconds": config["explorer"].get("max_runtime_seconds"),
        },
        "cases": summaries,
        "stopped_after_issue_skill_target": stopped_after_issue_skill_target,
        "issue_skill_target": args.stop_after_issue_skills or None,
    }
    final["report"] = build_report(summaries, skill_bank)
    write_json(out_dir / "summary.json", final)
    write_json(out_dir / "experiment_report.json", final["report"])
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-smith-py")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=20260804,
        help="OpenAI-compatible request seed for reproducible Explorer and evolution calls.",
    )
    parser.add_argument("--output-dir", default="runs/swe_smith_case_by_case_5_deepseek_v4_flash")
    parser.add_argument("--config", default="config/evolutefl.global.json")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--prefer-local-images", action="store_true", default=True)
    parser.add_argument("--selected-cases-file")
    parser.add_argument("--force-recopy-repo", action="store_true")
    parser.add_argument(
        "--force-evolution",
        action="store_true",
        help="Evolve every completed case. By default only completed Top-5 misses are eligible.",
    )
    parser.add_argument(
        "--reflect-success",
        action="store_true",
        help=(
            "Allow completed Top-5 hits to contribute Project Skills. "
            "Strategy Skills remain restricted to corrective misses."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-steps", default="none", help="Explorer step limit. Use 'none' for no step limit.")
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--reset-skill-bank", action="store_true")
    parser.add_argument("--enable-embedding", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"])
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-min-score", type=float)
    parser.add_argument(
        "--skill-bank-path",
        help="Optional isolated SkillBank path for this training run.",
    )
    parser.add_argument(
        "--embedding-cache-path",
        help="Optional embedding cache paired with --skill-bank-path.",
    )
    parser.add_argument("--rebuild-embeddings-after-case", action="store_true")
    parser.add_argument(
        "--stop-after-issue-skills",
        type=int,
        default=0,
        help="Stop training after this many active Issue Skills have been created or rewritten into the bank.",
    )
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument(
        "--disable-skills",
        action="store_true",
        help="Use an isolated empty SkillBank without modifying the configured bank.",
    )
    parser.add_argument(
        "--disable-skill-type",
        action="append",
        choices=["project_skill", "issue_skill", "strategy_skill"],
        default=[],
        help="Disable one Skill type while leaving the remaining SkillBank active.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip cases already completed in progress_summary.json.")
    return parser


def apply_embedding_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    embedding = config.setdefault("embedding", {})
    skill_bank = config.setdefault("skill_bank", {})
    if args.enable_embedding:
        embedding["enabled"] = True
    if args.retrieval_mode:
        skill_bank["retrieval_mode"] = args.retrieval_mode
    if args.embedding_base_url:
        embedding["base_url"] = args.embedding_base_url
        embedding["enabled"] = True
    if args.embedding_min_score is not None:
        skill_bank["embedding_min_score"] = args.embedding_min_score


def apply_skill_bank_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    skill_bank = config.setdefault("skill_bank", {})
    embedding = config.setdefault("embedding", {})
    if args.skill_bank_path:
        skill_bank["path"] = args.skill_bank_path
    if args.embedding_cache_path:
        embedding["cache_path"] = args.embedding_cache_path


def check_llm_config(llm: dict[str, Any], provider: str) -> None:
    if not llm.get("base_url"):
        raise RuntimeError(
            f"Provider {provider!r} has no base_url. Set DEEPSEEK_BASE_URL or update config/evolutefl.global.json."
        )
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


def _windows_host_path(path: Path) -> str:
    """Translate a WSL mount path for optional Docker Desktop integrations."""

    parts = path.parts
    if len(parts) >= 3 and parts[1].lower() == "mnt" and len(parts[2]) == 1:
        suffix = "\\".join(parts[3:])
        return f"{parts[2].upper()}:\\{suffix}" if suffix else f"{parts[2].upper()}:\\"
    return str(path)


def select_cases(sample_size: int, seed: int, dataset_name: str, split: str, prefer_local_images: bool) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for SWE-smith loading. Run this script in WSL where datasets is installed.") from exc
    local_images = available_images_from_docker_or_cache() if prefer_local_images else set()
    ds = load_dataset(dataset_name, split=split)
    unique_by_image: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    for row in rows:
        image = row.get("image_name", "")
        # V3 reveals the issue only after repository orientation. A blank
        # problem_statement cannot exercise its Issue Skill stage and would
        # turn the run into undirected repository browsing.
        if not str(row.get("problem_statement") or "").strip():
            continue
        if local_images and image not in local_images and f"{image}:latest" not in local_images:
            continue
        case = {
            "instance_id": row.get("instance_id"),
            "repo": row.get("repo", ""),
            "image_name": image,
            "problem_statement": row.get("problem_statement", ""),
            "patch": row.get("patch", ""),
            "base_commit": row.get("repo", "").split(".")[-1] if row.get("repo") else "",
        }
        if image not in seen_images:
            unique_by_image.append(case)
            seen_images.add(image)
        else:
            fallback.append(case)
        if len(unique_by_image) >= sample_size:
            break
    candidates = unique_by_image + fallback
    if len(candidates) < sample_size:
        raise RuntimeError(f"Only found {len(candidates)} SWE-smith cases with available local images.")
    return candidates[:sample_size]


def validate_case_issues(cases: list[dict[str, Any]]) -> None:
    """Reject manifests that cannot exercise issue-guided localization."""

    empty_issue_ids = [
        str(case.get("instance_id") or "<missing-instance-id>")
        for case in cases
        if not str(case.get("problem_statement") or "").strip()
    ]
    if empty_issue_ids:
        joined = ", ".join(empty_issue_ids)
        raise ValueError(
            "SWE-smith cases require a non-empty problem_statement; "
            f"refusing to run {len(empty_issue_ids)} invalid case(s): {joined}"
        )


def docker_images() -> set[str]:
    try:
        output = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    except Exception:
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def available_images_from_docker_or_cache() -> set[str]:
    images = docker_images()
    images.update(cached_repo_images())
    return images


def cached_repo_images() -> set[str]:
    images: set[str] = set()
    for repo_dir in ROOT.glob("runs/*/repos/*"):
        if not repo_dir.is_dir() or not any(repo_dir.iterdir()):
            continue
        images.add(_image_name_from_safe_name(repo_dir.name))
        images.add(f"{_image_name_from_safe_name(repo_dir.name)}:latest")
    return images


def materialize_repo(image_name: str, repo_dir: Path, *, force: bool = False) -> Path:
    if not force and is_usable_source_repo(repo_dir):
        return repo_dir
    cached = find_cached_repo(image_name, exclude=repo_dir)
    if cached is not None and not force:
        return cached
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    container_name = f"evolutefl_materialize_{_safe_name(image_name)}"
    run(["docker", "rm", "-f", container_name], check=False)
    run(["docker", "create", "--name", container_name, image_name, "bash", "-lc", "sleep 1"])
    try:
        materialize_container_testbed(container_name, repo_dir)
    finally:
        run(["docker", "rm", "-f", container_name], check=False)
    return repo_dir


def materialize_container_testbed(container_name: str, repo_dir: Path) -> None:
    """Extract ``/testbed`` through stdout so Docker Desktop never maps paths.

    In WSL, ``docker`` can be a bridge to ``docker.exe``.  A conventional
    ``docker cp`` then interprets Linux destinations as Windows paths and
    rejects symlinks from SWE-smith repositories.  Extracting directly under
    ``/mnt/d`` is also unreliable when Docker leaves an NTFS handle open, so
    stage the archive on the native WSL filesystem before copying it into the
    workspace.
    """
    completed = subprocess.run(
        ["docker", "cp", f"{container_name}:/testbed", "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: docker cp {container_name}:/testbed -\n"
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evolutefl_testbed_") as staging_dir:
        staging_root = Path(staging_dir).resolve()
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                destination = (staging_root / member.name).resolve()
                if destination != staging_root and staging_root not in destination.parents:
                    raise RuntimeError(f"Unsafe archive member from Docker: {member.name}")
            archive.extractall(staging_root)
        extracted = staging_root / "testbed"
        if not extracted.is_dir():
            raise RuntimeError("Docker archive did not contain /testbed")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.copytree(extracted, repo_dir, symlinks=True)


def find_cached_repo(image_name: str, *, exclude: Path) -> Path | None:
    safe = _safe_name(image_name)
    exclude = exclude.resolve()
    candidates = sorted(ROOT.glob(f"runs/*/repos/{safe}"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == exclude:
            continue
        if is_usable_source_repo(resolved):
            return resolved
    return None


def is_usable_source_repo(path: Path) -> bool:
    """Return whether a historical repo cache can support patch-ground-truth mapping."""

    if not path.is_dir() or not (path / ".git").exists():
        return False
    try:
        return any(path.rglob("*.py"))
    except OSError:
        return False


def run(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(command, text=True, capture_output=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stderr.strip()}")
    return completed.stdout


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reset_skill_bank(path: Path, out_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        backup = out_dir / "skill_bank_initial_backup.jsonl"
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("", encoding="utf-8")


def load_resume_summaries(out_dir: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recover completed cases even when an interrupted run truncated its summary.

    Each case writes Explorer and evolution artifacts independently.  Treat
    those artifacts as authoritative during ``--resume`` so an incomplete
    ``progress_summary.json`` cannot make the runner repeat completed cases
    and append duplicate SkillBank updates.
    """

    recovered: dict[str, dict[str, Any]] = {}
    summary_path = out_dir / "progress_summary.json"
    if summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        for item in prior.get("cases", []) if isinstance(prior, dict) else []:
            if isinstance(item, dict) and item.get("instance_id"):
                recovered[str(item["instance_id"])] = item

    for index, case in enumerate(cases, start=1):
        instance_id = str(case.get("instance_id") or "")
        if not instance_id:
            continue
        case_dir = out_dir / "cases" / instance_id
        result_path = case_dir / "result.json"
        evolution_path = case_dir / "case_evolution" / "case_evolution_summary.json"
        if not result_path.exists() or not evolution_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            evolution = json.loads(evolution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "completed" or not evolution.get("eligible"):
            continue
        repo_dir = out_dir / "repos" / _safe_name(str(case.get("image_name") or ""))
        try:
            ground_truth_functions = functions_from_patch(case.get("patch", ""), repo_dir)
        except Exception:  # noqa: BLE001 - preserve resume even if metric rebuild is unavailable.
            ground_truth_functions = []
        ranked_functions = result.get("ranked_functions", []) or []
        recovered[instance_id] = {
            "index": index,
            "instance_id": instance_id,
            "repo": case.get("repo", ""),
            "image_name": case.get("image_name", ""),
            "explorer_status": "completed",
            "evolution_status": "completed",
            "ground_truth_functions": ground_truth_functions,
            "ranked_functions": ranked_functions,
            "function_metrics": evaluate_ranked_functions(ranked_functions, ground_truth_functions),
            "skill_usage": read_skill_usage(case_dir / "trajectory.jsonl"),
            "evolution": {
                "eligible": evolution.get("eligible"),
                "reason": evolution.get("reason"),
                "outcome": evolution.get("outcome", {}),
                "updated_skill_ids": evolution.get("updated_skill_ids", []),
                "updated_skill_types": evolution.get("updated_skill_types", []),
                "skill_updates": evolution.get("skill_updates", {}),
            },
            "embedding_rebuild": {"recovered": True},
        }
    return recovered


def reset_embedding_cache(config: dict[str, Any], out_dir: Path) -> None:
    embedding_cfg = config.get("embedding", {}) or {}
    bank_cfg = config.get("skill_bank", {}) or {}
    cache_path = embedding_cfg.get("cache_path") or bank_cfg.get("embedding_cache_path")
    if not cache_path:
        cache_path = "skill_pools/skill_bank_v0/embeddings/jina_v3/skill_embeddings.jsonl"
    path = ROOT / cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        backup = out_dir / "skill_embedding_initial_backup.jsonl"
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("", encoding="utf-8")


def _skill_content_changed(applied_updates: Any) -> bool:
    """Return whether this case changed any persisted Skill text.

    Preserve operations acknowledge a useful card but do not alter its embedding
    document, so rebuilding the entire local cache after them is unnecessary.
    """

    return any(
        isinstance(update, dict)
        and str(update.get("action") or "") in {"create_new", "rewrite"}
        for update in (applied_updates or [])
    )


def rebuild_embeddings_after_case(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return make_skill_bank(config).rebuild_embeddings()
    except Exception as exc:  # noqa: BLE001 - evolution should continue even if cache warming fails.
        return {"error": str(exc)}


def configure_empty_skill_bank(config: dict[str, Any], out_dir: Path) -> None:
    empty_root = out_dir / "empty_skill_bank"
    empty_root.mkdir(parents=True, exist_ok=True)
    skills_path = empty_root / "skills.jsonl"
    embeddings_path = empty_root / "embeddings.jsonl"
    skills_path.write_text("", encoding="utf-8")
    embeddings_path.write_text("", encoding="utf-8")
    config.setdefault("skill_bank", {})["path"] = str(skills_path)
    config.setdefault("embedding", {})["cache_path"] = str(embeddings_path)


def read_skill_usage(path: Path) -> dict[str, Any]:
    usage = {
        "project_skill_attempted": False,
        "project_skill_id": None,
        "issue_skill_attempted": False,
        "issue_skill_id": None,
        "strategy_skill_attempted": False,
        "strategy_skill_id": None,
        "forced_finish": False,
    }
    if not path.exists():
        return usage
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_name = event.get("event")
        if event_name == "project_skill_request":
            usage["project_skill_attempted"] = True
        elif event_name == "project_skill_loaded":
            usage["project_skill_id"] = event.get("loaded_skill_id")
        elif event_name == "issue_skill_request":
            usage["issue_skill_attempted"] = True
        elif event_name == "issue_skill_loaded":
            usage["issue_skill_id"] = event.get("loaded_skill_id")
        elif event_name == "strategy_skill_request":
            usage["strategy_skill_attempted"] = True
        elif event_name == "strategy_skill_loaded":
            usage["strategy_skill_id"] = event.get("loaded_skill_id")
        elif event_name in {"forced_finish", "deterministic_finish"}:
            usage["forced_finish"] = True
    usage["any_skill_loaded"] = bool(
        usage["project_skill_id"] or usage["issue_skill_id"] or usage["strategy_skill_id"]
    )
    return usage


def build_report(
    summaries: list[dict[str, Any]],
    skill_bank: Any,
) -> dict[str, Any]:
    metric_rows = [
        item
        for item in summaries
        if item.get("ground_truth_functions") and isinstance(item.get("function_metrics"), dict)
    ]
    count = len(metric_rows)

    def rate(field: str) -> float:
        return (
            sum(bool(item["function_metrics"].get(field)) for item in metric_rows) / count
            if count
            else 0.0
        )

    with_skill = [
        item for item in metric_rows if (item.get("skill_usage") or {}).get("any_skill_loaded")
    ]
    without_skill = [
        item for item in metric_rows if not (item.get("skill_usage") or {}).get("any_skill_loaded")
    ]

    def subgroup(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "case_count": total,
            "top1": (
                sum(bool(item["function_metrics"].get("top1")) for item in items) / total
                if total
                else 0.0
            ),
            "top3": (
                sum(bool(item["function_metrics"].get("top3")) for item in items) / total
                if total
                else 0.0
            ),
            "top5": (
                sum(bool(item["function_metrics"].get("top5")) for item in items) / total
                if total
                else 0.0
            ),
            "mrr": (
                sum(float(item["function_metrics"].get("reciprocal_rank") or 0.0) for item in items)
                / total
                if total
                else 0.0
            ),
        }

    active_skills = skill_bank.active_skills()
    evolution_decisions: Counter[str] = Counter()
    evolution_operations: Counter[str] = Counter()
    for item in summaries:
        evolution = item.get("evolution") or {}
        project_decision = evolution.get("project_skill_decision")
        if project_decision:
            evolution_decisions[f"project_skill:{project_decision}"] += 1
        for skill_type, update in (evolution.get("skill_updates") or {}).items():
            if isinstance(update, dict) and update.get("decision"):
                evolution_decisions[f"{skill_type}:{update['decision']}"] += 1
        for update in evolution.get("applied_updates") or []:
            if isinstance(update, dict) and update.get("operation"):
                evolution_operations[str(update["operation"])] += 1
    return {
        "case_count": len(summaries),
        "completed_count": sum(item.get("explorer_status") == "completed" for item in summaries),
        "evolution_completed_count": sum(
            item.get("evolution_status") == "completed" for item in summaries
        ),
        "mapped_ground_truth_count": count,
        "function_metrics": {
            "top1": rate("top1"),
            "top3": rate("top3"),
            "top5": rate("top5"),
            "mrr": (
                sum(
                    float(item["function_metrics"].get("reciprocal_rank") or 0.0)
                    for item in metric_rows
                )
                / count
                if count
                else 0.0
            ),
        },
        "skill_usage": {
            "project_skill_loaded_count": sum(
                bool((item.get("skill_usage") or {}).get("project_skill_id"))
                for item in summaries
            ),
            "issue_skill_loaded_count": sum(
                bool((item.get("skill_usage") or {}).get("issue_skill_id"))
                for item in summaries
            ),
            "strategy_skill_loaded_count": sum(
                bool((item.get("skill_usage") or {}).get("strategy_skill_id"))
                for item in summaries
            ),
            "with_any_skill": subgroup(with_skill),
            "without_any_skill": subgroup(without_skill),
        },
        "forced_finish_count": sum(
            bool((item.get("skill_usage") or {}).get("forced_finish"))
            for item in summaries
        ),
        "active_skill_count": len(active_skills),
        "active_skill_count_by_type": {
            skill_type: sum(skill.skill_type == skill_type for skill in active_skills)
            for skill_type in ("project_skill", "issue_skill", "strategy_skill")
        },
        "active_skill_ids": [skill.skill_id for skill in active_skills],
        "evolution_decision_counts": dict(evolution_decisions),
        "evolution_operation_counts": dict(evolution_operations),
    }


def _parse_max_steps_arg(value: str) -> int | None:
    if value.strip().lower() in {"", "none", "null", "unbounded", "infinite"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _image_name_from_safe_name(value: str) -> str:
    # SWE-smith images use slash/dash/dot separators; this inverse is only for
    # cache filtering, where the exact safe-name transform is enough.
    if value.startswith("swebench_swesmith_x86_64_"):
        rest = value[len("swebench_swesmith_x86_64_") :]
        parts = rest.split("_1776_")
        if len(parts) == 2:
            owner = parts[0].replace("_", "-")
            repo_and_hash = parts[1]
            repo_parts = repo_and_hash.rsplit("_", 1)
            if len(repo_parts) == 2:
                repo, commit = repo_parts
                return f"swebench/swesmith.x86_64.{owner}_1776_{repo}.{commit}"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
