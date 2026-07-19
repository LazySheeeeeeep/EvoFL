from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import llm_config, load_config  # noqa: E402
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

    cases = (
        json.loads(Path(args.selected_cases_file).read_text(encoding="utf-8-sig"))
        if args.selected_cases_file
        else select_cases(args.sample_size, args.seed, args.dataset_name, args.split, args.prefer_local_images)
    )
    write_json(out_dir / "selected_cases.json", cases)
    previous_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and (out_dir / "progress_summary.json").exists():
        previous = json.loads((out_dir / "progress_summary.json").read_text(encoding="utf-8"))
        previous_by_id = {
            item["instance_id"]: item
            for item in previous.get("cases", [])
            if isinstance(item, dict) and item.get("instance_id")
        }
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
    apply_embedding_overrides(config, args)
    check_llm_config(config["llm"], args.provider)

    bank_cfg = config.get("skill_bank", {})
    skill_bank_path = ROOT / bank_cfg.get("path", "skill_pools/skill_bank_v0/skills.jsonl")
    if args.reset_skill_bank:
        reset_skill_bank(skill_bank_path, out_dir)
        reset_embedding_cache(config, out_dir)
    skill_bank = make_skill_bank(config)
    client = OpenAICompatibleClient.from_config(config["llm"])
    summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        instance_id = case["instance_id"]
        previous_summary = previous_by_id.get(instance_id)
        if (
            args.resume
            and previous_summary
            and previous_summary.get("explorer_status") == "completed"
            and previous_summary.get("evolution_status") == "completed"
        ):
            summaries.append(previous_summary)
            write_json(out_dir / "progress_summary.json", {"cases": summaries})
            continue
        case_dir = out_dir / "cases" / instance_id
        repo_dir = out_dir / "repos" / _safe_name(case["image_name"])
        case_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = materialize_repo(case["image_name"], repo_dir, force=args.force_recopy_repo)
        write_json(case_dir / "task.json", case)

        summary: dict[str, Any] = {
            "index": index,
            "instance_id": instance_id,
            "repo": case["repo"],
            "image_name": case["image_name"],
            "explorer_status": "not_started",
            "evolution_status": "not_started",
        }
        try:
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
            evolution = run_case_evolution(
                case_run_dir=case_dir,
                repo=case["repo"],
                issue=case["problem_statement"],
                config=config,
                llm_client=client,
                force=args.force_evolution,
                ground_truth_patch=case.get("patch", ""),
                ground_truth_functions=[],
                output_dir=case_dir / "case_evolution",
            )
            summary["evolution_status"] = "completed"
            summary["evolution"] = {
                "eligible": evolution.get("eligible"),
                "reason": evolution.get("reason"),
                "updated_skill_ids": evolution.get("updated_skill_ids", []),
                "updated_skill_types": evolution.get("updated_skill_types", []),
                "skill_updates": evolution.get("skill_updates", {}),
            }
            if args.rebuild_embeddings_after_case:
                summary["embedding_rebuild"] = rebuild_embeddings_after_case(config)
        except Exception as exc:  # noqa: BLE001 - batch isolation.
            summary["error"] = str(exc)
            if summary["explorer_status"] == "not_started":
                summary["explorer_status"] = "failed"
            elif summary["evolution_status"] == "not_started":
                summary["evolution_status"] = "failed"
        summaries.append(summary)
        write_json(out_dir / "progress_summary.json", {"cases": summaries})

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
    }
    write_json(out_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-smith-py")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--output-dir", default="runs/swe_smith_case_by_case_5_deepseek_v4_flash")
    parser.add_argument("--config", default="config/evolutefl.global.json")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--prefer-local-images", action="store_true", default=True)
    parser.add_argument("--selected-cases-file")
    parser.add_argument("--force-recopy-repo", action="store_true")
    parser.add_argument("--force-evolution", action="store_true", default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-steps", default="none", help="Explorer step limit. Use 'none' for no step limit.")
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--reset-skill-bank", action="store_true")
    parser.add_argument("--enable-embedding", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"])
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-min-score", type=float)
    parser.add_argument("--rebuild-embeddings-after-case", action="store_true")
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


def check_llm_config(llm: dict[str, Any], provider: str) -> None:
    if not llm.get("base_url"):
        raise RuntimeError(
            f"Provider {provider!r} has no base_url. Set DEEPSEEK_BASE_URL or update config/evolutefl.global.json."
        )
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


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
    if repo_dir.exists() and any(repo_dir.iterdir()) and not force:
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
        run(["docker", "cp", f"{container_name}:/testbed", str(repo_dir)])
    finally:
        run(["docker", "rm", "-f", container_name], check=False)
    return repo_dir


def find_cached_repo(image_name: str, *, exclude: Path) -> Path | None:
    safe = _safe_name(image_name)
    exclude = exclude.resolve()
    candidates = sorted(ROOT.glob(f"runs/*/repos/{safe}"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == exclude:
            continue
        if resolved.is_dir() and any(resolved.iterdir()):
            return resolved
    return None


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


def rebuild_embeddings_after_case(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return make_skill_bank(config).rebuild_embeddings()
    except Exception as exc:  # noqa: BLE001 - evolution should continue even if cache warming fails.
        return {"error": str(exc)}


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
