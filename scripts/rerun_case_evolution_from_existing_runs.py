from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import llm_config, load_config, resolve_path  # noqa: E402
from evolutefl.json_utils import read_json, write_json  # noqa: E402
from evolutefl.llm.client import OpenAICompatibleClient  # noqa: E402
from evolutefl.reflection import run_case_evolution, run_general_reflection  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    apply_embedding_overrides(config, args)
    check_llm_config(config["llm"], args.provider)

    skill_bank_path = resolve_path(config.get("skill_bank", {}).get("path", "skill_pools/skill_bank_v0/skills.jsonl"))
    ensure_skill_bank_file(skill_bank_path)
    if args.reset_skill_bank:
        skill_bank_path.write_text("", encoding="utf-8")
    if args.reset_embedding_cache:
        reset_embedding_cache(config)

    input_roots = args.input_roots or ["runs"]
    cases = discover_cases(
        input_roots=[Path(root) for root in input_roots],
        dataset_contains=args.dataset_contains,
        repo_contains=args.repo_contains,
        instance_contains=args.instance_contains,
        require_completed=not args.include_non_completed,
    )
    cases = cases[: args.limit] if args.limit else cases
    write_json(out_dir / "selected_existing_cases.json", [case["manifest"] for case in cases])

    client = OpenAICompatibleClient.from_config(config["llm"])
    summaries: list[dict[str, Any]] = []
    general_runs: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        task = case["task"]
        case_output_dir = out_dir / "cases" / safe_name(task["instance_id"]) / "case_evolution"
        summary: dict[str, Any] = {
            "index": index,
            "instance_id": task["instance_id"],
            "source_case_dir": str(case["case_dir"]),
            "repo": task.get("repo", ""),
            "status": "not_started",
        }
        try:
            evolution = run_case_evolution(
                case_run_dir=case["case_dir"],
                repo=task.get("repo", ""),
                issue=task.get("problem_statement") or task.get("issue") or "",
                config=config,
                llm_client=client,
                force=True,
                ground_truth_patch=task.get("patch", ""),
                ground_truth_functions=[],
                ground_truth_locations=[],
                output_dir=case_output_dir,
            )
            summary.update(
                {
                    "status": "completed",
                    "evolution": {
                        "eligible": evolution.get("eligible"),
                        "reason": evolution.get("reason"),
                        "outcome_type": evolution.get("outcome_type"),
                        "updated_skill_ids": evolution.get("updated_skill_ids", []),
                        "updated_dimensions": evolution.get("updated_dimensions", []),
                        "no_update_reason": evolution.get("no_update_reason"),
                        "residual_card_path": evolution.get("residual_card_path"),
                    },
                }
            )
            if args.rebuild_embeddings_after_case:
                summary["embedding_rebuild"] = rebuild_embeddings(config)
        except Exception as exc:  # noqa: BLE001 - isolate cases.
            summary["status"] = "failed"
            summary["error"] = str(exc)
        summaries.append(summary)
        write_json(out_dir / "progress_summary.json", {"cases": summaries, "general_runs": general_runs})

        if args.general_every and index % args.general_every == 0:
            general_runs.append(
                run_general_window(
                    out_dir=out_dir,
                    config=config,
                    client=client,
                    index=index,
                    window_size=args.general_window_size or args.general_every,
                    dry_run=args.general_dry_run,
                )
            )
            write_json(out_dir / "progress_summary.json", {"cases": summaries, "general_runs": general_runs})

    if args.run_final_general_reflection:
        general_runs.append(
            run_general_window(
                out_dir=out_dir,
                config=config,
                client=client,
                index=len(cases),
                window_size=args.general_window_size,
                dry_run=args.general_dry_run,
                final=True,
            )
        )

    final = {
        "output_dir": str(out_dir),
        "input_roots": input_roots,
        "case_count": len(cases),
        "completed_evolution_count": sum(1 for item in summaries if item.get("status") == "completed"),
        "failed_evolution_count": sum(1 for item in summaries if item.get("status") == "failed"),
        "provider": args.provider,
        "model": args.model,
        "skill_bank_path": str(skill_bank_path),
        "general_runs": general_runs,
        "cases": summaries,
    }
    write_json(out_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun case evolution from existing Explorer run directories.")
    parser.add_argument("--input-root", dest="input_roots", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/evolutefl.global.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-contains")
    parser.add_argument("--repo-contains")
    parser.add_argument("--instance-contains")
    parser.add_argument("--include-non-completed", action="store_true")
    parser.add_argument("--reset-skill-bank", action="store_true")
    parser.add_argument("--reset-embedding-cache", action="store_true")
    parser.add_argument("--enable-embedding", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"])
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-min-score", type=float)
    parser.add_argument("--rebuild-embeddings-after-case", action="store_true")
    parser.add_argument("--general-every", type=int, help="Run batch-level general reflection every N completed case attempts.")
    parser.add_argument("--general-window-size", type=int, help="How many latest residual cards to pass to each general reflection.")
    parser.add_argument("--run-final-general-reflection", action="store_true")
    parser.add_argument("--general-dry-run", action="store_true")
    return parser


def discover_cases(
    *,
    input_roots: list[Path],
    dataset_contains: str | None,
    repo_contains: str | None,
    instance_contains: str | None,
    require_completed: bool,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for root in input_roots:
        for result_path in iter_result_paths(root):
            case_dir = result_path.parent
            if str(case_dir) in seen:
                continue
            seen.add(str(case_dir))
            trajectory_path = case_dir / "trajectory.jsonl"
            task_path = case_dir / "task.json"
            if not trajectory_path.exists() or not task_path.exists():
                continue
            try:
                task = read_json(task_path)
                result = read_json(result_path)
            except Exception:
                continue
            instance_id = str(task.get("instance_id") or result.get("instance_id") or case_dir.name)
            task["instance_id"] = instance_id
            if dataset_contains and dataset_contains not in str(task.get("dataset_name", "")):
                continue
            if repo_contains and repo_contains not in str(task.get("repo", "")):
                continue
            if instance_contains and instance_contains not in instance_id:
                continue
            if require_completed and result.get("status") != "completed":
                continue
            if not (task.get("problem_statement") or task.get("issue")):
                continue
            cases.append(
                {
                    "case_dir": case_dir,
                    "task": task,
                    "result": result,
                    "manifest": {
                        "instance_id": instance_id,
                        "case_dir": str(case_dir),
                        "repo": task.get("repo", ""),
                        "dataset_name": task.get("dataset_name", ""),
                        "result_status": result.get("status"),
                    },
                }
            )
    return cases


def iter_result_paths(root: Path) -> list[Path]:
    result_paths: list[Path] = []
    if not root.exists():
        return result_paths
    skip_dirs = {"repos", "case_evolution", "general_reflection", ".git", "__pycache__", ".pytest_cache"}
    for current, dirs, files in os.walk(root, topdown=True, onerror=lambda _error: None):
        dirs[:] = [name for name in dirs if name not in skip_dirs]
        if "result.json" in files:
            result_paths.append(Path(current) / "result.json")
    return sorted(result_paths)


def run_general_window(
    *,
    out_dir: Path,
    config: dict[str, Any],
    client: OpenAICompatibleClient,
    index: int,
    window_size: int | None,
    dry_run: bool,
    final: bool = False,
) -> dict[str, Any]:
    name = "final" if final else f"after_{index:04d}"
    general_dir = out_dir / "general_reflection" / name
    return run_general_reflection(
        input_run_dir=out_dir,
        config=config,
        llm_client=client,
        output_dir=general_dir,
        window_size=window_size,
        apply_updates=not dry_run,
    )


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
        raise RuntimeError(f"Provider {provider!r} has no base_url.")
    api_key_env = llm.get("api_key_env")
    if api_key_env and not os.getenv(api_key_env) and not llm.get("api_key"):
        raise RuntimeError(f"Provider {provider!r} requires env var {api_key_env}.")


def ensure_skill_bank_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def reset_embedding_cache(config: dict[str, Any]) -> None:
    path_value = (config.get("embedding") or {}).get("cache_path") or (config.get("skill_bank") or {}).get("embedding_cache_path")
    if not path_value:
        return
    path = resolve_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def rebuild_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    bank = make_skill_bank(config)
    return bank.rebuild_embeddings()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
