"""Replay compatible staged Explorer trajectories into an isolated SkillBank.

This avoids rerunning Explorer when a prior run already contains the native
Project/Strategy stage events and function-level ground truth. The target bank
is always configured explicitly, so historical experiments remain untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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
    source_dir = Path(args.source_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _select_cases(source_dir, args.sample_size, args.seed, args.instance_id)
    (output_dir / "selected_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    if args.llm_timeout is not None:
        config["llm"]["timeout"] = args.llm_timeout
    if args.llm_max_attempts is not None:
        config["llm"]["max_attempts"] = args.llm_max_attempts
    config.setdefault("skill_bank", {})["path"] = args.skill_bank_path
    config["skill_bank"]["retrieval_mode"] = args.retrieval_mode
    config["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    config.setdefault("embedding", {})["enabled"] = True
    config["embedding"]["base_url"] = args.embedding_base_url
    config["embedding"]["cache_path"] = args.embedding_cache_path

    bank_path = ROOT / args.skill_bank_path
    cache_path = ROOT / args.embedding_cache_path
    if args.reset_target:
        # ``--reset-target`` starts a new isolated training run. Clear only
        # run-local artifacts under the explicit output directory so stale
        # case summaries cannot be mistaken for fresh evidence.
        cases_dir = output_dir / "cases"
        if cases_dir.exists():
            shutil.rmtree(cases_dir)
        for artifact_name in ("progress_summary.json", "summary.json"):
            artifact = output_dir / artifact_name
            if artifact.exists():
                artifact.unlink()
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        bank_path.write_text("", encoding="utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("", encoding="utf-8")

    client = OpenAICompatibleClient.from_config(config["llm"])
    summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        source_case_dir = Path(case["case_run_dir"])
        target_case_dir = output_dir / "cases" / str(case["instance_id"])
        task = _read_json(source_case_dir / "task.json")
        try:
            evolution = run_case_evolution(
                case_run_dir=source_case_dir,
                repo=str(case["repo"]),
                issue=str(task.get("problem_statement") or ""),
                config=config,
                llm_client=client,
                ground_truth_patch=str(task.get("patch") or ""),
                ground_truth_functions=list(case.get("ground_truth_functions") or []),
                output_dir=target_case_dir,
                reflect_success=args.reflect_success,
                reflect_failure=args.reflect_failure,
                fault_only=args.fault_only,
            )
            rebuild = make_skill_bank(config).rebuild_embeddings() if args.rebuild_embeddings_after_case else None
            summary = {
                "index": index,
                "instance_id": case["instance_id"],
                "status": "completed",
                "eligible": evolution.get("eligible"),
                "reason": evolution.get("reason"),
                "outcome": evolution.get("outcome"),
                "updated_skill_ids": evolution.get("updated_skill_ids", []),
                "updated_skill_types": evolution.get("updated_skill_types", []),
                "embedding_rebuild": rebuild,
            }
        except Exception as exc:  # noqa: BLE001 - preserve the remaining training cases.
            summary = {"index": index, "instance_id": case["instance_id"], "status": "failed", "error": str(exc)}
        summaries.append(summary)
        _write_json(output_dir / "progress_summary.json", {"cases": summaries})

    final = {
        "source_run_dir": str(source_dir),
        "target_skill_bank": str(bank_path),
        "target_embedding_cache": str(cache_path),
        "sample_size": len(cases),
        "provider": args.provider,
        "model": args.model,
        "fault_only": args.fault_only,
        "cases": summaries,
        "active_skill_count": len(make_skill_bank(config).active_skills()),
    }
    _write_json(output_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--provider", default="key77")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="Optional per-request transport timeout for recoverable replay runs.",
    )
    parser.add_argument(
        "--llm-max-attempts",
        type=int,
        default=None,
        help="Optional transport retry limit; failed cases are recorded and replay continues.",
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Replay only these instance IDs. May be supplied more than once.",
    )
    parser.add_argument("--skill-bank-path", required=True)
    parser.add_argument("--embedding-cache-path", required=True)
    parser.add_argument("--retrieval-mode", choices=["embedding", "lexical", "hybrid"], default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument("--reset-target", action="store_true")
    parser.add_argument("--rebuild-embeddings-after-case", action="store_true")
    parser.add_argument(
        "--fault-only",
        action="store_true",
        help="Replay only Fault reflection; skip Project Builder and Strategy reflection.",
    )
    parser.add_argument(
        "--reflect-success",
        action="store_true",
        help="Also evolve completed Top-5 hits. The default learns only from localization misses.",
    )
    parser.add_argument("--reflect-failure", action="store_true", default=True)
    return parser


def _select_cases(
    source_dir: Path,
    sample_size: int,
    seed: int,
    instance_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    summary = _read_json(source_dir / "summary.json")
    candidates: list[dict[str, Any]] = []
    for row in summary.get("cases") or []:
        if row.get("explorer_status") != "completed":
            continue
        instance_id = str(row.get("instance_id") or "")
        case_dir = source_dir / "cases" / instance_id
        if not instance_id or not (case_dir / "result.json").exists() or not (case_dir / "trajectory.jsonl").exists():
            continue
        candidates.append(
            {
                "instance_id": instance_id,
                "repo": str(row.get("repo") or ""),
                "ground_truth_functions": list(row.get("ground_truth_functions") or []),
                "case_run_dir": str(case_dir),
            }
        )
    requested_ids = {value.strip() for value in (instance_ids or []) if value.strip()}
    if requested_ids:
        candidates = [case for case in candidates if case["instance_id"] in requested_ids]
        missing = requested_ids - {case["instance_id"] for case in candidates}
        if missing:
            raise ValueError(f"Requested instance IDs lack compatible trajectories: {sorted(missing)}")
        return candidates
    random.Random(seed).shuffle(candidates)
    return candidates[:sample_size]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
