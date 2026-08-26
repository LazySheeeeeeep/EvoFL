"""Select reproducible SWE-Explore transfer probes from issue-only Project similarity.

This utility is deliberately a *discovery* step, not an accuracy evaluator.
It never reads a patch, ground-truth region, Explorer result, or Skill outcome.
It ranks only ``repo + problem_statement`` against active Project Skill cards,
then writes complete task records for a later frozen no-skill/Skill ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import load_config  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402
from evolutefl.skills.embedding_store import cosine_similarity, query_embedding_text  # noqa: E402


SWE_EXPLORE_DATASET = "SWE-Explore-Bench/SWE-Explore-Bench"
SWE_VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - WSL-only dependency.
        raise RuntimeError("The datasets package is required in WSL.") from exc

    config = load_config(args.config)
    config.setdefault("skill_bank", {})["path"] = args.skill_bank_path
    config["skill_bank"]["retrieval_mode"] = "embedding"
    config.setdefault("embedding", {})["enabled"] = True
    config["embedding"]["base_url"] = args.embedding_base_url
    config["embedding"]["cache_path"] = args.embedding_cache_path
    bank = make_skill_bank(config)
    skills = [skill for skill in bank.active_skills() if skill.skill_type == "project_skill"]
    if not skills or bank.embedding_client is None:
        raise RuntimeError("An active Project SkillBank and embedding client are required.")

    # Ensure passage embeddings are current before reusing the cache below.
    bank.rebuild_embeddings()
    skill_vectors = _skill_vectors(bank, skills)

    explore_rows = list(load_dataset(args.swe_explore_dataset, split=args.swe_explore_split))
    verified_rows = list(load_dataset(args.swe_verified_dataset, split=args.swe_verified_split))
    verified_by_id = {str(row["instance_id"]): row for row in verified_rows}
    excluded = set(_read_ids(args.exclude_instance_ids_file))
    candidates = [
        _merge_case(row, verified_by_id[str(row["instance_id"])])
        for row in explore_rows
        if str(row.get("instance_id") or "") in verified_by_id
        and str(row.get("instance_id") or "") not in excluded
        and (row.get("ground_truth") or {}).get("read_core_regions")
    ]
    if args.candidate_limit:
        # Bound CPU-only discovery deterministically without consulting any evaluation signal.
        candidates = sorted(candidates, key=lambda row: row["instance_id"])[: args.candidate_limit]
    scored = _score_cases(candidates, skills, skill_vectors, bank, batch_size=args.batch_size)
    scored.sort(key=lambda row: (-float(row["project_similarity"]), row["instance_id"]))
    selected = scored[: args.sample_size]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_cases.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "selection_method": "issue_only_project_embedding_similarity_v1",
        "skill_bank_path": str(Path(args.skill_bank_path).resolve()),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "excluded_instance_ids": sorted(excluded),
        "selected": [
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "project_similarity": row["project_similarity"],
                "selection_project_skill_id": row["selection_project_skill_id"],
            }
            for row in selected
        ],
    }
    (out_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--skill-bank-path", required=True)
    parser.add_argument("--embedding-cache-path", required=True)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Deterministically score only the first N eligible instance IDs (0 means all).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exclude-instance-ids-file", default="")
    parser.add_argument("--swe-explore-dataset", default=SWE_EXPLORE_DATASET)
    parser.add_argument("--swe-verified-dataset", default=SWE_VERIFIED_DATASET)
    parser.add_argument("--swe-explore-split", default="train")
    parser.add_argument("--swe-verified-split", default="test")
    return parser


def _skill_vectors(bank: Any, skills: list[Any]) -> dict[str, list[float]]:
    from evolutefl.skills.embedding_store import SkillEmbeddingStore

    store = SkillEmbeddingStore(bank.embedding_config.cache_path, model_id=bank.embedding_config.model_id)
    vectors, _ = store.ensure_skill_embeddings(
        skills,
        bank.embedding_client,
        task=bank.embedding_config.document_task,
        batch_size=bank.embedding_config.batch_size,
    )
    return vectors


def _score_cases(
    cases: list[dict[str, Any]],
    skills: list[Any],
    skill_vectors: dict[str, list[float]],
    bank: Any,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    queries = [query_embedding_text(str(case["repo"]), str(case["problem_statement"])) for case in cases]
    query_vectors: list[list[float]] = []
    for start in range(0, len(queries), max(1, batch_size)):
        query_vectors.extend(
            bank.embedding_client.embed_texts(
                queries[start : start + max(1, batch_size)],
                task=bank.embedding_config.query_task,
            )
        )
    scored: list[dict[str, Any]] = []
    for case, vector in zip(cases, query_vectors):
        best = max(
            (
                (cosine_similarity(vector, skill_vectors[skill.skill_id]), skill)
                for skill in skills
                if skill.skill_id in skill_vectors
            ),
            default=(0.0, None),
            key=lambda pair: pair[0],
        )
        score, skill = best
        scored.append(
            {
                **case,
                "project_similarity": float(score),
                "selection_project_skill_id": skill.skill_id if skill is not None else None,
            }
        )
    return scored


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


def _read_ids(path: str) -> list[str]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--exclude-instance-ids-file must contain a JSON list.")
    return [str(value) for value in data]


if __name__ == "__main__":
    raise SystemExit(main())
