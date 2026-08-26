from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.config import load_config  # noqa: E402
from evolutefl.skills import make_skill_bank  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect one staged SkillBank retrieval query.")
    parser.add_argument("skill_type", choices=["project_skill", "strategy_skill"])
    parser.add_argument("query")
    parser.add_argument("--config", default="runs/key77_gpt4omini_config_20260615.json")
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"], default="embedding")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--embedding-min-score", type=float, default=0.48)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--skill-bank-path", default="")
    parser.add_argument("--embedding-cache-path", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("skill_bank", {})
    config["skill_bank"]["retrieval_mode"] = args.retrieval_mode
    config["skill_bank"]["embedding_min_score"] = args.embedding_min_score
    if args.skill_bank_path:
        config["skill_bank"]["path"] = args.skill_bank_path
    config.setdefault("embedding", {})
    config["embedding"]["enabled"] = args.retrieval_mode in {"embedding", "hybrid"}
    config["embedding"]["base_url"] = args.embedding_base_url
    if args.embedding_cache_path:
        config["embedding"]["cache_path"] = args.embedding_cache_path

    result = make_skill_bank(config).search_for_stage(
        args.skill_type,
        args.query,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
