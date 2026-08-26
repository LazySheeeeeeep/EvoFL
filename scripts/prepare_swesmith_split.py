"""Create reproducible, disjoint SWE-smith manifests for a train/evaluation run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_swesmith_case_by_case import select_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-candidates", type=int, default=50)
    parser.add_argument("--evaluation-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--prefer-local-images", action="store_true")
    args = parser.parse_args()

    total = args.training_candidates + args.evaluation_size
    cases = select_cases(
        total,
        args.seed,
        "SWE-bench/SWE-smith-py",
        "train",
        args.prefer_local_images,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_candidates.json").write_text(
        json.dumps(cases[: args.training_candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_cases.json").write_text(
        json.dumps(cases[args.training_candidates :], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"training_candidates": args.training_candidates, "evaluation_cases": args.evaluation_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
