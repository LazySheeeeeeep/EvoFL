"""Select disjoint SWE-smith cases with an actual local source-repo cache."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _cached_repo_names() -> set[str]:
    names: set[str] = set()
    for path in ROOT.glob("runs/*/repos/*"):
        if path.is_dir() and (path / ".git").is_dir():
            names.add(path.name)
    return names


def _read_instance_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    return {str(row.get("instance_id")) for row in rows if row.get("instance_id")}


def _local_images() -> set[str]:
    completed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to enumerate local Docker images: {completed.stderr.strip()}")
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    cached_names = _cached_repo_names()
    local_images = _local_images()
    excluded = _read_instance_ids(Path(args.exclude_manifest) if args.exclude_manifest else None)
    dataset = load_dataset("SWE-bench/SWE-smith-py", split="train")
    candidates: list[dict[str, object]] = []
    seen_images: set[str] = set()
    for row in dataset:
        instance_id = str(row.get("instance_id") or "")
        image_name = str(row.get("image_name") or "")
        problem_statement = str(row.get("problem_statement") or "").strip()
        if not instance_id or instance_id in excluded or not image_name or not problem_statement:
            continue
        available = _safe_name(image_name) in cached_names or image_name in local_images or f"{image_name}:latest" in local_images
        if not available or image_name in seen_images:
            continue
        candidates.append({
            "instance_id": instance_id,
            "repo": str(row.get("repo") or ""),
            "image_name": image_name,
            "problem_statement": problem_statement,
            "patch": str(row.get("patch") or ""),
            "base_commit": str(row.get("repo") or "").split(".")[-1],
        })
        seen_images.add(image_name)
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.size]
    if len(selected) < args.size:
        raise RuntimeError(
            f"Only found {len(selected)} cached, disjoint SWE-smith cases; requested {args.size}."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "cached_candidates": len(candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
