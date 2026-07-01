from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = build_parser().parse_args()
    seen = load_seen(args.exclude_selected_cases)
    local_images = available_images()
    rows = list(load_rows(args.dataset_name, args.split))
    random.Random(args.seed).shuffle(rows)

    selected: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id or instance_id in seen:
            continue
        image_name = row.get("image_name", "")
        if local_images and image_name not in local_images and f"{image_name}:latest" not in local_images:
            continue
        case = {
            "instance_id": instance_id,
            "repo": row.get("repo", ""),
            "image_name": image_name,
            "problem_statement": row.get("problem_statement", ""),
            "patch": row.get("patch", ""),
            "base_commit": row.get("repo", "").split(".")[-1] if row.get("repo") else "",
        }
        if image_name not in used_images:
            selected.append(case)
            used_images.add(image_name)
        else:
            fallback.append(case)
        if len(selected) >= args.sample_size:
            break

    selected = (selected + fallback)[: args.sample_size]
    if len(selected) < args.sample_size:
        raise RuntimeError(f"Only selected {len(selected)} cases; requested {args.sample_size}.")

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(out), "selected_count": len(selected), "seen_count": len(seen)}, indent=2))
    for case in selected:
        print(case["instance_id"], case["image_name"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-smith-py")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--exclude-selected-cases", action="append", default=[])
    return parser


def load_seen(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        for case in json.loads(path.read_text(encoding="utf-8-sig")):
            instance_id = case.get("instance_id")
            if instance_id:
                seen.add(instance_id)
    return seen


def load_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment guard.
        raise RuntimeError("The datasets package is required in WSL.") from exc
    return list(load_dataset(dataset_name, split=split))


def available_images() -> set[str]:
    images = docker_images()
    images.update(cached_repo_images())
    return images


def docker_images() -> set[str]:
    try:
        output = subprocess.check_output(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], text=True)
    except Exception:
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def cached_repo_images() -> set[str]:
    images: set[str] = set()
    for repo_dir in ROOT.glob("runs/*/repos/*"):
        if not repo_dir.is_dir() or not any(repo_dir.iterdir()):
            continue
        image = image_name_from_safe_name(repo_dir.name)
        images.add(image)
        images.add(f"{image}:latest")
    return images


def image_name_from_safe_name(value: str) -> str:
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
