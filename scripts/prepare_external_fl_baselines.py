"""Freeze patch-free RQ1 inputs for external fault-localization baselines."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924"
SOURCES = ROOT / "runs/external_baseline_sources"
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
DEPENDENCIES = {
    "LocAgent": ("datasets", "litellm", "faiss", "bm25s", "llama_index", "networkx"),
    "RGFL": ("datasets", "openai", "anthropic", "libcst", "swebench", "unidiff"),
    "CoSIL": ("datasets", "litellm", "pandas", "tiktoken", "libcst"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run: Path = RUN, output: Path = OUTPUT, sources: Path = SOURCES) -> dict:
    protocol_path = run / "protocol.json"
    cases_path = run / "evaluation_cases.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    target = protocol["evaluation_target"]
    if len(cases) != target or target != 500:
        raise ValueError("Evaluation manifest does not match the frozen 500-case protocol")
    ids = [case["instance_id"] for case in cases]
    if len(set(ids)) != target:
        raise ValueError("Duplicate evaluation case IDs")

    public = []
    for case in cases:
        if any(not isinstance(case.get(key), str) or not case[key].strip()
               for key in PUBLIC_FIELDS):
            raise ValueError(f"Missing public field in {case.get('instance_id')}")
        if case["created_at"] < protocol["test_created_from"]:
            raise ValueError(f"Temporal boundary violated by {case['instance_id']}")
        public.append({key: case[key] for key in PUBLIC_FIELDS})

    output.mkdir(parents=True, exist_ok=True)
    public_path = output / "evaluation_public.jsonl"
    public_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public),
        encoding="utf-8",
    )

    source_audit = {}
    for name, modules in DEPENDENCIES.items():
        checkout = sources / name
        commit = None
        if (checkout / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            commit = result.stdout.strip()
        source_audit[name] = {
            "commit": commit,
            "dependencies_available": {
                module: importlib.util.find_spec(module) is not None for module in modules
            },
        }

    report = {
        "status": "input_prepared",
        "evaluation_count": len(public),
        "evaluation_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "protocol_sha256": sha256(protocol_path),
        "private_evaluation_sha256": sha256(cases_path),
        "public_manifest": str(public_path),
        "public_manifest_sha256": sha256(public_path),
        "public_fields": list(PUBLIC_FIELDS),
        "excluded_fields": ["patch", "test_patch", "function_ground_truth", "FAIL_TO_PASS", "PASS_TO_PASS"],
        "sources": source_audit,
        "note": "This preflight does not run or reproduce an external baseline.",
    }
    (output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    args = parser.parse_args()
    report = prepare(args.run, args.output, args.sources)
    print(json.dumps(report, ensure_ascii=False, indent=2))
