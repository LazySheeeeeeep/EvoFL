"""Check RGFL's author file-stage path on one audited base source without API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "runs/external_baseline_sources/RGFL"
INPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/astropy__astropy-13033"
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/mock_smoke"
PUBLIC_FIELDS = {"instance_id", "repo", "base_commit", "problem_statement"}


def load_public_case(prepared: Path) -> dict:
    prepared = prepared.resolve(strict=True)
    rows = [json.loads(line) for line in (prepared / "case.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or set(rows[0]) != PUBLIC_FIELDS:
        raise ValueError("RGFL smoke requires one patch-free public case")
    case = rows[0]
    structure = json.loads((prepared / "repo_structures" /
                            f"{case['instance_id']}.json").read_text(encoding="utf-8"))
    if any(structure.get(key) != case[key] for key in
           ("instance_id", "repo", "base_commit")):
        raise ValueError("RGFL base source does not match public case")
    return case


def smoke(prepared: Path = INPUT, output: Path = OUTPUT) -> dict:
    prepared = prepared.resolve(strict=True)
    output = output.resolve()
    case = load_public_case(prepared)
    os.environ["PROJECT_FILE_LOC"] = str(prepared / "repo_structures")
    sys.path.insert(0, str(AUTHOR))
    from rgfl.fl.localize import localize_instance

    output.mkdir(parents=True, exist_ok=True)
    (output / "localization_logs").mkdir(exist_ok=True)
    result_path = output / "loc_outputs.jsonl"
    if result_path.exists():
        raise ValueError("Mock output already exists; use a fresh output directory")
    args = SimpleNamespace(target_id=case["instance_id"], output_folder=str(output),
                           output_file=str(result_path), file_level=True,
                           related_level=False, fine_grain_line_level=False,
                           mock=True, model="deepseek-v4-flash", backend="deepseek")
    localize_instance(case, args, [case], None, set())
    result = json.loads(result_path.read_text(encoding="utf-8").strip())
    if result.get("instance_id") != case["instance_id"] or "prompt" not in result.get("file_traj", {}):
        raise ValueError("RGFL did not construct a file-stage prompt from the base source")
    report = {"status": "mock_completed", "instance_id": case["instance_id"],
              "prompt_chars": len(result["file_traj"]["prompt"]),
              "model_calls": 0, "output": str(result_path)}
    (output / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(smoke(args.prepared, args.output), indent=2))
