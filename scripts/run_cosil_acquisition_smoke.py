"""Run an isolated CoSIL file/function smoke on one acquisition case."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/external_baseline_sources/CoSIL"
RUNTIME = ROOT / "runs/external_baseline_runtime/cosil_vendor"
SMOKE = ROOT / "runs/external_fl_baseline_preflight_20260924/cosil_acquisition_smoke"
MODEL = "openai/deepseek-v4-flash"


def has_prediction(folder: Path, case_id: str, stage: str) -> bool:
    output_name = "loc_outputs.jsonl" if stage == "file" else "loc_outputs_func.jsonl"
    output = folder / output_name
    if not output.is_file():
        return False
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    match = [row for row in rows if row.get("instance_id") == case_id]
    field = "found_files" if stage == "file" else "found_related_locs"
    return len(match) == 1 and bool(match[0].get(field))


def run(smoke: Path = SMOKE, attempt: str = "nonthinking_v1") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", attempt):
        raise ValueError("Invalid attempt name")
    case = json.loads((smoke / "case.jsonl").read_text(encoding="utf-8"))
    case_id = case["instance_id"]
    if case_id != "psf__requests-1376":
        raise ValueError("Smoke runner is restricted to the audited acquisition case")
    env = os.environ.copy()
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY explicitly for this smoke process")
    env["OPENAI_API_KEY"] = key
    env["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
    env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]
    env["PROJECT_FILE_LOC"] = str(smoke / "repo_structures")
    env["PYTHONPATH"] = os.pathsep.join((str(SOURCE), str(SOURCE / "CoSIL/fl"),
                                         str(RUNTIME), env.get("PYTHONPATH", "")))
    attempt_dir = smoke / "attempts" / attempt
    stages = [
        ("file", "CoSIL.fl.CoSIL_localize_file", ["--file_level"]),
        ("function", "CoSIL.fl.CoSIL_localize_func", [
            "--loc_file", str(attempt_dir / "file/loc_outputs.jsonl"), "--temperature", "0.85",
        ]),
    ]
    report = {"case_id": case_id, "model": MODEL, "attempt": attempt,
              "reasoning_effort": "none", "stages": []}
    for stage, module, extra in stages:
        folder = attempt_dir / stage
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", module, "--dataset", str(smoke / "case.jsonl"),
                   "--output_folder", str(folder), "--model", MODEL,
                   "--num_threads", "1", "--target_id", case_id, "--skip_existing", *extra]
        log_path = folder / "smoke.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=smoke, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=900, check=False)
            status = "completed" if result.returncode == 0 else "failed"
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            exit_code = None
        if status == "completed" and not has_prediction(folder, case_id, stage):
            status = "empty_prediction"
        row = {"stage": stage, "status": status, "exit_code": exit_code, "log": str(log_path)}
        report["stages"].append(row)
        (attempt_dir / "smoke_result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if status != "completed":
            break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, default=SMOKE)
    parser.add_argument("--attempt", default="nonthinking_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.smoke, args.attempt), indent=2))
