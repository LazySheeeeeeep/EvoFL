"""Run CoSIL on the frozen, patch-free RQ1 pilot in an isolated directory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .normalize_cosil_functions import normalize
    from .prepare_cosil_acquisition_smoke import PUBLIC_FIELDS
    from .run_cosil_acquisition_smoke import MODEL, RUNTIME, SOURCE, has_prediction
else:
    from normalize_cosil_functions import normalize
    from prepare_cosil_acquisition_smoke import PUBLIC_FIELDS
    from run_cosil_acquisition_smoke import MODEL, RUNTIME, SOURCE, has_prediction


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/external_fl_baseline_preflight_20260924/cosil_pilot"
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/cosil_pilot_results"
EVALUATION = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922/evaluation_cases.json"


def read_json(path: Path) -> dict | list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def structure_identity(path: Path) -> dict:
    # Preparation writes identity fields before the multi-megabyte structure.
    with path.open("rb") as handle:
        header = handle.read(2048).decode("utf-8")
    marker = '"structure":'
    if marker not in header:
        raise ValueError(f"Missing source structure header: {path}")
    return json.loads(header.split(marker, 1)[0] + marker + "null}")


def selected_cases(pilot: Path, limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("Pilot limit must be positive")
    rows = read_jsonl(pilot / "selected_cases.jsonl")[:limit]
    if len(rows) != limit or len({row.get("instance_id") for row in rows}) != limit:
        raise ValueError("Missing or duplicate selected pilot case")
    for row in rows:
        case_id = row["instance_id"]
        if set(row) != set(PUBLIC_FIELDS) or not all(isinstance(value, str) and value.strip()
                                                    for value in row.values()):
            raise ValueError(f"Invalid patch-free pilot row: {case_id}")
        if Path(case_id).name != case_id or "/" in case_id or "\\" in case_id:
            raise ValueError("Unsafe case ID")
        prepared = pilot / "prepared" / case_id / "case.jsonl"
        if read_jsonl(prepared) != [row]:
            raise ValueError(f"Prepared public input changed: {case_id}")
        structure = structure_identity(pilot / "repo_structures" / f"{case_id}.json")
        if any(structure.get(field) != row[field] for field in
               ("instance_id", "repo", "base_commit")):
            raise ValueError(f"Prepared source structure changed: {case_id}")
    return rows


def author_environment(pilot: Path) -> dict[str, str]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY must be provided to this process")
    env = os.environ.copy()
    env.update({
        "OPENAI_API_KEY": key,
        "OPENAI_API_BASE": "https://api.deepseek.com/v1",
        "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        "PROJECT_FILE_LOC": str(pilot / "repo_structures"),
        "PYTHONPATH": os.pathsep.join((str(SOURCE), str(SOURCE / "CoSIL/fl"),
                                       str(RUNTIME), env.get("PYTHONPATH", ""))),
    })
    return env


def run_stage(case_id: str, case_file: Path, pilot: Path, folder: Path,
              stage: str, env: dict[str, str]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    module = ("CoSIL.fl.CoSIL_localize_file" if stage == "file"
              else "CoSIL.fl.CoSIL_localize_func")
    extra = (["--file_level"] if stage == "file" else
             ["--loc_file", str(folder.parent / "file/loc_outputs.jsonl"),
              "--temperature", "0.85"])
    command = [sys.executable, "-m", module, "--dataset", str(case_file),
               "--output_folder", str(folder), "--model", MODEL,
               "--num_threads", "1", "--target_id", case_id, *extra]
    log_path = folder / "author.log"
    with log_path.open("a", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, cwd=pilot, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=900,
                                    check=False)
            status = "completed" if result.returncode == 0 else "author_failed"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
    if status == "completed" and not has_prediction(folder, case_id, stage):
        status = "empty_prediction"
    return {"stage": stage, "status": status, "exit_code": code,
            "log": str(log_path)}


def run_case(row: dict, pilot: Path, output: Path, env: dict[str, str],
             reuse_pilot: Path | None = None, reuse_results: Path | None = None) -> dict:
    case_id = row["instance_id"]
    folder = output / "cases" / case_id
    result_path = folder / "result.json"
    if result_path.exists():
        return read_json(result_path)
    if folder.exists() and any(folder.iterdir()):
        return {"instance_id": case_id, "status": "interrupted_attempt",
                "note": "Use a new output directory; existing author output was preserved."}
    folder.mkdir(parents=True, exist_ok=True)
    stages = []
    author_output = folder / "function" / "loc_outputs_func.jsonl"
    reused = False
    if reuse_pilot is not None and reuse_results is not None:
        prior_case = reuse_results / "cases" / case_id
        if (prior_case / "result.json").exists():
            prior = read_json(prior_case / "result.json")
            missing_normalized_file = (
                prior["status"] == "invalid_author_output"
                and prior.get("error", "").startswith("[Errno 2] No such file or directory:")
                and prior["error"].endswith("/normalized.json'")
                and all(stage.get("status") == "completed" for stage in prior.get("stages", []))
            )
            if prior["status"] not in {"completed", "empty_normalized_prediction"} and not missing_normalized_file:
                raise ValueError(f"Prior CoSIL author run did not complete: {case_id}")
            old_input = reuse_pilot / "prepared" / case_id / "case.jsonl"
            new_input = pilot / "prepared" / case_id / "case.jsonl"
            old_structure = reuse_pilot / "repo_structures" / f"{case_id}.json"
            new_structure = pilot / "repo_structures" / f"{case_id}.json"
            if old_input.read_bytes() != new_input.read_bytes() or old_structure.read_bytes() != new_structure.read_bytes():
                raise ValueError(f"Reused CoSIL input or base source changed: {case_id}")
            author_output = (Path(prior["reused_author_output"])
                             if missing_normalized_file and prior.get("reused_author_output")
                             else prior_case / "function" / "loc_outputs_func.jsonl")
            if missing_normalized_file and hashlib.sha256(author_output.read_bytes()).hexdigest() != prior.get("reused_author_output_sha256"):
                raise ValueError(f"Reused author output hash mismatch: {case_id}")
            stages = prior["stages"]
            reused = True
    if not reused:
        case_file = pilot / "prepared" / case_id / "case.jsonl"
        for stage in ("file", "function"):
            stage_result = run_stage(case_id, case_file, pilot, folder / stage, stage, env)
            stages.append(stage_result)
            if stage_result["status"] != "completed":
                break
    result = {"instance_id": case_id, "status": stages[-1]["status"],
              "stages": stages, "ranked_functions": []}
    if reused:
        result["reused_author_output"] = str(author_output)
        result["reused_author_output_sha256"] = hashlib.sha256(author_output.read_bytes()).hexdigest()
    if result["status"] == "completed":
        try:
            rows = read_jsonl(author_output)
            if len(rows) != 1 or rows[0].get("instance_id") != case_id:
                raise ValueError("Expected one function-level output for this case")
            structure = read_json(pilot / "repo_structures" / f"{case_id}.json")
            mapped = normalize(rows[0], structure["structure"])
            result["ranked_functions"] = mapped["ranked_functions"]
            (folder / "normalized.json").write_text(
                json.dumps(mapped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if not result["ranked_functions"]:
                result["status"] = "empty_normalized_prediction"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result["status"] = "invalid_author_output"
            result["error"] = str(exc)
    folder.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return result


def score(results: list[dict], evaluation: Path) -> list[dict]:
    if __package__:
        from .run_swe_explore_v5_stratified import strict_metrics
    else:
        from run_swe_explore_v5_stratified import strict_metrics

    cases = {row["instance_id"]: row for row in read_json(evaluation)}
    scored = []
    for result in results:
        case = cases[result["instance_id"]]
        scored.append({**result, "metrics": strict_metrics(result.get("ranked_functions", [])[:5],
                                                          case["function_ground_truth"])})
    return scored


def run(pilot: Path = PILOT, output: Path = OUTPUT, evaluation: Path = EVALUATION,
        limit: int = 10, dry_run: bool = False, reuse_pilot: Path | None = None,
        reuse_results: Path | None = None) -> dict:
    pilot = pilot.resolve(strict=True)
    output = output.resolve()
    evaluation = evaluation.resolve()
    if (reuse_pilot is None) != (reuse_results is None):
        raise ValueError("Reuse requires both a pilot and its author results")
    if reuse_pilot is not None:
        reuse_pilot = reuse_pilot.resolve(strict=True)
        reuse_results = reuse_results.resolve(strict=True)
    rows = selected_cases(pilot, limit)
    if dry_run:
        return {"status": "ready", "case_count": len(rows),
                "case_ids": [row["instance_id"] for row in rows], "model_calls": 0}
    env = author_environment(pilot)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        results.append(run_case(row, pilot, output, env, reuse_pilot, reuse_results))
        scored = score(results, evaluation)
        count = len(scored)
        summary = {"method": "CoSIL",
                   "adapter": "deepseek_non_thinking_v1+strict_raw_xml_path_recovery_v1",
                   "processed": count, "target": len(rows),
                   "reused_author_case_count": sum("reused_author_output" in item for item in scored),
                   "statuses": dict(Counter(item["status"] for item in scored)),
                   "top1": sum(item["metrics"]["top1"] for item in scored) / count,
                   "top3": sum(item["metrics"]["top3"] for item in scored) / count,
                   "top5": sum(item["metrics"]["top5"] for item in scored) / count,
                   "mrr": sum(item["metrics"]["mrr"] for item in scored) / count,
                   "cases": scored}
        (output / "comparison_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-pilot", type=Path)
    parser.add_argument("--reuse-results", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pilot, args.output, args.evaluation, args.limit,
                         args.dry_run, args.reuse_pilot, args.reuse_results),
                     ensure_ascii=False, indent=2))
