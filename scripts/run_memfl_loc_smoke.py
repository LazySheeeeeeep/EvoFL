"""Minimal test-free MemFL-Loc smoke on one acquisition/evaluation pair.

This validates external-memory construction and injection, not benchmark
accuracy. The original Java/Defects4J failing-test, stack-trace, and coverage
inputs are intentionally omitted.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evolutefl.explorer import run_explorer
from evolutefl.json_utils import extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from rq1_memory_runtime import MemoryInjectedClient
from run_swe_explore_v5_stratified import safe_archive_filter, strict_metrics

EXPANDED = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
OUTPUT = ROOT / "runs/memfl_loc_smoke_20260930"
ACQUISITION_ID = "astropy__astropy-8005"
EVALUATION_ID = "astropy__astropy-13033"


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def extract_archive(case_id: str, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = EXPANDED / "sources" / f"{case_id}.tar.gz"
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter=safe_archive_filter)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one repository root")
    return roots[0]


def case_from_manifest(path: Path, case_id: str) -> dict:
    matches = [row for row in read(path, []) if row["instance_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Case not uniquely found in {path.name}: {case_id}")
    return matches[0]


def training_evidence(case: dict) -> tuple[dict, list[dict]]:
    result = read(EXPANDED / "training" / ACQUISITION_ID / "explorer" / "result.json", {})
    trajectory = []
    path = EXPANDED / "training" / ACQUISITION_ID / "explorer" / "trajectory.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") in {"assistant", "investigation_action", "finish"}:
            trajectory.append({
                key: row.get(key)
                for key in ("event", "step", "content", "tool", "arguments", "ranked_functions", "summary")
                if row.get(key) is not None
            })
    return result, trajectory[-20:]


def build_static_memory(root: Path, case: dict, result: dict, client) -> dict:
    predicted = result.get("ranked_functions") or []
    selected_paths = []
    for identity in predicted:
        path = identity.split("::", 1)[0]
        if path not in selected_paths:
            selected_paths.append(path)
    excerpts = []
    for path in selected_paths[:4]:
        source = root / path
        if source.is_file():
            excerpts.append(f"### {path}\n{source.read_text(encoding='utf-8', errors='replace')[:5000]}")
    tree = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.is_file() and not path.is_symlink()
    )[:400]
    prompt = (
        "Construct static external memory for software fault localization. "
        "Summarize the project and the reusable responsibilities or boundaries "
        "of the selected components. Do not mention a repair patch or a target function. "
        "Return JSON with keys project_summary and component_summaries. "
        "component_summaries must be a list of objects with scope, purpose, and boundary.\n\n"
        + json.dumps({"repo": case["repo"], "issue": case["problem_statement"],
                      "python_files": tree, "selected_excerpts": excerpts},
                     ensure_ascii=False)
    )
    response = client.chat(
        messages=[{"role": "system", "content": "You build reusable project knowledge."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, tool_choice="none", temperature=0,
        max_tokens=4096,
    )
    parsed = extract_json_object(response.get("content") or "{}")
    if not isinstance(parsed.get("project_summary"), str) or not parsed["project_summary"].strip():
        raise ValueError("Static MemFL memory is incomplete")
    return parsed


def build_dynamic_memory(case: dict, result: dict, trajectory: list[dict], client) -> dict:
    prompt = (
        "Derive one reusable dynamic debugging-guidance memory from a previous "
        "fault-localization attempt. Do not include ground-truth patches. "
        "Return JSON with keys issue_signature, symptom_pattern, "
        "investigation_guidance, and ranking_implication.\n\n"
        + json.dumps({"issue": case["problem_statement"],
                      "final_summary": result.get("final_summary", ""),
                      "predicted_functions": result.get("ranked_functions", []),
                      "investigation": trajectory}, ensure_ascii=False)
    )
    required = ("issue_signature", "symptom_pattern", "investigation_guidance", "ranking_implication")
    for attempt in range(2):
        messages = [
            {"role": "system", "content": "You distill reusable debugging guidance."},
            {"role": "user", "content": prompt},
        ]
        if attempt:
            messages.append({
                "role": "user",
                "content": (
                    "The previous response was incomplete. Return exactly one JSON object "
                    "with non-empty string fields: issue_signature, symptom_pattern, "
                    "investigation_guidance, ranking_implication."
                ),
            })
        response = client.chat(
            messages=messages,
            response_format={"type": "json_object"}, tool_choice="none", temperature=0,
            max_tokens=4096,
        )
        write_json(OUTPUT / "dynamic_response_attempt.json", response)
        parsed = extract_json_object(response.get("content") or "{}")
        if not isinstance(parsed, dict):
            continue
        for holder in (
            parsed,
            parsed.get("dynamic_memory") if isinstance(parsed.get("dynamic_memory"), dict) else {},
            parsed.get("guidance") if isinstance(parsed.get("guidance"), dict) else {},
            parsed.get("analysis") if isinstance(parsed.get("analysis"), dict) else {},
        ):
            if all(isinstance(holder.get(key), str) and holder[key].strip() for key in required):
                return {key: holder[key].strip() for key in required}
    raise ValueError(
        "Dynamic MemFL guidance is incomplete; inspect dynamic_response_attempt.json"
    )


def run_smoke() -> dict:
    acquisition = case_from_manifest(EXPANDED / "training_additions.json", ACQUISITION_ID)
    evaluation = case_from_manifest(EXPANDED / "evaluation_cases.json", EVALUATION_ID)
    config = copy.deepcopy(read(EXPANDED / "config.json"))
    client = OpenAICompatibleClient.from_config(config["llm"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="memfl-smoke-", dir=OUTPUT) as temporary:
        root = extract_archive(ACQUISITION_ID, Path(temporary) / "acquisition")
        training_result, trajectory = training_evidence(acquisition)
        static_path = OUTPUT / "static_memory.json"
        static_memory = read(static_path)
        if static_memory is None:
            static_memory = build_static_memory(root, acquisition, training_result, client)
            write_json(static_path, static_memory)
        dynamic_memory = build_dynamic_memory(acquisition, training_result, trajectory, client)
    memory = {
        "memory_type": "MemFL-Loc external memory (SWE test-free adaptation)",
        "static_memory": static_memory,
        "dynamic_memory": dynamic_memory,
        "source_instance_id": ACQUISITION_ID,
        "source_repo": acquisition["repo"],
        "phase_instruction": (
            "Use the memory in three phases: bug review generation, code condensation, "
            "and fault confirmation. Verify every candidate against the current repository."
        ),
    }
    write_json(OUTPUT / "memory.json", memory)

    evaluation_dir = OUTPUT / "evaluation" / EVALUATION_ID
    work = OUTPUT / "work" / EVALUATION_ID
    root = extract_archive(EVALUATION_ID, work)
    config["skill_bank"]["path"] = str(EXPANDED / "empty_skills.jsonl")
    config["skill_bank"]["enabled_skill_types"] = []
    guided = MemoryInjectedClient(
        client,
        instance_id=EVALUATION_ID,
        issue=evaluation["problem_statement"],
        memory=memory,
    )
    result = run_explorer(
        task={
            "instance_id": EVALUATION_ID,
            "repo": evaluation["repo"],
            "base_commit": evaluation["base_commit"],
            "bug_report": evaluation["problem_statement"],
            "repo_path": str(root),
            "run_dir": str(evaluation_dir),
        },
        config=config,
        llm_client=guided,
        skill_bank=make_skill_bank(config),
    )
    predictions = list(dict.fromkeys(result.get("ranked_functions") or []))[:5]
    report = {
        "status": "completed" if result.get("status") == "completed" else "failed",
        "acquisition_case": ACQUISITION_ID,
        "evaluation_case": EVALUATION_ID,
        "memory_injection_count": guided.injection_count,
        "explorer_status": result.get("status"),
        "predictions": predictions,
        "metrics": strict_metrics(predictions, evaluation["function_ground_truth"]),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "note": "MemFL-Loc smoke validates memory construction and injection; it is not a benchmark result.",
    }
    write_json(OUTPUT / "smoke_result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    print(json.dumps(run_smoke(), indent=2))
