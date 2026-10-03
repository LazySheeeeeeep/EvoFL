"""Build lessons from historical issue, oracle repair functions, and patch only.

This intentionally excludes Explorer trajectories. Oracle functions and patches
come only from the frozen 400-case acquisition set, never evaluation cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_rq1_memory_baselines import EXPANDED, HISTORICAL, OUTPUT, read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient


PROMPT = """You are preparing a short reusable function-level fault-localization lesson from one historical issue and its known repair location. You do not have an investigation trajectory.
Use the issue, the repaired function identities, and the historical patch to infer a useful investigation cue for a future, similar issue. Return exactly one JSON object with keys "applicability" and "lesson".
The lesson should tell a future investigator what to inspect or compare, not name the repaired function, copy patch text, or claim an investigation step was observed. If there is no transferable cue, return an empty lesson string. Keep the lesson concise.
"""
AUDIT_CACHE = EXPANDED.parent / "rq1_expanded400_eval500_deepseek_20260922" / "function_audit"


def load_oracle_evidence(episode: dict, case: dict, source: Path,
                         audit_cache: Path = AUDIT_CACHE) -> dict:
    cid = episode["instance_id"]
    path = source / "training" / cid / "evolution" / "trajectory_evidence.json"
    if path.is_file():
        return {**read_json(path), "ground_truth_source": "evolution_evidence"}
    functions = case.get("function_ground_truth")
    if functions:
        origin = "frozen_training_manifest"
    else:
        mapping = read_json(audit_cache / (cid + ".json"))
        if (mapping.get("instance_id") != cid or mapping.get("repo") != case["repo"]
                or mapping.get("base_commit") != case["base_commit"]
                or mapping.get("patch_sha256") != hashlib.sha256(
                    case["patch"].replace("\r\n", "\n").strip().encode()).hexdigest()):
            raise ValueError(f"Oracle audit mismatch: {cid}")
        for item in mapping.get("sources", []):
            cached = Path(item["cache_path"])
            if hashlib.sha256(cached.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Oracle source cache changed: {cid}")
        functions = mapping.get("existing_function_targets")
        origin = "audited_patch_function_mapping"
    return {"repo": case["repo"], "problem_statement": episode["issue"],
            "ground_truth": {"functions": functions, "patch": case["patch"]},
            "ground_truth_source": origin}


def oracle_input(episode: dict, case: dict, evidence: dict) -> dict:
    cid = episode["instance_id"]
    if case["instance_id"] != cid or evidence.get("repo") != episode["repo"]:
        raise ValueError(f"Oracle acquisition identity mismatch: {cid}")
    if evidence.get("problem_statement") != episode["issue"]:
        raise ValueError(f"Oracle issue mismatch: {cid}")
    truth = evidence.get("ground_truth") or {}
    functions = truth.get("functions")
    if not isinstance(functions, list) or not functions or any(
        not isinstance(name, str) or not name.strip() for name in functions
    ):
        raise ValueError(f"Missing oracle functions: {cid}")
    if truth.get("patch") != case["patch"]:
        raise ValueError(f"Oracle patch mismatch: {cid}")
    return {"repo": episode["repo"], "issue": episode["issue"],
            "oracle_repaired_functions": functions,
            "historical_repair_patch": case["patch"]}


def reflect_once(client, episode: dict, case: dict, evidence: dict) -> dict:
    payload = oracle_input(episode, case, evidence)
    response = client.chat(messages=[{"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        response_format={"type": "json_object"}, temperature=0, max_tokens=4096)
    output = json.loads(response.get("content") or "{}")
    if not isinstance(output, dict) or not all(isinstance(output.get(key), str)
                                                for key in ("applicability", "lesson")):
        raise ValueError("Oracle reflection did not return applicability and lesson")
    return {"instance_id": episode["instance_id"], "repo": episode["repo"],
            "issue": episode["issue"], "applicability": output["applicability"].strip(),
            "lesson": output["lesson"].strip(), "source_skill_conditioned": False,
            "ground_truth_source": evidence["ground_truth_source"],
            "source_episode_sha256": hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest(),
            "source_oracle_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def build(*, preparation: Path = OUTPUT, expanded: Path = EXPANDED,
          historical: Path = HISTORICAL, limit: int | None = None,
          case_id: str | None = None, nonthinking: bool = False,
          dry_run: bool = False, client=None) -> dict:
    summary = read_json(preparation / "preparation_summary.json")
    if summary.get("status") != "complete" or summary.get("acquisition_target") != 400:
        raise ValueError("Frozen 400-case acquisition is incomplete")
    for name, expected in summary["manifest_hashes"].items():
        if hashlib.sha256((expanded / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen RQ1 manifest changed: " + name)
    if hashlib.sha256((preparation / "episodes.jsonl").read_bytes()).hexdigest() != summary["prepared_hashes"]["episodes.jsonl"]:
        raise ValueError("Prepared acquisition episodes changed")
    episodes = [json.loads(line) for line in (preparation / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    originals = read_json(expanded / "original_training.json")
    additions = read_json(expanded / "training_additions.json")
    cases = {case["instance_id"]: (case, historical if index < len(originals) else expanded)
             for index, case in enumerate([*originals, *additions])}
    if len(episodes) != 400 or len(cases) != 400 or {row["instance_id"] for row in episodes} != set(cases):
        raise ValueError("Oracle acquisition cases do not match the frozen manifest")
    if case_id is not None and limit is not None:
        raise ValueError("Choose either case_id or limit")
    selected = ([row for row in episodes if row["instance_id"] == case_id]
                if case_id is not None else episodes if limit is None else episodes[:limit])
    if not selected:
        raise ValueError("No acquisition episode selected")
    inputs = []
    for episode in selected:
        cid = episode["instance_id"]
        case, source = cases[cid]
        inputs.append((episode, case, load_oracle_evidence(episode, case, source)))
        oracle_input(*inputs[-1])
    output_dir = preparation / "oracle_function_reflections"
    if dry_run:
        return {"status": "dry_run", "available": len(episodes), "selected": len(selected),
                "pending": sum(not (output_dir / (row["instance_id"] + ".json")).exists() for row in selected),
                "model_calls": 0}
    if client is None:
        llm = dict(read_json(expanded / "config.json")["llm"])
        if nonthinking:
            llm["extra_body"] = {**llm.get("extra_body", {}), "thinking": {"type": "disabled"}}
        client = OpenAICompatibleClient.from_config(llm)
    completed, failed = 0, []
    for episode, case, evidence in inputs:
        cid = episode["instance_id"]
        path = output_dir / (cid + ".json")
        if path.exists():
            existing = read_json(path)
            if existing.get("instance_id") != cid or not isinstance(existing.get("lesson"), str):
                raise ValueError(f"Invalid oracle reflection checkpoint: {cid}")
            completed += 1
            continue
        try:
            result = reflect_once(client, episode, case, evidence)
        except Exception as exc:
            failed.append({"instance_id": cid, "error": str(exc)})
            continue
        write_json(path, result)
        completed += 1
    report = {"status": "complete" if completed == len(episodes) == len(selected) and not failed else "provisional",
              "selected": len(selected), "completed": completed, "failed": failed,
              "acquisition_target": 400, "nonthinking": nonthinking,
              "data_lineage_warning": "Oracle repaired functions and patch are acquisition-only privileged information; evaluation cases never supply them."}
    write_json(preparation / "oracle_function_reflection_progress.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=OUTPUT)
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--nonthinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(preparation=args.preparation, expanded=args.expanded,
                           historical=args.historical, limit=args.limit,
                           case_id=args.case_id, nonthinking=args.nonthinking,
                           dry_run=args.dry_run), indent=2))
