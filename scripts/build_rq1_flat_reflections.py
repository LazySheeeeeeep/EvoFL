"""Build a one-call-per-case flat-reflection memory from audited acquisitions.

Unlike EvoluteFL v5, this arm uses no supplementary investigation, evidence-gap
search, skill taxonomy, or evolving bank. Outputs stay outside the active run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_rq1_memory_baselines import EXPANDED, OUTPUT, read_json, write_json
from evolutefl.llm.client import OpenAICompatibleClient


PROMPT = """You are preparing a short reusable fault-localization lesson from one historical case.
Compare the issue, the agent's actual investigation and prediction, and the historical repair.
Return exactly one JSON object with keys "applicability" and "lesson".
The lesson should describe a useful investigation decision, not the repaired function name,
patch text, or a claim that an unobserved step was performed. If the record supports no
transferable lesson, return an empty lesson string. Keep the lesson concise.
"""


def flat_reflection_input(episode: dict, case: dict) -> dict:
    if episode["instance_id"] != case["instance_id"]:
        raise ValueError("Acquisition case mismatch")
    return {
        "repo": episode["repo"],
        "issue": episode["issue"],
        "investigation": episode["investigation"],
        "prediction": episode["predicted_functions"],
        "final_summary": episode["final_summary"],
        "historical_repair_patch": case["patch"],
    }


def reflect_once(client, episode: dict, case: dict) -> dict:
    response = client.chat(messages=[{"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(flat_reflection_input(episode, case), ensure_ascii=False)}],
        response_format={"type": "json_object"}, temperature=0, max_tokens=4096)
    output = json.loads(response.get("content") or "{}")
    if not isinstance(output, dict) or not all(isinstance(output.get(key), str)
                                                for key in ("applicability", "lesson")):
        raise ValueError("Flat reflection did not return applicability and lesson")
    return {"instance_id": episode["instance_id"], "repo": episode["repo"], "issue": episode["issue"],
            "applicability": output["applicability"].strip(), "lesson": output["lesson"].strip(),
            "source_skill_conditioned": episode["skill_conditioned"],
            "source_episode_sha256": hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest()}


def build(*, preparation: Path = OUTPUT, expanded: Path = EXPANDED,
          limit: int | None = None, case_id: str | None = None,
          nonthinking: bool = False, dry_run: bool = False, client=None) -> dict:
    summary = read_json(preparation / "preparation_summary.json")
    expected = {name: hashlib.sha256((expanded / name).read_bytes()).hexdigest()
                for name in summary["manifest_hashes"]}
    if expected != summary["manifest_hashes"]:
        raise ValueError("Frozen RQ1 acquisition manifest changed")
    if hashlib.sha256((preparation / "episodes.jsonl").read_bytes()).hexdigest() != summary["prepared_hashes"]["episodes.jsonl"]:
        raise ValueError("Prepared episodes changed")
    episodes = [json.loads(line) for line in (preparation / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    originals = read_json(expanded / "original_training.json")
    additions = read_json(expanded / "training_additions.json")
    cases = {case["instance_id"]: case for case in [*originals, *additions]}
    if len(cases) != 400 or any(row["instance_id"] not in cases for row in episodes):
        raise ValueError("Prepared episodes do not match acquisition manifest")
    if case_id is not None and limit is not None:
        raise ValueError("Choose either case_id or limit")
    selected = ([row for row in episodes if row["instance_id"] == case_id]
                if case_id is not None else episodes if limit is None else episodes[:limit])
    if not selected:
        raise ValueError("No acquisition episode selected")
    if dry_run:
        return {"status": "dry_run", "available": len(episodes), "selected": len(selected),
                "pending": sum(not (preparation / "flat_reflections" / (row["instance_id"] + ".json")).exists()
                               for row in selected), "model_calls": 0}
    if client is None:
        config = read_json(expanded / "config.json")
        llm = dict(config["llm"])
        if nonthinking:
            llm["extra_body"] = {**llm.get("extra_body", {}),
                                 "thinking": {"type": "disabled"}}
        client = OpenAICompatibleClient.from_config(llm)
    completed, failed = 0, []
    for episode in selected:
        cid = episode["instance_id"]
        path = preparation / "flat_reflections" / (cid + ".json")
        if path.exists():
            existing = read_json(path)
            if existing.get("instance_id") != cid or not isinstance(existing.get("lesson"), str):
                raise ValueError(f"Invalid reflection checkpoint: {cid}")
            completed += 1
            continue
        try:
            result = reflect_once(client, episode, cases[cid])
        except Exception as exc:
            failed.append({"instance_id": cid, "error": str(exc)})
            continue
        write_json(path, result)
        completed += 1
    report = {"status": "complete" if completed == len(episodes) == len(selected)
              and summary["status"] == "complete"
              else "provisional", "selected": len(selected), "completed": completed, "failed": failed,
              "acquisition_target": summary["acquisition_target"],
              "nonthinking": nonthinking, "case_id": case_id,
              "data_lineage_warning": summary["data_lineage_warning"]}
    write_json(preparation / "flat_reflection_progress.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=OUTPUT)
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--nonthinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(preparation=args.preparation, expanded=args.expanded,
                           limit=args.limit, case_id=args.case_id,
                           nonthinking=args.nonthinking, dry_run=args.dry_run), indent=2))
