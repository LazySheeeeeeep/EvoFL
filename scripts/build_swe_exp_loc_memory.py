"""Build acquisition-only SWE-Exp-Loc experiences from frozen FL trajectories.

This is a localization adaptation of SWE-Exp, not the author's APR run. The
evaluation manifest is read only for identity/temporal separation checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evolutefl.llm.client import OpenAICompatibleClient
from prepare_rq1_memory_baselines import EXPANDED, OUTPUT as PREPARATION, read_json, write_json
from prepare_swe_exp_loc import DESTINATION, literal_prompts


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_PROMPTS = ROOT / "SWE-Exp/moatless/experience/prompts/exp_prompts.py"
FAILED_FL_PROMPT = """You are extracting a localization experience from a failed historical search.
The issue, actual investigation, wrong Top-5 prediction, and historical repair are provided.
Follow SWE-Exp's failed-trajectory perspective and positioning distinction, adapted to
function-level localization rather than code modification. Return one JSON object:
{"perspective": ["a transferable interpretation of the issue"],
 "positioning": ["a transferable way to investigate the missing code location"]}
Describe what the observed search missed and how to investigate it. Avoid copying
the repaired function name or patch text into the reusable experience.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _response_object(client, messages: list[dict]) -> dict:
    response = client.chat(messages=messages, response_format={"type": "json_object"},
                           temperature=0, max_tokens=4096,
                           extra_body={"thinking": {"type": "disabled"}})
    content = response.get("content") or ""
    if not content.strip():
        raw = response.get("raw") or {}
        choice = (raw.get("choices") or [{}])[0]
        usage = raw.get("usage") or {}
        raise ValueError("Empty SWE-Exp-Loc response: " + json.dumps({
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        }))
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("SWE-Exp-Loc model response is not an object")
    return parsed


def _strings(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("SWE-Exp-Loc experience must be text or a list of text")
    return [item.strip() for item in value if item.strip()]


def extract_one(client, episode: dict, case: dict, plan: dict, prompts: dict) -> dict:
    cid = episode["instance_id"]
    if cid != case["instance_id"] or cid != plan["instance_id"]:
        raise ValueError("Acquisition case mismatch")
    if not plan["eligible"] or episode["status"] != "completed":
        raise ValueError("Only completed acquisition trajectories can yield experiences")
    if not episode["issue"].strip() or episode["issue"].strip() != case["problem_statement"].strip():
        raise ValueError("Historical issue is empty or changed")
    if plan["branch"] not in {"success", "failure"}:
        raise ValueError("Invalid FL outcome branch")

    issue_type = _response_object(client, [
        {"role": "system", "content": prompts["issue_type_system_prompt"]},
        {"role": "user", "content": prompts["issue_type_user_prompt"].format(episode["issue"])},
    ])
    if not all(isinstance(issue_type.get(key), str) and issue_type[key].strip()
               for key in ("issue_type", "description")):
        raise ValueError("SWE-Exp issue type is incomplete")

    trajectory = json.dumps({"investigation": episode["investigation"],
                             "predicted_functions": episode["predicted_functions"][:5],
                             "final_summary": episode["final_summary"]}, ensure_ascii=False)
    user = (f"<issue>\n{episode['issue']}\n</issue>\n\n"
            f"<golden_patch>\n{case['patch']}\n</golden_patch>\n\n"
            f"<trajectory>\n{trajectory}\n</trajectory>")
    system = (prompts["encode_success_perspective_system_prompt"]
              if plan["branch"] == "success" else FAILED_FL_PROMPT)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(1, 4):
        experience = _response_object(client, messages)
        try:
            perspective = _strings(experience.get("perspective"))
            positioning = (_strings(experience.get("positioning"))
                           if plan["branch"] == "failure" else [])
            if not perspective and not positioning:
                raise ValueError("SWE-Exp-Loc extraction produced no experience")
            break
        except ValueError as exc:
            if attempt == 3:
                shape = {key: type(value).__name__ for key, value in experience.items()}
                raise ValueError(f"SWE-Exp-Loc experience shape invalid after 3 identical requests: {shape}") from exc
    return {"instance_id": cid, "repo": episode["repo"], "issue": episode["issue"],
            "issue_type": issue_type["issue_type"].strip(),
            "description": issue_type["description"].strip(),
            "flag": plan["branch"], "perspective": perspective,
            "positioning": positioning,
            "source_skill_conditioned": episode["skill_conditioned"],
            "source_episode_sha256": hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest(),
            "historical_patch_sha256": hashlib.sha256(case["patch"].encode()).hexdigest(),
            "experience_attempts": attempt,
            "adaptation": "SWE-Exp-Loc: function-localization perspective/positioning"}


def build(*, limit: int = 400, dry_run: bool = False, client=None,
          preparation: Path = PREPARATION, expanded: Path = EXPANDED,
          destination: Path = DESTINATION) -> dict:
    prepared = read_json(preparation / "preparation_summary.json")
    if prepared["status"] != "complete" or digest(preparation / "episodes.jsonl") != prepared["prepared_hashes"]["episodes.jsonl"]:
        raise ValueError("Frozen acquisition episodes changed or are incomplete")
    for name, expected in prepared["manifest_hashes"].items():
        if digest(expanded / name) != expected:
            raise ValueError("Frozen manifest changed: " + name)
    episodes = {row["instance_id"]: row for row in (
        json.loads(line) for line in (preparation / "episodes.jsonl").read_text(encoding="utf-8").splitlines())}
    plans = [json.loads(line) for line in (destination / "acquisition_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = {row["instance_id"]: row for name in ("original_training.json", "training_additions.json")
             for row in read_json(expanded / name)}
    eval_ids = {row["instance_id"] for row in read_json(expanded / "evaluation_cases.json")}
    if len(plans) != 400 or set(episodes) != set(cases) or set(episodes) & eval_ids:
        raise ValueError("Acquisition/evaluation manifest identity changed")
    if limit < 1 or limit > len(plans):
        raise ValueError("limit must select 1..400 acquisition cases")
    selected = [row for row in plans[:limit] if row["eligible"]]
    pending = []
    for plan in selected:
        cid = plan["instance_id"]
        path = destination / "experiences" / (cid + ".json")
        if not path.is_file():
            pending.append(plan)
            continue
        existing = read_json(path)
        episode = episodes[cid]
        expected_hash = hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest()
        if existing.get("instance_id") != cid or existing.get("source_episode_sha256") != expected_hash:
            raise ValueError("Existing experience does not match frozen acquisition: " + cid)
    if dry_run:
        return {"status": "dry_run", "eligible_selected": len(selected), "pending": len(pending),
                "estimated_model_calls": 2 * len(pending)}
    if client is None:
        client = OpenAICompatibleClient.from_config(read_json(expanded / "config.json")["llm"])
    prompts = literal_prompts(AUTHOR_PROMPTS)
    failures = []
    completed_count = len(selected) - len(pending)
    for plan in pending:
        cid = plan["instance_id"]
        episode, case = episodes[cid], cases[cid]
        trajectory = Path(plan["trajectory_path"])
        if not trajectory.is_file() or digest(trajectory) != plan["trajectory_sha256"]:
            raise ValueError("Historical trajectory changed: " + cid)
        try:
            extracted = extract_one(client, episode, case, plan, prompts)
        except Exception as exc:
            failures.append({"instance_id": cid, "error_type": type(exc).__name__, "error": str(exc)[:300]})
        else:
            write_json(destination / "experiences" / (cid + ".json"), extracted)
            completed_count += 1
        write_json(destination / "experience_build_progress.json", {
            "status": "running", "eligible_selected": len(selected),
            "completed": completed_count, "failures": failures,
            "adaptation": "SWE-Exp-Loc; not author APR reproduction",
        })
    report = {"status": "complete" if not failures else "partial",
              "eligible_selected": len(selected), "completed": completed_count,
              "newly_completed": len(pending) - len(failures),
              "failures": failures, "adaptation": "SWE-Exp-Loc; not author APR reproduction",
              "data_lineage_warning": prepared["data_lineage_warning"]}
    write_json(destination / "experience_build_progress.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(limit=args.limit, dry_run=args.dry_run), indent=2))
