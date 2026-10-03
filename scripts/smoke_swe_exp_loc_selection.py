"""Exercise E5 recall, author-style selection, and FL Instructor on acquisition only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_swe_exp_loc_memory import AUTHOR_PROMPTS, literal_prompts
from evolutefl.llm.client import OpenAICompatibleClient
from prepare_rq1_memory_baselines import EXPANDED, OUTPUT as PREPARATION, read_json, write_json
from prepare_swe_exp_loc import DESTINATION
from swe_exp_loc_retrieval import E5Encoder, load_index, query_text, recall
from swe_exp_loc_selector import instruct_once, select_one


def smoke(case_id: str, *, client=None, encoder=None, destination: Path = DESTINATION) -> dict:
    index_path = destination / "e5_issue_types.jsonl"
    index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    memories = [read_json(destination / "experiences" / (row["instance_id"] + ".json"))
                for row in index_rows]
    indexed = load_index(memories, index_path)
    by_id = {row["instance_id"]: row for row in memories}
    if case_id not in by_id:
        raise ValueError("Smoke case must be in acquisition index")
    query = by_id[case_id]
    encoder = encoder or E5Encoder()
    vector = encoder.encode([query_text(query)])[0]
    candidates = recall(query=query, current_repo=query["repo"], experiences=memories,
                        indexed=indexed, query_vector=vector, k=10)
    if not candidates:
        raise ValueError("Acquisition smoke needs a different-repo memory")
    client = client or OpenAICompatibleClient.from_config(read_json(EXPANDED / "config.json")["llm"])
    prompts = literal_prompts(AUTHOR_PROMPTS)
    selected = select_one(client, issue=query["issue"], candidates=candidates,
                          experiences=by_id, prompts=prompts)
    if not selected["selected_source_id"]:
        raise ValueError("SWE-Exp selector returned no experience")
    episodes = (json.loads(line) for line in (PREPARATION / "episodes.jsonl").read_text(encoding="utf-8").splitlines())
    episode = next(row for row in episodes if row["instance_id"] == case_id)
    history = episode["investigation"][:3]
    instruction = instruct_once(client, issue=query["issue"], history=history,
                                experience=by_id[selected["selected_source_id"]])
    report = {"status": "completed", "case_id": case_id, "source": "acquisition_only",
              "retrieval_model": "intfloat/multilingual-e5-large-instruct", "candidates": candidates,
              "selection": selected, "instructor": instruction, "model_calls": 2}
    write_json(destination / "selection_smoke.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="psf__requests-1713")
    args = parser.parse_args()
    print(json.dumps(smoke(args.case_id), indent=2))
