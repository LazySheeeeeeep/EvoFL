"""SWE-Exp-style candidate selection and FL-only Instructor guidance."""
from __future__ import annotations

import copy
import json


INSTRUCTOR_SYSTEM = """You are an instructor guiding a function-level fault-localization assistant.
Use the current issue, observed code-search history, and one selected historical
experience to recommend the next investigation action. The historical experience
is a hint, not evidence that a function in this repository is faulty.
Return one JSON object with exactly these fields:
{"thoughts":"brief synthesis of current observations", "instructions":"one actionable next objective",
 "context":"a search concept or a path/line span already observed in this repository",
 "type":"search | view | finish"}
Search and view mean inspect source; finish means rank functions supported by observed code.
This is a localization-only adaptation of SWE-Exp's Instructor. No patch action is available.
"""


def _object(client, messages: list[dict]) -> dict:
    response = client.chat(messages=messages, response_format={"type": "json_object"},
                           temperature=0, max_tokens=2048,
                           extra_body={"thinking": {"type": "disabled"}})
    content = response.get("content") or ""
    if not content.strip():
        raise ValueError("Empty SWE-Exp-Loc selector/Instructor response")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("SWE-Exp-Loc response must be a JSON object")
    return result


def select_one(client, *, issue: str, candidates: list[dict],
               experiences: dict[str, dict], prompts: dict) -> dict:
    if not candidates:
        return {"selected_source_id": None, "score": None, "reason": "no E5 candidates"}
    paragraphs = []
    for candidate in candidates:
        cid = candidate["instance_id"]
        experience = experiences[cid]
        perspectives = experience.get("perspective") or []
        if not perspectives:
            raise ValueError("Candidate has no perspective: " + cid)
        lines = [f"***issue_id***: {cid}",
                 f"***issue_description***: {experience['issue']}",
                 "***Experiences***:"]
        lines.extend(f" - Perspective: {item}" for item in perspectives)
        paragraphs.append("\n".join(lines))
    messages = [
        {"role": "system", "content": prompts["select_exp_system_prompt"].format(k=1)},
        {"role": "user", "content": prompts["select_exp_user_prompt"].format("\n\n".join(paragraphs), issue)},
    ]
    result = _object(client, messages)
    allowed = {item["instance_id"]: item for item in candidates}
    chosen = [(cid, detail) for cid, detail in result.items() if cid in allowed]
    if len(chosen) != 1 or not isinstance(chosen[0][1], dict):
        raise ValueError("Selector must choose exactly one recalled experience ID")
    cid, detail = chosen[0]
    return {"selected_source_id": cid, "score": allowed[cid]["score"],
            "reason": str(detail.get("reason") or "")}


def instruct_once(client, *, issue: str, history: list[dict], experience: dict) -> dict:
    if not issue.strip():
        raise ValueError("Current issue is empty")
    evidence = {
        "past_issue": experience["issue"],
        "perspective": experience.get("perspective") or [],
        "positioning": experience.get("positioning") or [],
    }
    for key in ("past_investigation", "past_predicted_functions", "past_final_summary"):
        if key in experience:
            evidence[key] = experience[key]
    messages = [{"role": "system", "content": INSTRUCTOR_SYSTEM},
                {"role": "user", "content": json.dumps({"current_issue": issue,
                    "recent_code_observations": history,
                    "historical_experience": evidence}, ensure_ascii=False)}]
    result = _object(client, messages)
    if result.get("type") not in {"search", "view", "finish"}:
        raise ValueError("Invalid FL Instructor action type")
    if not all(isinstance(result.get(key), str) for key in ("thoughts", "instructions", "context")):
        raise ValueError("Incomplete FL Instructor guidance")
    return {key: result[key] for key in ("thoughts", "instructions", "context", "type")}


class InstructorGuidedClient:
    """Add a separate Instructor turn before Explorer's main action decision."""

    def __init__(self, client, *, issue: str, selected_experience: dict,
                 max_instructions: int = 20):
        self.client = client
        self.issue = issue.strip()
        self.experience = copy.deepcopy(selected_experience)
        self.max_instructions = max_instructions
        self.guidance: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, **kwargs):
        messages = kwargs.get("messages") or []
        if (len(self.guidance) < self.max_instructions and kwargs.get("tools")
                and len(messages) >= 2 and messages[1].get("role") == "user"):
            try:
                payload = json.loads(messages[1]["content"])
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and "fault_families" in payload:
                if (payload.get("bug_report") or "").strip() != self.issue:
                    raise ValueError("Instructor issue changed during Explorer run")
                history = [{"observation": str(row.get("content") or "")[:2000]}
                           for row in messages if row.get("role") == "tool"][-6:]
                advice = instruct_once(self.client, issue=self.issue, history=history,
                                       experience=self.experience)
                self.guidance.append(advice)
                kwargs = dict(kwargs)
                kwargs["messages"] = copy.deepcopy(messages)
                kwargs["messages"].append({"role": "user", "content":
                    "Instructor next-step guidance (verify against current source): "
                    + json.dumps(advice, ensure_ascii=False)})
        return self.client.chat(**kwargs)
