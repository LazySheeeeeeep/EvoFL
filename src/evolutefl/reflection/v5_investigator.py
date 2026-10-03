"""Training-only, bounded, read-only investigation of evidence gaps."""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.investigation import InvestigationLog, read_observations, delivered_observation
from evolutefl.tools.symbols import register_symbol_tools
from evolutefl.json_utils import append_jsonl, extract_json_object, write_json
from evolutefl.tools import ToolRegistry, register_builtin_tools
from evolutefl.tools.registry import openai_function_tool


LEARNING_FOCI = {"evidence_acquisition", "evidence_interpretation", "candidate_ranking",
                "supported_practice", "uncertain"}


def _conclusion_object(text: str) -> dict:
    # Explanatory prose may contain code such as state={} before the JSON.
    # Preserve a single fenced answer verbatim instead of asking the LLM to rewrite it.
    fenced = re.findall(r"```(?:json)?[ \t]*\r?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return extract_json_object(fenced[0] if len(fenced) == 1 else text)


def _finalization_messages(messages: list[dict], response: dict, error: str) -> list[dict]:
    """Retry against the same delivered evidence and retain the rejected answer."""
    retry = copy.deepcopy(messages)
    previous = response.get("content")
    if isinstance(previous, str) and previous:
        retry.append({"role": "assistant", "content": previous})
    retry.append({"role": "user", "content": (
        "Return the conclusion using the original JSON schema. Validation error: " + error + ". "
        "Correct the indicated format or reference using the tool observations already in this conversation. "
        "Preserve supported findings and exact source facts, including paths, names, and observation IDs. "
        "Keep investigation_advice focused on source inspection and candidate discrimination. "
        "If the evidence is insufficient, return unresolved."
    )})
    return retry


def validate_conclusion(value: dict, original: dict, supplementary: dict) -> dict:
    if value.get("status") not in {"resolved", "unresolved"}:
        raise ValueError("status must be resolved or unresolved")
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        raise ValueError("summary is required")
    findings = value.get("findings")
    if not isinstance(findings, list):
        raise ValueError("findings must be an array")
    if value["status"] == "resolved" and not findings:
        raise ValueError("resolved requires an observation-supported finding")
    all_observations = {**original, **supplementary}
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("finding must be an object")
        # Missing attribution in older/custom prompts stays unknown, never inferred from Top-5.
        focus = finding.get("learning_focus", "uncertain")
        if focus not in LEARNING_FOCI:
            raise ValueError("Invalid finding learning_focus")
        anchor = original.get(finding.get("anchor_observation_id", ""))
        if not anchor or not anchor["ok"] or anchor["tool"] not in {"grep", "read_file", "find_symbol", "read_symbol"}:
            raise ValueError("anchor must be a successful original repository observation")
        reviewed = any(o.get("tool") == "read_observation" and o.get("ok")
                       and isinstance(o.get("result"), dict)
                       and o["result"].get("observation_id") == finding["anchor_observation_id"]
                       for o in supplementary.values())
        if not reviewed:
            raise ValueError("Review the anchor with read_observation before using it in a finding")
        for field in ("available_clue", "investigation_advice", "explanation"):
            if not isinstance(finding.get(field), str) or not finding[field].strip():
                raise ValueError(f"finding requires {field}")
        refs = finding.get("supporting_observation_ids")
        if not isinstance(refs, list) or not refs:
            raise ValueError("supporting_observation_ids is required")
        for ref in refs:
            observation = all_observations.get(ref) if isinstance(ref, str) else None
            if not observation or not observation["ok"] or observation["tool"] not in {"grep", "read_file", "read_observation", "find_symbol", "read_symbol"}:
                raise ValueError("Support must reference an existing successful observation")
    return {"status": value["status"], "summary": value["summary"], "findings": findings,
            "unresolved_questions": value.get("unresolved_questions", [])}


def investigate(*, client: Any, config: dict, root: Path, case_dir: Path, directory: Path, evidence: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    original = read_observations(case_dir / "observations.jsonl")
    log = InvestigationLog(directory, prefix="supp",
                           model_observation_char_limit=int(config.get("reflection", {}).get("model_observation_char_limit", 2400)))
    registry = register_builtin_tools(ToolRegistry(), repo_path=str(root))
    register_symbol_tools(registry, root)
    schemas = [s for s in registry.list_openai_tools() if s["function"]["name"] in {"grep", "read_file", "find_symbol", "read_symbol"}]
    schemas.append(openai_function_tool(name="read_observation", description="Read the exact result previously returned to Explorer, by observation ID.",
        properties={"observation_id": {"type": "string"}}, required=["observation_id"]))
    cfg = config.get("reflection", {})
    prompt = resolve_path(cfg.get("investigator_prompt_path", "prompt_records/reflection/investigator_v5.txt")).read_text(encoding="utf-8")
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
    write_json(directory / "initial_payload.json", evidence)
    maximum = max(0, int(cfg.get("investigation_tool_budget", 16)))
    retries = max(1, int(cfg.get("investigation_response_attempts", 2)))
    started, count, invalid, turn, conclude_now = time.monotonic(), 0, 0, 0, False
    failure_stage = None
    conclusion = {"status": "unresolved", "summary": "No supported conclusion within the investigation budget.", "findings": [], "unresolved_questions": []}
    try:
        while turn < maximum + retries + 1:
            if time.monotonic() - started >= float(cfg.get("investigation_timeout_seconds", 900)):
                failure_stage = "investigation_timeout"
                break
            turn += 1
            budget_left = count < maximum and not conclude_now
            kwargs = ({"tools": schemas, "tool_choice": "auto"} if budget_left else {
                "response_format": {"type": "json_object"},
                # Reserve enough room for reasoning and the small structured
                # conclusion after the read-only evidence budget is spent.
                "max_tokens": int(cfg.get("investigation_conclusion_max_tokens", 8192)),
            })
            response = client.chat(messages=messages, **kwargs)
            append_jsonl(directory / "llm_trace.jsonl", {"turn": turn, "response": response, "remaining_tool_calls": maximum - count})
            calls = response.get("tool_calls") or []
            if calls:
                if not budget_left:
                    invalid += 1
                    if invalid >= retries:
                        failure_stage = "investigation_protocol"
                        break
                    messages.append({"role": "user", "content": "Tool budget exhausted. Return the conclusion JSON only."})
                    continue
                assistant = {"role": "assistant", "content": response.get("content"), "tool_calls": calls}
                messages.append(assistant)
                append_jsonl(directory / "trajectory.jsonl", {"event": "assistant", "step": turn, **assistant})
                for position, call in enumerate(calls):
                    if position:
                        # A single evidence operation per turn keeps a tool-heavy
                        # response from crowding out the conclusion turn.
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
                            "error": "Use one read-only investigation action per assistant response."})})
                        continue
                    if count >= maximum:
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"error": "Investigation tool budget exhausted"})})
                        continue
                    count += 1
                    record = None
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be an object")
                        args, record = log.begin(turn, call, args)
                        name = call["function"]["name"]
                        if name == "read_observation":
                            result = delivered_observation(original, args["observation_id"])
                        elif name in {"grep", "read_file", "find_symbol", "read_symbol"}:
                            if "repo_path" in args:
                                raise ValueError("Repository is fixed for this investigation")
                            result = registry.call_tool(name, args)
                        else:
                            raise ValueError("Only read-only investigation tools are available")
                        messages.append(log.observe(record, result))
                    except Exception as exc:
                        if record is None:
                            _, record = log.begin(turn, call, {})
                        messages.append(log.observe(record, None, error=str(exc)))
                if count >= maximum:
                    messages.append({"role": "user", "content": "Tool budget exhausted. Return a supported conclusion or unresolved as JSON."})
                continue
            try:
                conclusion = validate_conclusion(_conclusion_object(response.get("content") or ""), original, log.observations)
                break
            except (ValueError, TypeError, KeyError) as exc:
                invalid += 1
                if invalid >= retries:
                    failure_stage = "investigation_protocol"
                    conclusion["summary"] = str(exc)
                    break
                conclude_now = True
                messages = _finalization_messages(messages, response, str(exc))
                append_jsonl(directory / "conclusion_retries.jsonl", {
                    "after_turn": turn, "error": str(exc), "messages": messages})
    except Exception as exc:
        failure_stage = "investigation_llm_request"
        conclusion["summary"] = str(exc)
    conclusion.update(tool_call_count=count, failure_stage=failure_stage, budget=maximum,
                      invalid_response_count=invalid, budget_exhausted=count >= maximum)
    log.save()
    write_json(directory.parent / "investigation_conclusion.json", conclusion)
    return conclusion
