"""Lossless observations and a lightweight, explicit investigation timeline."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from evolutefl.json_utils import append_jsonl, write_json
from evolutefl.tools.builtin import _iter_text_files

TRACE_VERSION = "v5"


def _model_observation_payload(payload: dict[str, Any], limit: int) -> tuple[dict[str, Any], bool]:
    """Deliver the tool's complete page; pagination belongs to the source tool.

    limit remains a compatibility argument for older callers, not a second
    truncation boundary over serialized source code or Skill knowledge.
    """
    return payload, False


def repository_identity(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    count = 0
    for file in sorted(_iter_text_files(root)):
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            continue
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
        count += 1
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    return {"source_sha256": digest.hexdigest(), "text_file_count": count,
            "git_head": head.stdout.strip() if head.returncode == 0 else None}


def instrument_tools(schemas: list[dict]) -> list[dict]:
    schemas = copy.deepcopy(schemas)
    for schema in schemas:
        props = schema["function"]["parameters"]["properties"]
        props.update({
            "purpose": {"type": "string", "description": "Brief investigation question this action addresses."},
            "based_on": {"type": "array", "items": {"type": "string"},
                         "description": "Prior observation IDs motivating this action; empty for issue-only reasoning."},
            "candidate_updates": {"type": "array", "description": "Optional changes to explicit function hypotheses.",
                "items": {"type": "object", "properties": {
                    "function": {"type": "string"}, "decision": {"type": "string", "enum": ["retain", "exclude"]},
                    "reason": {"type": "string"}, "observation_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["function", "decision", "reason", "observation_ids"], "additionalProperties": False}},
        })
    return schemas


class InvestigationLog:
    def __init__(self, directory: Path, prefix: str = "obs", model_observation_char_limit: int = 2400) -> None:
        self.directory, self.prefix = directory, prefix
        self.model_observation_char_limit = max(400, int(model_observation_char_limit))
        self.observations: dict[str, dict] = {}
        self.timeline: list[dict] = []

    def begin(self, step: int, call: dict, arguments: dict) -> tuple[dict, dict]:
        arguments = dict(arguments)
        purpose = arguments.pop("purpose", None)
        references = arguments.pop("based_on", [])
        updates = arguments.pop("candidate_updates", [])
        references = references if isinstance(references, list) else []
        available = {ref for ref, observation in self.observations.items() if observation["step"] < step}
        valid_refs = [ref for ref in references if isinstance(ref, str) and ref in available]
        valid_updates, invalid_updates = [], []
        for item in updates if isinstance(updates, list) else []:
            refs = item.get("observation_ids", []) if isinstance(item, dict) else []
            if (isinstance(item, dict) and isinstance(item.get("function"), str)
                    and "::" in item["function"] and item.get("decision") in {"retain", "exclude"}
                    and isinstance(refs, list) and all(isinstance(r, str) and r in available for r in refs)):
                valid_updates.append(item)
            else:
                invalid_updates.append(item)
        record = {"event": "investigation_action", "step": step, "call_id": call["id"],
                  "tool": call["function"]["name"], "arguments": arguments,
                  "purpose": purpose if isinstance(purpose, str) and purpose.strip() else None,
                  "based_on": valid_refs, "unresolved_references": [r for r in references if r not in valid_refs],
                  "candidate_updates": valid_updates, "invalid_candidate_updates": invalid_updates}
        self.timeline.append(record)
        append_jsonl(self.directory / "trajectory.jsonl", record)
        return arguments, record

    def observe(self, record: dict, result: Any, *, error: str | None = None) -> dict:
        observation_id = f"{self.prefix}-{len(self.observations) + 1:05d}"
        payload = {"observation_id": observation_id, "ok": error is None, "result": result}
        if error is not None:
            payload["error"] = error
        model_payload, truncated = _model_observation_payload(payload, self.model_observation_char_limit)
        observation = {"step": record["step"], "call_id": record["call_id"], "tool": record["tool"], **payload,
                       "model_payload": model_payload, "model_payload_truncated": truncated}
        self.observations[observation_id] = observation
        record["observation_id"] = observation_id
        append_jsonl(self.directory / "observations.jsonl", observation)
        message = {"role": "tool", "tool_call_id": record["call_id"],
                   "content": json.dumps(model_payload, ensure_ascii=False)}
        append_jsonl(self.directory / "trajectory.jsonl", {"event": "tool_result", "step": record["step"],
                     "observation_id": observation_id, "message": {"name": record["tool"], **message}})
        self.save()
        return message

    def save(self) -> None:
        timeline = copy.deepcopy(self.timeline)
        for event in timeline:
            clipped = []
            for key, value in event["arguments"].items():
                if isinstance(value, str) and len(value) > 800:
                    event["arguments"][key] = value[:800]
                    clipped.append(key)
            if clipped:
                event["argument_previews"] = clipped
        write_json(self.directory / "investigation_index.json", {
            "trace_version": TRACE_VERSION, "timeline": timeline,
            "observations": [{"observation_id": key, "step": value["step"], "tool": value["tool"], "ok": value["ok"],
                              "delivered_source": {field: value["model_payload"].get("result", {}).get(field)
                                  for field in ("path", "start_line", "end_line", "next_start_line")}
                                  if isinstance(value["model_payload"].get("result"), dict)
                                  and "content" in value["model_payload"]["result"] else None}
                             for key, value in self.observations.items()]})


def read_observations(path: Path) -> dict[str, dict]:
    return {item["observation_id"]: item for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() for item in [json.loads(line)]}


def delivered_observation(observations: dict[str, dict], observation_id: str) -> dict:
    """Replay exactly what the original agent saw, including legacy previews."""
    observation = observations[observation_id]
    if "model_payload" not in observation:
        raise ValueError("Observation lacks an auditable delivered payload")
    return copy.deepcopy(observation["model_payload"])
