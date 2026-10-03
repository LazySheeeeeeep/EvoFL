"""Retrieval and first-turn injection for RQ1 memory-format ablations.

The active RQ1 Explorer is not modified. A separate client wrapper adds one
bounded memory item to its initial user payload for the memory-only arms.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
import re


_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+")
_STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
         "it", "of", "on", "or", "that", "the", "this", "to", "was", "with"}


def tokens(value: str) -> set[str]:
    return {word for word in _TOKEN.findall(value.lower()) if word not in _STOP}


class EpisodeRetriever:
    """Deterministic IDF-weighted issue retrieval over acquisition episodes."""

    def __init__(self, episodes: list[dict]):
        self.episodes = list(episodes)
        self.documents = [tokens(row["issue"]) for row in episodes]
        counts = Counter(word for document in self.documents for word in document)
        self.idf = {word: math.log1p((len(episodes) + 1) / (count + 1))
                    for word, count in counts.items()}

    def retrieve(self, *, issue: str, instance_id: str, limit: int = 1) -> list[dict]:
        query = tokens(issue)
        scored = []
        for episode, document in zip(self.episodes, self.documents):
            if episode["instance_id"] == instance_id:
                continue
            overlap = query & document
            if not overlap:
                continue
            numerator = sum(self.idf[word] for word in overlap)
            denominator = math.sqrt(sum(self.idf[word] ** 2 for word in document)) or 1
            score = numerator / denominator
            scored.append((score, episode["instance_id"], episode))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [{"score": round(score, 6), "episode": episode}
                for score, _, episode in scored[:limit]]


def episodic_memory_item(episode: dict, *, max_actions: int = 8) -> dict:
    """Source-linked history, not a synthesized lesson or repair patch."""
    actions = episode.get("investigation") or []
    return {
        "source_instance_id": episode["instance_id"],
        "source_repo": episode["repo"],
        "past_issue": episode["issue"],
        "past_actions": [{key: step.get(key) for key in ("tool", "purpose", "path", "pattern")
                          if step.get(key) is not None} for step in actions[:max_actions]],
        "past_final_summary": episode.get("final_summary") or "",
        "past_predicted_functions": episode.get("predicted_functions") or [],
        "source_skill_conditioned": bool(episode.get("skill_conditioned")),
    }


class MemoryInjectedClient:
    """Add memory to the Explorer's initial user payload on every main turn.

    Selector and validator calls have different payloads and are left intact.
    The caller must save the selected item and source IDs separately because
    the Explorer's own initial_payload.json describes its unmodified input.
    """

    def __init__(self, client, *, instance_id: str, issue: str, memory: dict):
        self.client = client
        self.instance_id = instance_id
        self.issue = issue.strip()
        self.memory = copy.deepcopy(memory)
        self.injection_count = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, **kwargs):
        messages = kwargs.get("messages")
        if messages and len(messages) >= 2 and messages[1].get("role") == "user":
            try:
                payload = json.loads(messages[1]["content"])
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and payload.get("bug_report") and "fault_families" in payload:
                if payload.get("instance_id") not in (None, self.instance_id):
                    raise ValueError("Memory injection case mismatch")
                if payload["bug_report"].strip() != self.issue:
                    raise ValueError("Memory injection issue mismatch")
                kwargs = dict(kwargs)
                kwargs["messages"] = copy.deepcopy(messages)
                kwargs["messages"][1]["content"] = json.dumps({**payload,
                    "historical_memory": self.memory,
                    "historical_memory_instruction": (
                        "This is a past case, not evidence for this repository. "
                        "Use it only if its investigation pattern helps; verify candidates in current source.")},
                    ensure_ascii=False)
                self.injection_count += 1
        return self.client.chat(**kwargs)
