from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .schema import SKILL_TYPES, DimensionSkill


DEFAULT_SKILL_TYPE_QUOTA = {
    "project_skill": 1,
    "fault_skill": 1,
    "strategy_skill": 1,
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "when",
    "where",
    "into",
    "does",
    "not",
    "are",
    "was",
    "bug",
    "issue",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    tokens = re.split(r"[^a-zA-Z0-9]+|_+", text.lower())
    return [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]


def skill_text(skill: DimensionSkill) -> str:
    parts = [
        skill.skill_id,
        skill.skill_type,
        skill.value,
        skill.title,
        skill.trigger,
        " ".join(skill.knowledge_texts),
    ]
    return " ".join(parts)


def score_skill(query_text: str, skill: DimensionSkill) -> tuple[float, dict[str, Any]]:
    query_tokens = Counter(tokenize(query_text))
    if not query_tokens:
        return 0.0, {"overlap_tokens": []}
    skill_tokens = set(tokenize(skill_text(skill)))
    overlap_tokens = sorted(token for token in query_tokens if token in skill_tokens)
    overlap = sum(count for token, count in query_tokens.items() if token in skill_tokens)
    score = float(overlap)

    lower_query = query_text.lower()
    bonuses: list[str] = []
    if skill.value and skill.value != "unknown" and skill.value.lower() in lower_query:
        score += 3.0
        bonuses.append("value")
    trigger_tokens = set(tokenize(skill.trigger))
    if trigger_tokens.intersection(query_tokens):
        score += 1.0
        bonuses.append("trigger")
    return score, {"overlap_tokens": overlap_tokens, "bonuses": bonuses}


def select_skills(
    query_text: str,
    skills: list[DimensionSkill],
    *,
    max_matched_skills: int = 5,
    max_per_skill_type: dict[str, int] | None = None,
    min_score: float = 4.0,
    project_skill_min_score: float = 6.0,
) -> tuple[list[DimensionSkill], dict[str, Any]]:
    quotas = {**DEFAULT_SKILL_TYPE_QUOTA, **(max_per_skill_type or {})}
    scored = []
    match_reasons: dict[str, dict[str, Any]] = {}
    for skill in skills:
        score, reasons = score_skill(query_text, skill)
        scored.append((score, skill))
        match_reasons[skill.skill_id] = reasons
    candidates = []
    filtered: list[dict[str, Any]] = []
    for score, skill in scored:
        threshold = project_skill_min_score if skill.skill_type == "project_skill" else min_score
        if score >= threshold:
            candidates.append((score, skill))
        elif score > 0:
            filtered.append({"skill_id": skill.skill_id, "score": score, "threshold": threshold})
    candidates.sort(key=lambda pair: (-pair[0], pair[1].skill_type, pair[1].skill_id))

    selected: list[DimensionSkill] = []
    used_by_skill_type: dict[str, int] = defaultdict(int)
    for score, skill in candidates:
        skill_type = skill.skill_type
        if skill_type not in SKILL_TYPES or used_by_skill_type[skill_type] >= quotas.get(skill_type, 1):
            continue
        selected.append(skill)
        used_by_skill_type[skill_type] += 1
        if len(selected) >= max_matched_skills:
            break

    trace = {
        "query": query_text,
        "retrieval_mode": "lexical_skill_type_retrieval_v1",
        "candidate_skill_ids": [skill.skill_id for _, skill in candidates],
        "selected_skill_ids": [skill.skill_id for skill in selected],
        "scores": {skill.skill_id: score for score, skill in candidates},
        "top_scores": [
            {
                "skill_id": skill.skill_id,
                "score": score,
                "skill_type": skill.skill_type,
                "value": skill.value,
            }
            for score, skill in sorted(scored, key=lambda pair: (-pair[0], pair[1].skill_id))[:10]
        ],
        "match_reasons": {skill.skill_id: match_reasons.get(skill.skill_id, {}) for _, skill in candidates},
        "filtered_below_threshold": filtered,
        "notes": [f"min_score={min_score}, project_skill_min_score={project_skill_min_score}"],
    }
    if not selected:
        trace["notes"].append("No lexical skill match found.")
    return selected, trace
