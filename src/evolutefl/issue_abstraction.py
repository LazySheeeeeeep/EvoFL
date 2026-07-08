from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolutefl.json_utils import extract_json_object, write_json


ABSTRACTION_FIELDS = (
    "abstract_problem_signature",
    "project_type_query",
    "fault_mode_query",
    "strategy_type_query",
    "key_symptoms",
)


def abstract_issue(
    *,
    repo: str,
    issue: str,
    llm_client: Any,
    prompt: str,
    attempts: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a compact issue abstraction for dimension-specific skill retrieval."""

    payload = {"repo": repo, "issue": issue}
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    attempts_debug: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        response = llm_client.chat(
            messages=messages,
            tool_choice="none",
            response_format={"type": "json_object"},
        )
        content = response.get("content") or ""
        try:
            abstraction = normalize_issue_abstraction(extract_json_object(content), repo=repo, issue=issue)
            if output_path:
                write_json(output_path, {"attempt": attempt, "issue_abstraction": abstraction, "raw_response": content})
            return abstraction
        except Exception as exc:  # noqa: BLE001 - invalid abstraction should degrade to fallback.
            last_error = exc
            attempts_debug.append({"attempt": attempt, "error": str(exc), "raw_response": content})
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        json.dumps(payload, ensure_ascii=False, indent=2)
                        + "\n\nReturn the same schema as strict JSON. "
                        + f"Validation feedback: {exc}"
                    ),
                },
            ]
    fallback = fallback_issue_abstraction(repo, issue, error=last_error)
    if output_path:
        write_json(output_path, {"failed": True, "attempts": attempts_debug, "issue_abstraction": fallback})
    return fallback


def normalize_issue_abstraction(payload: dict[str, Any], *, repo: str, issue: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Issue abstraction must be a JSON object.")
    normalized = {
        "abstract_problem_signature": _clean_string(payload.get("abstract_problem_signature")),
        "project_type_query": _clean_string(payload.get("project_type_query")),
        "fault_mode_query": _clean_string(payload.get("fault_mode_query")),
        "strategy_type_query": _clean_string(payload.get("strategy_type_query")),
        "key_symptoms": _clean_list(payload.get("key_symptoms")),
    }
    if not any(normalized[field] for field in ABSTRACTION_FIELDS if field != "key_symptoms"):
        raise ValueError("Issue abstraction must include at least one non-empty query field.")
    fallback = fallback_issue_abstraction(repo, issue)
    for key, value in fallback.items():
        if key == "key_symptoms":
            continue
        if not normalized[key]:
            normalized[key] = value
    return normalized


def fallback_issue_abstraction(repo: str, issue: str, *, error: Exception | None = None) -> dict[str, Any]:
    signature = _first_non_empty_line(issue) or f"Issue in {repo}"
    fallback = {
        "abstract_problem_signature": signature[:800],
        "project_type_query": repo,
        "fault_mode_query": issue[:1200],
        "strategy_type_query": issue[:1200],
        "key_symptoms": [],
    }
    if error is not None:
        fallback["_fallback_reason"] = str(error)
    return fallback


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_non_empty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""
