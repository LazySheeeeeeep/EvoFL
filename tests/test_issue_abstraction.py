from __future__ import annotations

import json

from evolutefl.issue_abstraction import abstract_issue, fallback_issue_abstraction
from evolutefl.llm.client import FakeLLMClient


def test_abstract_issue_returns_normalized_json(tmpdir) -> None:
    fake = FakeLLMClient(
        [
            {
                "content": json.dumps(
                    {
                        "abstract_problem_signature": "Configured backend option is accepted but ignored downstream.",
                        "project_type_query": "web framework backend adapter",
                        "fault_mode_query": "configured option ignored before downstream invocation",
                        "strategy_type_query": "trace option propagation from entrypoint to backend consumer",
                        "key_symptoms": ["configured option ignored"],
                    }
                )
            }
        ]
    )

    result = abstract_issue(
        repo="demo/repo",
        issue="specific issue",
        llm_client=fake,
        prompt="Return JSON.",
        output_path=str(tmpdir.join("debug.json")),
    )

    assert result["fault_mode_query"] == "configured option ignored before downstream invocation"
    assert result["key_symptoms"] == ["configured option ignored"]
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_abstract_issue_falls_back_after_bad_json() -> None:
    fake = FakeLLMClient([{"content": ""}, {"content": ""}])

    result = abstract_issue(repo="demo/repo", issue="Bug title\nMore detail", llm_client=fake, prompt="Return JSON.")

    assert result["abstract_problem_signature"] == "Bug title"
    assert "_fallback_reason" in result


def test_fallback_issue_abstraction_has_dimension_queries() -> None:
    result = fallback_issue_abstraction("demo/repo", "Bug title")

    assert result["project_type_query"] == "demo/repo"
    assert result["fault_mode_query"] == "Bug title"
