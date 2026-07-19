from __future__ import annotations

import json

from evolutefl.issue_abstraction import abstract_issue, fallback_issue_abstraction
from evolutefl.llm.client import FakeLLMClient


def test_abstract_issue_returns_two_skill_queries(tmpdir) -> None:
    fake = FakeLLMClient(
        [
            {
                "content": json.dumps(
                    {
                        "abstract_problem_signature": "Configured backend option is accepted but ignored downstream.",
                        "project_type_key": "configuration_driven_web_system",
                        "project_type_description": "A web system that normalizes configuration before backend dispatch.",
                        "project_skill_query": "web framework backend adapter and option propagation boundary",
                        "strategy_skill_query": "trace option from entrypoint producer to backend consumer",
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

    assert result["project_skill_query"].startswith("web framework")
    assert result["strategy_skill_query"].startswith("trace option")
    assert result["project_type_key"] == "configuration_driven_web_system"
    assert result["project_type_description"].startswith("A web system")
    assert "fault_mode_query" not in result


def test_fallback_issue_abstraction_has_two_skill_queries() -> None:
    result = fallback_issue_abstraction("demo/repo", "Bug title")

    assert "demo/repo" in result["project_skill_query"]
    assert result["strategy_skill_query"] == "Bug title"
    assert result["abstract_problem_signature"] == "Bug title"
    assert result["project_type_key"] == "unknown"
