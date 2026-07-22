from __future__ import annotations

from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection.reflector import run_reflector


def test_reflector_protocol_failure_returns_no_update(tmpdir) -> None:
    output_path = tmpdir.join("reflector_debug.json")
    fake = FakeLLMClient([{"content": ""}, {"content": ""}])

    output = run_reflector(
        trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case1"}},
        skill_search_context={"skill_type_slots": {}},
        llm_client=fake,
        prompt="Return JSON.",
        attempts=2,
        output_path=str(output_path),
    )

    assert output["materialized_updates"] == []
    assert set(output["skill_updates"]) == {"project_skill", "strategy_skill"}
    assert output["optimization_intent"] == "no_update"
    assert "protocol failed" in output["no_update_reason"]
    assert output_path.exists()
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
