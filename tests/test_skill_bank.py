from __future__ import annotations

import json
from pathlib import Path

from evolutefl.skills.bank import SkillBankV0


def _seed(path: Path) -> None:
    records = [
        {"skill_id": "project_config_v1", "status": "active", "version": 1, "skill_type": "project_skill", "value": "configuration_adapter", "skill": {"title": "Configuration adapter", "trigger": "Use for external configuration normalization.", "knowledge": "Adapters normalize external values before backend consumption."}},
        {"skill_id": "strategy_trace_v1", "status": "active", "version": 1, "skill_type": "strategy_skill", "value": "producer_consumer_trace", "skill": {"title": "Producer consumer trace", "trigger": "Use when values are ignored downstream.", "knowledge": "Trace producer and consumer and rank the first divergence."}},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_stage_search_is_type_limited(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path, min_score=1, project_skill_min_score=1)
    project = bank.search_for_stage("project_skill", "configuration adapter normalize backend", limit=1)
    assert [item["skill_id"] for item in project["matched_skills"]] == ["project_config_v1"]
    catalog = bank.strategy_catalog()
    assert catalog == [
        {
            "skill_id": "strategy_trace_v1",
            "title": "Producer consumer trace",
            "trigger": "Use when values are ignored downstream.",
        }
    ]
    strategy = bank.select_strategy_skill(
        "strategy_trace_v1",
        query="Values are ignored by the downstream reader.",
    )
    assert [item["skill_id"] for item in strategy["matched_skills"]] == ["strategy_trace_v1"]
    assert (
        strategy["skill_search_trace"]["retrieval_mode"]
        == "llm_trigger_catalog_selection_v2"
    )


def test_strategy_catalog_selection_uses_trigger_evidence_not_embedding_threshold(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path, min_score=100, project_skill_min_score=100)

    result = bank.select_strategy_skill(
        "strategy_trace_v1",
        query="Values are ignored by the downstream reader.",
    )

    assert [item["skill_id"] for item in result["matched_skills"]] == ["strategy_trace_v1"]
    assert result["skill_search_trace"]["retrieval_mode"] == "llm_trigger_catalog_selection_v2"


def test_strategy_catalog_resolution_defers_applicability_to_validator(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path)

    result = bank.select_strategy_skill(
        "strategy_trace_v1",
        query="A parser serializes tokens into a syntax tree.",
    )

    assert [item["skill_id"] for item in result["matched_skills"]] == ["strategy_trace_v1"]
    assert "selected this card" in result["skill_search_trace"]["notes"][0]


def test_strategy_catalog_resolution_preserves_llm_choice_for_semantic_validation(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_adapter_contract_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "value": "environment_adapter_contract",
                "skill": {
                    "title": "Environment adapter contract",
                    "trigger": "Use when an environment detector selects an adapter with an incompatible interface.",
                    "knowledge": "Compare detector output with the selected adapter interface.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = SkillBankV0(path)

    result = bank.select_strategy_skill(
        "strategy_adapter_contract_v1",
        query="A consumer observes an adapter exposing an incorrect metadata value.",
    )

    assert [item["skill_id"] for item in result["matched_skills"]] == ["strategy_adapter_contract_v1"]


def test_strategy_catalog_resolution_does_not_require_literal_trigger_overlap(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_generation_execution_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "value": "generation_execution_ordering",
                "skill": {
                    "title": "Generation versus execution ordering",
                    "trigger": "Use when a generator emits work items and an executor applies them in order.",
                    "knowledge": "Compare generation and execution order.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = SkillBankV0(path)

    result = bank.select_strategy_skill(
        "strategy_generation_execution_v1",
        query="An operation has unexpected behavior.",
    )
    assert [item["skill_id"] for item in result["matched_skills"]] == [
        "strategy_generation_execution_v1"
    ]


def test_evolution_search_is_independent_per_type(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path, min_score=1, project_skill_min_score=1)
    result = bank.search_for_evolution(
        project_skill_query="configuration adapter normalize backend",
        selected_strategy_skill_id="strategy_trace_v1",
        strategy_diagnostic_context="A producer emits values that are ignored by the downstream reporter.",
        limit_per_type=5,
    )
    assert result["candidates_by_skill_type"]["project_skill"][0]["skill_id"] == "project_config_v1"
    assert result["candidates_by_skill_type"]["strategy_skill"][0]["skill_id"] == "strategy_trace_v1"


def test_evolution_search_recalls_strategy_candidates_from_diagnostic_context(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path, min_score=100, project_skill_min_score=100)
    evolution = bank.search_for_evolution(
        project_skill_query="configuration adapter",
        selected_strategy_skill_id="strategy_trace_v1",
        strategy_diagnostic_context="A producer emits values that are ignored by the downstream reporter.",
        limit_per_type=5,
    )
    assert [item["skill_id"] for item in evolution["candidates_by_skill_type"]["strategy_skill"]] == [
        "strategy_trace_v1"
    ]
    strategy_trace = evolution["search_traces"]["strategy_skill"]
    assert strategy_trace["candidate_recall"] == "independent_strategy_evolution_recall_v2"
    assert strategy_trace["runtime_catalog_selection_id"] == "strategy_trace_v1"
    assert strategy_trace["selected_skill_ids"] == ["strategy_trace_v1"]


def test_evolution_search_preserves_runtime_catalog_choice_for_reflector_judgment(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_environment_selector_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "value": "environment_selector",
                "skill": {
                    "title": "Environment signal selector diagnosis",
                    "trigger": "Use when an environment signal from an input source selects a display adapter.",
                    "knowledge": "Compare the signal source with the selected display adapter.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = SkillBankV0(path)

    evolution = bank.search_for_evolution(
        project_skill_query="lazy version property",
        selected_strategy_skill_id="strategy_environment_selector_v1",
        strategy_diagnostic_context=(
            "A lazy property and a version selector compete over an off-by-one capacity result."
        ),
    )

    assert [item["skill_id"] for item in evolution["candidates_by_skill_type"]["strategy_skill"]] == [
        "strategy_environment_selector_v1"
    ]
    assert evolution["search_traces"]["strategy_skill"]["runtime_catalog_selection_id"] == (
        "strategy_environment_selector_v1"
    )


def test_evolution_search_recalls_strategy_candidates_without_runtime_selection(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path, min_score=1, project_skill_min_score=1)

    evolution = bank.search_for_evolution(
        project_skill_query="configuration adapter",
        selected_strategy_skill_id=None,
        strategy_diagnostic_context="A producer competes with a downstream reporter.",
    )

    assert [item["skill_id"] for item in evolution["candidates_by_skill_type"]["strategy_skill"]] == [
        "strategy_trace_v1"
    ]
    assert evolution["search_traces"]["strategy_skill"]["candidate_recall"] == (
        "independent_strategy_evolution_recall_v2"
    )


def test_strategy_none_selection_returns_no_candidate(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path)
    result = bank.select_strategy_skill("none")
    assert result["matched_skills"] == []
    assert result["skill_search_trace"]["selected_skill_ids"] == []


def test_active_skills_can_be_limited_to_one_type_for_ablation(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    project_only = SkillBankV0(path, enabled_skill_types=["project_skill"])

    assert [skill.skill_id for skill in project_only.active_skills()] == ["project_config_v1"]
    assert project_only.strategy_catalog() == []


def test_apply_create_rewrite_preserve(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path)
    created = bank.apply_update({"operation": "create", "skill_type": "strategy_skill", "target_skill_id": None, "skill": {"value": "path_contrast", "title": "Path contrast", "trigger": "Use for divergent paths.", "knowledge": ["Compare paths at the first shared boundary.", "Rank the first path that creates divergent state."]}})
    assert created["action"] == "create_new"
    rewritten = bank.apply_update({"operation": "rewrite", "skill_type": "project_skill", "target_skill_id": "project_config_v1", "skill": {"value": "configuration_adapter", "title": "Configuration-driven system", "trigger": "Use when external configuration is normalized before backend consumption.", "knowledge": ["Adapters normalize external values before backend consumption.", "Validation belongs at the adapter responsibility boundary."]}})
    assert rewritten["version"] == 2
    assert bank.apply_update({"operation": "preserve", "skill_type": "strategy_skill", "target_skill_id": "strategy_trace_v1", "skill": None})["action"] == "preserve"


def test_create_with_existing_type_and_value_does_not_duplicate_active_card(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path)

    result = bank.apply_update(
        {
            "operation": "create",
            "skill_type": "project_skill",
            "target_skill_id": None,
            "skill": {
                "value": "configuration_adapter",
                "title": "A duplicate center",
                "trigger": "Use for another configuration adapter description.",
                "knowledge": [
                    "External values cross an adapter boundary before consumption.",
                    "The adapter owns normalization before backend behavior diverges.",
                ],
            },
        }
    )

    assert result["action"] == "preserve_duplicate_value"
    assert [skill.skill_id for skill in bank.active_skills()].count("project_config_v1") == 1


def test_rewrite_rejects_a_new_semantic_identity(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    _seed(path)
    bank = SkillBankV0(path)

    try:
        bank.apply_update(
            {
                "operation": "rewrite",
                "skill_type": "project_skill",
                "target_skill_id": "project_config_v1",
                "skill": {
                    "value": "storage_lifecycle",
                    "title": "Storage lifecycle",
                    "trigger": "Use for persisted artifacts.",
                    "knowledge": [
                        "Store canonical artifacts before resolution.",
                        "Consumers read the resolved artifact after storage.",
                    ],
                },
            }
        )
    except ValueError as exc:
        assert "preserve the target semantic identity" in str(exc)
    else:
        raise AssertionError("Expected rewrite with a new semantic identity to fail.")


def test_project_skill_uses_exact_repo_identity_and_independent_add(tmpdir) -> None:
    path = Path(str(tmpdir)) / "skills.jsonl"
    path.write_text("", encoding="utf-8")
    bank = SkillBankV0(path)

    created = bank.apply_project_update({
        "decision": "create_new",
        "repo_id": "Pallets/Click",
        "skill": {
            "repo_id": "pallets/click",
            "title": "Click repository architecture",
            "knowledge": ["Command declarations become parser state before callback invocation."],
        },
    })

    assert created["action"] == "create_new"
    assert bank.search_project_for_repo("https://github.com/pallets/click.git")[
        "matched_skills"
    ][0]["repo_id"] == "pallets/click"
    assert bank.search_project_for_repo("pallets/flask")["matched_skills"] == []

    added = bank.apply_project_update({
        "decision": "add",
        "repo_id": "pallets/click",
        "knowledge_to_add": [
            "Context carries resolved parameters from parsing into command dispatch."
        ],
    })

    assert added["action"] == "add_project_knowledge"
    current = bank.get_project_skill("pallets/click")
    assert current is not None
    assert len(current["knowledge"]) == 2
    assert "trigger" not in current
