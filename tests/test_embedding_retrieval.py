from __future__ import annotations

import json
from pathlib import Path

from evolutefl.skills.bank import SkillBankV0
from evolutefl.skills.embedding_store import skill_embedding_text
from evolutefl.skills.schema import DimensionSkill


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def embed_texts(self, texts: list[str], *, task: str | None = None) -> list[list[float]]:
        self.calls.append((texts, task))
        return [_vector(text) for text in texts]


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "config" in lowered else 0.0,
        1.0 if "schema" in lowered else 0.0,
        1.0 if "parser" in lowered else 0.0,
        1.0 if "token" in lowered else 0.0,
    ]


def _write_skill(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "skill_id": "project_config_adapter_v1",
                "status": "active",
                "version": 1,
                "skill_type": "project_skill",
                "value": "configuration_adapter",
                "skill": {
                    "title": "Config schema mismatch",
                    "trigger": "Use for config schema mismatch.",
                    "knowledge": "Inspect config schema normalization and default merge behavior before downstream consumers.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_embedding_retrieval_uses_cache_and_threshold(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    _write_skill(bank_path)
    client = FakeEmbeddingClient()
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=client,
        embedding_cache_path=cache_path,
        embedding_min_score=0.2,
    )

    result = bank.search_for_explorer("demo/repo", "config option ignored by schema")

    assert result["matched_skills"][0]["skill_id"] == "project_config_adapter_v1"
    assert result["skill_search_trace"]["retrieval_mode"] == "embedding_skill_type_retrieval_v1"
    assert cache_path.exists()
    assert result["skill_search_trace"]["embedding_cache"]["generated_count"] == 1

    second = bank.search_for_explorer("demo/repo", "config option ignored by schema")

    assert second["skill_search_trace"]["embedding_cache"]["cached_count"] == 1


def test_embedding_retrieval_does_not_fallback_to_lexical_by_default_when_unconfigured(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(bank_path, retrieval_mode="embedding", embedding_client=None)

    result = bank.search_for_explorer("demo/repo", "config option default merge is ignored")

    assert result["matched_skills"] == []
    assert result["skill_search_trace"]["retrieval_mode"] == "embedding_unconfigured_v1"


def test_embedding_retrieval_can_explicitly_fallback_to_lexical_when_unconfigured(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=None,
        embedding_fallback_to_lexical=True,
        project_skill_min_score=4.0,
    )

    result = bank.search_for_explorer("demo/repo", "config option default merge is ignored")

    assert result["matched_skills"][0]["skill_id"] == "project_config_adapter_v1"
    assert result["skill_search_trace"]["retrieval_mode"] == "embedding_unconfigured_fallback_to_lexical_v1"


def test_project_skill_embedding_includes_bounded_knowledge() -> None:
    text = skill_embedding_text(
        DimensionSkill(
            skill_id="project_config_adapter_v1",
            status="active",
            version=1,
            skill_type="project_skill",
            value="configuration_driven_system",
            title="Configuration-driven system",
            trigger="Use when external configuration is normalized before backend consumption.",
            knowledge=["Inspect configuration schema normalization before downstream consumers."],
        )
    )

    assert "project_type: configuration_driven_system" in text
    assert "architecture_signature:" in text
    assert "Inspect configuration schema normalization" in text
    assert "retrieval_text:" not in text


def test_strategy_skill_embedding_includes_bounded_knowledge() -> None:
    text = skill_embedding_text(
        DimensionSkill(
            skill_id="strategy_first_divergence_v1",
            status="active",
            version=1,
            skill_type="strategy_skill",
            value="rank_first_divergence",
            title="Rank the first divergence",
            trigger="Use when several functions propagate the same wrong state.",
            knowledge=[
                "Compare state across shared boundaries.",
                "Rank the first function that creates the wrong state.",
            ],
        )
    )
    assert "applicability_trigger: Use when several functions propagate the same wrong state." in text
    assert "Compare state across shared boundaries." in text
    assert "Rank the first function" in text


def test_project_and_strategy_embedding_thresholds_are_independent(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
        embedding_min_score=0.2,
        project_skill_embedding_min_score=1.1,
        strategy_skill_embedding_min_score=0.2,
    )
    result = bank.search_for_stage("project_skill", "config schema system", limit=1)
    assert result["matched_skills"] == []
    assert "embedding_min_score=1.1" in result["skill_search_trace"]["notes"]


def test_project_candidate_recall_excludes_skills_below_candidate_threshold(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
        project_skill_embedding_min_score=1.1,
        project_skill_candidate_min_score=1.1,
    )

    result = bank.search_project_candidates("config schema system", limit=3)

    assert result["candidate_skills"] == []
    assert result["skill_search_trace"]["candidate_selected_skill_ids"] == []
    assert result["skill_search_trace"]["ranked_candidate_skill_ids"] == []
    assert result["skill_search_trace"]["top_scores"][0]["skill_id"] == "project_config_adapter_v1"


def test_project_candidate_recall_can_be_broader_than_runtime_injection(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
        project_skill_embedding_min_score=1.1,
        project_skill_candidate_min_score=0.2,
    )

    runtime = bank.search_for_stage("project_skill", "config schema system", limit=1)
    candidates = bank.search_project_candidates("config schema system", limit=3)

    assert runtime["matched_skills"] == []
    assert [item["skill_id"] for item in candidates["candidate_skills"]] == [
        "project_config_adapter_v1"
    ]
    assert candidates["skill_search_trace"]["retrieval_mode"] == "embedding_project_candidate_recall_v2"


def test_project_candidate_recall_defaults_to_selector_gated_top_candidates(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    _write_skill(bank_path)
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
    )

    # The runtime threshold remains independent. Candidate recall keeps the
    # top Project cards for the structural LLM selector to accept or reject.
    candidates = bank.search_project_candidates("unrelated subsystem", limit=3)

    assert [item["skill_id"] for item in candidates["candidate_skills"]] == [
        "project_config_adapter_v1"
    ]
    assert "project_skill_candidate_min_score=0.0" in candidates["skill_search_trace"]["notes"]


def test_fault_catalog_is_family_scoped_and_does_not_use_embedding(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    bank_path.write_text(
        json.dumps(
            {
                "skill_id": "fault_config_boundary_v1",
                "status": "active",
                "version": 1,
                "skill_type": "fault_skill",
                "fault_family": "transformation_representation",
                "fault_subtype": "configuration_boundary_mismatch",
                "skill": {
                    "title": "Configuration boundary mismatch",
                    "trigger": "Use when configuration values diverge before backend consumption.",
                    "knowledge": ["Follow configuration through its first transformation boundary."],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
    )

    candidates = bank.fault_catalog("transformation_representation")
    unrelated = bank.fault_catalog("algorithm_computation")

    assert [item["skill_id"] for item in candidates["candidate_skills"]] == [
        "fault_config_boundary_v1"
    ]
    assert candidates["candidate_skills"][0]["fault_subtype"] == "configuration_boundary_mismatch"
    assert "knowledge" not in candidates["candidate_skills"][0]
    assert candidates["skill_search_trace"]["retrieval_mode"] == "llm_full_family_catalog_v1"
    assert unrelated["candidate_skills"] == []


def test_apply_update_warms_embedding_cache_when_client_configured(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    bank_path.write_text("", encoding="utf-8")
    client = FakeEmbeddingClient()
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=client,
        embedding_cache_path=cache_path,
    )

    result = bank.apply_update(
        {
            "operation": "create",
            "skill_type": "project_skill",
            "target_skill_id": None,
            "skill": {
                "value": "configuration_driven_system",
                "title": "Configuration-driven system",
                "trigger": "Use when external configuration is normalized before backend consumption.",
                "knowledge": [
                    "Configuration adapters normalize external values.",
                    "Inspect normalization before downstream consumers.",
                ],
            },
            "source_cases": ["case1"],
            "outcome_type": "failure",
        }
    )

    assert result["embedding_cache"]["enabled"] is True
    assert result["embedding_cache"]["generated_count"] == 1
    assert cache_path.exists()


def test_rebuild_embeddings_prunes_stale_skill_versions(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    cache_path = tmp_path / "embeddings.jsonl"
    records = [
        {
            "skill_id": "project_config_adapter_v1",
            "status": "active",
            "version": 1,
            "skill_type": "project_skill",
            "value": "configuration_adapter",
            "skill": {
                "title": "Old config schema",
                "trigger": "Use for old config schema bugs.",
                "knowledge": "Old knowledge.",
            },
        },
        {
            "skill_id": "project_config_adapter_v1",
            "status": "active",
            "version": 2,
            "skill_type": "project_skill",
            "value": "configuration_adapter",
            "skill": {
                "title": "New config schema",
                "trigger": "Use for new config schema bugs.",
                "knowledge": "New knowledge.",
            },
        },
    ]
    bank_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    cache_path.write_text(
        json.dumps(
            {
                "skill_id": "project_config_adapter_v1",
                "version": 1,
                "skill_type": "project_skill",
                "value": "configuration_adapter",
                "model_id": "jina-embeddings-v3",
                "text_hash": "stale",
                "text_preview": "old",
                "embedding": [1.0, 0.0, 0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = SkillBankV0(
        bank_path,
        retrieval_mode="embedding",
        embedding_client=FakeEmbeddingClient(),
        embedding_cache_path=cache_path,
    )

    result = bank.rebuild_embeddings()
    cache_records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["embedded_project_skill_count"] == 1
    assert result["embedding_cache"]["pruned_count"] == 1
    assert len(cache_records) == 1
    assert cache_records[0]["skill_id"] == "project_config_adapter_v1"
    assert cache_records[0]["version"] == 2
