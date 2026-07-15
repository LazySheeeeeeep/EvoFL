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
                "skill_id": "config_schema_v1",
                "status": "active",
                "version": 1,
                "dimension": "fault_mode",
                "value": "config_schema_mismatch",
                "retrieval_text": "configuration schema normalization default merge",
                "skill": {
                    "title": "Config schema mismatch",
                    "trigger": "Use for config schema mismatch.",
                    "knowledge": "Inspect config schema normalization before downstream consumers.",
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

    assert result["matched_skills"][0]["skill_id"] == "config_schema_v1"
    assert result["skill_search_trace"]["retrieval_mode"] == "embedding_dimension_skill_retrieval_v1"
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
    )

    result = bank.search_for_explorer("demo/repo", "config option default merge is ignored")

    assert result["matched_skills"][0]["skill_id"] == "config_schema_v1"
    assert result["skill_search_trace"]["retrieval_mode"] == "embedding_unconfigured_fallback_to_lexical_v1"


def test_skill_embedding_text_uses_compact_retrieval_view() -> None:
    text = skill_embedding_text(
        DimensionSkill(
            skill_id="config_schema_v1",
            status="active",
            version=1,
            dimension="fault_mode",
            value="config_schema_mismatch",
            retrieval_text="configuration schema normalization default merge",
            title="Config schema mismatch",
            trigger="Use for config schema mismatch.",
            knowledge="Very long case-specific knowledge that should be loaded after retrieval, not embedded.",
        )
    )

    assert "retrieval_text: configuration schema normalization default merge" in text
    assert "knowledge:" not in text
    assert "anti_patterns:" not in text


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
            "operation": "add",
            "target": {"skill_id": None, "dimension": "fault_mode", "value": "config_schema_mismatch"},
            "content": {
                "title": "Config schema",
                "trigger": "Use for config schema bugs.",
                "text": "Inspect schema normalization before consumers.",
                "retrieval_text": "config schema normalization",
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
            "skill_id": "config_schema_v1",
            "status": "active",
            "version": 1,
            "dimension": "fault_mode",
            "value": "config_schema_mismatch",
            "retrieval_text": "old config schema",
            "skill": {
                "title": "Old config schema",
                "trigger": "Use for old config schema bugs.",
                "knowledge": "Old knowledge.",
            },
        },
        {
            "skill_id": "config_schema_v1",
            "status": "active",
            "version": 2,
            "dimension": "fault_mode",
            "value": "config_schema_mismatch",
            "retrieval_text": "new config schema",
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
                "skill_id": "config_schema_v1",
                "version": 1,
                "dimension": "fault_mode",
                "value": "config_schema_mismatch",
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

    assert result["active_skill_count"] == 1
    assert result["embedding_cache"]["pruned_count"] == 1
    assert len(cache_records) == 1
    assert cache_records[0]["skill_id"] == "config_schema_v1"
    assert cache_records[0]["version"] == 2
