from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from evolutefl.embedding import EmbeddingClient
from evolutefl.json_utils import read_jsonl, write_jsonl

from .embedding_store import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    EmbeddingRetrievalConfig,
    SkillEmbeddingStore,
    query_embedding_text,
    select_embedding_skills,
)
from .retrieval import DEFAULT_SKILL_TYPE_QUOTA, select_skills
from .schema import (
    SKILL_TYPES,
    DimensionSkill,
    UnsupportedLegacySkillError,
    slugify,
    validate_atomic_knowledge,
    validate_issue_knowledge,
)


class SkillBankV0:
    def __init__(
        self,
        path: str | Path,
        *,
        max_matched_skills: int = 5,
        max_per_skill_type: dict[str, int] | None = None,
        min_score: float = 4.0,
        project_skill_min_score: float = 6.0,
        retrieval_mode: str = "lexical",
        embedding_client: EmbeddingClient | None = None,
        embedding_cache_path: str | Path | None = None,
        embedding_model_id: str = "jina-embeddings-v3",
        embedding_query_task: str = "retrieval.query",
        embedding_document_task: str = "retrieval.passage",
        embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
        project_skill_embedding_min_score: float | None = None,
        issue_skill_embedding_min_score: float | None = None,
        strategy_skill_embedding_min_score: float | None = None,
        project_skill_candidate_min_score: float = 0.0,
        issue_skill_candidate_min_score: float = 0.0,
        evolution_candidate_min_score: float = 0.25,
        embedding_fallback_to_lexical: bool = False,
        embedding_batch_size: int = 16,
        enabled_skill_types: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_matched_skills = int(max_matched_skills)
        self.max_per_skill_type = {**DEFAULT_SKILL_TYPE_QUOTA, **(max_per_skill_type or {})}
        self.min_score = float(min_score)
        self.project_skill_min_score = float(project_skill_min_score)
        self.retrieval_mode = _valid_retrieval_mode(retrieval_mode)
        self.embedding_client = embedding_client
        default_cache = self.path.parent / "embeddings" / "jina_v3" / "skill_embeddings.jsonl"
        self.embedding_config = EmbeddingRetrievalConfig(
            cache_path=Path(embedding_cache_path) if embedding_cache_path else default_cache,
            model_id=embedding_model_id,
            query_task=embedding_query_task,
            document_task=embedding_document_task,
            min_score=float(embedding_min_score),
            fallback_to_lexical=bool(embedding_fallback_to_lexical),
            batch_size=int(embedding_batch_size),
        )
        self.embedding_min_scores = {
            "project_skill": float(
                embedding_min_score
                if project_skill_embedding_min_score is None
                else project_skill_embedding_min_score
            ),
            "issue_skill": float(
                embedding_min_score
                if issue_skill_embedding_min_score is None
                else issue_skill_embedding_min_score
            ),
            "strategy_skill": float(
                embedding_min_score
                if strategy_skill_embedding_min_score is None
                else strategy_skill_embedding_min_score
            ),
        }
        self.evolution_candidate_min_score = float(evolution_candidate_min_score)
        self.project_skill_candidate_min_score = float(project_skill_candidate_min_score)
        self.issue_skill_candidate_min_score = float(issue_skill_candidate_min_score)
        requested_types = enabled_skill_types or SKILL_TYPES
        self.enabled_skill_types = {
            _valid_skill_type(skill_type) for skill_type in requested_types
        }

    def load(self) -> list[DimensionSkill]:
        latest: dict[str, DimensionSkill] = {}
        order: list[str] = []
        for record in read_jsonl(self.path):
            try:
                skill = DimensionSkill.from_dict(record)
            except (UnsupportedLegacySkillError, ValueError, TypeError):
                continue
            if skill.skill_id not in latest:
                order.append(skill.skill_id)
            current = latest.get(skill.skill_id)
            if current is None or skill.version >= current.version:
                latest[skill.skill_id] = skill
        return [latest[skill_id] for skill_id in order]

    def active_skills(self) -> list[DimensionSkill]:
        active = [
            skill
            for skill in self.load()
            if skill.status == "active" and skill.skill_type in self.enabled_skill_types
        ]
        # A card's semantic identity is its type plus its stable value. Older
        # experimental banks may contain duplicate IDs for the same center;
        # expose only the newest one so retrieval and embedding stay unbiased.
        latest: dict[tuple[str, str], DimensionSkill] = {}
        order: list[tuple[str, str]] = []
        for skill in active:
            identity = (skill.skill_type, skill.value)
            current = latest.get(identity)
            if current is None:
                order.append(identity)
            if current is None or skill.version >= current.version:
                latest[identity] = skill
        return [latest[identity] for identity in order]

    def strategy_catalog(self) -> list[dict[str, Any]]:
        """Return the compact applicability catalog shown to an LLM selector."""

        return [
            {
                "skill_id": skill.skill_id,
                "title": skill.title,
                "trigger": skill.trigger,
            }
            for skill in self.active_skills()
            if skill.skill_type == "strategy_skill"
        ]

    def select_strategy_skill(self, skill_id: str | None, *, query: str = "") -> dict[str, Any]:
        """Resolve one LLM-selected Strategy Skill from the compact trigger catalog.

        Strategy cards are selected after the agent has formed concrete competing
        hypotheses. At that point, a compact trigger comparison is more faithful
        than re-embedding the full card knowledge. Applicability is deliberately
        checked by Explorer's dedicated LLM validator, which can assess semantic
        evidence across repository-specific vocabulary. This resolver must not
        reintroduce a lexical-overlap gate that rejects an already selected card.
        """

        selected_id = str(skill_id or "").strip()
        catalog = self.strategy_catalog()
        catalog_ids = [item["skill_id"] for item in catalog]
        trace = {
            "retrieval_mode": "llm_trigger_catalog_selection_v2",
            "catalog_skill_ids": catalog_ids,
            "selected_skill_ids": [],
            "notes": [],
        }
        if selected_id in {"", "none", "null"}:
            trace["notes"].append("The selector chose no applicable Strategy Skill.")
            return {"matched_skills": [], "skill_search_trace": trace}
        selected = next(
            (
                skill
                for skill in self.active_skills()
                if skill.skill_type == "strategy_skill" and skill.skill_id == selected_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Selected Strategy Skill {selected_id!r} is not present in the active trigger catalog."
            )
        diagnostic_context = str(query or "").strip()
        if diagnostic_context:
            trace["diagnostic_context"] = diagnostic_context
        trace["selected_skill_ids"] = [selected.skill_id]
        trace["notes"].append(
            "The LLM selected this card from the compact trigger catalog; semantic applicability requires the dedicated validator."
        )
        return {
            "matched_skills": [selected.compact_dict()],
            "skill_search_trace": trace,
        }

    def search_for_stage(self, skill_type: str, query: str, *, limit: int = 1) -> dict[str, Any]:
        skill_type = _valid_skill_type(skill_type)
        if skill_type == "strategy_skill":
            raise ValueError(
                "Strategy Skills are selected from strategy_catalog(); use select_strategy_skill()."
            )
        query = str(query or "").strip()
        if not query:
            raise ValueError("Stage skill query must be non-empty.")
        limit = max(1, int(limit))
        active = [skill for skill in self.active_skills() if skill.skill_type == skill_type]
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=limit,
            max_per_skill_type={skill_type: limit},
            skill_type=skill_type,
        )
        trace["requested_skill_type"] = skill_type
        trace["runtime_limit"] = limit
        return {
            "matched_skills": [skill.compact_dict() for skill in selected],
            "skill_search_trace": trace,
        }

    def search_project_candidates(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        """Recall only threshold-qualified Project Skill candidates.

        Embedding similarity is the first safety gate. The Explorer's Project
        selector then decides whether one of those semantically close cards
        actually matches the observed architecture before knowledge is exposed.
        Low-scoring neighbors stay in the trace for audit, not in the LLM
        catalog, so they cannot be injected merely through shared role words.
        """

        query = str(query or "").strip()
        if not query:
            raise ValueError("Project Skill candidate query must be non-empty.")
        limit = max(1, int(limit))
        active = [
            skill for skill in self.active_skills() if skill.skill_type == "project_skill"
        ]
        candidate_selected, trace = self._select_project_candidates(
            query,
            active,
            max_matched_skills=limit,
            max_per_skill_type={"project_skill": limit},
        )
        by_id = {skill.skill_id: skill for skill in active}
        ranked_ids = [skill.skill_id for skill in candidate_selected[:limit]]
        scores = trace.get("scores") or {}
        candidates = [
            {
                "skill_id": by_id[skill_id].skill_id,
                "value": by_id[skill_id].value,
                "title": by_id[skill_id].title,
                "trigger": by_id[skill_id].trigger,
                "retrieval_score": scores.get(skill_id),
            }
            for skill_id in ranked_ids
        ]
        trace["requested_skill_type"] = "project_skill"
        trace["candidate_limit"] = limit
        trace["ranked_candidate_skill_ids"] = ranked_ids
        trace["candidate_selected_skill_ids"] = [
            skill.skill_id for skill in candidate_selected
        ]
        return {
            "candidate_skills": candidates,
            "skill_search_trace": trace,
        }

    def search_issue_candidates(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        """Recall Issue candidates before the Issue selector and validator.

        Runtime similarity must not decide applicability by itself. It only
        provides a bounded semantic neighborhood; the selector compares compact
        title/trigger cards and the validator decides whether knowledge is safe
        to inject for the current issue and observed repository boundary.
        """
        query = str(query or "").strip()
        if not query:
            raise ValueError("Issue Skill candidate query must be non-empty.")
        limit = max(1, int(limit))
        active = [skill for skill in self.active_skills() if skill.skill_type == "issue_skill"]
        selected, trace = self._select_issue_candidates(
            query,
            active,
            max_matched_skills=limit,
            max_per_skill_type={"issue_skill": limit},
        )
        trace["requested_skill_type"] = "issue_skill"
        trace["candidate_limit"] = limit
        return {
            "candidate_skills": [
                {
                    "skill_id": skill.skill_id,
                    "value": skill.value,
                    "title": skill.title,
                    "trigger": skill.trigger,
                    "retrieval_score": (trace.get("scores") or {}).get(skill.skill_id),
                }
                for skill in selected
            ],
            "skill_search_trace": trace,
        }

    def get_active_skill(self, skill_id: str, *, skill_type: str | None = None) -> dict[str, Any] | None:
        selected_id = str(skill_id or "").strip()
        for skill in self.active_skills():
            if skill.skill_id != selected_id:
                continue
            if skill_type is not None and skill.skill_type != _valid_skill_type(skill_type):
                return None
            return skill.compact_dict()
        return None

    def search_for_evolution(
        self,
        *,
        project_skill_query: str,
        selected_strategy_skill_id: str | None,
        strategy_diagnostic_context: str = "",
        limit_per_type: int = 5,
    ) -> dict[str, Any]:
        project_query = str(project_skill_query or "").strip()
        selected_strategy_skill_id = str(selected_strategy_skill_id or "").strip() or None
        active = {skill.skill_id: skill for skill in self.active_skills()}
        limit = max(1, int(limit_per_type))
        project_skills = [
            skill for skill in active.values() if skill.skill_type == "project_skill"
        ]
        project_selected, project_trace = self._select_for_evolution(
            project_query,
            project_skills,
            max_matched_skills=limit,
            max_per_skill_type={"project_skill": limit},
        )
        # Evolution recall is intentionally broader than runtime loading. The
        # Reflector receives a compact, same-type neighborhood and decides
        # whether a card shares the semantic center; low runtime similarity
        # never causes the Explorer to inject a card during localization.
        project_scores = project_trace.get("scores", {}) or {}
        project_ranked = []
        for skill in project_selected[:limit]:
            compact = skill.compact_dict()
            compact["retrieval_score"] = project_scores.get(skill.skill_id)
            project_ranked.append(compact)

        # Strategy evolution must not be limited to the card (if any) that
        # Explorer selected at runtime. A missing runtime selection means only
        # that no card was safe to inject for that partially explored case; it
        # does not mean the completed trajectory has no existing decision
        # family worth consolidating. Recall a same-type neighborhood from the
        # post-hoc diagnostic context, then retain the runtime catalog choice
        # as an additional exact candidate when it was outside that neighborhood.
        strategy_query = str(strategy_diagnostic_context or "").strip()
        strategy_skills = [
            skill for skill in active.values() if skill.skill_type == "strategy_skill"
        ]
        strategy_selected, strategy_trace = self._select_for_evolution(
            strategy_query,
            strategy_skills,
            max_matched_skills=limit,
            max_per_skill_type={"strategy_skill": limit},
        )
        strategy_scores = strategy_trace.get("scores", {}) or {}
        strategy_ranked: list[dict[str, Any]] = []
        ranked_ids: set[str] = set()
        for skill in strategy_selected[:limit]:
            compact = skill.compact_dict()
            compact["retrieval_score"] = strategy_scores.get(skill.skill_id)
            strategy_ranked.append(compact)
            ranked_ids.add(skill.skill_id)

        runtime_selected = None
        if selected_strategy_skill_id:
            runtime_selected = next(
                (
                    skill
                    for skill in strategy_skills
                    if skill.skill_id == selected_strategy_skill_id
                ),
                None,
            )
        if runtime_selected is not None and runtime_selected.skill_id not in ranked_ids:
            compact = runtime_selected.compact_dict()
            compact["retrieval_score"] = None
            compact["candidate_source"] = "runtime_catalog_selection"
            strategy_ranked.append(compact)
            ranked_ids.add(runtime_selected.skill_id)

        strategy_trace = dict(strategy_trace)
        strategy_trace.update(
            {
                "query": strategy_query,
                "diagnostic_context": strategy_query,
                "runtime_catalog_selection_id": selected_strategy_skill_id,
                "candidate_recall": "independent_strategy_evolution_recall_v2",
            }
        )
        strategy_trace["candidate_skill_ids"] = [item["skill_id"] for item in strategy_ranked]
        strategy_trace["selected_skill_ids"] = [item["skill_id"] for item in strategy_ranked]
        strategy_trace.setdefault("notes", [])
        if runtime_selected is not None:
            strategy_trace["notes"].append(
                "The runtime trigger-catalog choice was retained alongside post-hoc Strategy candidates."
            )
        elif not selected_strategy_skill_id:
            strategy_trace["notes"].append(
                "No runtime Strategy card was selected; candidates come from post-hoc diagnostic-context recall."
            )
        else:
            strategy_trace["notes"].append(
                "The runtime Strategy selection no longer exists in the active bank."
            )
        return {
            "queries": {
                "project_skill": project_query,
                "strategy_skill": str(strategy_diagnostic_context or "").strip(),
            },
            "strategy_catalog": self.strategy_catalog(),
            "selected_strategy_skill_id": selected_strategy_skill_id,
            "candidates_by_skill_type": {
                "project_skill": project_ranked,
                "strategy_skill": strategy_ranked,
            },
            "candidate_target_skills": [*project_ranked, *strategy_ranked],
            "search_traces": {
                "project_skill": project_trace,
                "strategy_skill": strategy_trace,
            },
        }

    def search_for_explorer(self, repo: str, issue: str) -> dict[str, Any]:
        """Compatibility preview; the staged Explorer does not call this method."""
        query = query_embedding_text(repo, issue)
        active = [
            skill for skill in self.active_skills() if skill.skill_type == "project_skill"
        ]
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=self.max_matched_skills,
            max_per_skill_type={"project_skill": self.max_matched_skills},
            skill_type="project_skill",
        )
        return {
            "matched_skills": [skill.compact_dict() for skill in selected],
            "skill_search_trace": trace,
        }

    def rebuild_embeddings(self) -> dict[str, Any]:
        if self.embedding_client is None:
            raise RuntimeError("Embedding client is not configured.")
        active = self.active_skills()
        store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
        _, trace = store.ensure_skill_embeddings(
            active,
            self.embedding_client,
            task=self.embedding_config.document_task,
            batch_size=self.embedding_config.batch_size,
            prune_to_skills=True,
        )
        return {"skill_bank_path": str(self.path), "active_skill_count": len(active), "embedding_cache": trace}

    def _select_for_query(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
        skill_type: str | None = None,
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        if self.retrieval_mode == "lexical":
            return self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
        embedding_selected, embedding_trace = self._select_embedding(
            query, active, max_matched_skills, max_per_skill_type, skill_type
        )
        if self.retrieval_mode == "embedding" or embedding_selected:
            return embedding_selected, embedding_trace
        lexical_selected, lexical_trace = self._select_lexical(
            query, active, max_matched_skills, max_per_skill_type
        )
        lexical_trace["retrieval_mode"] = "hybrid_embedding_then_lexical_v1"
        lexical_trace["embedding_trace"] = embedding_trace
        return lexical_selected, lexical_trace

    def _select_for_evolution(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        """Retrieve a small same-type neighborhood for Reflector consolidation."""

        if self.retrieval_mode == "lexical":
            return self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
        if self.embedding_client is None:
            return self._select_for_query(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
        try:
            config = replace(
                self.embedding_config,
                min_score=self.evolution_candidate_min_score,
            )
            selected, trace = select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=config,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
            trace["retrieval_mode"] = "embedding_evolution_candidate_recall_v1"
            trace.setdefault("notes", []).append(
                f"evolution_candidate_min_score={self.evolution_candidate_min_score}"
            )
            return selected, trace
        except Exception as exc:  # noqa: BLE001 - a failed recall must not stop evolution.
            return [], _failed_trace(query, f"Evolution candidate retrieval failed: {exc}")

    def _select_project_candidates(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        """Recall plausible Project cards before the structural LLM verifier."""

        if self.retrieval_mode == "lexical":
            return self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
        if self.embedding_client is None:
            return self._select_for_query(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
                skill_type="project_skill",
            )
        try:
            config = replace(
                self.embedding_config,
                min_score=self.project_skill_candidate_min_score,
            )
            selected, trace = select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=config,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
            trace["retrieval_mode"] = "embedding_project_candidate_recall_v2"
            trace.setdefault("notes", []).append(
                f"project_skill_candidate_min_score={self.project_skill_candidate_min_score}"
            )
            return selected, trace
        except Exception as exc:  # noqa: BLE001 - an empty candidate set is safe.
            return [], _failed_trace(query, f"Project candidate retrieval failed: {exc}")

    def _select_issue_candidates(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        """Recall a bounded Issue neighborhood for semantic applicability gates."""

        if self.retrieval_mode == "lexical":
            return self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
        if self.embedding_client is None:
            return self._select_for_query(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
                skill_type="issue_skill",
            )
        try:
            config = replace(
                self.embedding_config,
                min_score=self.issue_skill_candidate_min_score,
            )
            selected, trace = select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=config,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
            trace["retrieval_mode"] = "embedding_issue_candidate_recall_v1"
            trace.setdefault("notes", []).append(
                f"issue_skill_candidate_min_score={self.issue_skill_candidate_min_score}"
            )
            return selected, trace
        except Exception as exc:  # noqa: BLE001 - an empty candidate set is safe.
            return [], _failed_trace(query, f"Issue candidate retrieval failed: {exc}")

    def _select_lexical(
        self,
        query: str,
        active: list[DimensionSkill],
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        return select_skills(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_skill_type=max_per_skill_type,
            min_score=self.min_score,
            project_skill_min_score=self.project_skill_min_score,
        )

    def _select_embedding(
        self,
        query: str,
        active: list[DimensionSkill],
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
        skill_type: str | None,
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        if self.embedding_client is None:
            if self.embedding_config.fallback_to_lexical:
                selected, trace = self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
                trace["retrieval_mode"] = "embedding_unconfigured_fallback_to_lexical_v1"
                return selected, trace
            trace = _failed_trace(query, "Embedding client is not configured.")
            trace["retrieval_mode"] = "embedding_unconfigured_v1"
            return [], trace
        try:
            embedding_config = (
                replace(self.embedding_config, min_score=self.embedding_min_scores[skill_type])
                if skill_type in self.embedding_min_scores
                else self.embedding_config
            )
            return select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=embedding_config,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval failure should not stop Explorer.
            if self.embedding_config.fallback_to_lexical:
                selected, trace = self._select_lexical(query, active, max_matched_skills, max_per_skill_type)
                trace["retrieval_mode"] = "embedding_failed_fallback_to_lexical_v1"
                trace.setdefault("notes", []).append(str(exc))
                return selected, trace
            return [], _failed_trace(query, f"Embedding retrieval failed: {exc}")

    def search_for_v3_evolution(
        self,
        *,
        issue_skill_query: str,
        selected_strategy_skill_id: str | None,
        strategy_diagnostic_context: str,
        limit_per_type: int = 5,
    ) -> dict[str, Any]:
        """Recall Issue and Strategy candidates independently of runtime loading."""
        active = {skill.skill_id: skill for skill in self.active_skills()}
        limit = max(1, int(limit_per_type))
        by_type: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for skill_type, query in (
            ("issue_skill", str(issue_skill_query or "")),
            ("strategy_skill", str(strategy_diagnostic_context or "")),
        ):
            selected, trace = self._select_for_evolution(
                query,
                [skill for skill in active.values() if skill.skill_type == skill_type],
                max_matched_skills=limit,
                max_per_skill_type={skill_type: limit},
            )
            scores = trace.get("scores") or {}
            by_type[skill_type] = [
                {
                    **skill.compact_dict(include_knowledge_ids=(skill_type == "issue_skill")),
                    "retrieval_score": scores.get(skill.skill_id),
                }
                for skill in selected[:limit]
            ]
            traces[skill_type] = trace
        selected_id = str(selected_strategy_skill_id or "").strip()
        selected = active.get(selected_id)
        if selected and selected.skill_type == "strategy_skill" and not any(
            item["skill_id"] == selected_id for item in by_type["strategy_skill"]
        ):
            by_type["strategy_skill"].append({**selected.compact_dict(), "candidate_source": "runtime_catalog_selection"})
        return {
            "queries": {"issue_skill": issue_skill_query, "strategy_skill": strategy_diagnostic_context},
            "candidates_by_skill_type": by_type,
            "candidate_target_skills": [item for cards in by_type.values() for item in cards],
            "search_traces": traces,
            "selected_strategy_skill_id": selected_strategy_skill_id,
        }

    def apply_update(self, update: dict[str, Any]) -> dict[str, Any]:
        operation = update.get("operation")
        if operation == "preserve":
            return self._apply_preserve(update)
        if operation == "create":
            return self._apply_create(update)
        if operation == "rewrite":
            return self._apply_rewrite(update)
        raise ValueError(f"Unsupported skill update operation: {operation!r}")

    def _apply_create(self, update: dict[str, Any]) -> dict[str, Any]:
        skill_type = _valid_skill_type(update.get("skill_type"))
        if skill_type == "issue_skill" and update.get("knowledge_edit") is not None:
            card = _apply_issue_knowledge_edit(old, update["knowledge_edit"])
        else:
            card = _valid_complete_skill(update.get("skill"), skill_type)
        existing = self.load()
        duplicate = next(
            (
                skill
                for skill in self.active_skills()
                if skill.skill_type == skill_type and skill.value == card["value"]
            ),
            None,
        )
        if duplicate is not None:
            # Creation is reserved for a new semantic center. Rewriting a
            # known center must be explicit, otherwise repeated reflections
            # would create near-identical active cards and bias retrieval.
            return {
                "action": "preserve_duplicate_value",
                "updated_skill_id": duplicate.skill_id,
                "version": duplicate.version,
                "embedding_cache": {"enabled": False},
            }
        skill = DimensionSkill(
            skill_id=_unique_skill_id(existing, f"{skill_type}_{card['value']}_v1"),
            status="active",
            version=1,
            skill_type=skill_type,
            **card,
        )
        records = read_jsonl(self.path)
        records.append(skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": "create_new",
            "updated_skill_id": skill.skill_id,
            "version": 1,
            "embedding_cache": self._warm_embedding(skill),
        }

    def _apply_rewrite(self, update: dict[str, Any]) -> dict[str, Any]:
        target = str(update.get("target_skill_id") or "").strip()
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target)
        if old is None:
            raise ValueError(f"No active skill found for target_skill_id={target!r}.")
        skill_type = _valid_skill_type(update.get("skill_type"))
        if old.skill_type != skill_type:
            raise ValueError("rewrite must preserve skill_type.")
        if skill_type == "issue_skill" and update.get("knowledge_edit") is not None:
            card = _apply_issue_knowledge_edit(old, update["knowledge_edit"])
        else:
            card = _valid_complete_skill(update.get("skill"), skill_type)
        if card["value"] != old.value:
            raise ValueError(
                "rewrite must preserve the target skill value; use create_new for a new semantic center."
            )
        new = DimensionSkill(
            skill_id=old.skill_id,
            status="active",
            version=old.version + 1,
            skill_type=skill_type,
            **card,
        )
        superseded = deepcopy(old)
        superseded.status = "superseded"
        records = read_jsonl(self.path)
        records.extend([superseded.to_dict(), new.to_dict()])
        write_jsonl(self.path, records)
        return {
            "action": "rewrite",
            "updated_skill_id": new.skill_id,
            "version": new.version,
            "embedding_cache": self._warm_embedding(new),
        }

    def _apply_preserve(self, update: dict[str, Any]) -> dict[str, Any]:
        target = str(update.get("target_skill_id") or "").strip()
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target)
        return {
            "action": "preserve",
            "updated_skill_id": old.skill_id if old else None,
            "version": old.version if old else None,
            "embedding_cache": {"enabled": False},
        }

    def _warm_embedding(self, skill: DimensionSkill) -> dict[str, Any]:
        if self.embedding_client is None:
            return {"enabled": False}
        try:
            store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
            _, trace = store.ensure_skill_embeddings(
                [skill], self.embedding_client, task=self.embedding_config.document_task, batch_size=1
            )
            return {"enabled": True, **trace}
        except Exception as exc:  # noqa: BLE001 - cache warming is non-critical.
            return {"enabled": True, "error": str(exc)}


def _valid_skill_type(value: Any) -> str:
    skill_type = str(value or "")
    if skill_type not in SKILL_TYPES:
        raise ValueError(f"Invalid skill_type: {skill_type!r}")
    return skill_type


def _valid_complete_skill(value: Any, skill_type: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("create/rewrite requires a complete skill object.")
    card = {
        "value": str(value.get("value") or "").strip(),
        "title": str(value.get("title") or "").strip(),
        "trigger": str(value.get("trigger") or "").strip(),
        "knowledge": (
            validate_issue_knowledge(value.get("knowledge"))
            if skill_type == "issue_skill"
            else validate_atomic_knowledge(value.get("knowledge"))
        ),
    }
    missing = [key for key in ("value", "title", "trigger", "knowledge") if not card[key]]
    if missing:
        raise ValueError(f"Complete skill object missing: {', '.join(missing)}")
    return card


def _apply_issue_knowledge_edit(old: DimensionSkill, raw_edit: Any) -> dict[str, Any]:
    """Apply one stable-ID Issue proposition edit while preserving card identity."""

    if not isinstance(raw_edit, dict):
        raise ValueError("Issue rewrite requires a knowledge_edit object.")
    operation = str(raw_edit.get("operation") or "").strip()
    if operation not in {"add", "replace", "delete"}:
        raise ValueError("Issue knowledge_edit.operation must be add, replace, or delete.")
    raw_target = raw_edit.get("target_knowledge_id")
    try:
        target_id = int(raw_target) if raw_target is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Issue knowledge_edit.target_knowledge_id must be a number or null.") from exc
    items = [dict(item) for item in old.knowledge]
    by_id = {int(item["id"]): index for index, item in enumerate(items)}
    text = str(raw_edit.get("text") or "").strip()
    if operation == "add":
        if target_id is not None:
            raise ValueError("Issue add requires target_knowledge_id=null.")
        if not text:
            raise ValueError("Issue add requires text.")
        items.append({"id": max(by_id, default=0) + 1, "text": text})
    else:
        if target_id is None or target_id not in by_id:
            raise ValueError("Issue replace/delete must target an existing knowledge ID.")
        index = by_id[target_id]
        if operation == "replace":
            if not text:
                raise ValueError("Issue replace requires text.")
            items[index] = {"id": target_id, "text": text}
        else:
            items.pop(index)
            if not items:
                raise ValueError("Issue delete cannot remove the last knowledge proposition.")
    return {
        "value": old.value,
        "title": old.title,
        "trigger": old.trigger,
        "knowledge": validate_issue_knowledge(items),
    }


def _unique_skill_id(existing: list[DimensionSkill], base: str) -> str:
    root = slugify(base)
    used = {skill.skill_id for skill in existing}
    if root not in used:
        return root
    suffix = 2
    while f"{root}_{suffix}" in used:
        suffix += 1
    return f"{root}_{suffix}"


def _valid_retrieval_mode(value: Any) -> str:
    mode = str(value or "lexical").strip().lower()
    if mode not in {"lexical", "embedding", "hybrid"}:
        raise ValueError(f"Invalid retrieval_mode: {mode!r}")
    return mode


def _failed_trace(query: str, note: str) -> dict[str, Any]:
    return {
        "query": query,
        "retrieval_mode": "embedding_failed_v1",
        "candidate_skill_ids": [],
        "selected_skill_ids": [],
        "scores": {},
        "top_scores": [],
        "notes": [note],
    }
