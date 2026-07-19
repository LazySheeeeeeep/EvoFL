from __future__ import annotations

from copy import deepcopy
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
    build_retrieval_text,
    slugify,
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
        embedding_fallback_to_lexical: bool = False,
        embedding_batch_size: int = 16,
    ) -> None:
        self.path = Path(path)
        self.max_matched_skills = max_matched_skills
        self.max_per_skill_type = {**DEFAULT_SKILL_TYPE_QUOTA, **(max_per_skill_type or {})}
        self.min_score = min_score
        self.project_skill_min_score = project_skill_min_score
        self.retrieval_mode = _valid_retrieval_mode(retrieval_mode)
        self.embedding_client = embedding_client
        default_cache = self.path.parent / "embeddings" / "jina_v3" / "skill_embeddings.jsonl"
        self.embedding_config = EmbeddingRetrievalConfig(
            cache_path=Path(embedding_cache_path) if embedding_cache_path else default_cache,
            model_id=embedding_model_id,
            query_task=embedding_query_task,
            document_task=embedding_document_task,
            min_score=embedding_min_score,
            fallback_to_lexical=embedding_fallback_to_lexical,
            batch_size=embedding_batch_size,
        )

    def load(self) -> list[DimensionSkill]:
        records = read_jsonl(self.path)
        latest_by_id: dict[str, DimensionSkill] = {}
        ordered_ids: list[str] = []
        for record in records:
            try:
                skill = DimensionSkill.from_dict(record)
            except UnsupportedLegacySkillError:
                continue
            if skill.skill_id not in latest_by_id:
                ordered_ids.append(skill.skill_id)
            current = latest_by_id.get(skill.skill_id)
            if current is None or _skill_sort_key(skill) >= _skill_sort_key(current):
                latest_by_id[skill.skill_id] = skill
        return [latest_by_id[skill_id] for skill_id in ordered_ids]

    def active_skills(self) -> list[DimensionSkill]:
        return [skill for skill in self.load() if skill.status == "active"]

    def search_for_explorer(
        self,
        repo: str,
        issue: str,
        issue_abstraction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if issue_abstraction:
            return self._search_for_explorer_with_abstraction(repo, issue, issue_abstraction)

        query = query_embedding_text(repo, issue)
        active = self.active_skills()
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=self.max_matched_skills,
            max_per_skill_type=self.max_per_skill_type,
        )
        scores = trace.get("scores", {})
        reasons = trace.get("match_reasons", {})
        matched_skills = [
            skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
            for skill in selected
        ]
        skill_type_slots = _build_skill_type_slots(active, matched_skills, trace)
        trace["skill_type_slot_summary"] = {
            skill_type: slot["status"] for skill_type, slot in skill_type_slots.items()
        }
        return {
            "matched_skills": matched_skills,
            "skill_type_slots": skill_type_slots,
            "skill_search_trace": trace,
        }

    def _search_for_explorer_with_abstraction(
        self,
        repo: str,
        issue: str,
        issue_abstraction: dict[str, Any],
    ) -> dict[str, Any]:
        active = self.active_skills()
        queries = _skill_type_queries_from_abstraction(repo, issue, issue_abstraction)
        selected_by_id: dict[str, DimensionSkill] = {}
        scores_by_id: dict[str, float] = {}
        reasons_by_id: dict[str, dict[str, Any]] = {}
        skill_type_traces: dict[str, dict[str, Any]] = {}

        for skill_type in SKILL_TYPES:
            query = queries.get(skill_type) or query_embedding_text(repo, issue)
            type_active = [skill for skill in active if skill.skill_type == skill_type]
            quota = self.max_per_skill_type.get(skill_type, 1)
            selected, trace = self._select_for_query(
                query,
                type_active,
                max_matched_skills=quota,
                max_per_skill_type={skill_type: quota},
            )
            skill_type_traces[skill_type] = trace
            trace_scores = trace.get("scores", {}) or {}
            trace_reasons = trace.get("match_reasons", {}) or {}
            for skill in selected:
                selected_by_id[skill.skill_id] = skill
                if skill.skill_id in trace_scores:
                    scores_by_id[skill.skill_id] = trace_scores[skill.skill_id]
                if skill.skill_id in trace_reasons:
                    reasons_by_id[skill.skill_id] = trace_reasons[skill.skill_id]

        selected_skills = _merge_selected_by_quota(
            list(selected_by_id.values()),
            max_matched_skills=self.max_matched_skills,
            max_per_skill_type=self.max_per_skill_type,
        )
        matched_skills = [
            skill.compact_dict(score=scores_by_id.get(skill.skill_id), match_reasons=reasons_by_id.get(skill.skill_id, {}))
            for skill in selected_skills
        ]
        skill_type_slots = _build_skill_type_slots(
            active,
            matched_skills,
            _merge_skill_type_traces(queries, skill_type_traces, selected_skills),
        )
        trace = _merge_skill_type_traces(queries, skill_type_traces, selected_skills)
        trace["skill_type_slot_summary"] = {
            skill_type: slot["status"] for skill_type, slot in skill_type_slots.items()
        }
        return {
            "matched_skills": matched_skills,
            "skill_type_slots": skill_type_slots,
            "skill_search_trace": trace,
        }

    def search_for_reflector(
        self,
        repo: str,
        issue: str,
        insight: dict[str, Any] | None = None,
        *,
        issue_abstraction: dict[str, Any] | None = None,
        explorer_search: dict[str, Any] | None = None,
        project_candidate_limit: int = 5,
        strategy_candidate_limit: int = 5,
    ) -> dict[str, Any]:
        if explorer_search is None and issue_abstraction:
            explorer_search = self.search_for_explorer(
                repo,
                issue,
                issue_abstraction=issue_abstraction,
            )
        if explorer_search is not None:
            return self._reflector_context_from_explorer_search(
                explorer_search,
                insight=insight,
                issue_abstraction=issue_abstraction,
                project_candidate_limit=project_candidate_limit,
                strategy_candidate_limit=strategy_candidate_limit,
            )

        insight = insight or {}
        analysis = insight.get("analysis", {}) if isinstance(insight, dict) else {}
        target_skill_type = analysis.get("missing_or_reinforced_skill_type")
        value = analysis.get("target_value_hint", "unknown")
        query = f"{repo}\n{issue}\n{analysis.get('transferable_lesson', '')}\n{value}"
        active = self.active_skills()
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=4,
            max_per_skill_type={"project_skill": 2, "strategy_skill": 2},
        )
        scores = trace.get("scores", {})
        reasons = trace.get("match_reasons", {})
        same_type = [skill for skill in selected if skill.skill_type == target_skill_type]
        compact = [
            skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
            for skill in selected
        ]
        skill_type_slots = _build_skill_type_slots(active, compact, trace)
        trace["skill_type_slot_summary"] = {
            skill_type: slot["status"] for skill_type, slot in skill_type_slots.items()
        }
        explorer_like_search = {
            "matched_skills": compact,
            "skill_type_slots": skill_type_slots,
            "skill_search_trace": trace,
        }
        context = self._reflector_context_from_explorer_search(
            explorer_like_search,
            insight=insight,
            issue_abstraction=issue_abstraction,
            project_candidate_limit=project_candidate_limit,
            strategy_candidate_limit=strategy_candidate_limit,
        )
        context.update({
            "skill_type_query": {
                "target_skill_type": target_skill_type,
                "target_value": value,
            },
            "same_type_skills": [
                skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
                for skill in same_type
            ],
        })
        return context

    def _reflector_context_from_explorer_search(
        self,
        explorer_search: dict[str, Any],
        *,
        insight: dict[str, Any] | None,
        issue_abstraction: dict[str, Any] | None,
        project_candidate_limit: int,
        strategy_candidate_limit: int,
    ) -> dict[str, Any]:
        active_by_id = {skill.skill_id: skill for skill in self.active_skills()}
        matched = [
            skill for skill in explorer_search.get("matched_skills", []) or [] if isinstance(skill, dict)
        ]
        slots = explorer_search.get("skill_type_slots", {}) or {}
        trace = explorer_search.get("skill_search_trace", {}) or {}
        trace_scores = trace.get("scores", {}) or {}
        candidates_by_id: dict[str, dict[str, Any]] = {}

        def add_candidate(candidate: dict[str, Any], role: str, score: Any = None) -> dict[str, Any]:
            skill_id = str(candidate.get("skill_id") or "")
            existing = candidates_by_id.get(skill_id)
            if existing is not None:
                roles = existing.setdefault("candidate_roles", [existing.get("candidate_role")])
                if role not in roles:
                    roles.append(role)
                if score is not None:
                    try:
                        existing["retrieval_score"] = float(score)
                    except (TypeError, ValueError):
                        pass
                return existing
            item = dict(candidate)
            item["candidate_role"] = role
            item["candidate_roles"] = [role]
            if score is not None:
                try:
                    item["retrieval_score"] = float(score)
                except (TypeError, ValueError):
                    pass
            if skill_id:
                candidates_by_id[skill_id] = item
            return item

        for candidate in matched:
            add_candidate(
                candidate,
                "explorer_matched",
                trace_scores.get(str(candidate.get("skill_id") or "")),
            )
        for skill_type, slot in slots.items():
            if not isinstance(slot, dict):
                continue
            for candidate in slot.get("weak_candidates", []) or []:
                if isinstance(candidate, dict):
                    add_candidate(
                        candidate,
                        f"{skill_type}_weak_candidate",
                        candidate.get("retrieval_score") or candidate.get("score"),
                    )

        project_type_key = str((issue_abstraction or {}).get("project_type_key") or "unknown").strip()
        project_type_description = str(
            (issue_abstraction or {}).get("project_type_description") or ""
        ).strip()
        project_identity_trace: dict[str, Any] = {}
        if project_type_description and project_candidate_limit > 0:
            project_skills = [
                skill for skill in active_by_id.values() if skill.skill_type == "project_skill"
            ]
            _, project_identity_trace = self._select_for_query(
                project_type_description,
                project_skills,
                max_matched_skills=project_candidate_limit,
                max_per_skill_type={"project_skill": project_candidate_limit},
            )

        project_trace = (trace.get("skill_type_traces", {}) or {}).get("project_skill", {}) or {}
        project_top_scores = project_identity_trace.get("top_scores", []) or []
        if not project_top_scores:
            project_top_scores = project_trace.get("top_scores", []) or []
        if not project_top_scores:
            project_top_scores = [
                item
                for item in trace.get("top_scores", []) or []
                if isinstance(item, dict)
                and (item.get("query_skill_type") == "project_skill" or item.get("skill_type") == "project_skill")
            ]
        project_scores_by_id = {
            str(item.get("skill_id") or ""): item.get("score")
            for item in project_top_scores
            if isinstance(item, dict) and item.get("skill_id")
        }
        project_type_candidates: list[dict[str, Any]] = []

        def add_project_type_candidate(skill: DimensionSkill, role: str, score: Any = None) -> None:
            if len(project_type_candidates) >= max(0, project_candidate_limit):
                return
            candidate = add_candidate(skill.compact_dict(), role, score)
            if candidate not in project_type_candidates:
                project_type_candidates.append(candidate)

        if project_type_key and project_type_key != "unknown":
            normalized_key = slugify(project_type_key)
            for skill in active_by_id.values():
                if skill.skill_type == "project_skill" and slugify(skill.value) == normalized_key:
                    add_project_type_candidate(
                        skill,
                        "project_type_exact_candidate",
                        project_scores_by_id.get(skill.skill_id),
                    )
        for score_item in project_top_scores:
            if len(project_type_candidates) >= max(0, project_candidate_limit):
                break
            if not isinstance(score_item, dict):
                continue
            skill = active_by_id.get(str(score_item.get("skill_id") or ""))
            if skill is None or skill.skill_type != "project_skill":
                continue
            add_project_type_candidate(
                skill,
                "project_type_semantic_dedup_candidate",
                score_item.get("score"),
            )

        strategy_trace = (trace.get("skill_type_traces", {}) or {}).get("strategy_skill", {}) or {}
        strategy_top_scores = strategy_trace.get("top_scores", []) or []
        if not strategy_top_scores:
            strategy_top_scores = [
                item
                for item in trace.get("top_scores", []) or []
                if isinstance(item, dict)
                and (item.get("query_skill_type") == "strategy_skill" or item.get("skill_type") == "strategy_skill")
            ]
        strategy_dedup_candidates: list[dict[str, Any]] = []
        for score_item in strategy_top_scores:
            if len(strategy_dedup_candidates) >= max(0, strategy_candidate_limit):
                break
            if not isinstance(score_item, dict):
                continue
            skill_id = str(score_item.get("skill_id") or "")
            skill = active_by_id.get(skill_id)
            if skill is None or skill.skill_type != "strategy_skill":
                continue
            candidate = add_candidate(
                skill.compact_dict(),
                "strategy_semantic_dedup_candidate",
                score_item.get("score"),
            )
            if candidate not in strategy_dedup_candidates:
                strategy_dedup_candidates.append(candidate)

        analysis = (insight or {}).get("analysis", {}) if isinstance(insight, dict) else {}
        target_skill_type = analysis.get("missing_or_reinforced_skill_type")
        same_type = [
            candidate
            for candidate in candidates_by_id.values()
            if candidate.get("skill_type") == target_skill_type
        ]
        return {
            "skill_type_policy": (
                "Explorer-matched skills passed the execution threshold. Strategy dedup candidates are lower-threshold "
                "neighbors supplied only to decide whether the same reusable strategy already exists."
            ),
            "issue_abstraction": issue_abstraction,
            "abstraction_used_for_retrieval": issue_abstraction is not None,
            "skill_type_queries": trace.get("skill_type_queries", {}),
            "skill_type_slots": slots,
            "matched_case_skills": matched,
            "project_type": {
                "key": project_type_key,
                "description": project_type_description,
                "identity_query": project_type_description,
            },
            "project_type_candidates": project_type_candidates,
            "project_identity_trace": project_identity_trace,
            "strategy_dedup_candidates": strategy_dedup_candidates,
            "candidate_target_skills": list(candidates_by_id.values()),
            "same_type_skills": same_type,
            "search_trace": trace,
            "notes": [
                "A strategy dedup candidate is not automatically applicable to Explorer.",
                "A project type candidate represents an existing system-type container and may be updated even when it did not pass Explorer's execution threshold.",
                "Use it as an update or preserve target only when the evidence condition, diagnostic action, and ranking consequence have the same semantic identity.",
            ],
        }

    def rebuild_embeddings(self) -> dict[str, Any]:
        if self.embedding_client is None:
            raise RuntimeError("Embedding client is not configured.")
        active = self.active_skills()
        store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
        _, cache_trace = store.ensure_skill_embeddings(
            active,
            self.embedding_client,
            task=self.embedding_config.document_task,
            batch_size=self.embedding_config.batch_size,
            prune_to_skills=True,
        )
        return {
            "skill_bank_path": str(self.path),
            "active_skill_count": len(active),
            "embedding_cache": cache_trace,
        }

    def _select_for_query(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        if self.retrieval_mode == "lexical":
            return self._select_lexical(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
        if self.retrieval_mode == "embedding":
            return self._select_embedding_or_fallback(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
        embedding_selected, embedding_trace = self._select_embedding_or_fallback(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_skill_type=max_per_skill_type,
            fallback_to_lexical=False,
        )
        lexical_selected, lexical_trace = self._select_lexical(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_skill_type=max_per_skill_type,
        )
        selected = _merge_selected_by_quota(
            [*embedding_selected, *lexical_selected],
            max_matched_skills=max_matched_skills,
            max_per_skill_type=max_per_skill_type,
        )
        trace = _merge_hybrid_trace(query, selected, embedding_trace, lexical_trace)
        return selected, trace

    def _select_lexical(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
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

    def _select_embedding_or_fallback(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_skill_type: dict[str, int],
        fallback_to_lexical: bool | None = None,
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        should_fallback = self.embedding_config.fallback_to_lexical if fallback_to_lexical is None else fallback_to_lexical
        if self.embedding_client is None:
            if should_fallback:
                selected, trace = self._select_lexical(
                    query,
                    active,
                    max_matched_skills=max_matched_skills,
                    max_per_skill_type=max_per_skill_type,
                )
                trace["retrieval_mode"] = "embedding_unconfigured_fallback_to_lexical_v1"
                trace.setdefault("notes", []).append("Embedding client is not configured.")
                return selected, trace
            return [], {
                "query": query,
                "retrieval_mode": "embedding_unconfigured_v1",
                "candidate_skill_ids": [],
                "selected_skill_ids": [],
                "scores": {},
                "match_reasons": {},
                "filtered_below_threshold": [],
                "notes": ["Embedding client is not configured."],
            }
        try:
            return select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=self.embedding_config,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval should degrade, not break Explorer.
            if not should_fallback:
                return [], {
                    "query": query,
                    "retrieval_mode": "embedding_failed_v1",
                    "candidate_skill_ids": [],
                    "selected_skill_ids": [],
                    "scores": {},
                    "match_reasons": {},
                    "filtered_below_threshold": [],
                    "notes": [f"Embedding retrieval failed: {exc}"],
                }
            selected, trace = self._select_lexical(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_skill_type=max_per_skill_type,
            )
            trace["retrieval_mode"] = "embedding_failed_fallback_to_lexical_v1"
            trace.setdefault("notes", []).append(f"Embedding retrieval failed: {exc}")
            return selected, trace

    def apply_update(self, edit: dict[str, Any]) -> dict[str, Any]:
        operation = edit.get("operation")
        if operation == "preserve":
            return self._apply_preserve(edit)
        if operation == "add" and not (edit.get("target") or {}).get("skill_id"):
            return self._apply_create_new(edit)
        if operation in ("add", "replace", "delete"):
            return self._apply_modify_existing(edit)
        raise ValueError(f"Unsupported skill edit operation: {operation!r}")

    def _apply_create_new(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        skill_type = _valid_skill_type(target.get("skill_type"))
        scope = _valid_scope(skill_type, target.get("scope"))
        value = str(target.get("value") or "unknown")
        content = edit.get("content") or {}
        text = str(content.get("text") or content.get("new_text") or "").strip()
        if not text:
            raise ValueError("add edit requires content.text for new skill.")
        title = str(content.get("title") or _title_from_value(value))
        trigger = str(content.get("trigger") or f"Use when the issue matches {value.replace('_', ' ')} localization patterns.")
        retrieval_text = str(
            content.get("retrieval_text")
            or build_retrieval_text(title, trigger, skill_type, value, scope)
        )
        existing = self.load()
        skill_id = _unique_skill_id(existing, f"{skill_type}_{value}_v1")
        skill = DimensionSkill(
            skill_id=skill_id,
            status="active",
            version=1,
            skill_type=skill_type,
            scope=scope,
            value=value,
            retrieval_text=retrieval_text,
            title=title,
            trigger=trigger,
            knowledge=text,
        )
        records = read_jsonl(self.path)
        records.append(skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": "create_new",
            "updated_skill_id": skill.skill_id,
            "version": skill.version,
            "embedding_cache": self._warm_embedding_for_skill(skill),
        }

    def _apply_modify_existing(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        target_skill_id = str(target.get("skill_id") or "")
        if not target_skill_id:
            raise ValueError(f"{edit.get('operation')} edit requires target.skill_id.")
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target_skill_id)
        if not old:
            raise ValueError(f"No active skill found for target.skill_id={target_skill_id!r}.")

        new_skill = deepcopy(old)
        new_skill.version = old.version + 1
        field = str(target.get("field") or "skill.knowledge")
        operation = str(edit.get("operation"))
        _apply_text_edit(new_skill, field, operation, edit.get("content") or {})
        if not new_skill.retrieval_text:
            new_skill.retrieval_text = build_retrieval_text(
                new_skill.title,
                new_skill.trigger,
                new_skill.skill_type,
                new_skill.value,
                new_skill.scope,
            )

        superseded = deepcopy(old)
        superseded.status = "superseded"

        records = read_jsonl(self.path)
        records.append(superseded.to_dict())
        records.append(new_skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": operation,
            "updated_skill_id": new_skill.skill_id,
            "version": new_skill.version,
            "field": field,
            "embedding_cache": self._warm_embedding_for_skill(new_skill),
        }

    def _apply_preserve(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        target_skill_id = str(target.get("skill_id") or "")
        if not target_skill_id:
            return {"action": "preserve", "updated_skill_id": None, "reason": "preserve edit has no target skill"}
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target_skill_id)
        if not old:
            return {"action": "preserve", "updated_skill_id": None, "reason": "target skill not active"}
        return {
            "action": "preserve",
            "updated_skill_id": old.skill_id,
            "version": old.version,
            "embedding_cache": {"enabled": False, "reason": "preserve does not change skill text"},
        }

    def _warm_embedding_for_skill(self, skill: DimensionSkill) -> dict[str, Any]:
        if self.embedding_client is None:
            return {"enabled": False}
        store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
        try:
            _, trace = store.ensure_skill_embeddings(
                [skill],
                self.embedding_client,
                task=self.embedding_config.document_task,
                batch_size=1,
            )
            return {"enabled": True, **trace}
        except Exception as exc:  # noqa: BLE001 - skill update should not fail because the cache is cold.
            return {"enabled": True, "error": str(exc)}


def _valid_skill_type(value: Any) -> str:
    skill_type = str(value or "")
    if skill_type not in SKILL_TYPES:
        raise ValueError(f"Invalid target skill_type: {skill_type!r}")
    return skill_type


def _valid_scope(skill_type: str, value: Any) -> str:
    defaults = {"project_skill": "architecture_family", "strategy_skill": "contextual"}
    allowed = {
        "project_skill": {"repository", "architecture_family"},
        "strategy_skill": {"global", "contextual"},
    }
    scope = str(value or defaults[skill_type])
    if scope not in allowed[skill_type]:
        raise ValueError(f"Invalid scope {scope!r} for {skill_type}.")
    return scope


def _valid_retrieval_mode(value: Any) -> str:
    mode = str(value or "lexical").strip().lower()
    if mode not in {"lexical", "embedding", "hybrid"}:
        raise ValueError(f"Invalid skill retrieval_mode: {mode!r}")
    return mode


def _skill_sort_key(skill: DimensionSkill) -> tuple[int]:
    return (int(skill.version),)


def _merge_selected_by_quota(
    candidates: list[DimensionSkill],
    *,
    max_matched_skills: int,
    max_per_skill_type: dict[str, int],
) -> list[DimensionSkill]:
    selected: list[DimensionSkill] = []
    seen: set[str] = set()
    used_by_skill_type = {skill_type: 0 for skill_type in SKILL_TYPES}
    for skill in candidates:
        if skill.skill_id in seen:
            continue
        skill_type = skill.skill_type
        if used_by_skill_type[skill_type] >= max_per_skill_type.get(skill_type, 1):
            continue
        selected.append(skill)
        seen.add(skill.skill_id)
        used_by_skill_type[skill_type] += 1
        if len(selected) >= max_matched_skills:
            break
    return selected


def _merge_hybrid_trace(
    query: str,
    selected: list[DimensionSkill],
    embedding_trace: dict[str, Any],
    lexical_trace: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    reasons: dict[str, Any] = {}
    for source, trace in (("embedding", embedding_trace), ("lexical", lexical_trace)):
        for skill_id, score in (trace.get("scores") or {}).items():
            try:
                scores[skill_id] = max(float(score), float(scores.get(skill_id, 0.0)))
            except (TypeError, ValueError):
                scores.setdefault(skill_id, 0.0)
        for skill_id, reason in (trace.get("match_reasons") or {}).items():
            reasons.setdefault(skill_id, {})[source] = reason
    filtered = []
    filtered.extend(embedding_trace.get("filtered_below_threshold") or [])
    filtered.extend(lexical_trace.get("filtered_below_threshold") or [])
    notes = ["hybrid retrieval = embedding candidates first, lexical candidates fill remaining slots"]
    notes.extend(embedding_trace.get("notes") or [])
    notes.extend(lexical_trace.get("notes") or [])
    return {
        "query": query,
        "retrieval_mode": "hybrid_skill_type_retrieval_v1",
        "candidate_skill_ids": list(
            dict.fromkeys(
                [
                    *(embedding_trace.get("candidate_skill_ids") or []),
                    *(lexical_trace.get("candidate_skill_ids") or []),
                ]
            )
        ),
        "selected_skill_ids": [skill.skill_id for skill in selected],
        "scores": scores,
        "match_reasons": reasons,
        "filtered_below_threshold": filtered,
        "notes": notes,
        "embedding_trace": embedding_trace,
        "lexical_trace": lexical_trace,
    }


def _skill_type_queries_from_abstraction(
    repo: str,
    issue: str,
    issue_abstraction: dict[str, Any],
) -> dict[str, str]:
    signature = str(issue_abstraction.get("abstract_problem_signature") or "").strip()
    project_query = str(issue_abstraction.get("project_skill_query") or signature or repo).strip()
    project_type_description = str(issue_abstraction.get("project_type_description") or "").strip()
    return {
        "project_skill": "\n".join(
            part for part in (project_type_description, project_query) if part
        ),
        "strategy_skill": str(issue_abstraction.get("strategy_skill_query") or signature or issue).strip(),
    }


def _merge_skill_type_traces(
    skill_type_queries: dict[str, str],
    skill_type_traces: dict[str, dict[str, Any]],
    selected: list[DimensionSkill],
) -> dict[str, Any]:
    candidate_skill_ids: list[str] = []
    top_scores: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    match_reasons: dict[str, dict[str, Any]] = {}
    filtered_below_threshold: list[dict[str, Any]] = []
    notes: list[str] = ["issue_abstraction_used=true"]
    for skill_type, trace in skill_type_traces.items():
        for skill_id in trace.get("candidate_skill_ids") or []:
            if skill_id not in candidate_skill_ids:
                candidate_skill_ids.append(skill_id)
        for item in trace.get("top_scores") or []:
            if isinstance(item, dict):
                top_scores.append({"query_skill_type": skill_type, **item})
        for skill_id, score in (trace.get("scores") or {}).items():
            try:
                scores[skill_id] = max(float(score), float(scores.get(skill_id, 0.0)))
            except (TypeError, ValueError):
                scores.setdefault(skill_id, 0.0)
        for skill_id, reason in (trace.get("match_reasons") or {}).items():
            if isinstance(reason, dict):
                match_reasons.setdefault(skill_id, {})[skill_type] = reason
        for item in trace.get("filtered_below_threshold") or []:
            if isinstance(item, dict):
                filtered_below_threshold.append({"query_skill_type": skill_type, **item})
        for note in trace.get("notes") or []:
            note_text = f"{skill_type}: {note}"
            if note_text not in notes:
                notes.append(note_text)
    return {
        "query": "\n".join(f"{skill_type}: {query}" for skill_type, query in skill_type_queries.items()),
        "retrieval_mode": "abstracted_skill_type_retrieval_v1",
        "skill_type_queries": skill_type_queries,
        "skill_type_traces": skill_type_traces,
        "candidate_skill_ids": candidate_skill_ids,
        "selected_skill_ids": [skill.skill_id for skill in selected],
        "top_scores": top_scores[:20],
        "scores": scores,
        "match_reasons": match_reasons,
        "filtered_below_threshold": filtered_below_threshold,
        "notes": notes,
    }


def _apply_text_edit(skill: DimensionSkill, field: str, operation: str, content: dict[str, Any]) -> None:
    text = str(content.get("text") or "").strip()
    old_text = str(content.get("old_text") or "").strip()
    new_text = str(content.get("new_text") or text).strip()
    if field == "skill.knowledge":
        skill.knowledge = _edit_string(skill.knowledge, operation, text=text, old_text=old_text, new_text=new_text)
    elif field == "skill.trigger":
        skill.trigger = _edit_string(skill.trigger, operation, text=text, old_text=old_text, new_text=new_text)
    elif field == "retrieval_text":
        skill.retrieval_text = _edit_string(skill.retrieval_text, operation, text=text, old_text=old_text, new_text=new_text)
    else:
        raise ValueError(f"Unsupported edit target field: {field!r}")


def _edit_string(current: str, operation: str, *, text: str, old_text: str, new_text: str) -> str:
    if operation == "add":
        addition = text or new_text
        if not addition:
            raise ValueError("add edit requires text.")
        return f"{current.rstrip()}\n\n{addition}".strip() if current else addition
    if operation == "replace":
        if not old_text or not new_text:
            raise ValueError("replace edit requires exact old_text and new_text.")
        if old_text not in current:
            raise ValueError("replace edit old_text was not found in target.")
        return current.replace(old_text, new_text, 1)
    if operation == "delete":
        target = text or old_text
        if not target:
            raise ValueError("delete edit requires text or old_text.")
        if target not in current:
            raise ValueError("delete edit text was not found in target.")
        return current.replace(target, "").strip()
    raise ValueError(f"Unsupported string operation: {operation!r}")


def _title_from_value(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Localization Knowledge"


def _unique_skill_id(existing: list[DimensionSkill], base: str) -> str:
    normalized = slugify(base, fallback="skill_v1")
    existing_ids = {skill.skill_id for skill in existing}
    if normalized not in existing_ids:
        return normalized
    index = 2
    while f"{normalized}_{index}" in existing_ids:
        index += 1
    return f"{normalized}_{index}"


def _build_skill_type_slots(
    active_skills: list[DimensionSkill],
    matched_skills: list[dict[str, Any]],
    trace: dict[str, Any],
    *,
    weak_limit_per_skill_type: int = 2,
) -> dict[str, Any]:
    active_by_id = {skill.skill_id: skill for skill in active_skills}
    matched_by_type: dict[str, list[dict[str, Any]]] = {skill_type: [] for skill_type in SKILL_TYPES}
    for skill in matched_skills:
        skill_type = skill.get("skill_type")
        if skill_type in SKILL_TYPES:
            matched_by_type[skill_type].append(skill)

    scores = trace.get("scores", {})
    reasons = trace.get("match_reasons", {})
    weak_by_type: dict[str, list[dict[str, Any]]] = {skill_type: [] for skill_type in SKILL_TYPES}
    for item in trace.get("filtered_below_threshold", []) or []:
        skill_id = str(item.get("skill_id") or "")
        skill = active_by_id.get(skill_id)
        if not skill:
            continue
        weak = skill.compact_dict(score=scores.get(skill_id, item.get("score")), match_reasons=reasons.get(skill_id, {}))
        weak["threshold"] = item.get("threshold")
        weak_by_type[skill.skill_type].append(weak)

    slots: dict[str, Any] = {}
    for skill_type in SKILL_TYPES:
        weak_candidates = sorted(
            weak_by_type[skill_type],
            key=lambda skill: (-float(skill.get("score") or 0.0), skill.get("skill_id", "")),
        )[:weak_limit_per_skill_type]
        matched = sorted(
            matched_by_type[skill_type],
            key=lambda skill: (-float(skill.get("score") or 0.0), skill.get("skill_id", "")),
        )
        if matched:
            status = "matched"
        elif weak_candidates:
            status = "weak"
        else:
            status = "missing"
        slots[skill_type] = {
            "skill_type": skill_type,
            "status": status,
            "matched_skills": matched,
            "weak_candidates": weak_candidates,
            "guidance": (
                "update the matched skill when it represents the same reusable concept"
                if status == "matched"
                else "update the weak candidate only when semantic identity is clear; otherwise create a new skill"
                if status == "weak"
                else "create a new skill only when the case provides a transferable lesson of this type"
            ),
        }
    return slots
