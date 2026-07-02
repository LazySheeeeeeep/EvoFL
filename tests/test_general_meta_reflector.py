from __future__ import annotations

import json
from pathlib import Path

from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection.general_meta_reflector import collect_residual_cards, run_general_reflection
from evolutefl.skills.bank import SkillBankV0


def _config(skill_path: Path) -> dict:
    return {
        "llm": {},
        "reflection": {
            "general_reflector_prompt_path": "prompt_records/reflection/general_meta_reflector_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
    }


def _write_card(path: Path, instance_id: str, lesson: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "outcome_type": "failure",
                "dimension_coverage": {
                    "project_type": {"verdict": "irrelevant", "value": "unknown", "evidence": "Not project-specific."},
                    "fault_mode": {"verdict": "weak", "value": "unknown", "evidence": "No single fault mode explains it."},
                    "strategy_type": {"verdict": "weak", "value": "unknown", "evidence": "No single strategy explains it."},
                },
                "best_explaining_dimension": "none",
                "residual_lesson": lesson,
                "residual_reason": "The lesson crosses ordinary dimensions.",
                "supporting_evidence": ["Explorer over-trusted surface reporter."],
            }
        ),
        encoding="utf-8",
    )


def test_general_meta_reflection_creates_general_skill_from_residual_cards(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_card(run_dir / "case1" / "case_evolution" / "residual_card.json", "case1", "Reporter functions can mask root-cause evidence.")
    _write_card(run_dir / "case2" / "case_evolution" / "residual_card.json", "case2", "Reporter functions can mask root-cause evidence.")
    reflector = {
        "batch_summary": "Two cases share a residual reporter/root-cause confusion.",
        "dimension_coverage_summary": "The lesson is not tied to one project, fault mode, or strategy label.",
        "residual_patterns": [
            {
                "pattern_name": "surface_reporter_residual",
                "supporting_cases": ["case1", "case2"],
                "why_not_project_type": "Cases are not repository-family specific.",
                "why_not_fault_mode": "The symptom appears across mechanisms.",
                "why_not_strategy_type": "It is broader than one named strategy.",
                "general_lesson": "Treat reporter-like functions as weaker candidates until causal evidence links them to state creation.",
                "decision": "create_new",
                "target_skill_id": None,
            }
        ],
        "proposed_general_updates": [
            {
                "decision": "create_new",
                "target_value": "surface_reporter_residual",
                "target_skill_id": None,
                "rationale": "Multiple residual cards support the same cross-dimensional lesson.",
                "supporting_cases": ["case1", "case2"],
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "evidence": ["Both cards describe reporter/root-cause confusion."],
                    "content": {
                        "text": "Reporter-like functions are weaker candidates until code evidence shows they create or transform the faulty state.",
                        "title": "Surface reporter residual",
                        "trigger": "Use when issue wording or final outputs point strongly at reporter functions.",
                        "retrieval_text": "surface reporter root cause residual causal evidence",
                    },
                    "expected_effect": "Reduce over-ranking of symptom reporters.",
                    "risk": "May under-rank reporters that also mutate state.",
                },
            }
        ],
        "no_update_reason": None,
    }
    fake = FakeLLMClient([{"content": json.dumps(reflector)}])

    summary = run_general_reflection(
        input_run_dir=run_dir,
        config=_config(skill_path),
        llm_client=fake,
        output_dir=tmp_path / "general_out",
    )

    assert summary["residual_card_count"] == 2
    assert summary["updated_skill_ids"]
    active = SkillBankV0(skill_path).active_skills()
    assert len(active) == 1
    assert active[0].dimension == "general"
    assert active[0].supported_by_cases == ["case1", "case2"]


def test_collect_residual_cards_supports_window_size(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    run_dir = tmp_path / "run"
    _write_card(run_dir / "case1" / "case_evolution" / "residual_card.json", "case1", "first")
    _write_card(run_dir / "case2" / "case_evolution" / "residual_card.json", "case2", "second")

    cards = collect_residual_cards(run_dir, window_size=1)

    assert len(cards) == 1
    assert cards[0]["instance_id"] == "case2"


def test_general_meta_reflection_merges_multiple_minibatch_candidates(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_card(run_dir / "case1" / "case_evolution" / "residual_card.json", "case1", "Reporter functions can mask root-cause evidence.")
    _write_card(run_dir / "case2" / "case_evolution" / "residual_card.json", "case2", "Surface outputs can hide the causal state writer.")
    analyst_1 = {
        "batch_summary": "case1 suggests reporter/root-cause confusion",
        "dimension_coverage_summary": "",
        "common_residual_patterns": [],
        "general_patch_candidates": [
            {
                "decision": "create_new",
                "target_value": "surface_reporter_residual",
                "target_skill_id": None,
                "rationale": "case1 supports the pattern",
                "supporting_cases": ["case1"],
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": "Treat reporter-like functions as weaker candidates until causal evidence links them to state creation.",
                        "title": "Surface reporter residual",
                        "trigger": "Use when issue wording points at reporter functions.",
                        "retrieval_text": "surface reporter root cause causal state",
                    },
                },
            }
        ],
    }
    analyst_2 = {
        "batch_summary": "case2 suggests the same residual pattern",
        "dimension_coverage_summary": "",
        "common_residual_patterns": [],
        "general_patch_candidates": [
            {
                "decision": "create_new",
                "target_value": "surface_reporter_residual",
                "target_skill_id": None,
                "rationale": "case2 supports the pattern",
                "supporting_cases": ["case2"],
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": "Prefer the first state writer over output/reporting functions when both are plausible.",
                        "title": "Surface reporter residual",
                        "trigger": "Use when final outputs point at reporters but upstream state writers exist.",
                        "retrieval_text": "surface reporter upstream state writer causal evidence",
                    },
                },
            }
        ],
    }
    merged = {
        "batch_summary": "Two minibatches agree on reporter/root-cause confusion.",
        "dimension_coverage_summary": "",
        "residual_patterns": [],
        "proposed_general_updates": [
            {
                "decision": "create_new",
                "target_value": "surface_reporter_residual",
                "target_skill_id": None,
                "rationale": "Both analyst candidates support the same general residual principle.",
                "supporting_cases": ["case1", "case2"],
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": "Treat reporter-like functions as weaker candidates until code evidence shows they create or transform the faulty state.",
                        "title": "Surface reporter residual",
                        "trigger": "Use when issue wording or final outputs point strongly at reporter functions.",
                        "retrieval_text": "surface reporter root cause residual causal evidence",
                    },
                },
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {"content": json.dumps(analyst_1)},
            {"content": json.dumps(analyst_2)},
            {"content": json.dumps(merged)},
        ]
    )

    summary = run_general_reflection(
        input_run_dir=run_dir,
        config=_config(skill_path),
        llm_client=fake,
        output_dir=tmp_path / "general_out",
        minibatch_size=1,
    )

    assert len(fake.calls) == 3
    assert summary["analyst_batch_count"] == 2
    assert summary["analyst_candidate_count"] == 2
    assert summary["updated_skill_ids"]
    assert (tmp_path / "general_out" / "compact_residual_cards.json").exists()
    assert (tmp_path / "general_out" / "general_analyst_outputs.json").exists()
