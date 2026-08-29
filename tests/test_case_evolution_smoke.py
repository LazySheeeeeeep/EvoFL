from __future__ import annotations

import json
from pathlib import Path

from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection import run_case_evolution
from evolutefl.explorer import run_explorer
from evolutefl.skills.bank import SkillBankV0


def _write_case(path: Path, *, staged: bool = True) -> None:
    path.mkdir()
    (path / "result.json").write_text(json.dumps({"instance_id": "case1", "status": "completed", "ranked_functions": ["wrong.py::report"], "final_summary": "wrong"}), encoding="utf-8")
    events = [{"event": "tool_result", "step": 1, "name": "grep", "content": "{}"}]
    if staged:
        events.extend([
            {"event": "project_skill_request", "step": 2, "query": "service state boundary"},
            {"event": "project_skill_loaded", "step": 2, "loaded_skill_id": None, "search_trace": {}},
            {"event": "strategy_skill_request", "step": 4, "query": "producer reporter contrast"},
            {"event": "strategy_skill_loaded", "step": 4, "loaded_skill_id": None, "search_trace": {}},
        ])
    events.append({"event": "finish", "step": 5})
    (path / "trajectory.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _config(skill_path: Path) -> dict:
    return {
        "llm": {},
        "reflection": {
            "evolution_query_prompt_path": "prompt_records/reflection/evolution_query_v0.txt",
            "reflector_prompt_path": "prompt_records/reflection/reflector_skill_v0.txt",
            "reflector_candidate_top_k": 5,
        },
        "skill_bank": {"path": str(skill_path), "min_score": 1, "project_skill_min_score": 1},
    }


def _no_update() -> dict:
    return {"decision": "no_update", "target_skill_id": None, "rationale": "No lesson.", "skill": None, "no_update_reason": "No lesson."}


def test_case_evolution_generates_queries_and_skill(tmpdir) -> None:
    root = Path(str(tmpdir))
    case = root / "case"
    _write_case(case)
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    reflector = {
        "case_summary": "Create a strategy.",
        "skill_updates": {
            "project_skill": _no_update(),
            "strategy_skill": {
                "decision": "create_new", "target_skill_id": None, "rationale": "Reusable ranking lesson.",
                "skill": {"value": "producer_before_reporter", "title": "Producer before reporter", "trigger": "Use when reporters compete with state producers.", "knowledge": ["Compare where invalid state is first created with where it is later exposed.", "Rank the causal producer above the reporter."]},
                "no_update_reason": None,
            },
        },
        "no_update_reason": None,
    }
    fake = FakeLLMClient([
        {"content": json.dumps({
            "project_skill_query": "service state responsibility boundary",
            "strategy_diagnostic_context": "A producer competes with a downstream reporter.",
            "selected_strategy_skill_id": None,
            "strategy_selection_reason": "The empty catalog has no applicable trigger.",
        })},
            {"content": json.dumps(reflector)},
            {"content": json.dumps({
                "semantic_center": "Determine whether an invalid state originates at an upstream producer or is only exposed by a later reporter.",
                "roles": ["upstream state producer", "later symptom reporter"],
                "contract": "The earliest role that violates the state contract should rank first.",
            })},
            {"content": json.dumps({
            "skill": {
                "value": "source_state_propagation_contrast",
                "title": "Source-state propagation contrast",
                "trigger": "Use when a source role and a later reporter compete as the cause of an incorrect state.",
                "knowledge": [
                    "Compare where the state is first produced with where it is later exposed.",
                    "A state that is already invalid at production supports the source role.",
                    "Rank the earliest role whose observed state violates the expected contract.",
                ],
            }
        })},
    ])
    summary = run_case_evolution(case_run_dir=case, repo="demo/repo", issue="wrong state", config=_config(skills), llm_client=fake, ground_truth_functions=["right.py::produce"], ground_truth_patch="diff --git a/right.py b/right.py")
    assert summary["eligible"] is True
    assert summary["evolution_queries"]["project_skill_query"]
    assert summary["updated_skill_types"] == ["strategy_skill"]


def test_v3_case_evolution_separates_project_builder_from_case_reflector(tmpdir) -> None:
    root = Path(str(tmpdir))
    case = root / "v3_case"
    case.mkdir()
    (case / "result.json").write_text(json.dumps({
        "instance_id": "v3_case", "status": "completed", "ranked_functions": ["wrong.py::report"],
    }), encoding="utf-8")
    events = [
        {"event": "project_skill_request", "step": 1, "request": {"project_type": "service"}},
        {"event": "project_skill_loaded", "step": 1, "loaded_skill_id": None},
        {"event": "issue_revealed", "step": 1, "issue_payload": {"issue": "wrong state"}},
        {"event": "issue_skill_request", "step": 2, "request": {"issue_type": "wrong state"}},
        {"event": "issue_skill_loaded", "step": 2, "loaded_skill_id": None},
        {"event": "strategy_skill_request", "step": 3, "request": {}},
        {"event": "strategy_skill_loaded", "step": 3, "loaded_skill_id": None},
        {"event": "finish", "step": 4, "result": {"action": {"ranked_functions": ["wrong.py::report"]}}},
    ]
    (case / "trajectory.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    config = _config(skills)
    config["explorer"] = {"workflow_version": "v3"}
    fake = FakeLLMClient([
        {"content": json.dumps({"decision": "preserve_existing", "reason": "orientation is insufficient"})},
        {"content": json.dumps({"issue_skill_query": "incorrect state through adapter boundary", "strategy_diagnostic_context": "producer competes with reporter", "selected_strategy_skill_id": None, "strategy_selection_reason": "no catalog match"})},
        {"content": json.dumps({"case_summary": "issue lesson", "skill_updates": {
            "issue_skill": {"decision": "create_new", "target_skill_id": None, "rationale": "reusable path", "skill": {"value": "wrong_state_boundary", "title": "Wrong state boundary", "trigger": "Use when a reported state differs after an adapter boundary.", "knowledge": [{"text": "When reported state diverges, inspect the adaptation boundary that first changes its representation."}]}, "knowledge_edit": None, "no_update_reason": None},
        }, "no_update_reason": None})},
        {"content": json.dumps({"case_summary": "ranking not applicable", "skill_updates": {
            "strategy_skill": _no_update(),
        }, "no_update_reason": "Ground truth was absent from Top-5."})},
    ])
    summary = run_case_evolution(case_run_dir=case, repo="demo/repo", issue="wrong state", config=config,
                                  llm_client=fake, ground_truth_functions=["right.py::produce"])
    assert summary["eligible"] is True
    assert (case / "case_evolution" / "project_knowledge_evidence.json").exists()
    project_evidence = json.loads(
        (case / "case_evolution" / "project_knowledge_evidence.json").read_text(encoding="utf-8")
    )
    assert "wrong state" not in json.dumps(project_evidence)
    assert any(item.get("updated_skill_id") for item in summary["applied_updates"])
    records = [json.loads(line) for line in skills.read_text(encoding="utf-8").splitlines() if line]
    assert [record["skill_type"] for record in records] == ["issue_skill"]
    assert (case / "case_evolution" / "reflector_skill_search_context.json").exists()
    assert (case / "case_evolution" / "issue_reflector_output.json").exists()
    assert (case / "case_evolution" / "strategy_reflector_output.json").exists()
    assert summary["learning_signals"]["issue_skill"]["label"] == "failure"
    assert summary["learning_signals"]["strategy_skill"]["label"] == "not_applicable"
    issue_call_payload = json.loads(fake.calls[2]["messages"][1]["content"])
    strategy_call_payload = json.loads(fake.calls[3]["messages"][1]["content"])
    assert set(issue_call_payload["skill_search_context"]["candidates_by_skill_type"]) == {"issue_skill"}
    assert set(strategy_call_payload["skill_search_context"]["candidates_by_skill_type"]) == {"strategy_skill"}
    assert issue_call_payload["trajectory_evidence"]["reflection_scope"].startswith("Issue-guided")
    assert strategy_call_payload["trajectory_evidence"]["reflection_scope"].startswith("Comparison")
    assert len(json.loads(skills.read_text(encoding="utf-8").splitlines()[0])) > 0


def test_old_trajectory_is_rejected(tmpdir) -> None:
    root = Path(str(tmpdir))
    case = root / "case"
    _write_case(case, staged=False)
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    fake = FakeLLMClient([])
    summary = run_case_evolution(case_run_dir=case, repo="demo/repo", issue="bug", config=_config(skills), llm_client=fake, force=True)
    assert summary["eligible"] is False
    assert "incompatible trajectory" in summary["reason"]
    assert fake.calls == []


def test_case_evolution_uses_normalized_function_identity_for_outcome(tmpdir) -> None:
    root = Path(str(tmpdir))
    case = root / "case"
    _write_case(case)
    (case / "result.json").write_text(
        json.dumps(
            {
                "instance_id": "case1",
                "status": "completed",
                "ranked_functions": ["package.module.Writer.write"],
            }
        ),
        encoding="utf-8",
    )
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")

    summary = run_case_evolution(
        case_run_dir=case,
        repo="demo/repo",
        issue="wrong output",
        config=_config(skills),
        llm_client=FakeLLMClient([]),
        ground_truth_functions=["package/module.py::Writer.write"],
    )

    assert summary["outcome"]["top5_hit"] is True
    assert summary["eligible"] is False
    assert "top-5 hit" in summary["reason"]


def test_generated_skill_can_load_in_next_staged_case(tmpdir) -> None:
    root = Path(str(tmpdir))
    case = root / "case"
    _write_case(case)
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    reflector = {
        "case_summary": "Create strategy.",
        "skill_updates": {
            "project_skill": _no_update(),
            "strategy_skill": {
                "decision": "create_new", "target_skill_id": None, "rationale": "Reusable contrast.",
                "skill": {"value": "producer_reporter_contrast", "title": "Producer reporter contrast", "trigger": "Use when a reporter competes with a state producer.", "knowledge": ["Compare the first invalid state producer with later symptom reporting.", "Rank the causal producer above symptom reporters."]},
                "no_update_reason": None,
            },
        },
        "no_update_reason": None,
    }
    evolution_fake = FakeLLMClient([
        {"content": json.dumps({
            "project_skill_query": "service boundary",
            "strategy_diagnostic_context": "A producer competes with a downstream reporter.",
            "selected_strategy_skill_id": None,
            "strategy_selection_reason": "The empty catalog has no applicable trigger.",
        })},
            {"content": json.dumps(reflector)},
            {"content": json.dumps({
                "semantic_center": "Determine whether an invalid state originates at an upstream producer or is only exposed by a later reporter.",
                "roles": ["upstream state producer", "later symptom reporter"],
                "contract": "The earliest role that violates the state contract should rank first.",
            })},
            {"content": json.dumps({
            "skill": {
                "value": "producer_reporter_contrast",
                "title": "State-origin and symptom contrast",
                "trigger": "Use when a source role and a later reporter compete as the cause of an incorrect state.",
                "knowledge": [
                    "Compare where the state is first produced with where it is later exposed.",
                    "A state that is already invalid at production supports the source role.",
                    "Rank the earliest role whose observed state violates the expected contract.",
                ],
            }
        })},
    ])
    run_case_evolution(case_run_dir=case, repo="demo/repo", issue="bug", config=_config(skills), llm_client=evolution_fake, ground_truth_functions=["right.py::produce"])

    repo_dir = root / "repo"
    repo_dir.mkdir()
    (repo_dir / "right.py").write_text("def produce():\n    return 1\n", encoding="utf-8")
    def call(name: str, arguments: dict, call_id: str) -> dict:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    explorer_fake = FakeLLMClient([
        {"tool_calls": [call("load_project_skill", {"project_type": "layered service system", "architecture_signature": "State producers feed response consumers.", "component_roles": ["state producer", "response consumer"], "responsibility_boundary": "state"}, "p")]},
        {"tool_calls": [call("load_strategy_skill", {"selected_skill_id": "strategy_skill_producer_reporter_contrast_v1", "candidate_functions": ["right.py::produce"], "candidate_relationship": "A source role and later reporter compete.", "unresolved_question": "Which role causes the incorrect state?", "observed_evidence": "The source produces invalid state before the later reporter exposes it.", "selection_reason": "The source-versus-reporter trigger applies."}, "s")]},
        {"content": json.dumps({"approved": True, "condition_evidence": ["The producer and reporter are competing candidates.", "The reporter exposes state already produced by the source candidate."], "reason": "The source-versus-reporter trigger is established."})},
        {"tool_calls": [call("finish_localization", {"ranked_functions": ["right.py::produce"], "summary": "evidence"}, "f")]},
    ])
    run_dir = root / "next"
    explorer_config = _config(skills)
    explorer_config["explorer"] = {"max_steps": 6, "finalization_steps": 3, "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt"}
    run_explorer(
        task={"instance_id": "next", "repo_path": str(repo_dir), "repo": "demo/repo", "base_commit": "", "bug_report": "wrong state", "run_dir": str(run_dir)},
        config=explorer_config,
        llm_client=explorer_fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )
    loaded = json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8"))
    assert loaded["strategy_skill"]["value"] == "producer_reporter_contrast"
