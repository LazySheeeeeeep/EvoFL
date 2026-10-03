from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evolutefl.explorer import run_explorer
from evolutefl.investigation import InvestigationLog, read_observations, repository_identity
from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection import run_case_evolution
from evolutefl.reflection.v5_investigator import investigate, validate_conclusion
from evolutefl.skills import make_skill_bank


def call(name, args, id="call"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def tool(name, args):
    return {"content": None, "tool_calls": [call(name, args)]}


def content(value):
    return {"content": json.dumps(value), "tool_calls": []}


def config(tmp_path):
    cfg = json.loads(Path("config/evolutefl.global.json").read_text())
    cfg["skill_bank"]["path"] = str(tmp_path / "skills.jsonl")
    cfg["explorer"]["max_steps"] = 8
    cfg["reflection"]["investigation_response_attempts"] = 1
    return cfg


def fault_args():
    return {"fault_family": "state_assignment", "fault_signature": "wrong stored value", "project_context": "state adapter", "suspected_path": "value assignment"}


def prepare(tmp_path, ranked=None):
    root, directory = tmp_path / "repo", tmp_path / "case"
    root.mkdir()
    (root / "a.py").write_text("def adapt(x):\n    return 0\n\ndef other():\n    return 1\n")
    cfg = config(tmp_path)
    client = FakeLLMClient([
        tool("read_file", {"path": "a.py", "purpose": "Inspect the adapter", "based_on": []}),
        tool("load_fault_skill", {**fault_args(), "based_on": ["obs-00001"], "candidate_updates": [
            {"function": "a.py::adapt", "decision": "retain", "reason": "Drops input", "observation_ids": ["obs-00001"]}]}),
        tool("finish_localization", {"ranked_functions": ranked if ranked is not None else ["a.py::other"], "summary": "Original conclusion"}),
    ])
    task = {"repo": "demo/repo", "repo_path": str(root), "run_dir": str(directory),
            "instance_id": "secret_mutation_identifier", "bug_report": "Input value is lost."}
    result = run_explorer(task=task, config=cfg, llm_client=client)
    assert result["status"] == "completed"
    return root, directory, cfg, client


def conclusion():
    return {"status": "resolved", "summary": "Trace the dropped value", "findings": [{
        "anchor_observation_id": "obs-00001", "available_clue": "Adapter returns a constant", "supporting_observation_ids": ["obs-00001", "supp-00001"],
        "investigation_advice": "Compare the adapter output with the incoming value", "explanation": "The original source shows loss of the input"}], "unresolved_questions": []}


def card():
    return {"fault_family": "state_assignment", "fault_subtype": "lost_input", "title": "Input lost at an adapter",
            "trigger": "Provided values are lost across an adapter", "knowledge": ["Compare the input and output at the first adapter that drops the value."]}


def evolve(root, directory, cfg, client):
    return run_case_evolution(case_run_dir=directory, repo_path=root, repo="demo/repo", issue="Input value is lost.",
        config=cfg, llm_client=client, ground_truth_functions=["a.py::adapt"], ground_truth_patch="training-only-secret-patch",
        patch_metadata={"source": "fixture", "direction": "clean_to_buggy"})


@pytest.mark.parametrize("ranked,focus", [
    (["a.py::adapt"], "evidence_interpretation"),
    (["a.py::other", "a.py::adapt"], "candidate_ranking"),
    (["a.py::other"], "supported_practice"),
    (["a.py::other"], "evidence_acquisition"),
])
def test_v5_reflection_focus_is_independent_of_top5(tmp_path, ranked, focus):
    root, directory, cfg, _ = prepare(tmp_path, ranked=ranked)
    # Old success/failure switches must not select prompts in the active v5 path.
    cfg["reflection"]["fault_success_reflector_prompt_path"] = "nonexistent-success.txt"
    cfg["reflection"]["fault_failure_reflector_prompt_path"] = "nonexistent-failure.txt"
    finding = conclusion()
    finding["findings"][0]["learning_focus"] = focus
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(finding),
        content({"fault_family": "state_assignment", "fault_subtype_query": "inspect input loss"}),
        content({"skill_updates": {"fault_skill": {"decision": "create_new", "target_skill_id": None,
                 "rationale": "Supported investigation lesson", "skill": card()}}})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    assert summary["outcome"]["top5_hit"] == ("a.py::adapt" in ranked)
    assert summary["learning_foci"] == [focus]
    assert summary["reflection_mode"] == "evidence_driven"
    assert fake.calls[-1]["messages"][0]["content"] == Path(cfg["reflection"]["fault_reflector_prompt_path"]).read_text()
    data = json.loads(fake.calls[-1]["messages"][1]["content"])["trajectory_evidence"]
    assert data["investigation_conclusion"]["findings"][0]["learning_focus"] == focus
    assert "training-only-secret-patch" not in json.dumps(data)
    assert "function_metrics" in data["outcome"]


def test_v5_top5_hit_without_supported_lesson_does_not_update(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path, ranked=["a.py::adapt"])
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}),
        content({"status": "unresolved", "summary": "Correct ranking does not establish a supported lesson",
                 "findings": [], "unresolved_questions": ["The basis for the ranking is unclear"]})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["outcome"]["top5_hit"]
    assert summary["decision"] == "no_update" and summary["status"] == "completed"
    assert not make_skill_bank(cfg).active_skills()
    assert len(fake.calls) == 2


def test_v5_rejects_invalid_focus_without_replacing_it_with_outcome(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    bad = conclusion()
    bad["findings"][0]["learning_focus"] = "success"
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(bad)])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "failed"
    assert summary["failure_stage"] == "investigation_protocol"
    assert not make_skill_bank(cfg).active_skills()


def test_v5_records_native_observations_without_answer_or_old_stages(tmp_path):
    root, directory, cfg, client = prepare(tmp_path)
    initial = json.loads((directory / "initial_payload.json").read_text())
    assert "secret_mutation_identifier" not in json.dumps(initial)
    assert "skill_context" not in initial and "issue_abstraction" not in initial
    names = {s["function"]["name"] for s in client.calls[0]["tools"]}
    assert names == {"grep", "read_file", "write", "find_symbol", "read_symbol", "read_observation", "load_fault_skill"}
    assert "finish_localization" in {s["function"]["name"] for s in client.calls[-1]["tools"]}
    observations = read_observations(directory / "observations.jsonl")
    assert observations["obs-00001"]["result"]["content"][0]["text"] == "def adapt(x):"
    timeline = json.loads((directory / "investigation_index.json").read_text())["timeline"]
    assert timeline[0]["candidate_updates"] == []
    assert timeline[1]["candidate_updates"][0]["function"] == "a.py::adapt"
    assert any(m["role"] == "tool" and "obs-00001" in m["content"] for m in client.calls[1]["messages"])


def truncated_response(calls=None):
    return {"content": "", "tool_calls": calls or [], "raw": {"choices": [{
        "finish_reason": "length", "message": {"reasoning_content": "unfinished-reasoning"}}],
        "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}}}


def run_recovery_fixture(tmp_path, responses, *, base_url="https://api.deepseek.com", cfg_overrides=None):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0\n")
    cfg = config(tmp_path)
    cfg["explorer"].update(cfg_overrides or {})
    fake = FakeLLMClient(responses)
    fake.base_url, fake.model, fake.max_tokens = base_url, "deepseek-v4-flash", 4096
    result = run_explorer(task={"repo": "demo", "repo_path": str(root),
        "run_dir": str(tmp_path / "case"), "bug_report": "Wrong value"}, config=cfg, llm_client=fake)
    return result, fake


def test_v5_length_retries_same_step_without_executing_partial_call(tmp_path):
    partial = truncated_response([call("write", {"output_path": "wrong.txt", "content": "wrong"})])
    result, fake = run_recovery_fixture(tmp_path, [partial,
        tool("read_file", {"path": "a.py"}), tool("load_fault_skill", fault_args()),
        tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "Observed"})])
    assert result["status"] == "completed" and result["steps"] == 3
    assert result["llm_request_count"] == 4
    assert result["truncated_response_count"] == result["truncation_recovery_count"] == 1
    assert fake.calls[0]["messages"] == fake.calls[1]["messages"]
    assert fake.calls[0]["tools"] == fake.calls[1]["tools"]
    assert fake.calls[0]["tool_choice"] == fake.calls[1]["tool_choice"]
    assert fake.calls[1]["extra_body"]["thinking"] == {"type": "disabled"}
    assert fake.calls[1]["max_tokens"] == 8192
    assert "extra_body" not in fake.calls[0]
    assert fake.calls[2]["extra_body"]["thinking"] == {"type": "disabled"}
    assert fake.calls[3]["extra_body"]["thinking"] == {"type": "disabled"}
    obs = read_observations(tmp_path / "case/observations.jsonl")
    assert len(obs) == 3 and obs["obs-00001"]["tool"] == "read_file"
    assert "unfinished-reasoning" not in json.dumps(fake.calls)
    trace = json.loads((tmp_path / "case/llm_trace.json").read_text())
    assert [(t["step"], t["attempt"]) for t in trace] == [(1, 1), (1, 2), (2, 1), (3, 1)]


def test_v5_length_exhaustion_has_distinct_error_and_no_side_effects(tmp_path):
    result, fake = run_recovery_fixture(tmp_path, [truncated_response(), truncated_response()])
    assert result["error"] == "response_truncated"
    assert result["status"] == "agent_interrupted" and result["steps"] == 1
    assert len(fake.calls) == 2 and result["truncation_recovery_count"] == 0
    assert result["truncated_response_count"] == 2
    assert not result["fault_skill_attempted"]
    assert not (tmp_path / "case/observations.jsonl").exists()


@pytest.mark.parametrize("url", ["https://api.key77qiqi.com/v1", "https://api.deepseek.com.example.invalid/v1"])
def test_v5_length_recovery_does_not_send_deepseek_options_to_other_providers(tmp_path, url):
    result, fake = run_recovery_fixture(tmp_path, [truncated_response(),
        tool("load_fault_skill", fault_args()),
        tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})], base_url=url)
    assert result["status"] == "completed"
    assert fake.calls[1]["max_tokens"] == 8192
    assert "thinking" not in fake.calls[1]["extra_body"]


def test_v5_thinking_recovery_can_be_disabled(tmp_path):
    _, fake = run_recovery_fixture(tmp_path, [truncated_response(), truncated_response()],
        cfg_overrides={"deepseek_disable_thinking_on_truncation": False})
    assert "thinking" not in fake.calls[1]["extra_body"]


def test_v5_non_length_protocol_error_remains_bounded(tmp_path):
    result, fake = run_recovery_fixture(tmp_path, [{"content": "No tool"}] * 3)
    assert result["error"] == "native_tool_protocol_failed"
    assert len(fake.calls) == 3 and result["truncated_response_count"] == 0


def test_v5_forced_finish_length_recovery_keeps_forced_tool(tmp_path):
    result, fake = run_recovery_fixture(tmp_path, [tool("load_fault_skill", fault_args()),
        truncated_response(), tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})],
        cfg_overrides={"max_steps": 3})
    assert result["status"] == "completed" and result["forced_finish"]
    assert fake.calls[1]["tools"] == fake.calls[2]["tools"]
    assert fake.calls[2]["tool_choice"]["function"]["name"] == "finish_localization"
    assert result["steps"] == 2


def test_v5_successful_deepseek_reasoning_round_trips_without_becoming_observation(tmp_path):
    first = tool("read_file", {"path": "a.py"})
    first["raw"] = {"choices": [{"finish_reason": "tool_calls", "message": {"reasoning_content": "protocol-reasoning"}}]}
    result, fake = run_recovery_fixture(tmp_path, [first, tool("load_fault_skill", fault_args()),
        tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})])
    assert result["status"] == "completed"
    assistants = [m for m in fake.calls[1]["messages"] if m["role"] == "assistant"]
    assert assistants[0]["reasoning_content"] == "protocol-reasoning"
    assert "protocol-reasoning" not in (tmp_path / "case/observations.jsonl").read_text()


def test_v5_length_recovery_checks_runtime_deadline(tmp_path, monkeypatch):
    from evolutefl.explorer.v5_agent import ExplorerDeadlineExceeded, V5ExplorerAgent
    fake = FakeLLMClient([truncated_response()])
    cfg = config(tmp_path)
    agent = V5ExplorerAgent(llm_client=fake, skill_bank=None, config=cfg, system_prompt="localize")
    times = iter([0, 11])
    monkeypatch.setattr("evolutefl.explorer.v5_agent.time.monotonic", lambda: next(times))
    with pytest.raises(ExplorerDeadlineExceeded):
        agent._chat_step({"messages": [], "tools": [], "tool_choice": "auto"},
            step=1, traces=[], directory=tmp_path, deadline=10)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("ranked", [["a.py::other"], ["a.py::adapt"]])
def test_v5_failure_and_success_can_create_skill(tmp_path, ranked):
    root, directory, cfg, _ = prepare(tmp_path, ranked)
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(conclusion()),
        content({"fault_family": "state_assignment", "fault_subtype_query": "trace input loss"}),
        content({"skill_updates": {"fault_skill": {"decision": "create_new", "target_skill_id": None, "rationale": "A transferable investigation", "skill": card()}}})])
    before = repository_identity(root)
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    assert len(make_skill_bank(cfg).active_skills()) == 1
    assert summary["investigation_tool_calls"] == 1
    assert repository_identity(root) == before
    assert "training-only-secret-patch" in fake.calls[0]["messages"][1]["content"]
    assert "training-only-secret-patch" not in fake.calls[-1]["messages"][1]["content"]
    assert set(json.loads((directory / "case_evolution/fault_reflector_output.json").read_text())["skill_updates"]) == {"fault_skill"}


@pytest.mark.parametrize("decision", ["rewrite_existing", "preserve_existing", "no_update"])
def test_v5_remaining_update_decisions(tmp_path, decision):
    root, directory, cfg, _ = prepare(tmp_path)
    bank = make_skill_bank(cfg)
    created = bank.apply_update({"operation": "create", "skill_type": "fault_skill", "skill": card()})
    target = created["updated_skill_id"]
    updated = card()
    updated["knowledge"] = ["Read the adapter before ranking the downstream reporter."]
    update = {"decision": decision, "target_skill_id": target if decision != "no_update" else None,
              "rationale": "Compared the existing pattern", "skill": updated if decision == "rewrite_existing" else None}
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(conclusion()),
        content({"fault_family": "state_assignment", "fault_subtype_query": "trace input loss"}),
        content({"selected_skill_id": target, "reason": "Same pattern"}), content({"skill_updates": {"fault_skill": update}})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    assert summary["decision"] == decision
    assert make_skill_bank(cfg).active_skills()[0].version == (2 if decision == "rewrite_existing" else 1)


def test_v5_unresolved_does_not_write_bank(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    fake = FakeLLMClient([content({"status": "unresolved", "summary": "No supported connection", "findings": []})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["decision"] == "no_update"
    assert make_skill_bank(cfg).active_skills() == []
    assert len(fake.calls) == 1


def test_v5_readonly_budget_and_midtrace_recall(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    cfg["reflection"]["investigation_tool_budget"] = 2
    fake = FakeLLMClient([{"tool_calls": [call("write", {"output_path": "a.py", "content": "bad"}, "w"),
        call("read_observation", {"observation_id": "obs-00001"}, "r"), call("read_file", {"path": "a.py"}, "overflow")]},
        content({"status": "unresolved", "summary": "Budget reached", "findings": []})])
    result = investigate(client=fake, config=cfg, root=root, case_dir=directory, directory=directory / "case_evolution/supplementary", evidence={})
    assert result["tool_call_count"] == 1
    assert fake.calls[-1].get("tools")
    obs = read_observations(directory / "case_evolution/supplementary/observations.jsonl")
    assert obs["supp-00001"]["ok"] is False
    assert "supp-00002" not in obs
    assert (root / "a.py").read_text().startswith("def adapt")


def test_v5_rejects_old_trace_and_changed_source_without_llm(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    (root / "a.py").write_text("changed")
    fake = FakeLLMClient([])
    summary = evolve(root, directory, cfg, fake)
    assert summary["failure_stage"] == "repository_identity"
    (directory / "run_context.json").unlink()
    assert evolve(root, directory, cfg, fake)["failure_stage"] == "eligibility"
    assert fake.calls == []


def test_v5_empty_issue_before_artifacts(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        run_explorer(task={"bug_report": " "}, config=cfg, llm_client=FakeLLMClient([]))


def test_v5_forces_fault_then_finish_and_retries_bad_family(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0")
    cfg = config(tmp_path)
    cfg["explorer"]["max_steps"] = 3
    fake = FakeLLMClient([tool("load_fault_skill", {**fault_args(), "fault_family": "invalid"}),
        tool("load_fault_skill", fault_args()), tool("finish_localization", {"ranked_functions": [], "summary": "No supported candidate"})])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "case"), "bug_report": "wrong value"}, config=cfg, llm_client=fake)
    assert result["status"] == "completed" and result["forced_finish"]
    assert fake.calls[1]["tool_choice"]["function"]["name"] == "load_fault_skill"
    assert any(m["role"] == "user" and "next protocol action is load_fault_skill" in m["content"] for m in fake.calls[1]["messages"])
    assert fake.calls[2]["tool_choice"]["function"]["name"] == "finish_localization"
    assert fake.calls[2]["max_tokens"] == 8192


def test_v5_knowledge_loaded_in_tool_message(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    bank = make_skill_bank(cfg)
    target = bank.apply_update({"operation": "create", "skill_type": "fault_skill", "skill": card()})["updated_skill_id"]
    fake = FakeLLMClient([tool("load_fault_skill", fault_args()), content({"selected_skill_id": target, "reason": "Same symptom"}),
        content({"applicable": True, "reason": "Input loss fits"}), tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "Input dropped"})])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "next"), "bug_report": "Input lost"}, config=cfg, llm_client=fake)
    assert result["status"] == "completed"
    assert any(m["role"] == "tool" and target in m["content"] for m in fake.calls[-1]["messages"])
    assert "knowledge" not in fake.calls[1]["messages"][1]["content"]


def test_v5_cross_family_reflection_updates_one_card_and_next_explorer_loads_alias(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    bank = make_skill_bank(cfg)
    target = bank.apply_update({"operation": "create", "skill_type": "fault_skill", "skill": card()})["updated_skill_id"]
    replacement = {**card(), "retrieval_families": ["state_assignment", "validation_control"],
                   "knowledge": ["Inspect the branch that leaves an accepted input unused."]}
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(conclusion()),
        content({"fault_family": "validation_control", "fault_subtype_query": "branch skips assigning input"}),
        content({"selected_skill_id": target, "reason": "Same input loss through a control-flow entry"}),
        content({"skill_updates": {"fault_skill": {"decision": "rewrite_existing", "target_skill_id": target,
            "rationale": "Both retrieval perspectives support this check", "skill": replacement}}})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    assert not summary["fault_family_consistent"]
    assert summary["updated_skill_ids"] == [target]
    assert summary["retrieval_entry_changes"][0]["added"] == ["validation_control"]
    assert len(bank.active_skills()) == 1 and bank.active_skills()[0].version == 2
    selection = json.loads(fake.calls[3]["messages"][1]["content"])
    assert len(selection["candidates"]) == 1
    assert selection["candidates"][0]["matched_retrieval_families"] == ["state_assignment"]
    assert "knowledge" not in selection["candidates"][0]
    reflection = json.loads(fake.calls[4]["messages"][1]["content"])
    assert reflection["trajectory_evidence"]["runtime_fault_request"]["fault_family"] == "state_assignment"
    assert reflection["skill_search_context"]["candidate_target_skills"][0]["skill_id"] == target
    subsequent = FakeLLMClient([
        tool("load_fault_skill", {**fault_args(), "fault_family": "validation_control"}),
        content({"selected_skill_id": target, "reason": "Branch leaves value unset"}),
        content({"applicable": True, "reason": "Inspecting branch assignment can distinguish the cause"}),
        tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "Input loss confirmed"})])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "next"),
        "bug_report": "Accepted input value is lost"}, config=cfg, llm_client=subsequent)
    assert result["status"] == "completed"
    search = json.loads((tmp_path / "next/fault_skill_search.json").read_text())
    assert search["matched_skill"]["skill_id"] == target
    assert search["matched_skill"]["knowledge"] == replacement["knowledge"]
    assert search["search_trace"]["loaded_via_alias"]


@pytest.mark.parametrize("decision", ["rewrite_existing", "preserve_existing", "no_update"])
def test_v5_disagreement_does_not_automatically_add_retrieval_entry(tmp_path, decision):
    root, directory, cfg, _ = prepare(tmp_path)
    bank = make_skill_bank(cfg)
    target = bank.apply_update({"operation": "create", "skill_type": "fault_skill", "skill": card()})["updated_skill_id"]
    update = {"decision": decision, "target_skill_id": target if decision != "no_update" else None,
              "rationale": "Keep the existing entry; the other classification is not an applicable retrieval perspective",
              "skill": card() if decision == "rewrite_existing" else None}
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(conclusion()),
        content({"fault_family": "validation_control", "fault_subtype_query": "input loss"}),
        content({"selected_skill_id": target, "reason": "Possible matching card"}),
        content({"skill_updates": {"fault_skill": update}})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    assert not bank.fault_catalog("validation_control")["candidate_skills"]
    assert bank.get_active_skill(target)["retrieval_families"] == ["state_assignment"]


def test_v5_new_shared_skill_is_not_duplicated_across_directories(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    value = {**card(), "fault_family": "validation_control",
             "retrieval_families": ["validation_control", "state_assignment"]}
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), content(conclusion()),
        content({"fault_family": "validation_control", "fault_subtype_query": "input lost due to branch"}),
        content({"skill_updates": {"fault_skill": {"decision": "create_new", "target_skill_id": None,
            "rationale": "Both input-state and control-flow entries fit", "skill": value}}})])
    summary = evolve(root, directory, cfg, fake)
    assert summary["status"] == "completed", summary
    bank = make_skill_bank(cfg)
    assert len(bank.active_skills()) == 1
    assert bank.fault_catalog("validation_control")["candidate_skills"][0]["skill_id"] == bank.fault_catalog("state_assignment")["candidate_skills"][0]["skill_id"]


def test_v5_invalid_evidence_reference_rejected(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    raw = read_observations(directory / "observations.jsonl")
    with pytest.raises(ValueError, match="Review"):
        validate_conclusion(conclusion(), raw, {})
    bad = copy.deepcopy(conclusion())
    bad["findings"][0]["anchor_observation_id"] = "not_real"
    with pytest.raises(ValueError, match="anchor"):
        validate_conclusion(bad, raw, {})


def test_v5_mutation_patch_uses_buggy_coordinates(tmp_path):
    from evolutefl.evaluation import functions_from_patch
    (tmp_path / "a.py").write_text("# inserted\n# inserted\n# inserted\ndef target():\n    return 0\n\ndef later():\n    return 1\n")
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +4,2 @@\n def target():\n-    return 2\n+    return 0\n"
    assert functions_from_patch(patch, tmp_path, source_side="new") == ["a.py::target"]


def test_v5_same_turn_future_reference_not_accepted(tmp_path):
    log = InvestigationLog(tmp_path)
    _, first = log.begin(1, call("read_file", {}), {})
    log.observe(first, {"content": []})
    _, second = log.begin(1, call("grep", {}), {"based_on": ["obs-00001"]})
    assert second["based_on"] == []
    assert second["unresolved_references"] == ["obs-00001"]


def test_v5_delivers_full_tool_page_despite_legacy_character_limit(tmp_path):
    log = InvestigationLog(tmp_path, model_observation_char_limit=500)
    _, record = log.begin(1, call("read_file", {}), {})
    message = log.observe(record, {"content": "x" * 3000})
    saved = read_observations(tmp_path / "observations.jsonl")["obs-00001"]
    delivered = json.loads(message["content"])
    assert saved["result"]["content"] == "x" * 3000
    assert saved["model_payload_truncated"] is False
    assert delivered == saved["model_payload"]
    assert delivered["result"]["content"] == "x" * 3000


def test_v5_full_readonly_budget_and_supplementary_source(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    fake = FakeLLMClient([tool("read_file", {"path": "a.py"}) for _ in range(16)] + [
        content({"status": "unresolved", "summary": "Could not connect to an original clue", "findings": []})])
    before = repository_identity(root)
    result = investigate(client=fake, config=cfg, root=root, case_dir=directory,
        directory=directory / "case_evolution/supplementary", evidence={})
    assert result["tool_call_count"] == 16
    assert not fake.calls[-1].get("tools")
    assert fake.calls[-1]["max_tokens"] == 8192
    assert len(read_observations(directory / "case_evolution/supplementary/observations.jsonl")) == 16
    assert repository_identity(root) == before


@pytest.mark.parametrize("previous_content", ["", '{"status":"resolved","summary":"Exact draft: a.py adapt",'])
def test_v5_conclusion_retry_retains_evidence_and_previous_answer(tmp_path, previous_content):
    root, directory, cfg, _ = prepare(tmp_path)
    cfg["reflection"]["investigation_tool_budget"] = 1
    cfg["reflection"]["investigation_response_attempts"] = 2
    blank = {"content": previous_content, "tool_calls": [], "raw": {"choices": [{"message": {
        "reasoning_content": "An unverified claim about fictional.py."}}]}}
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}), blank, content(conclusion())])
    result = investigate(client=fake, config=cfg, root=root, case_dir=directory,
        directory=directory / "case_evolution/supplementary", evidence={"problem_statement": "Input lost"})
    assert result["status"] == "resolved"
    final = fake.calls[-1]
    assert not final.get("tools") and final["max_tokens"] == 8192
    prior = fake.calls[-2]["messages"]
    assert final["messages"][:len(prior)] == prior
    assert any(m["role"] == "tool" and "return 0" in m["content"] for m in final["messages"])
    assert "fictional.py" not in json.dumps(final["messages"])
    assert "Validation error:" in final["messages"][-1]["content"]
    if previous_content:
        assert final["messages"][-2] == {"role": "assistant", "content": previous_content}
    assert result["invalid_response_count"] == 1
    assert (directory / "case_evolution/supplementary/conclusion_retries.jsonl").exists()


def test_v5_prose_with_code_before_fenced_conclusion_needs_no_retry(tmp_path):
    root, directory, cfg, _ = prepare(tmp_path)
    answer = conclusion()
    fake = FakeLLMClient([tool("read_observation", {"observation_id": "obs-00001"}),
        {"content": "The caller passes state={}.\n```json\n" + json.dumps(answer) + "\n```"}])
    result = investigate(client=fake, config=cfg, root=root, case_dir=directory,
        directory=directory / "case_evolution/supplementary", evidence={})
    assert result["findings"] == answer["findings"]
    assert result["invalid_response_count"] == 0
    assert len(fake.calls) == 2


def test_v5_reminds_required_action_after_hallucinated_old_tool(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0")
    cfg = config(tmp_path)
    cfg["explorer"]["max_steps"] = 2
    fake = FakeLLMClient([tool("grep", {"pattern": "adapt"}), tool("load_fault_skill", fault_args()),
        tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})])
    fake.supports_tool_choice = False
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "case"), "bug_report": "wrong value"},
        config=cfg, llm_client=fake)
    assert result["status"] == "completed"
    error = read_observations(tmp_path / "case/observations.jsonl")["obs-00001"]
    assert error["ok"] is False and "Available tools: load_fault_skill" in error["error"]
    assert all(len(c["tools"]) == 1 for c in fake.calls)


def test_v5_executes_one_repository_action_per_turn(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0")
    cfg = config(tmp_path)
    fake = FakeLLMClient([{"tool_calls": [call("read_file", {"path": "a.py"}, "first"),
        call("grep", {"query": "adapt"}, "second")]}, tool("load_fault_skill", fault_args()),
        tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "case"), "bug_report": "wrong value"},
        config=cfg, llm_client=fake)
    assert result["status"] == "completed"
    observations = read_observations(tmp_path / "case/observations.jsonl")
    assert list(observations) == ["obs-00001", "obs-00002", "obs-00003"]


def test_v5_allows_repeated_source_reads_without_candidate_gate(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0")
    cfg = config(tmp_path)
    cfg["explorer"].update(max_steps=7, fault_skill_attempt_step=3, candidate_consolidation_step=4)
    fake = FakeLLMClient([
        tool("read_file", {"path": "a.py"}),
        tool("load_fault_skill", fault_args()),
        tool("grep", {"query": "adapt"}),
        tool("read_file", {"path": "a.py"}),
        tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"}),
    ])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "case"), "bug_report": "wrong value"},
        config=cfg, llm_client=fake)
    assert result["status"] == "completed"
    review_call = fake.calls[3]
    assert {t["function"]["name"] for t in review_call["tools"]} == {
        "grep", "read_file", "write", "find_symbol", "read_symbol", "read_observation", "finish_localization"}
    assert not any(m["role"] == "user" and "candidate-consolidation review" in m["content"] for m in review_call["messages"])
    assert result["forced_fault_skill_attempt"] is False
    assert result["forced_finish"] is False


def test_v5_forces_fault_request_at_early_stage_boundary(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(): return 0")
    cfg = config(tmp_path)
    cfg["explorer"].update(max_steps=8, fault_skill_attempt_step=3)
    fake = FakeLLMClient([tool("read_file", {"path": "a.py"}), tool("grep", {"query": "adapt"}),
        tool("load_fault_skill", fault_args()), tool("finish_localization", {"ranked_functions": [], "summary": "No evidence"})])
    result = run_explorer(task={"repo": "demo", "repo_path": str(root), "run_dir": str(tmp_path / "case"), "bug_report": "wrong value"},
        config=cfg, llm_client=fake)
    assert result["status"] == "completed"
    assert fake.calls[2]["tool_choice"]["function"]["name"] == "load_fault_skill"
    assert result["forced_fault_skill_attempt"] is True
    assert result["forced_finish"] is False


def test_v5_symbol_observation_replay_and_evolution(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(x):\n    return 0\n")
    directory = tmp_path / "case"
    cfg = config(tmp_path)
    client = FakeLLMClient([
        tool("find_symbol", {"name": "adapt"}),
        tool("read_symbol", {"symbol_id": "a.py::adapt"}),
        tool("load_fault_skill", fault_args()),
        tool("read_observation", {"observation_id": "obs-00002"}),
        tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "Adapter drops input"}),
    ])
    result = run_explorer(task={"repo": "demo/repo", "repo_path": str(root), "run_dir": str(directory),
                                "bug_report": "Input value is lost."}, config=cfg, llm_client=client)
    assert result["status"] == "completed" and not result["forced_finish"]
    observations = read_observations(directory / "observations.jsonl")
    assert observations["obs-00004"]["result"] == observations["obs-00002"]["model_payload"]
    finding = conclusion()
    finding["findings"][0].update(anchor_observation_id="obs-00002",
                                  supporting_observation_ids=["obs-00002", "supp-00001"])
    fake = FakeLLMClient([
        tool("read_observation", {"observation_id": "obs-00002"}), content(finding),
        content({"fault_family": "state_assignment", "fault_subtype_query": "trace input loss"}),
        content({"skill_updates": {"fault_skill": {"decision": "create_new", "target_skill_id": None,
                 "rationale": "Trace input loss", "skill": card()}}}),
    ])
    assert evolve(root, directory, cfg, fake)["status"] == "completed"
    assert len(make_skill_bank(cfg).active_skills()) == 1
