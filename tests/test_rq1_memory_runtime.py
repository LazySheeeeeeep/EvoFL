import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rq1_memory_runtime.py"
SPEC = importlib.util.spec_from_file_location("rq1_memory_runtime", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self):
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        return {"content": "{}"}


class MemoryRuntimeTest(unittest.TestCase):
    def test_retrieval_excludes_same_case_and_is_deterministic(self):
        episodes = [
            {"instance_id": "same", "issue": "cookie mapping fails", "repo": "a/b"},
            {"instance_id": "relevant", "issue": "cookie mapping fails for jar", "repo": "c/d"},
            {"instance_id": "unrelated", "issue": "polynomial interpolation", "repo": "e/f"},
        ]
        retriever = MODULE.EpisodeRetriever(episodes)
        selected = retriever.retrieve(issue="cookie mapping fails", instance_id="same")
        self.assertEqual(selected[0]["episode"]["instance_id"], "relevant")
        self.assertGreater(selected[0]["score"], 0)

    def test_injection_only_changes_main_explorer_payload(self):
        fake = FakeClient()
        wrapped = MODULE.MemoryInjectedClient(fake, instance_id="case-a", issue="issue",
                                              memory={"source_instance_id": "past"})
        original = [{"role": "system", "content": "system"},
                    {"role": "user", "content": json.dumps({"bug_report": "issue", "fault_families": []})}]
        wrapped.chat(messages=original)
        sent = json.loads(fake.requests[0]["messages"][1]["content"])
        self.assertEqual(sent["historical_memory"]["source_instance_id"], "past")
        self.assertNotIn("historical_memory", json.loads(original[1]["content"]))
        selector = [{"role": "system", "content": "selector"},
                    {"role": "user", "content": json.dumps({"candidates": []})}]
        wrapped.chat(messages=selector)
        self.assertIs(fake.requests[1]["messages"], selector)
        self.assertEqual(wrapped.injection_count, 1)

    def test_episode_item_omits_ground_truth_and_patch(self):
        item = MODULE.episodic_memory_item({"instance_id": "past", "repo": "a/b", "issue": "bug",
            "investigation": [{"tool": "grep", "purpose": "find", "path": "a.py"}],
            "final_summary": "hypothesis", "predicted_functions": ["a.py::f"],
            "patch": "secret", "ground_truth": ["a.py::g"]})
        self.assertEqual(item["past_actions"][0]["tool"], "grep")
        self.assertNotIn("patch", item)
        self.assertNotIn("ground_truth", item)

    def test_issue_mismatch_fails_closed(self):
        wrapped = MODULE.MemoryInjectedClient(FakeClient(), instance_id="case-a", issue="correct",
                                              memory={"source_instance_id": "past"})
        with self.assertRaisesRegex(ValueError, "issue mismatch"):
            wrapped.chat(messages=[{"role": "system", "content": "localize"},
                {"role": "user", "content": json.dumps({"bug_report": "wrong", "fault_families": []})}])


def test_memory_reaches_v5_explorer_without_enabling_skill(tmp_path):
    from evolutefl.explorer import run_explorer
    from evolutefl.llm.client import FakeLLMClient

    def tool(name, args):
        return {"content": None, "tool_calls": [{"id": "call", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n")
    empty = tmp_path / "empty_skills.jsonl"
    empty.touch()
    config = json.loads((Path(__file__).resolve().parents[1] / "config/evolutefl.global.json").read_text())
    config["skill_bank"].update(path=str(empty), enabled_skill_types=[])
    config["explorer"].update(max_steps=5, fault_skill_attempt_step=2)
    fake = FakeLLMClient([
        tool("read_file", {"path": "a.py", "purpose": "inspect", "based_on": []}),
        tool("load_fault_skill", {"fault_family": "state_assignment", "fault_signature": "wrong value",
                                  "project_context": "adapter", "suspected_path": "a.py::f"}),
        tool("finish_localization", {"ranked_functions": ["a.py::f"], "summary": "seen"}),
    ])
    wrapped = MODULE.MemoryInjectedClient(fake, instance_id="current", issue="Wrong value",
                                          memory={"source_instance_id": "historical"})
    result = run_explorer(task={"instance_id": "current", "repo": "owner/repo",
        "repo_path": str(root), "run_dir": str(tmp_path / "run"), "bug_report": "Wrong value"},
        config=config, llm_client=wrapped)
    assert result["status"] == "completed"
    assert wrapped.injection_count == 3
    assert json.loads(fake.calls[0]["messages"][1]["content"])["historical_memory"]["source_instance_id"] == "historical"
    assert json.loads((tmp_path / "run/loaded_skills.json").read_text()) == {"fault_skill": None}


def test_memory_evaluation_requires_finished_acquisition(tmp_path):
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_rq1_memory_baseline",
                                                   scripts / "run_rq1_memory_baseline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (tmp_path / "expanded").mkdir()
    (tmp_path / "preparation").mkdir()
    (tmp_path / "expanded/protocol.json").write_text("{}")
    (tmp_path / "preparation/preparation_summary.json").write_text('{"status":"provisional"}')
    try:
        runner.validate_inputs(tmp_path / "expanded", tmp_path / "preparation", "episodic")
    except ValueError as exc:
        assert "Finish 400" in str(exc)
    else:
        raise AssertionError("Incomplete acquisition must not launch evaluation")


def test_memory_materialization_creates_output_before_disk_check(tmp_path, monkeypatch):
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_rq1_memory_baseline",
                                                   scripts / "run_rq1_memory_baseline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "new-output"

    def check_directory(path):
        assert path == output
        assert path.is_dir()
        raise RuntimeError("disk check reached")

    monkeypatch.setattr(runner.shutil, "disk_usage", check_directory)
    try:
        runner.materialize({}, output, tmp_path, tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "disk check reached"
    else:
        raise AssertionError("Expected disk check")


def test_memory_arms_use_separate_workspaces(tmp_path, monkeypatch):
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_rq1_memory_baseline",
                                                   scripts / "run_rq1_memory_baseline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    roots = []

    def fake_materialize(case, output, expanded, historical):
        roots.append(output)
        return output / "work" / case["instance_id"], tmp_path

    monkeypatch.setattr(runner, "materialize", fake_materialize)
    monkeypatch.setattr(runner, "read_json", lambda path: {"llm": {}, "skill_bank": {"path": ""}})
    monkeypatch.setattr(runner.OpenAICompatibleClient, "from_config", lambda config: object())
    monkeypatch.setattr(runner, "make_skill_bank", lambda config: object())
    monkeypatch.setattr(runner, "run_explorer", lambda **kwargs: {"status": "completed", "ranked_functions": []})

    def fake_clean(workspace):
        assert workspace.is_relative_to(runner.bench.OUT / "work")

    monkeypatch.setattr(runner.bench, "clean_workspace", fake_clean)
    case = {"instance_id": "case", "repo": "owner/repo", "base_commit": "abc",
            "problem_statement": "issue", "function_ground_truth": []}
    selected = {"memory": None, "selected_source_id": None, "score": None}
    for arm in ("episodic", "flat_reflection"):
        runner.evaluate_one(case, arm, selected, expanded=tmp_path, historical=tmp_path,
                            output=tmp_path / "output")
    assert roots == [tmp_path / "output" / "episodic", tmp_path / "output" / "flat_reflection"]


def test_memory_interrupted_case_is_preserved_before_retry(tmp_path, monkeypatch):
    import hashlib
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_rq1_memory_baseline",
                                                   scripts / "run_rq1_memory_baseline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    selected = {"memory": None, "selected_source_id": None, "score": None}
    selection_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    directory = tmp_path / "output/episodic/cases/case"
    directory.mkdir(parents=True)
    (directory / "memory_selection.json").write_text(json.dumps({"selection_sha256": selection_hash}))
    (directory / "trajectory.jsonl").write_text('partial trace\n')

    def stop_after_archive(*args, **kwargs):
        assert not directory.exists()
        archived = directory.with_name("case.interrupted_1")
        assert (archived / "trajectory.jsonl").read_text() == 'partial trace\n'
        raise RuntimeError("archive verified")

    monkeypatch.setattr(runner, "materialize", stop_after_archive)
    case = {"instance_id": "case", "function_ground_truth": []}
    try:
        runner.evaluate_one(case, "episodic", selected, expanded=tmp_path,
                            historical=tmp_path, output=tmp_path / "output")
    except RuntimeError as exc:
        assert str(exc) == "archive verified"
    else:
        raise AssertionError("Expected archive verification")


if __name__ == "__main__":
    unittest.main()
