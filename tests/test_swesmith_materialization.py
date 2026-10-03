from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_swesmith_case_by_case as runner


def test_resume_keeps_completed_exploration_and_archives_failed_evolution(tmp_path):
    case = tmp_path / "cases" / "example"
    evolution = case / "case_evolution"
    evolution.mkdir(parents=True)
    result = {"status": "completed", "ranked_functions": ["a.py::f"]}
    (case / "result.json").write_text(json.dumps(result))
    (evolution / "debug.json").write_text("old attempt")
    assert runner.resume_case_artifacts(case, tmp_path) == result
    assert (case / "result.json").exists()
    assert not evolution.exists()
    assert len(list((tmp_path / "resume_attempts").rglob("debug.json"))) == 1


def test_resume_archives_incomplete_explorer_without_deleting_trace(tmp_path):
    case = tmp_path / "cases" / "example"
    case.mkdir(parents=True)
    (case / "trajectory.jsonl").write_text("original observation")
    assert runner.resume_case_artifacts(case, tmp_path) is None
    assert not case.exists()
    archived = list((tmp_path / "resume_attempts").rglob("trajectory.jsonl"))
    assert archived[0].read_text() == "original observation"


PATCH = """diff --git a/pkg/worker.py b/pkg/worker.py
--- a/pkg/worker.py
+++ b/pkg/worker.py
@@ -1,2 +1,2 @@
 def transform(value):
-    return value
+    return None
"""


def _clean_repo(path: Path) -> None:
    (path / "pkg").mkdir(parents=True)
    (path / "pkg" / "worker.py").write_text(
        "def transform(value):\n    return value\n", encoding="utf-8"
    )
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "base"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_swesmith_mutation_is_applied_in_case_specific_repo(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []

    def fake_materialize(image_name: str, repo_dir: Path, *, force: bool, allow_cache: bool) -> Path:
        assert image_name == "example:image"
        assert force is True
        assert allow_cache is False
        calls.append(repo_dir)
        _clean_repo(repo_dir)
        return repo_dir

    monkeypatch.setattr(runner, "materialize_repo", fake_materialize)
    case = {"instance_id": "owner__repo.mutation-a", "image_name": "example:image", "patch": PATCH}

    repo, state = runner.materialize_swesmith_case(case, tmp_path / "repos")

    assert repo.name == "owner__repo_mutation_a"
    assert "return None" in (repo / "pkg" / "worker.py").read_text(encoding="utf-8")
    assert state["source_state"] == "buggy"
    assert state["patch_direction"] == "clean_to_buggy"
    assert json.loads((repo / ".evolutefl_swesmith_case.json").read_text(encoding="utf-8"))["applied"] is True

    reused_repo, reused_state = runner.materialize_swesmith_case(case, tmp_path / "repos")
    assert reused_repo == repo
    assert reused_state["patch_sha256"] == state["patch_sha256"]
    assert calls == [repo]

    second_case = {**case, "instance_id": "owner__repo.mutation-b"}
    second_repo, _ = runner.materialize_swesmith_case(second_case, tmp_path / "repos")
    assert second_repo != repo
    assert calls == [repo, second_repo]


def test_swesmith_mutation_verification_rejects_clean_repository(tmp_path: Path) -> None:
    _clean_repo(tmp_path)

    try:
        runner._verify_applied_mutation(tmp_path, PATCH)
    except RuntimeError as error:
        assert "expected buggy SWE-smith state" in str(error)
    else:
        raise AssertionError("A clean repository must not pass buggy-state verification")
