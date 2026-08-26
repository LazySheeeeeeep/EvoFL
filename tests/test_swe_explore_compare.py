import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_swe_explore_explorer_compare import (
    build_parser,
    build_retrieval_rows,
    make_type_limited_skill_config,
    score_function_case,
)
from run_swebench_verified_explorer_compare import _windows_host_path as verified_windows_host_path
from run_swesmith_case_by_case import _windows_host_path, is_usable_source_repo


def test_score_function_case_uses_patch_function_ground_truth() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        source = repo / "pkg" / "worker.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "class Worker:\n"
            "    def transform(self, value):\n"
            "        return value\n",
            encoding="utf-8",
        )
        result_path = root / "result.json"
        result_path.write_text(
            json.dumps({"ranked_functions": ["pkg/worker.py::Worker.transform"]}),
            encoding="utf-8",
        )
        case = {
            "instance_id": "example__repo-1",
            "patch": (
                "diff --git a/pkg/worker.py b/pkg/worker.py\n"
                "--- a/pkg/worker.py\n"
                "+++ b/pkg/worker.py\n"
                "@@ -1,3 +1,3 @@\n"
                " class Worker:\n"
                "     def transform(self, value):\n"
                "-        return value\n"
                "+        return normalize(value)\n"
            ),
        }

        row = score_function_case(case, result_path, repo)

        assert row["evaluable"] is True
        assert row["ground_truth_functions"] == ["pkg/worker.py::Worker.transform"]
        assert row["top1"] is True


def test_retrieval_rows_read_staged_loaded_skills() -> None:
    with TemporaryDirectory() as temporary:
        arm_dir = Path(temporary)
        case_dir = arm_dir / "cases" / "example__repo-1"
        case_dir.mkdir(parents=True)
        (case_dir / "loaded_skills.json").write_text(
            json.dumps(
                {
                    "project_skill": {
                        "skill_id": "project_skill_example_v1",
                        "skill_type": "project_skill",
                    },
                    "strategy_skill": {
                        "skill_id": "strategy_skill_example_v1",
                        "skill_type": "strategy_skill",
                    },
                }
            ),
            encoding="utf-8",
        )
        (case_dir / "project_skill_search.json").write_text(
            json.dumps({"search_trace": {"top_scores": [{"score": 0.72}]}}),
            encoding="utf-8",
        )
        (case_dir / "strategy_skill_search.json").write_text(
            json.dumps({"search_trace": {"retrieval_mode": "llm_trigger_catalog_selection_v1"}}),
            encoding="utf-8",
        )

        rows = build_retrieval_rows(
            arm_dir,
            [
                {
                    "case": "example__repo-1",
                    "evaluable": True,
                    "top1": False,
                    "top3": True,
                    "top5": True,
                    "rank": 2,
                    "reciprocal_rank": 0.5,
                }
            ],
        )

        assert rows[0]["matched_skill_count"] == 2
        assert rows[0]["has_project_skill"] is True
        assert rows[0]["has_strategy_skill"] is True
        assert rows[0]["project_top_score"] == 0.72
        assert rows[0]["mrr"] == 0.5


def test_type_limited_skill_config_keeps_one_skill_type_visible() -> None:
    config = {"skill_bank": {"path": "skills.jsonl"}}

    project_only = make_type_limited_skill_config(config, "project_skill")

    assert project_only["skill_bank"]["enabled_skill_types"] == ["project_skill"]
    assert "enabled_skill_types" not in config["skill_bank"]


def test_compare_uses_arm_local_repo_copies_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.repo_mode == "copy"
    assert args.runtime_repo_root == ""
    assert args.llm_max_attempts is None
    assert args.llm_timeout is None


def test_swesmith_cache_requires_a_git_repo_with_python_source() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / ".git").mkdir()
        (root / "generated.pyc").write_bytes(b"cache")
        assert is_usable_source_repo(root) is False

        (root / "module.py").write_text("def run():\n    return 1\n", encoding="utf-8")
        assert is_usable_source_repo(root) is True


def test_wsl_mount_is_converted_for_docker_exe_copy() -> None:
    assert _windows_host_path(Path("/mnt/d/projects/EvoluteFL/runs/repo")) == r"D:\projects\EvoluteFL\runs\repo"
    assert verified_windows_host_path(Path("/mnt/d/projects/EvoluteFL/runs/repo")) == r"D:\projects\EvoluteFL\runs\repo"
