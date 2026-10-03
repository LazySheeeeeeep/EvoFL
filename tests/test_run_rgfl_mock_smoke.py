import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.run_rgfl_mock_smoke import smoke
from scripts.run_rgfl_file_smoke import map_unique_suffix_files, with_non_thinking
from scripts.run_rgfl_element_smoke import load_file_candidates


class RGFLMockInputTests(TestCase):
    def test_element_stage_rejects_unmatched_or_empty_file_candidates(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "files.jsonl"
            path.write_text(json.dumps({"instance_id": "wrong", "found_files": ["a.py"]}) + "\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_file_candidates(path, "case")
            path.write_text(json.dumps({"instance_id": "case", "found_files": []}) + "\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_file_candidates(path, "case")

    def test_path_adapter_only_accepts_unique_suffixes(self):
        files = [("repo/pkg/core.py", []), ("repo/other/core.py", []),
                 ("repo/pkg/extra.py", [])]

        def exact(names, entries):
            available = {entry[0] for entry in entries}
            return [name for name in names if name in available]

        self.assertEqual(map_unique_suffix_files(
            ["pkg/core.py", "core.py", "repo/pkg/extra.py"], files, original=exact),
            ["repo/pkg/core.py", "repo/pkg/extra.py"])

    def test_deepseek_adapter_sets_non_thinking_without_changing_prompt(self):
        seen = {}

        def request(config, logger, **kwargs):
            seen.update(config=config, kwargs=kwargs)
            return "response"

        config = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "issue"}]}
        self.assertEqual(with_non_thinking(config, None, original=request,
                                           base_url="https://api.deepseek.com"), "response")
        self.assertEqual(seen["config"]["messages"], config["messages"])
        self.assertEqual(seen["config"]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", config)

    def test_rejects_patch_in_public_input(self):
        with TemporaryDirectory() as root:
            path = Path(root)
            (path / "case.jsonl").write_text(json.dumps({
                "instance_id": "repo-1", "repo": "owner/repo", "base_commit": "abc",
                "problem_statement": "issue", "patch": "secret",
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke(path, path / "out")

    def test_rejects_mismatched_base_commit_before_importing_author(self):
        with TemporaryDirectory() as root:
            path = Path(root)
            (path / "case.jsonl").write_text(json.dumps({
                "instance_id": "repo-1", "repo": "owner/repo", "base_commit": "abc",
                "problem_statement": "issue",
            }) + "\n", encoding="utf-8")
            (path / "repo_structures").mkdir()
            (path / "repo_structures/repo-1.json").write_text(json.dumps({
                "instance_id": "repo-1", "repo": "owner/repo", "base_commit": "wrong",
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke(path, path / "out")
