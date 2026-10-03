import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_external_baseline_assets as assets
import prepare_rgfl_reasoning as rgfl


class PreparationTests(unittest.TestCase):
    def test_cached_structure_keeps_runner_contract_and_checks_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, base, output = root / "run", root / "external", root / "full"
            (run / "sources").mkdir(parents=True)
            (run / "materialized").mkdir()
            public = root / "public.jsonl"
            row = {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "abc", "problem_statement": "bug"}
            public.write_text(json.dumps(row) + "\n")
            archive = run / "sources/a__b-1.tar.gz"
            archive.write_bytes(b"audited archive fixture")
            (run / "materialized/a__b-1.json").write_text(json.dumps({"original_archive_sha256": assets.digest(archive)}))
            cached = base / "cosil_pilot_30/prepared/a__b-1"
            (cached / "repo_structures").mkdir(parents=True)
            (cached / "case.jsonl").write_text(json.dumps(row) + "\n")
            (cached / "preparation.json").write_text(json.dumps({"source_archive_sha256": assets.digest(archive)}))
            (cached / "repo_structures/a__b-1.json").write_text(json.dumps({"base_commit": "abc", "structure": {}}))
            with patch.multiple(assets, RUN=run, PUBLIC=public, BASE=base), patch.object(assets, "prepare") as prepare:
                report = assets.run(output, 1)
                self.assertEqual(report["prepared_count"], 1)
                prepare.assert_not_called()
                self.assertEqual(json.loads((output / "prepared/a__b-1/case.jsonl").read_text()), row)
                self.assertTrue((output / "repo_structures/a__b-1.json").is_file())
                archive.write_bytes(b"modified")
                report = assets.run(output, 1)
                self.assertEqual(report["prepared_count"], 0)
                self.assertIn("checksum mismatch", report["cases"][0]["error"])

    def test_author_cost_audit_counts_colliding_method_names(self):
        entries = rgfl.elements("class A:\n def f(self): pass\nclass B:\n def f(self): pass\nx=1\n")
        self.assertEqual(len(entries), 5)
        self.assertEqual(sum(entry["name"] == "f" for entry in entries), 2)

    def test_structure_paths_preserve_repository_prefix(self):
        files = rgfl.source_files({"repo": {"pkg": {"x.py": {"text": ["def f():", " pass"]}}}})
        self.assertEqual(files, {"repo/pkg/x.py": "def f():\n pass"})

    def test_author_prompt_capture_makes_no_real_request(self):
        messages = rgfl.exact_prompt("file_reasoning", "def f(): pass", "Issue fixture")
        self.assertIn("Issue fixture", messages[0]["content"])
        self.assertIn("def f(): pass", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
