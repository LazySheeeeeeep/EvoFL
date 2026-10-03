import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.run_cosil_acquisition_smoke import has_prediction


class CosilSmokeOutputTests(TestCase):
    def test_file_and_function_use_distinct_author_output_names(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "loc_outputs.jsonl").write_text(json.dumps({
                "instance_id": "case", "found_files": ["repo/module.py"],
            }) + "\n", encoding="utf-8")
            self.assertTrue(has_prediction(folder, "case", "file"))
            self.assertFalse(has_prediction(folder, "case", "function"))
            (folder / "loc_outputs_func.jsonl").write_text(json.dumps({
                "instance_id": "case", "found_related_locs": {
                    "repo/module.py": ["function: target"],
                },
            }) + "\n", encoding="utf-8")
            self.assertTrue(has_prediction(folder, "case", "function"))

    def test_empty_prediction_is_not_success(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "loc_outputs.jsonl").write_text(json.dumps({
                "instance_id": "case", "found_files": [],
            }) + "\n", encoding="utf-8")
            self.assertFalse(has_prediction(folder, "case", "file"))
