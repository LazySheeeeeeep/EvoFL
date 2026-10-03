import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.prepare_cosil_acquisition_smoke import load_public_case


class PrepareCosilAcquisitionSmokeTests(TestCase):
    def test_public_manifest_selects_only_requested_case(self):
        with TemporaryDirectory() as root:
            manifest = Path(root) / "public.jsonl"
            rows = [
                {"instance_id": "repo__case-1", "repo": "owner/repo",
                 "base_commit": "abc", "problem_statement": "Issue one"},
                {"instance_id": "repo__case-2", "repo": "owner/repo",
                 "base_commit": "def", "problem_statement": "Issue two"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                                encoding="utf-8")
            self.assertEqual(load_public_case("repo__case-2", Path(root), manifest), rows[1])

    def test_duplicate_case_and_unsafe_id_rejected(self):
        with TemporaryDirectory() as root:
            manifest = Path(root) / "public.jsonl"
            row = {"instance_id": "repo__case-1"}
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n",
                                encoding="utf-8")
            with self.assertRaises(ValueError):
                load_public_case("repo__case-1", Path(root), manifest)
            with self.assertRaises(ValueError):
                load_public_case("../outside", Path(root), manifest)
