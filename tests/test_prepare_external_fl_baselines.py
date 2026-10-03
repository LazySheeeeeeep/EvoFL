import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.prepare_external_fl_baselines import prepare


class PrepareExternalFlBaselinesTests(TestCase):
    def test_public_manifest_excludes_labels_and_patch(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            run = root / "run"
            run.mkdir()
            (run / "protocol.json").write_text(json.dumps({
                "evaluation_target": 500,
                "test_created_from": "2024-01-01",
            }), encoding="utf-8")
            cases = [{
                "instance_id": f"repo__case-{i}", "repo": "owner/repo",
                "base_commit": "abc123", "problem_statement": "A failing behavior",
                "created_at": "2024-02-01", "patch": "secret patch",
                "function_ground_truth": ["secret::function"],
            } for i in range(500)]
            (run / "evaluation_cases.json").write_text(json.dumps(cases), encoding="utf-8")
            report = prepare(run, root / "out", root / "sources")
            self.assertEqual(report["evaluation_count"], 500)
            rows = [json.loads(line) for line in (root / "out/evaluation_public.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 500)
            self.assertEqual(set(rows[0]), {"instance_id", "repo", "base_commit", "problem_statement"})
            self.assertNotIn("secret patch", (root / "out/evaluation_public.jsonl").read_text())
