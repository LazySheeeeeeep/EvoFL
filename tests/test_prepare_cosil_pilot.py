import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.prepare_cosil_pilot import audit, select_cases


def row(case_id, repo):
    return {"instance_id": case_id, "repo": repo,
            "base_commit": "abc123", "problem_statement": "Public issue text"}


class PrepareCosilPilotTests(TestCase):
    def test_selects_distinct_repositories_before_filling(self):
        rows = [row("one-1", "one/repo"), row("one-2", "one/repo"),
                row("two-1", "two/repo"), row("three-1", "three/repo")]
        self.assertEqual([item["instance_id"] for item in select_cases(rows, 3)],
                         ["one-1", "two-1", "three-1"])
        self.assertEqual(len(select_cases(rows, 4)), 4)

    def test_audit_freezes_public_selection_without_labels(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            run, output = root / "run", root / "output"
            (run / "sources").mkdir(parents=True)
            (run / "materialized").mkdir()
            rows = [row("one-1", "one/repo"), row("two-1", "two/repo")]
            public = root / "public.jsonl"
            public.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            (run / "sources" / "one-1.tar.gz").write_bytes(b"base source")
            (run / "materialized" / "one-1.json").write_text("{}", encoding="utf-8")
            report = audit(run, public, output, limit=2)
            self.assertEqual(report["ready_ids"], ["one-1"])
            self.assertEqual(report["prepared_count"], 0)
            self.assertEqual(len((output / "selected_cases.jsonl").read_text().splitlines()), 2)
            self.assertNotIn("patch", (output / "selected_cases.jsonl").read_text())

    def test_rejects_nonpublic_fields(self):
        with self.assertRaises(ValueError):
            select_cases([{**row("one-1", "one/repo"), "patch": "secret"}])
