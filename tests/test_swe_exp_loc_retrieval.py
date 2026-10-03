import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from swe_exp_loc_retrieval import build_index, load_index, read_experiences, recall


class Encoder:
    def encode(self, texts):
        return [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"instance_id": "a", "repo": "r/a", "issue_type": "TypeA", "description": "one"},
                     {"instance_id": "b", "repo": "r/b", "issue_type": "TypeB", "description": "two"},
                     {"instance_id": "c", "repo": "r/c", "issue_type": "TypeC", "description": "three"}]

    def test_index_cache_and_same_repo_exclusion(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "index.jsonl"
            self.assertEqual(build_index(self.rows, path, encoder=Encoder())["status"], "built")
            self.assertEqual(build_index(self.rows, path)["status"], "cached")
            indexed = load_index(self.rows, path)
            chosen = recall(query={"issue_type": "TypeA", "description": "one"},
                            current_repo="r/a", experiences=self.rows,
                            indexed=indexed, query_vector=[1.0, 0.0])
            self.assertEqual([row["instance_id"] for row in chosen], ["c", "b"])
            path.write_text(path.read_text() + "tamper")
            with self.assertRaisesRegex(ValueError, "Changed E5 index"):
                load_index(self.rows, path)

    def test_stale_index_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "index.jsonl"
            build_index(self.rows, path, encoder=Encoder())
            with self.assertRaisesRegex(ValueError, "Stale"):
                load_index([*self.rows[:-1], {**self.rows[-1], "description": "changed"}], path)

    def test_skill_conditioned_history_is_excluded_by_default(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for row in self.rows:
                (root / (row["instance_id"] + ".json")).write_text(json.dumps({
                    **row, "source_skill_conditioned": row["instance_id"] == "b"}), encoding="utf-8")
            self.assertEqual([row["instance_id"] for row in read_experiences(root)], ["a", "c"])
            self.assertEqual(len(read_experiences(root, include_skill_conditioned=True)), 3)


if __name__ == "__main__":
    unittest.main()
