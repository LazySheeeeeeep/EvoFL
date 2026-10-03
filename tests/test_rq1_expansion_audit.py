import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_rq1_expansion_candidates import visible_issue_groups
from audit_rq1_temporal_expansion import safe_source_path, stratified_order, temporal_train_ok


class ExpansionAuditTests(unittest.TestCase):
    def test_template_reference_removed_but_real_1234_kept(self):
        case = {'repo': 'owner/repo'}
        self.assertEqual(visible_issue_groups(case, {'body': '<!-- Fixes #1234 -->\nCloses #56'}), {'owner/repo#56'})
        self.assertEqual(visible_issue_groups(case, {'body': 'Fixes #1234'}), {'owner/repo#1234'})

    def test_shared_cross_repo_reference(self):
        self.assertEqual(visible_issue_groups({'repo': 'a/b'}, {'body': 'See https://github.com/c/d/issues/4'}), {'c/d#4'})

    def test_both_creation_and_merge_must_be_early(self):
        self.assertTrue(temporal_train_ok({'created_at': '2019-12-01T00:00:00Z', 'merged_at': '2019-12-31T23:59:59Z'}))
        self.assertFalse(temporal_train_ok({'created_at': '2019-12-01T00:00:00Z', 'merged_at': '2020-01-01T00:00:00Z'}))
        self.assertFalse(temporal_train_ok({'created_at': '2019-12-01T00:00:00Z', 'merged_at': None}))

    def test_order_is_complete_deterministic_and_proportional(self):
        cases = [{'instance_id': f'{repo}-{i}', 'repo': repo} for repo, n in [('a', 20), ('b', 10)] for i in range(n)]
        ordered = stratified_order(cases, 10)
        self.assertEqual(ordered, stratified_order(list(reversed(cases)), 10))
        self.assertEqual(len({c['instance_id'] for c in ordered}), 30)
        self.assertEqual(sum(c['repo'] == 'a' for c in ordered[:15]), 10)

    def test_patch_paths_are_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(safe_source_path(root, 'pkg/code.py'), root / 'pkg/code.py')
            for invalid in ('../escape.py', '/tmp/escape.py', 'a/../../escape.py', 'a\\..\\escape.py'):
                with self.assertRaises(ValueError):
                    safe_source_path(root, invalid)


if __name__ == '__main__':
    unittest.main()
