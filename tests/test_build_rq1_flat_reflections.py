import importlib.util
import json
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("build_rq1_flat_reflections", SCRIPTS / "build_rq1_flat_reflections.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": json.dumps({"applicability": "configuration boundary",
                                        "lesson": "Compare external values before and after normalization."})}


class FlatReflectionTest(unittest.TestCase):
    def test_one_direct_call_produces_short_memory_without_patch(self):
        client = FakeClient()
        episode = {"instance_id": "case-1", "repo": "owner/repo", "issue": "Config ignored",
                   "investigation": [{"tool": "grep", "purpose": "search config"}],
                   "predicted_functions": ["a.py::f"], "final_summary": "looked at consumer",
                   "skill_conditioned": False}
        result = MODULE.reflect_once(client, episode, {"instance_id": "case-1", "patch": "secret patch"})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("secret patch", client.calls[0]["messages"][1]["content"])
        self.assertNotIn("secret patch", json.dumps(result))
        self.assertEqual(result["lesson"], "Compare external values before and after normalization.")

    def test_malformed_reflection_is_rejected(self):
        client = FakeClient()
        client.chat = lambda **kwargs: {"content": "{}"}
        episode = {"instance_id": "case-1", "repo": "owner/repo", "issue": "Bug",
                   "investigation": [], "predicted_functions": [], "final_summary": "", "skill_conditioned": False}
        with self.assertRaisesRegex(ValueError, "did not return"):
            MODULE.reflect_once(client, episode, {"instance_id": "case-1", "patch": "patch"})


if __name__ == "__main__":
    unittest.main()
