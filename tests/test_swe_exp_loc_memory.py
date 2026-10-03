import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_swe_exp_loc_memory import extract_one


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def chat(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return {"content": json.dumps(next(self.outputs))}


class SweExpLocMemoryTests(unittest.TestCase):
    def setUp(self):
        self.episode = {"instance_id": "p__r-1", "repo": "p/r", "issue": "Wrong result",
                        "status": "completed", "investigation": [{"tool": "grep"}],
                        "predicted_functions": ["p.py::f"], "final_summary": "Observed output",
                        "skill_conditioned": False}
        self.case = {"instance_id": "p__r-1", "problem_statement": "Wrong result",
                     "patch": "historical repair only"}
        self.prompts = {"issue_type_system_prompt": "Classify issue", "issue_type_user_prompt": "{}",
                        "encode_success_perspective_system_prompt": "Extract perspective"}

    def test_success_extracts_perspective_and_never_stores_patch(self):
        client = FakeClient([{"issue_type": "LogicError", "description": "Wrong output"},
                             {"perspective": "Inspect boundary", "entry_point": {"entry": "p.py::f"}}])
        result = extract_one(client, self.episode, self.case,
                             {"instance_id": "p__r-1", "eligible": True, "branch": "success"}, self.prompts)
        self.assertEqual(result["perspective"], ["Inspect boundary"])
        self.assertEqual(result["positioning"], [])
        self.assertNotIn("historical repair only", json.dumps(result))
        self.assertIn("historical repair only", client.messages[1][1]["content"])

    def test_failure_extracts_positioning_without_modification(self):
        client = FakeClient([{"issue_type": "LogicError", "description": "Wrong output"},
                             {"perspective": ["Understand mismatch"],
                              "positioning": ["Trace upstream values"], "modification": ["ignore"]}])
        result = extract_one(client, self.episode, self.case,
                             {"instance_id": "p__r-1", "eligible": True, "branch": "failure"}, self.prompts)
        self.assertEqual(result["positioning"], ["Trace upstream values"])
        self.assertEqual(result["flag"], "failure")
        self.assertNotIn("modification", result)

    def test_rejects_issue_mismatch_before_any_model_request(self):
        client = FakeClient([])
        bad_case = dict(self.case, problem_statement="Changed issue")
        with self.assertRaisesRegex(ValueError, "changed"):
            extract_one(client, self.episode, bad_case,
                        {"instance_id": "p__r-1", "eligible": True, "branch": "success"}, self.prompts)
        self.assertEqual(client.messages, [])

    def test_same_input_retry_on_invalid_experience_shape(self):
        client = FakeClient([{"issue_type": "LogicError", "description": "Wrong output"},
                             {"perspective": {"unexpected": "shape"}},
                             {"perspective": "Inspect the boundary"}])
        result = extract_one(client, self.episode, self.case,
                             {"instance_id": "p__r-1", "eligible": True, "branch": "success"}, self.prompts)
        self.assertEqual(result["experience_attempts"], 2)
        self.assertEqual(client.messages[1], client.messages[2])


if __name__ == "__main__":
    unittest.main()
