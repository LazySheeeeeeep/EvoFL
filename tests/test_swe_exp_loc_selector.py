import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from swe_exp_loc_selector import InstructorGuidedClient, instruct_once, select_one


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def chat(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return {"content": json.dumps(next(self.outputs))}


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.exp = {"a": {"issue": "Past issue", "perspective": ["Trace input flow"],
                          "positioning": ["Inspect the boundary"]}}
        self.prompts = {"select_exp_system_prompt": "Select up to {k}",
                        "select_exp_user_prompt": "Past:\n{}\nCurrent:\n{}"}

    def test_selects_only_recalled_id(self):
        client = FakeClient([{"a": {"reason": "same failure shape"}}])
        result = select_one(client, issue="Current issue",
                            candidates=[{"instance_id": "a", "score": 89.0}],
                            experiences=self.exp, prompts=self.prompts)
        self.assertEqual(result["selected_source_id"], "a")
        self.assertIn("Trace input flow", client.messages[0][1]["content"])
        self.assertNotIn("Inspect the boundary", client.messages[0][1]["content"])

    def test_rejects_non_recalled_id(self):
        client = FakeClient([{"unseen": {"reason": "guess"}}])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_one(client, issue="Current issue",
                       candidates=[{"instance_id": "a", "score": 89.0}],
                       experiences=self.exp, prompts=self.prompts)

    def test_instructor_uses_selected_knowledge_not_patch(self):
        client = FakeClient([{"thoughts": "Need boundary", "instructions": "Inspect caller",
                              "context": "client.py", "type": "view"}])
        result = instruct_once(client, issue="Current issue", history=[{"observation": "read file"}],
                               experience=self.exp["a"])
        self.assertEqual(result["type"], "view")
        self.assertIn("Inspect the boundary", client.messages[0][1]["content"])

    def test_wrapper_guides_main_turn_only(self):
        advice = {"thoughts": "Need source", "instructions": "Search", "context": "adapter",
                  "type": "search"}
        base = FakeClient([advice, {"action": "tool"}])
        wrapper = InstructorGuidedClient(base, issue="Current issue", selected_experience=self.exp["a"])
        initial = [{"role": "system", "content": "system"},
                   {"role": "user", "content": json.dumps({"bug_report": "Current issue", "fault_families": []})}]
        wrapper.chat(messages=initial, tools=[{"type": "function"}])
        self.assertEqual(len(wrapper.guidance), 1)
        self.assertEqual(len(base.messages[1]), 3)
        self.assertEqual(len(initial), 2)


if __name__ == "__main__":
    unittest.main()
