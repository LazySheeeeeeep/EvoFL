import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_swe_exp_loc_baseline as baseline


class Client:
    def __init__(self):
        self.messages = []
        self.responses = iter([
            {"issue_type": "LogicError", "description": "Incorrect value"},
            {"past": {"reason": "relevant diagnosis"}},
        ])

    def chat(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return {"content": json.dumps(next(self.responses))}


class Encoder:
    def encode(self, texts):
        return [[1.0, 0.0]]


class BaselineTests(unittest.TestCase):
    def test_eval_case_does_not_send_patch_or_truth_to_memory_selection(self):
        case = {"instance_id": "eval", "repo": "new/repo", "base_commit": "abc",
                "problem_statement": "Incorrect value", "patch": "SECRET_REPAIR",
                "function_ground_truth": ["pkg.py::f"]}
        past = {"instance_id": "past", "repo": "old/repo", "issue": "Past issue",
                "issue_type": "LogicError", "description": "Wrong output",
                "perspective": ["Trace the boundary"], "positioning": []}
        client = Client()
        prompts = {"issue_type_system_prompt": "Classify", "issue_type_user_prompt": "{}",
                   "select_exp_system_prompt": "Select {k}",
                   "select_exp_user_prompt": "Candidates {} Issue {}"}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"skill_bank": {"path": "old",
                "enabled_skill_types": ["fault_skill"]}}), encoding="utf-8")
            with patch.object(baseline, "materialize", return_value=(root / "work", root / "repo")), \
                 patch.object(baseline, "run_explorer", return_value={"status": "completed",
                     "ranked_functions": ["pkg.py::f"]}), \
                 patch.object(baseline, "make_skill_bank", return_value=object()), \
                 patch.object(baseline.bench, "clean_workspace"):
                result = baseline.evaluate_one(case, client=client, encoder=Encoder(),
                    experiences=[past], indexed=[{"instance_id": "past", "vector": [1.0, 0.0]}],
                    prompts=prompts, expanded=root, historical=root, output=root / "out")
            self.assertEqual(result["memory_source_id"], "past")
            self.assertNotIn("SECRET_REPAIR", json.dumps(client.messages))
            selection = json.loads((root / "out/cases/eval/experience_selection.json").read_text())
            self.assertNotIn("SECRET_REPAIR", json.dumps(selection))


if __name__ == "__main__":
    unittest.main()
