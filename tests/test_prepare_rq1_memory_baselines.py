"""Read-only preparation must reject leakage and preserve trajectory provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_rq1_memory_baselines.py"
SPEC = importlib.util.spec_from_file_location("prepare_rq1_memory_baselines", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PrepareMemoryBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.expanded, self.historical, self.output = root / "expanded", root / "historical", root / "output"
        save(self.expanded / "protocol.json", {"training_target": 400, "evaluation_target": 500,
            "train_before": "2020-01-01", "test_created_from": "2021-01-01"})
        self.old = [self.case("old", index, "2018-01-01") for index in range(200)]
        self.new = [self.case("new", index, "2019-01-01") for index in range(200)]
        self.evaluation = [self.case("eval", index, "2022-01-01") for index in range(500)]
        self.manifests()
        case = self.old[0]
        directory = self.historical / "training" / case["instance_id"]
        save(directory / "record.json", {"instance_id": case["instance_id"], "metrics": {"top1": True}})
        save(directory / "explorer/result.json", {"trace_version": "v5", "instance_id": case["instance_id"],
            "status": "completed", "ranked_functions": ["module.py::f"], "final_summary": "look here"})
        save(directory / "explorer/initial_payload.json", {"bug_report": "Bug report"})
        save(directory / "explorer/loaded_skills.json", {"fault_skill": None})
        save(directory / "explorer/investigation_index.json", {"trace_version": "v5", "timeline": [
            {"event": "investigation_action", "step": 1, "tool": "grep", "purpose": "find symbol",
             "arguments": {"pattern": "f"}, "observation_id": "obs-00001"}]})

    @staticmethod
    def case(prefix, index, created):
        return {"instance_id": f"{prefix}-{index}", "repo": "owner/repo", "created_at": created,
                "problem_statement": "Bug report\n", "patch": "patch"}

    def manifests(self):
        save(self.expanded / "original_training.json", self.old)
        save(self.expanded / "training_additions.json", self.new)
        save(self.expanded / "evaluation_cases.json", self.evaluation)

    def test_provisional_episode_has_no_patch_or_reference_answer(self):
        summary = MODULE.prepare(self.expanded, self.historical, self.output)
        self.assertEqual(summary["available_episode_count"], 1)
        self.assertEqual(summary["missing_count"], 399)
        episode = json.loads((self.output / "episodes.jsonl").read_text().splitlines()[0])
        self.assertFalse(episode["skill_conditioned"])
        self.assertEqual(episode["issue"], "Bug report")
        self.assertEqual(episode["investigation"][0]["observation_id"], "obs-00001")
        self.assertNotIn("patch", episode)
        self.assertNotIn("ground_truth", episode)

    def test_evaluation_overlap_rejected_before_writing(self):
        self.evaluation[0]["instance_id"] = self.old[0]["instance_id"]
        self.manifests()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            MODULE.prepare(self.expanded, self.historical, self.output)
        self.assertFalse((self.output / "episodes.jsonl").exists())

    def test_temporal_violation_rejected(self):
        self.new[0]["created_at"] = "2020-01-01"
        self.manifests()
        with self.assertRaisesRegex(ValueError, "temporal cutoff"):
            MODULE.prepare(self.expanded, self.historical, self.output)


if __name__ == "__main__":
    unittest.main()
