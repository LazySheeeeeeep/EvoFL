import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.run_cosil_pilot import author_environment, run, run_case, run_stage, selected_cases, structure_identity


CASE = {"instance_id": "owner__repo-1", "repo": "owner/repo",
        "base_commit": "abc123", "problem_statement": "A public issue"}


class CosilPilotTests(TestCase):
    def _pilot(self, root: Path) -> Path:
        pilot = root / "pilot"
        prepared = pilot / "prepared" / CASE["instance_id"]
        structures = pilot / "repo_structures"
        prepared.mkdir(parents=True)
        structures.mkdir()
        row = json.dumps(CASE) + "\n"
        (pilot / "selected_cases.jsonl").write_text(row, encoding="utf-8")
        (prepared / "case.jsonl").write_text(row, encoding="utf-8")
        (structures / f"{CASE['instance_id']}.json").write_text(json.dumps({
            "instance_id": CASE["instance_id"], "repo": CASE["repo"],
            "base_commit": CASE["base_commit"], "structure": {"repo": {}},
        }), encoding="utf-8")
        return pilot

    def test_dry_run_checks_public_inputs_without_key_or_model_calls(self):
        with TemporaryDirectory() as root:
            pilot = self._pilot(Path(root))
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
                result = run(pilot=pilot, output=Path(root) / "out", limit=1,
                             dry_run=True)
            self.assertEqual(result["case_count"], 1)
            self.assertEqual(result["model_calls"], 0)

    def test_private_field_in_public_input_is_rejected(self):
        with TemporaryDirectory() as root:
            pilot = self._pilot(Path(root))
            (pilot / "selected_cases.jsonl").write_text(json.dumps({
                **CASE, "patch": "secret",
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                selected_cases(pilot, 1)

    def test_source_version_mismatch_is_rejected(self):
        with TemporaryDirectory() as root:
            pilot = self._pilot(Path(root))
            path = pilot / "repo_structures" / f"{CASE['instance_id']}.json"
            structure = json.loads(path.read_text(encoding="utf-8"))
            structure["base_commit"] = "wrong"
            path.write_text(json.dumps(structure), encoding="utf-8")
            with self.assertRaises(ValueError):
                selected_cases(pilot, 1)

    def test_structure_identity_only_reads_header(self):
        with TemporaryDirectory() as root:
            pilot = self._pilot(Path(root))
            path = pilot / "repo_structures" / f"{CASE['instance_id']}.json"
            path.write_text(json.dumps({"instance_id": CASE["instance_id"],
                                        "repo": CASE["repo"],
                                        "base_commit": CASE["base_commit"],
                                        "structure": {"large": "x" * 10000}}), encoding="utf-8")
            self.assertEqual(structure_identity(path)["base_commit"], CASE["base_commit"])
            self.assertEqual(selected_cases(pilot, 1), [CASE])

    def test_paid_run_requires_process_only_key(self):
        with TemporaryDirectory() as root:
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
                with self.assertRaises(RuntimeError):
                    author_environment(Path(root))

    def test_author_stage_receives_absolute_output_outside_pilot_cwd(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            pilot = self._pilot(base)
            folder = base / "outputs" / "file"
            case_file = pilot / "prepared" / CASE["instance_id"] / "case.jsonl"
            with patch("scripts.run_cosil_pilot.subprocess.run") as process, patch(
                "scripts.run_cosil_pilot.has_prediction", return_value=True
            ):
                process.return_value.returncode = 0
                result = run_stage(CASE["instance_id"], case_file, pilot, folder,
                                   "file", {})
            command = process.call_args.args[0]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(command[command.index("--output_folder") + 1], str(folder))
            self.assertTrue(Path(command[command.index("--output_folder") + 1]).is_absolute())
            self.assertEqual(process.call_args.kwargs["cwd"], pilot)

    def test_reuses_identical_public_case_and_author_output_without_model_call(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            old = self._pilot(base / "old")
            new = self._pilot(base / "new")
            structure = json.loads((new / "repo_structures" /
                                    f"{CASE['instance_id']}.json").read_text(encoding="utf-8"))
            structure["structure"] = {"repo": {"api.py": {
                "text": ["def locate():", "    pass"], "functions": [], "classes": [],
            }}}
            for pilot in (old, new):
                (pilot / "repo_structures" / f"{CASE['instance_id']}.json").write_text(
                    json.dumps(structure), encoding="utf-8")
            prior = base / "prior" / "cases" / CASE["instance_id"]
            (prior / "function").mkdir(parents=True)
            (prior / "result.json").write_text(json.dumps({
                "status": "completed", "stages": [{"stage": "file", "status": "completed"},
                                                  {"stage": "function", "status": "completed"}],
            }), encoding="utf-8")
            (prior / "function" / "loc_outputs_func.jsonl").write_text(json.dumps({
                "instance_id": CASE["instance_id"],
                "found_related_locs": {"repo/api.py": ["function: locate"]},
            }) + "\n", encoding="utf-8")
            with patch("scripts.run_cosil_pilot.run_stage") as stage:
                result = run_case(CASE, new, base / "results", {}, old, base / "prior")
            stage.assert_not_called()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["ranked_functions"], ["api.py::locate"])
            self.assertIn("reused_author_output_sha256", result)
            self.assertTrue((base / "results" / "cases" / CASE["instance_id"] /
                             "normalized.json").exists())
