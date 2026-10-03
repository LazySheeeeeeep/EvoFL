"""Run RGFL's author element stage after the isolated file-stage smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__:
    from .normalize_cosil_functions import normalize
    from .run_rgfl_file_smoke import MODEL, with_non_thinking
    from .run_rgfl_mock_smoke import AUTHOR, INPUT, load_public_case
else:
    from normalize_cosil_functions import normalize
    from run_rgfl_file_smoke import MODEL, with_non_thinking
    from run_rgfl_mock_smoke import AUTHOR, INPUT, load_public_case


ROOT = Path(__file__).resolve().parents[1]
FILE_OUTPUT = ROOT / (
    "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/"
    "live_file_smoke_path_adapter/loc_outputs.jsonl"
)
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/live_element_smoke"


def load_file_candidates(path: Path, case_id: str) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1 or rows[0].get("instance_id") != case_id or not rows[0].get("found_files"):
        raise ValueError("Element smoke requires one nonempty matching file-stage result")
    return rows[0]


def run(prepared: Path = INPUT, file_output: Path = FILE_OUTPUT,
        output: Path = OUTPUT) -> dict:
    prepared = prepared.resolve(strict=True)
    output = output.resolve()
    case = load_public_case(prepared)
    start = load_file_candidates(file_output.resolve(strict=True), case["instance_id"])
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY must be provided to this process")
    os.environ["PROJECT_FILE_LOC"] = str(prepared / "repo_structures")
    os.environ["OPENAI_API_KEY"] = key
    sys.path.insert(0, str(AUTHOR))
    from rgfl.fl.localize import localize_instance
    from rgfl.util import model as author_model

    original = author_model.request_chatgpt_engine
    author_model.request_chatgpt_engine = lambda config, logger, base_url=None: with_non_thinking(
        config, logger, original=original, base_url=base_url)
    output.mkdir(parents=True, exist_ok=True)
    (output / "localization_logs").mkdir(exist_ok=True)
    output_file = output / "loc_outputs.jsonl"
    if output_file.exists():
        raise ValueError("Element output already exists; use a fresh output directory")
    args = SimpleNamespace(
        target_id=case["instance_id"], output_folder=str(output), output_file=str(output_file),
        file_level=False, related_level=True, fine_grain_line_level=False, mock=False,
        model=MODEL, backend="deepseek", top_n=3, temperature=0.0,
        compress=True, related_level_separate_file=False, compress_assign=True,
        compress_assign_total_lines=30, compress_assign_prefix_lines=10,
        compress_assign_suffix_lines=10, keep_old_order=False,
    )
    localize_instance(case, args, [case], [start], set())
    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1 or rows[0].get("instance_id") != case["instance_id"]:
        raise ValueError("RGFL did not produce one element-stage result")
    structure = json.loads((prepared / "repo_structures" /
                            f"{case['instance_id']}.json").read_text(encoding="utf-8"))
    mapped = normalize(rows[0], structure["structure"])
    (output / "normalized.json").write_text(json.dumps(mapped, indent=2) + "\n",
                                            encoding="utf-8")
    result = {"status": "completed" if mapped["ranked_functions"] else "empty_prediction",
              "instance_id": case["instance_id"], "model": MODEL,
              "adapter": "reasoning_effort_none+unique_suffix_file_path",
              "ranked_functions": mapped["ranked_functions"], "output": str(output_file),
              "note": "Author element stage only; RGFL reasoning rerank is not included."}
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=INPUT)
    parser.add_argument("--file-output", type=Path, default=FILE_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.file_output, args.output), indent=2))
