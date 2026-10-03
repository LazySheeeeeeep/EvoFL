"""Run RGFL's author file stage on one patch-free case using DeepSeek."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__:
    from .run_rgfl_mock_smoke import AUTHOR, INPUT, load_public_case
else:
    from run_rgfl_mock_smoke import AUTHOR, INPUT, load_public_case


OUTPUT = Path(__file__).resolve().parents[1] / (
    "runs/external_fl_baseline_preflight_20260924/rgfl_pilot/live_file_smoke"
)
MODEL = "deepseek-v4-flash"


def map_unique_suffix_files(model_found_files, files, *, original):
    available = [item[0] for item in files]
    mapped = []
    for raw in model_found_files:
        exact = original([raw], files)
        if exact:
            match = exact[0]
        else:
            name = str(raw).replace("\\", "/").removeprefix("./")
            matches = [path for path in available if path.endswith("/" + name)]
            match = matches[0] if len(matches) == 1 else None
        if match and match not in mapped:
            mapped.append(match)
    return mapped


def with_non_thinking(config: dict, logger, *, original, base_url: str):
    if config.get("model") != MODEL:
        raise ValueError("RGFL smoke model changed unexpectedly")
    return original({**config, "reasoning_effort": "none"}, logger, base_url=base_url,
                    max_retries=2, timeout=100)


def run(prepared: Path = INPUT, output: Path = OUTPUT) -> dict:
    prepared = prepared.resolve(strict=True)
    output = output.resolve()
    case = load_public_case(prepared)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY must be provided to this process")
    os.environ["PROJECT_FILE_LOC"] = str(prepared / "repo_structures")
    os.environ["OPENAI_API_KEY"] = key
    sys.path.insert(0, str(AUTHOR))
    from rgfl.fl.localize import localize_instance
    from rgfl.util import model as author_model
    from rgfl.fl import FL as author_fl

    original = author_model.request_chatgpt_engine
    author_model.request_chatgpt_engine = lambda config, logger, base_url=None: with_non_thinking(
        config, logger, original=original, base_url=base_url)
    original_file_mapper = author_fl.correct_file_paths
    author_fl.correct_file_paths = lambda names, files: map_unique_suffix_files(
        names, files, original=original_file_mapper)
    output.mkdir(parents=True, exist_ok=True)
    (output / "localization_logs").mkdir(exist_ok=True)
    result_path = output / "loc_outputs.jsonl"
    if result_path.exists():
        raise ValueError("Live output already exists; use a fresh output directory")
    args = SimpleNamespace(target_id=case["instance_id"], output_folder=str(output),
                           output_file=str(result_path), file_level=True,
                           related_level=False, fine_grain_line_level=False,
                           mock=False, model=MODEL, backend="deepseek")
    localize_instance(case, args, [case], None, set())
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1 or rows[0].get("instance_id") != case["instance_id"]:
        raise ValueError("RGFL did not produce one file-stage result")
    result = {"status": "completed" if rows[0].get("found_files") else "empty_prediction",
              "instance_id": case["instance_id"], "model": MODEL,
              "adapter": "reasoning_effort_none+unique_suffix_file_path",
              "found_files": rows[0].get("found_files") or [],
              "output": str(result_path)}
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.output), indent=2))
