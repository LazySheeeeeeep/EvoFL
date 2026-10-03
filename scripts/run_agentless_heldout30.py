"""Pinned Agentless file->related-function baseline; no repair or skill access."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "5ce5888b9f149beaace393957a55ea8ee46c9f71"
UPSTREAM = ROOT / ".tools" / ("Agentless-" + COMMIT)
SOURCE = ROOT / "runs/evidence_reflector_heldout30_deepseek_20260914"
OUT = ROOT / "runs/agentless_fl_heldout30_deepseek_20260915"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_names(code):
    names, classes = [], []
    def walk(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join((*parents, child.name))
                (classes if isinstance(child, ast.ClassDef) else names).append(name)
                walk(child, (*parents, child.name))
            else:
                walk(child, parents)
    try:
        walk(ast.parse(code))
    except SyntaxError:
        pass
    return names, classes


def rank_functions(locations, contents):
    """Output order, exact/unique leaf resolution; never expand a class into methods."""
    predictions, ignored = [], []
    for path, blocks in locations.items():
        names, classes = function_names(contents.get(path, ""))
        for block in blocks:
            for line in block.splitlines():
                kind, sep, value = line.partition(":")
                value = value.strip().strip("`").replace("::", ".")
                if not sep or kind.strip() != "function":
                    ignored.append({"file": path, "location": line, "reason": "not_function"})
                    continue
                if value in classes:
                    ignored.append({"file": path, "location": line, "reason": "class_not_function"})
                    continue
                matches = [n for n in names if n == value]
                if not matches and "." not in value:
                    matches = [n for n in names if n.split(".")[-1] == value]
                if len(matches) > 1:
                    ignored.append({"file": path, "location": line, "reason": "ambiguous_function"})
                    continue
                # Preserve unknown explicit function predictions as misses, not free rank removal.
                name = matches[0] if matches else value
                if not name:
                    continue
                pred = f"{path}::{name}"
                if pred not in predictions:
                    predictions.append(pred)
    return predictions[:5], ignored


class Bridge:
    def __init__(self, cfg, directory):
        from evolutefl.llm.client import OpenAICompatibleClient
        self.client = OpenAICompatibleClient.from_config(cfg)
        self.directory = directory
        self.calls = 0
        self.stage = "file"

    def factory(self, **kwargs):
        return self

    def codegen(self, message, num_samples=1, **kwargs):
        if num_samples != 1:
            raise ValueError("Single-sample baseline")
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}]
        result = None
        for attempt in range(2):
            self.calls += 1
            start = time.monotonic()
            budget = 4096 if attempt == 0 else 8192
            extra = None if attempt == 0 else {"thinking": {"type": "disabled"}}
            record = {"stage": self.stage, "messages": messages, "temperature": 0,
                      "max_tokens": budget, "extra_body": extra}
            try:
                result = self.client.chat(messages=messages, temperature=0, max_tokens=budget, extra_body=extra)
                record["response"] = result
            except Exception as exc:
                record["error"] = type(exc).__name__
                raise
            finally:
                record["runtime_seconds"] = time.monotonic() - start
                write(self.directory / "requests" / f"{self.calls:03d}.json", record)
            choice = (result["raw"].get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length" and attempt == 0:
                continue
            if choice.get("finish_reason") == "length" or not result["content"].strip():
                raise ValueError("Empty or truncated Agentless output")
            break
        return [{"response": result["content"], "usage": result["raw"].get("usage", {})}]


def build_structure(root):
    from get_repo_structure.get_repo_structure import parse_python_file
    # Use the official parser with true repo-relative paths (archive names are not code paths).
    structure = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        node = structure
        parts = path.relative_to(root).parts
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        classes, functions, text = parse_python_file(str(path))
        node[parts[-1]] = {"classes": classes, "functions": functions, "text": text}
    return structure


def run_case(case, cfg):
    from agentless.fl.FL import LLMFL
    import agentless.util.model as upstream_model
    from agentless.util.preprocess_data import filter_none_python, filter_out_test_files, get_repo_files
    from run_swe_explore_v5_stratified import safe_archive_filter
    directory = OUT / "cases" / case["instance_id"]
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return read(result_path)
    start = time.monotonic()
    archive = SOURCE / "sources" / (case["instance_id"] + ".tar.gz")
    expected = read(SOURCE / "materialized" / (case["instance_id"] + ".json"))["original_archive_sha256"]
    if sha(archive) != expected:
        raise ValueError("Source archive checksum changed")
    write(directory / "input.json", {k: case[k] for k in ("repo", "base_commit", "problem_statement")})
    bridge = Bridge(cfg, directory)
    original_factory = upstream_model.make_model
    upstream_model.make_model = bridge.factory
    logger = logging.getLogger(case["instance_id"])
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(directory / "agentless.log")
    logger.addHandler(handler)
    result = {"instance_id": case["instance_id"], "source_sha256": expected, "predictions": []}
    try:
        with tempfile.TemporaryDirectory(prefix="agentless-", dir=OUT / "work") as temp:
            with tarfile.open(archive) as tar:
                tar.extractall(temp, filter=safe_archive_filter)
            roots = list(Path(temp).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("Invalid repository archive")
            with (directory / "parse.log").open("w") as parse_log, contextlib.redirect_stdout(parse_log):
                structure = build_structure(roots[0])
            filter_none_python(structure)
            filter_out_test_files(structure)
            fl = LLMFL(case["instance_id"], structure, case["problem_statement"], cfg["model"], "openai", logger)
            fl.max_tokens = 4096
            stage_file = directory / "file_stage.json"
            if stage_file.exists():
                files = read(stage_file)["found_files"]
                bridge.calls = len(list((directory / "requests").glob("*.json")))
            else:
                files, artifact, trajectory = fl.localize()
                write(stage_file, {"found_files": files, "artifact": artifact, "trajectory": trajectory})
            result["found_files"] = files
            if files:
                bridge.stage = "related"
                locs, artifact, trajectory = fl.localize_function_from_compressed_files(
                    files[:3], temperature=0, keep_old_order=False, compress_assign=True)
                write(directory / "related_stage.json", {"found_related_locs": locs,
                      "artifact": artifact, "trajectory": trajectory})
                predictions, ignored = rank_functions(locs, get_repo_files(structure, files[:3]))
                result.update(predictions=predictions, ignored_locations=ignored)
            result["status"] = "completed" if result["predictions"] else "finish_empty_prediction"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        upstream_model.make_model = original_factory
        logger.removeHandler(handler)
        handler.close()
    result["runtime_seconds"] = time.monotonic() - start
    result["llm_call_count"] = len(list((directory / "requests").glob("*.json")))
    write(result_path, result)
    return result


def summarize(cases):
    from run_swe_explore_v5_stratified import strict_metrics
    old = {c["instance_id"]: c for c in read(SOURCE / "comparison_summary.json")["cases"]}
    rows = []
    for case in cases:
        cid = case["instance_id"]
        path = OUT / "cases" / cid / "result.json"
        if not path.exists():
            continue
        result = read(path)
        truth = old[cid]["function_ground_truth"]
        arms = {a: old[cid]["arms"][a] for a in ("no_skill", "expanded_bank")}
        arms["agentless_fl"] = {**result, "metrics": strict_metrics(result["predictions"][:5], truth)}
        rows.append({"instance_id": cid, "function_ground_truth": truth, "arms": arms})
    eligible = [r for r in rows if r["function_ground_truth"]]
    summary = {"planned": len(cases), "finished": len(rows), "function_eligible": len(eligible), "arms": {}, "cases": rows}
    for arm in ("no_skill", "expanded_bank", "agentless_fl"):
        group = [r["arms"][arm] for r in eligible]
        summary["arms"][arm] = {key: sum(r["metrics"][key] for r in group) / len(group) if group else None
                                  for key in ("top1", "top3", "top5", "mrr")}
        summary["arms"][arm]["completed_count"] = sum(r["arms"][arm]["status"] == "completed" for r in rows)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
    for path in (OUT / "cases").glob("*/requests/*.json"):
        usage["llm_calls"] += 1
        u = read(path).get("response", {}).get("raw", {}).get("usage", {})
        for key in ("prompt_tokens", "completion_tokens"):
            usage[key] += u.get(key, 0)
    summary["agentless_usage"] = usage
    summary["cost_note"] = "Tokens retained; monetary cost unavailable without verified provider rates. Historical arms are reused, not contemporaneous."
    write(OUT / "comparison_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(exist_ok=True)
    cases = read(SOURCE / "selected_cases.json")
    if len(cases) != 30 or any(not c.get("problem_statement", "").strip() for c in cases):
        raise ValueError("Expected 30 nonempty issues")
    cfg = read(SOURCE / "config.json")["llm"]
    if cfg.get("api_key"):
        raise ValueError("Use environment credential only")
    snapshots = [SOURCE / "selected_cases.json", SOURCE / "config.json", SOURCE / "comparison_summary.json",
                 SOURCE / "expanded_skills.jsonl", *sorted(UPSTREAM.rglob("*.py"))]
    protocol = {"method": "Agentless-FL (LLM file + compressed related-function localization)",
        "upstream_commit": COMMIT, "model": cfg["model"], "temperature": 0, "top_n_files": 3,
        "samples": 1, "max_tokens": 4096, "truncation_retry": "8192, thinking disabled, identical messages",
        "excluded_stages": ["embedding retrieval/merge", "fine-grained line localization", "repair", "tests"],
        "ranking": "Original related-output order; exact/unique function leaf normalization; dedup; top5; classes not expanded.",
        "retry_note": "Transport and truncation retries only; no temperature escalation for empty locations.",
        "preprocessing": "Official Python parser/filter/skeleton; true repo-relative archive paths.",
        "hashes": {str(p): sha(p) for p in snapshots}, "runner_sha256": sha(Path(__file__))}
    p = OUT / "protocol.json"
    if p.exists() and read(p) != protocol:
        raise ValueError("Frozen baseline protocol changed")
    if not p.exists():
        write(p, protocol)
        write(OUT / "selected_cases.json", cases)
    if args.prepare_only:
        print("Prepared frozen 30-case protocol", flush=True)
        return
    if not os.getenv(cfg["api_key_env"]):
        raise ValueError("Missing environment credential")
    for i, case in enumerate(cases[:args.limit], 1):
        write(OUT / "current_case.json", {"index": i, "instance_id": case["instance_id"]})
        print(f"START {i}/30 {case['instance_id']}", flush=True)
        result = run_case(case, cfg)
        summary = summarize(cases)
        print(f"END {i}/30 {result['status']} predictions={len(result['predictions'])}", flush=True)
        # Gate the first three on valid protocol execution, never on localization accuracy.
        if i <= 3 and result["status"] == "failed":
            raise RuntimeError("Smoke failed; inspect before continuing")
    print(json.dumps({k: v for k, v in summarize(cases).items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
