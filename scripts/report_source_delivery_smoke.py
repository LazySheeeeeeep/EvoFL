"""Audit delivered source pages and verify a SWE-smith smoke repository."""
import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def analyze(case):
    result = json.loads((case / "result.json").read_text())
    observations = [json.loads(line) for line in (case / "observations.jsonl").read_text().splitlines() if line.strip()]
    timeline = json.loads((case / "investigation_index.json").read_text())["timeline"]
    seen = defaultdict(set)
    pages, overlapping, new_lines, repeated_lines, clipped = 0, 0, 0, 0, 0
    for observation in observations:
        if observation["tool"] not in {"read_file", "read_symbol"} or not observation["ok"]:
            continue
        pages += 1
        clipped += bool(observation.get("model_payload_truncated"))
        payload = observation["model_payload"]
        source = payload.get("result", {})
        lines = source.get("content", [])
        path = source.get("path")
        if "result_preview" in payload:
            # Count only complete serialized lines actually present in the preview.
            preview = payload["result_preview"]
            path_match = re.search(r'"path":\s*("(?:[^"\\]|\\.)*")', preview)
            path = json.loads(path_match.group(1)) if path_match else None
            lines = []
            for match in re.finditer(r'\{"line":', preview):
                try:
                    item, _ = json.JSONDecoder().raw_decode(preview[match.start():])
                    lines.append(item)
                except ValueError:
                    pass
        delivered = {line["line"] for line in lines if isinstance(line, dict) and "line" in line}
        repeated = delivered & seen[path]
        overlapping += bool(delivered) and len(repeated) / len(delivered) >= .8
        repeated_lines += len(repeated)
        new_lines += len(delivered - seen[path])
        seen[path].update(delivered)
    source_actions = [a for a in timeline if a["tool"] in {"grep", "read_file", "find_symbol", "read_symbol", "write"}]
    return {"steps": result["steps"], "forced_finish": result["forced_finish"],
            "runtime_seconds": result["runtime_seconds"],
            "fault_request_step": next(a["step"] for a in timeline if a["tool"] == "load_fault_skill"),
            "forced_fault_request": result["forced_fault_skill_attempt"],
            "tool_counts": dict(Counter(a["tool"] for a in timeline)),
            "source_pages": pages, "clipped_model_pages": clipped,
            "pages_with_at_least_80_percent_previously_delivered_lines": overlapping,
            "repeated_delivered_lines": repeated_lines, "unique_delivered_lines": new_lines,
            "source_actions_with_candidate_updates": sum(bool(a["candidate_updates"]) for a in source_actions),
            "source_action_count": len(source_actions), "ranked_functions": result["ranked_functions"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    task = json.loads((args.after / "task.json").read_text())
    context = json.loads((args.after / "run_context.json").read_text())
    checks = {}
    for name, extra in (("forward", []), ("reverse", ["--reverse"])):
        check = subprocess.run(["git", "-C", context["repo_path"], "apply", "--check", *extra, "-"],
                               input=task["patch"], text=True, capture_output=True)
        checks[name] = {"applicable": check.returncode == 0, "stderr": check.stderr.strip()}
    before_identity = json.loads((args.before / "run_context.json").read_text())["repository_identity"]
    report = {"before": analyze(args.before), "after": analyze(args.after),
              "same_source_identity": before_identity == context["repository_identity"],
              "patch_checks": checks,
              "evaluation_valid": not checks["forward"]["applicable"] and checks["reverse"]["applicable"],
              "note": "For this clean_to_buggy patch, forward-applicable and reverse-inapplicable indicates the mutation is absent. Read-overlap counts use delivered complete lines, not requested ranges."}
    (args.after.parent.parent / "delivery_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
