"""Read-only audit of stored predictions, patch scopes and issue specificity."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolutefl.evaluation.patch_ground_truth import (
    functions_from_patch, parse_patch_files, split_function_identity,
    normalize_python_module_identity, python_symbol_spans,
)


def canonical(prediction, truth):
    path, name = split_function_identity(prediction)
    path, name = normalize_python_module_identity(prediction, path, name.replace("::", "."), truth.split("::")[0])
    return path + "::" + name


def rank(predictions, truth):
    return next((i for i, p in enumerate(predictions, 1)
                 if any(canonical(p, t) == t for t in truth)), None)


def metrics(rows, field):
    return {"n": len(rows), **{f"top{k}": sum(bool(r[field] and r[field] <= k) for r in rows) / len(rows)
            if rows else None for k in (1, 3, 5)},
            "mrr": sum(1 / r[field] if r[field] else 0 for r in rows) / len(rows) if rows else None}


def spans_from_text(text):
    result = []
    def visit(node, parents):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join(parents + [child.name])
                result.append({"name": name, "kind": "class" if isinstance(child, ast.ClassDef) else "function",
                               "start": min([child.lineno] + [d.lineno for d in child.decorator_list]),
                               "end": child.end_lineno, "depth": len(parents)})
                visit(child, parents + [child.name])
            else:
                visit(child, parents)
    try:
        visit(ast.parse(text), [])
    except SyntaxError:
        pass
    return result


def containing(spans, number):
    matches = [s for s in spans if s["kind"] == "function" and s["start"] <= number <= s["end"]]
    return max(matches, key=lambda s: s["depth"])["name"] if matches else None


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text())
    selected = {r["instance_id"]: r for r in json.loads((directory / "selected_cases.json").read_text())}
    rows = []
    for case in summary["cases"]:
        cid = case["instance_id"]
        task = selected[cid]
        case_dir = directory / "cases" / cid
        context = json.loads((case_dir / "run_context.json").read_text())
        repo = Path(context["repo_path"])
        truth = case["ground_truth_functions"]
        pred = case["ranked_functions"]
        span_cache = {t.split("::")[0]: python_symbol_spans(repo / t.split("::")[0]) for t in truth}
        kinds = {t: next((s["kind"] for s in span_cache[t.split("::")[0]]
                         if s["qualified_name"] == t.split("::")[1]), "unresolved") for t in truth}
        func_truth = [t for t in truth if kinds[t] == "function"]
        direct_funcs = set()
        old_failed = []
        for file in parse_patch_files(task["patch"]):
            path = file["new_path"]
            if not path.endswith(".py"):
                continue
            new_spans = spans_from_text((repo / path).read_text(errors="replace")) if (repo / path).exists() else []
            old = subprocess.run(["git", "show", "HEAD:" + file["old_path"]], cwd=repo,
                                 capture_output=True, text=True)
            if old.returncode:
                old_failed.append(file["old_path"])
            old_spans = spans_from_text(old.stdout) if not old.returncode else []
            for hunk in file["hunks"]:
                before, after = hunk["old_start"], hunk["new_start"]
                for line in hunk["lines"]:
                    if line.startswith("\\"):
                        continue
                    name = None
                    if line.startswith("-"):
                        name = containing(old_spans, before)
                        before += 1
                    elif line.startswith("+"):
                        name = containing(new_spans, after)
                        after += 1
                    else:
                        before += 1
                        after += 1
                    if name:
                        direct_funcs.add(path + "::" + name)
        issue = task["problem_statement"]
        named = [t for t in func_truth if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t.split(".")[-1].split("::")[-1]) + r"(?![A-Za-z0-9_])", issue)]
        full_named = [t for t in func_truth if t.split("::")[1] in issue]
        initial = json.loads((case_dir / "initial_payload.json").read_text())
        trace_text = (case_dir / "observations.jsonl").read_text() if (case_dir / "observations.jsonl").exists() else ""
        rows.append({"index": case["index"], "instance_id": cid, "ground_truth": truth, "predictions": pred,
                     "kinds": kinds, "reported_rank": case["function_metrics"]["rank"],
                     "exact_rank": rank(pred, truth), "function_only_rank": rank(pred, func_truth),
                     "direct_patch_functions": sorted(direct_funcs), "direct_patch_rank": rank(pred, direct_funcs),
                     "mapper_only": sorted(set(func_truth) - direct_funcs), "omitted_direct": sorted(direct_funcs - set(func_truth)),
                     "old_source_unavailable": old_failed,
                     "named_functions_in_issue": named, "qualified_names_in_issue": full_named,
                     "issue_code_link": bool(re.search(r"github.com/[^\s]+(?:#L\d+|/blob/)", issue)),
                     "initial_keys": list(initial), "issue_unchanged": str(initial.get("bug_report", "")).strip() == issue.strip(),
                     "metadata_marker_observed": ".evolutefl_swesmith_case.json" in trace_text,
                     "issue": issue})
    eligible = [r for r in rows if "function" in r["kinds"].values()]
    result = {"reported": metrics(rows, "reported_rank"), "exact_names_including_classes": metrics(rows, "exact_rank"),
              "function_only_all_cases": metrics(rows, "function_only_rank"),
              "function_only_evaluable": metrics(eligible, "function_only_rank"),
              "direct_changed_lines_both_sides": metrics(rows, "direct_patch_rank"),
              "named_function_issue_cases": sum(bool(r["named_functions_in_issue"]) for r in rows),
              "qualified_name_issue_cases": sum(bool(r["qualified_names_in_issue"]) for r in rows),
              "issue_code_link_cases": sum(r["issue_code_link"] for r in rows),
              "class_ground_truth_cases": [r["index"] for r in rows if "class" in r["kinds"].values()],
              "class_only_cases": [r["index"] for r in rows if set(r["kinds"].values()) == {"class"}],
              "strict_matching_changed_cases": [r["index"] for r in rows if r["exact_rank"] != r["reported_rank"]],
              "all_issues_unchanged": all(r["issue_unchanged"] for r in rows),
              "marker_observed_cases": [r["index"] for r in rows if r["metadata_marker_observed"]],
              "rows": rows}
    output = directory / "accuracy_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print("mapping_differences", json.dumps([{k:r[k] for k in ("index", "mapper_only", "omitted_direct", "reported_rank", "direct_patch_rank")}
                                            for r in rows if r["mapper_only"] or r["omitted_direct"]]))


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
