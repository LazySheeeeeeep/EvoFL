"""Read-only label, input, and scoring audit of the frozen held-out experiment."""
import ast
import json
import math
import re
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolutefl.evaluation.patch_ground_truth import parse_patch_files, symbols_from_patch

OUT = ROOT / "runs/evidence_reflector_heldout30_deepseek_20260914"
read = lambda p: json.loads(p.read_text())
cases = {c["instance_id"]: c for c in read(OUT / "selected_cases.json")}
rows = read(OUT / "comparison_summary.json")["cases"]
report = {"cases": [], "arms": {}, "input_anomalies": [], "label_anomalies": []}
for row in rows:
    cid = row["instance_id"]
    case = cases[cid]
    truth = row["function_ground_truth"]
    old_symbols = {}
    with tarfile.open(OUT / "sources" / f"{cid}.tar.gz") as archive, tempfile.TemporaryDirectory() as td:
        members = {m.name.split("/", 1)[1]: m for m in archive.getmembers() if m.isfile() and "/" in m.name}
        for f in parse_patch_files(case["patch"]):
            path = f["old_path"]
            if not path.endswith(".py") or f.get("new_file"):
                continue
            dest = Path(td) / path
            if not dest.resolve().is_relative_to(Path(td).resolve()):
                raise ValueError("Unsafe patch path")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(members[path]).read())
        mapped = symbols_from_patch(case["patch"], td, source_side="old")
        for path in {t.split("::")[0] for t in truth}:
            f = Path(td) / path
            if not f.exists():
                continue
            def walk(node, parents=()):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = ".".join((*parents, child.name))
                        old_symbols[f"{path}::{name}"] = "class" if isinstance(child, ast.ClassDef) else "function"
                        walk(child, (*parents, child.name))
                    else:
                        walk(child, parents)
            walk(ast.parse(f.read_text()))
    bad_classes = [t for t in truth if old_symbols.get(t) == "class"]
    if set(mapped["functions"]) != set(truth) or bad_classes:
        report["label_anomalies"].append({"case": cid, "remapped": mapped, "class_targets": bad_classes})
    issue = case["problem_statement"]
    mentions = [t for t in truth if not t.split("::")[-1].split(".")[-1].startswith("__") and
                re.search(r"(?<!\w)" + re.escape(t.split("::")[-1].split(".")[-1]) + r"(?!\w)", issue)]
    record = {"instance_id": cid, "repo": case["repo"], "target_count": len(truth),
              "targets": truth, "absent_old_targets": [t for t in truth if t not in old_symbols],
              "issue_mentions_target_leaf_heuristic": mentions,
              "issue_mentions_target_file": [p for p in {t.split("::")[0] for t in truth} if p in issue],
              "traceback_present": "Traceback" in issue, "arms": {}}
    for arm in ("no_skill", "old_bank", "expanded_bank"):
        d = OUT / arm / "cases" / cid
        payload = read(d / "initial_payload.json")
        if set(payload) - {"repo", "base_commit", "bug_report", "fault_families"} or payload["bug_report"] != issue.strip():
            report["input_anomalies"].append({"case": cid, "arm": arm})
        preds = row["arms"][arm]["predictions"]
        rank = next((i for i, p in enumerate(preds, 1) if p in truth), None)
        results = read(d / "result.json")
        record["arms"][arm] = {"rank": rank, "saved_rank": row["arms"][arm]["metrics"]["rank"],
            "steps": results.get("steps"), "runtime_seconds": results.get("runtime_seconds"),
            "all_hit5": bool(truth) and set(truth).issubset(preds[:5]),
            "recall5": len(set(preds[:5]) & set(truth)) / len(truth) if truth else None}
    report["cases"].append(record)
eligible = [r for r in report["cases"] if r["target_count"]]
for arm in ("no_skill", "old_bank", "expanded_bank"):
    group = [r["arms"][arm] for r in eligible]
    report["arms"][arm] = {"n": len(group),
        "literal_vs_saved_differences": [r["instance_id"] for r in eligible if r["arms"][arm]["rank"] != r["arms"][arm]["saved_rank"]],
        "rank_distribution": dict(Counter(str(r["rank"]) for r in group)),
        "all_hit5": sum(r["all_hit5"] for r in group) / len(group),
        "macro_recall5": sum(r["recall5"] for r in group) / len(group),
        "average_steps": sum(r["arms"][arm]["steps"] for r in report["cases"]) / len(rows)}
report["target_counts"] = dict(Counter(r["target_count"] for r in report["cases"]))
for name, predicate in {
    "single_target": lambda r: r["target_count"] == 1,
    "multiple_targets": lambda r: r["target_count"] > 1,
    "issue_mentions_target_leaf": lambda r: bool(r["issue_mentions_target_leaf_heuristic"]),
    "issue_no_target_leaf": lambda r: not r["issue_mentions_target_leaf_heuristic"],
}.items():
    group = [r for r in eligible if predicate(r)]
    report[name] = {"n": len(group), "baseline_top1": sum(r["arms"]["no_skill"]["rank"] == 1 for r in group)}
report["projects"] = {p: {"n": sum(r["repo"] == p for r in eligible),
    "baseline_top1": sum(r["repo"] == p and r["arms"]["no_skill"]["rank"] == 1 for r in eligible)}
    for p in sorted({r["repo"] for r in eligible})}
n = len(eligible)
p = sum(r["arms"]["no_skill"]["rank"] == 1 for r in eligible) / n
z = 1.96
center = (p + z*z/(2*n)) / (1+z*z/n)
half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
report["baseline_wilson95_descriptive"] = [center-half, center+half]
(OUT / "accuracy_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
