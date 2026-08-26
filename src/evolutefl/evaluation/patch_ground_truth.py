from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


DIFF_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def functions_from_patch(patch: str, repo_root: str | Path) -> list[str]:
    """Map Python patch hunks to enclosing function or class identities."""

    root = Path(repo_root)
    identities: set[str] = set()
    for file_info in parse_patch_files(patch):
        path = str(file_info["new_path"])
        if not path.endswith(".py"):
            continue
        spans = python_symbol_spans(root / path)
        for hunk in file_info["hunks"]:
            for line in changed_base_lines(hunk):
                qualified_name = enclosing_symbol(spans, line)
                if qualified_name:
                    identities.add(f"{path}::{qualified_name}")
    return sorted(identities)


def parse_patch_files(patch: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in str(patch or "").splitlines():
        match = DIFF_RE.match(line)
        if match:
            current = {"old_path": match.group(1), "new_path": match.group(2), "hunks": []}
            files.append(current)
            continue
        if current is None:
            continue
        hunk_match = HUNK_RE.match(line)
        if hunk_match:
            current["hunks"].append(
                {
                    "old_start": int(hunk_match.group(1)),
                    "new_start": int(hunk_match.group(2)),
                    "lines": [],
                }
            )
            continue
        if current["hunks"]:
            current["hunks"][-1]["lines"].append(line)
    return files


def changed_base_lines(hunk: dict[str, Any]) -> list[int]:
    """Return line numbers that can be resolved against the unpatched repository."""

    old_line = int(hunk["old_start"])
    changed: list[int] = []
    pending_addition = False
    for raw in hunk["lines"]:
        if raw.startswith("\\ No newline"):
            continue
        prefix = raw[:1]
        if prefix == "-":
            changed.append(old_line)
            old_line += 1
            pending_addition = False
        elif prefix == "+":
            # Added lines do not exist in the base tree. Associate an addition
            # block with the closest surviving base line once.
            if not pending_addition:
                changed.append(max(1, old_line))
                pending_addition = True
        else:
            old_line += 1
            pending_addition = False
    return sorted(set(changed))


def python_symbol_spans(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    spans: list[dict[str, Any]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = ".".join([*parents, child.name])
                spans.append(
                    {
                        "qualified_name": qualified_name,
                        "kind": "class" if isinstance(child, ast.ClassDef) else "function",
                        "start": int(getattr(child, "lineno", 1)),
                        "end": _node_end_line(child),
                        "depth": len(parents),
                    }
                )
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return spans


def _node_end_line(node: ast.AST) -> int:
    """Return a reliable inclusive end line on Python versions without end_lineno."""

    start = int(getattr(node, "lineno", 1) or 1)
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int) and end >= start:
        return end
    return max(
        (int(getattr(descendant, "lineno", start) or start) for descendant in ast.walk(node)),
        default=start,
    )


def enclosing_symbol(spans: list[dict[str, Any]], line: int) -> str | None:
    functions = [
        span
        for span in spans
        if span["kind"] == "function" and span["start"] <= line <= span["end"]
    ]
    if functions:
        # Prefer the most specific nested function.
        functions.sort(key=lambda span: (-span["depth"], span["end"] - span["start"]))
        return str(functions[0]["qualified_name"])
    classes = [
        span
        for span in spans
        if span["kind"] == "class" and span["start"] <= line <= span["end"]
    ]
    if classes:
        classes.sort(key=lambda span: (-span["depth"], span["end"] - span["start"]))
        return str(classes[0]["qualified_name"])
    return None


def evaluate_ranked_functions(
    ranked_functions: list[str],
    ground_truth_functions: list[str],
) -> dict[str, Any]:
    rank: int | None = None
    matched_ground_truth: str | None = None
    for index, prediction in enumerate(ranked_functions, start=1):
        for ground_truth in ground_truth_functions:
            if function_identity_matches(prediction, ground_truth):
                rank = index
                matched_ground_truth = ground_truth
                break
        if rank is not None:
            break
    return {
        "rank": rank,
        "matched_ground_truth": matched_ground_truth,
        "top1": rank == 1,
        "top3": rank is not None and rank <= 3,
        "top5": rank is not None and rank <= 5,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


def function_identity_matches(prediction: str, ground_truth: str) -> bool:
    prediction_file, prediction_name = split_function_identity(prediction)
    truth_file, truth_name = split_function_identity(ground_truth)
    prediction_file, prediction_name = normalize_python_module_identity(
        prediction,
        prediction_file,
        prediction_name,
        truth_file,
    )
    if prediction_file != truth_file:
        return False
    return (
        prediction_name == truth_name
        or bool(prediction_name and truth_name.endswith("." + prediction_name))
        or bool(truth_name and prediction_name.endswith("." + truth_name))
    )


def split_function_identity(value: str) -> tuple[str, str]:
    text = str(value or "").strip().replace("\\", "/")
    if "::" not in text:
        return text, ""
    path, qualified_name = text.split("::", 1)
    return path, qualified_name


def normalize_python_module_identity(
    raw_prediction: str,
    prediction_file: str,
    prediction_name: str,
    truth_file: str,
) -> tuple[str, str]:
    """Recognize a dotted Python identity for the exact ground-truth module.

    Explorer outputs may use ``package.module.symbol`` instead of the
    evaluation format ``package/module.py::symbol``.  Only normalize when the
    dotted prefix exactly names the patched Python module, so this cannot turn
    a merely similar symbol from another file into a hit.
    """

    if not truth_file.endswith(".py"):
        return prediction_file, prediction_name

    module = truth_file[:-3].replace("/", ".")
    if prediction_file == module:
        return truth_file, prediction_name
    if prediction_name or "::" in raw_prediction:
        return prediction_file, prediction_name
    prefix = f"{module}."
    normalized = str(raw_prediction or "").strip().replace("\\", "/")
    if normalized.startswith(prefix):
        return truth_file, normalized[len(prefix) :]
    path_prefix = f"{truth_file[:-3]}."
    if normalized.startswith(path_prefix):
        return truth_file, normalized[len(path_prefix) :]
    return prediction_file, prediction_name
