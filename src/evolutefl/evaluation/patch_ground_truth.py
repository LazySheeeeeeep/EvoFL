from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


DIFF_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def functions_from_patch(patch: str, repo_root: str | Path, *, source_side: str = "old") -> list[str]:
    """Return function identities from actual changed lines on both patch sides."""
    return symbols_from_patch(patch, repo_root, source_side=source_side)["functions"]


def symbols_from_patch(patch: str, repo_root: str | Path, *, source_side: str = "old") -> dict[str, Any]:
    """Reconstruct the other side in memory; never modify the inspected tree."""

    root = Path(repo_root)
    if source_side not in {"old", "new"}:
        raise ValueError("source_side must be old or new")
    groups: dict[str, set[str]] = {"functions": set(), "classes": set(), "module_files": set()}
    evidence = []
    for file_info in parse_patch_files(patch):
        path = str(file_info["new_path"] if source_side == "new" else file_info["old_path"])
        if not path.endswith(".py"):
            continue
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"Patch path escapes repository: {path}")
        exists_on_side = not file_info.get("new_file" if source_side == "old" else "deleted_file", False)
        if exists_on_side and not resolved.is_file():
            raise ValueError(f"Patch source is missing: {path}")
        current = resolved.read_text(encoding="utf-8", errors="replace").splitlines() if exists_on_side else []
        other = reconstruct_other_side(current, file_info["hunks"], source_side)
        sources = {source_side: current, "old" if source_side == "new" else "new": other}
        spans_by_side = {side: _python_symbol_spans_text("\n".join(lines), strict=True)
                         for side, lines in sources.items()}
        for hunk in file_info["hunks"]:
            positions = {"old": hunk["old_start"], "new": hunk["new_start"]}
            for raw in hunk["lines"]:
                prefix = raw[:1]
                if prefix not in {"+", "-", " "}:
                    continue
                side = "old" if prefix == "-" else "new"
                if prefix in {"+", "-"} and raw[1:].strip() and not raw[1:].lstrip().startswith("#"):
                    line = positions[side]
                    matches = [s for s in spans_by_side[side] if s["start"] <= line <= s["end"]]
                    symbol = max(matches, key=lambda s: s["depth"]) if matches else None
                    kind = symbol["kind"] if symbol else "module"
                    name = symbol["qualified_name"] if symbol else None
                    groups[{"function": "functions", "class": "classes", "module": "module_files"}[kind]].add(
                        f"{path}::{name}" if name else path)
                    evidence.append({"path": path, "side": side, "line": line, "kind": kind, "symbol": name})
                if prefix in {"-", " "}:
                    positions["old"] += 1
                if prefix in {"+", " "}:
                    positions["new"] += 1
    return {**{key: sorted(values) for key, values in groups.items()}, "evidence": evidence}


def reconstruct_other_side(current: list[str], hunks: list[dict], side: str) -> list[str]:
    result, cursor = [], 0
    own = "-" if side == "old" else "+"
    opposite = "+" if side == "old" else "-"
    for hunk in hunks:
        start = int(hunk[side + "_start"])
        expected = [s[1:] for s in hunk["lines"] if s[:1] in {" ", own}]
        replacement = [s[1:] for s in hunk["lines"] if s[:1] in {" ", opposite}]
        offset = max(0, start - 1) if expected else start
        if offset < cursor or offset > len(current) or current[offset:offset + len(expected)] != expected:
            # git apply can relocate a hunk. Accept only a unique exact match,
            # never fuzzy context or an inferred neighboring function.
            matches = [i for i in range(cursor, len(current) - len(expected) + 1)
                       if expected and current[i:i + len(expected)] == expected]
            if len(matches) != 1:
                raise ValueError(f"Patch does not match supplied {side} source at line {start}")
            offset = matches[0]
        other_offset = offset + len(result) - cursor
        hunk[side + "_start"] = offset + 1 if expected else offset
        other_side = "new" if side == "old" else "old"
        hunk[other_side + "_start"] = other_offset + 1 if replacement else other_offset
        result.extend(current[cursor:offset])
        result.extend(replacement)
        cursor = offset + len(expected)
    result.extend(current[cursor:])
    return result


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
        if line.startswith("new file mode"):
            current["new_file"] = True
        if line.startswith("deleted file mode"):
            current["deleted_file"] = True
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
            # block with the closest *preceding* surviving base line once.
            # Using ``old_line`` points at the following line. At a function
            # boundary that can falsely attribute a deleted tail of one
            # function to the definition of the next one.
            if not pending_addition:
                changed.append(max(1, old_line - 1))
                pending_addition = True
        else:
            old_line += 1
            pending_addition = False
    return sorted(set(changed))


def python_symbol_spans(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _python_symbol_spans_text(path.read_text(encoding="utf-8", errors="replace"))


def _python_symbol_spans_text(text: str, *, strict: bool = False) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        if strict:
            raise ValueError(f"Cannot map Python patch scopes: {exc}") from exc
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
                        "start": min([child.lineno] + [d.lineno for d in child.decorator_list]),
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
    *, strict: bool = False,
) -> dict[str, Any]:
    rank: int | None = None
    matched_ground_truth: str | None = None
    for index, prediction in enumerate(ranked_functions, start=1):
        for ground_truth in ground_truth_functions:
            if function_identity_matches(prediction, ground_truth, strict=strict):
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


def function_identity_matches(prediction: str, ground_truth: str, *, strict: bool = False) -> bool:
    prediction_file, prediction_name = split_function_identity(prediction)
    truth_file, truth_name = split_function_identity(ground_truth)
    # The outer ``path::qualified_name`` separator is required by Explorer's
    # output contract. Some providers nevertheless render a class method as
    # ``Class::method`` inside the qualified name. Both forms identify the
    # same Python symbol, so canonicalize only the inner spelling here.
    prediction_name = prediction_name.replace("::", ".")
    truth_name = truth_name.replace("::", ".")
    prediction_file, prediction_name = normalize_python_module_identity(
        prediction,
        prediction_file,
        prediction_name,
        truth_file,
    )
    if prediction_file != truth_file:
        return False
    if strict:
        return bool(prediction_name and truth_name and prediction_name == truth_name)
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
