"""Normalize CoSIL element predictions to strict base-source function IDs."""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
import re
from xml.etree import ElementTree


def _source_files(structure: dict) -> dict[str, list[str]]:
    files = {}

    def visit(node: dict, parts: tuple[str, ...]) -> None:
        for name, child in node.items():
            if not isinstance(child, dict):
                continue
            path = (*parts, name)
            if name.endswith(".py") and isinstance(child.get("text"), list):
                files["/".join(path)] = child["text"]
            elif "text" not in child:
                visit(child, path)

    visit(structure, ())
    return files


def _function_names(lines: list[str]) -> set[str]:
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return set()
    names = set()

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (*parents, child.name)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(".".join(qualified))
                visit(child, qualified)
            else:
                visit(child, parents)

    visit(tree, ())
    return names


def _recover_raw_locations(row: dict, files: dict[str, list[str]],
                           repository_root: str) -> dict[str, list[str]]:
    """Recover author XML only when its parsed function locations are empty."""
    response = (row.get("func_traj") or {}).get("response")
    if not isinstance(response, str):
        return {}
    block = re.search(r"<locations>.*?</locations>", response, re.DOTALL)
    if not block:
        return {}
    try:
        root = ElementTree.fromstring(block.group())
    except ElementTree.ParseError:
        return {}
    candidates = set(row.get("found_files") or [])
    recovered: dict[str, list[str]] = {}
    for location in root.findall("location"):
        if location.findtext("type", "").strip() != "function":
            continue
        raw_path = location.findtext("file", "").strip().replace("\\", "/").removeprefix("./")
        name = location.findtext("name", "").strip()
        source_path = raw_path if raw_path in files else f"{repository_root}/{raw_path}"
        if not name or source_path not in files or source_path not in candidates:
            continue
        recovered.setdefault(source_path, []).append(f"function: {name}")
    return recovered


def normalize(row: dict, structure: dict, limit: int = 5) -> dict:
    files = _source_files(structure)
    if len(structure) != 1:
        raise ValueError("Expected a single repository root in CoSIL structure")
    repository_root = next(iter(structure))
    predictions = []
    mappings = []
    locations_by_file = row.get("found_related_locs") or {}
    recovered = not any(
        line.startswith("function:")
        for blocks in locations_by_file.values()
        for block in blocks
        for line in str(block).splitlines()
    )
    if recovered:
        locations_by_file = _recover_raw_locations(row, files, repository_root)
    for raw_path, locations in locations_by_file.items():
        raw_path = str(raw_path).replace("\\", "/").removeprefix("./")
        source_path = raw_path if raw_path in files else f"{repository_root}/{raw_path}"
        if source_path not in files:
            mappings.append({"source": raw_path, "status": "unknown_file"})
            continue
        relative_path = source_path[len(repository_root) + 1:]
        names = _function_names(files[source_path])
        simple = defaultdict(set)
        for qualified in names:
            simple[qualified.rsplit(".", 1)[-1]].add(qualified)
        for block in locations:
            for line in str(block).splitlines():
                if not line.startswith("function:"):
                    continue
                raw_name = line.partition(":")[2].strip()
                if raw_name in names:
                    name, status = raw_name, "exact"
                elif len(simple[raw_name]) == 1:
                    name, status = next(iter(simple[raw_name])), "unique_file_symbol"
                else:
                    name, status = f"__unresolved__.{raw_name}", "unresolved"
                if recovered and status == "unresolved":
                    continue
                identity = f"{relative_path}::{name}"
                mappings.append({"source": f"{raw_path}::{raw_name}",
                                 "normalized": identity,
                                 "status": f"raw_xml_recovered_{status}" if recovered else status})
                if identity not in predictions:
                    predictions.append(identity)
                if len(predictions) >= limit:
                    return {"instance_id": row["instance_id"],
                            "ranked_functions": predictions, "mappings": mappings}
    return {"instance_id": row["instance_id"],
            "ranked_functions": predictions, "mappings": mappings}


def normalize_file(predictions: Path, structures: Path, output: Path) -> dict:
    rows = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        structure = json.loads((structures / f"{row['instance_id']}.json").read_text(encoding="utf-8"))
        rows.append(normalize(row, structure["structure"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    return {"case_count": len(rows), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_file(args.predictions, args.structures, args.output), indent=2))
