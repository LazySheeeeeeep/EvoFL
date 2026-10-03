"""Map LocAgent's reported locations to base-graph function identities."""
from __future__ import annotations

import argparse
import ast
import hashlib
from io import BytesIO
import json
import pickle
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile
import tokenize

_PATH = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.py)(?::(?P<name>[A-Za-z_][\w.]*))?")
_FUNCTION = re.compile(r"function:\s*(?P<name>[A-Za-z_][\w.]*)", re.IGNORECASE)
_CLASS = re.compile(r"class:\s*(?P<name>[A-Za-z_][\w.]*)", re.IGNORECASE)
_LINE = re.compile(r"line:\s*(?P<line>\d+)", re.IGNORECASE)


def _location_blocks(answer: str) -> list[str]:
    fenced = re.findall(r"```(?:[A-Za-z0-9_-]+)?\s*\n(.*?)```", answer, re.DOTALL)
    return fenced or [answer]


def source_functions(archive: Path, answer: str) -> dict[str, dict]:
    mentioned = {match.group("path") for block in _location_blocks(answer)
                 for line in block.splitlines()
                 if (match := _PATH.fullmatch(line.strip()))}
    functions: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as contents:
        for member in contents:
            path = PurePosixPath(member.name)
            if not member.isfile() or len(path.parts) < 2:
                continue
            relative = "/".join(path.parts[1:])
            if relative not in mentioned:
                continue
            with contents.extractfile(member) as stream:
                raw = stream.read()
            try:
                encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
                tree = ast.parse(raw.decode(encoding), filename=relative)
            except (SyntaxError, UnicodeError):
                continue

            def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified = (*parents, child.name)
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions[f"{relative}:{'.'.join(qualified)}"] = {
                                "type": "function", "start_line": child.lineno,
                                "end_line": child.end_lineno,
                            }
                        visit(child, qualified)
                    else:
                        visit(child, parents)

            visit(tree, ())
    return functions


def normalize(answer: str, graph, limit: int = 5,
              extra_functions: dict[str, dict] | None = None) -> dict:
    functions = {
        str(node): data for node, data in graph.nodes(data=True)
        if data.get("type") == "function" and ":" in str(node)
    }
    functions.update(extra_functions or {})
    ranked: list[str] = []
    mappings: list[dict] = []

    def resolve(path: str, name: str | None, class_name: str | None,
                lines: list[int]) -> None:
        path = path.removeprefix("./")
        possible = []
        if name:
            possible.append(f"{path}:{name}")
            if class_name and not name.startswith(f"{class_name}."):
                possible.insert(0, f"{path}:{class_name}.{name}")
            exact = [node for node in possible if node in functions]
            if len(exact) == 1:
                matched, status = exact[0], "exact"
            elif not exact and not class_name:
                suffix = f".{name}"
                simple = [node for node in functions
                          if node.startswith(f"{path}:") and node.rsplit(":", 1)[-1].endswith(suffix)]
                matched, status = (simple[0], "unique_file_symbol") if len(simple) == 1 else (None, "unresolved")
            else:
                matched, status = None, "unresolved"
        else:
            enclosing = [node for node, data in functions.items()
                         if node.startswith(f"{path}:") and any(
                             data.get("start_line", -1) <= line <= data.get("end_line", -1)
                             for line in lines)]
            matched, status = (enclosing[0], "unique_line_scope") if len(enclosing) == 1 else (None, "unresolved")
        mapping = {"source": f"{path}:{name or ','.join(map(str, lines))}", "status": status}
        if matched:
            identity = matched.replace(":", "::", 1)
            mapping["normalized"] = identity
            if identity not in ranked and len(ranked) < limit:
                ranked.append(identity)
        mappings.append(mapping)

    for block in _location_blocks(answer):
        path = None
        class_name = None
        names: list[str] = []
        lines: list[int] = []

        def flush() -> None:
            if path:
                if names:
                    for name in names:
                        resolve(path, name, class_name, lines)
                else:
                    resolve(path, None, class_name, lines)

        for raw_line in block.splitlines():
            line = raw_line.strip()
            path_match = _PATH.fullmatch(line)
            if path_match:
                flush()
                path = path_match.group("path")
                class_name = None
                names = [path_match.group("name")] if path_match.group("name") else []
                lines = []
                continue
            if path is None:
                continue
            if match := _CLASS.fullmatch(line):
                class_name = match.group("name")
            elif match := _FUNCTION.fullmatch(line):
                names.append(match.group("name"))
            elif match := _LINE.fullmatch(line):
                lines.append(int(match.group("line")))
        flush()
    return {"ranked_functions": ranked, "mappings": mappings}


def normalize_file(result_path: Path, graph_path: Path, output_path: Path,
                   source_archive: Path | None = None,
                   expected_archive_sha256: str | None = None) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with graph_path.open("rb") as stream:
        graph = pickle.load(stream)  # Trusted local graph produced by prepare_locagent_graph.py.
    if source_archive is not None and expected_archive_sha256 is not None:
        if hashlib.sha256(source_archive.read_bytes()).hexdigest() != expected_archive_sha256:
            raise ValueError("Source archive checksum mismatch")
    extra = source_functions(source_archive, result["final_output"]) if source_archive else {}
    normalized = {"instance_id": result["instance_id"],
                  **normalize(result["final_output"], graph, extra_functions=extra)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    arguments = parser.parse_args()
    print(json.dumps(normalize_file(arguments.result, arguments.graph, arguments.output,
                                    arguments.source_archive, arguments.expected_archive_sha256), indent=2))
