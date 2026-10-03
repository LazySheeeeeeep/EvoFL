"""Python definition navigation using the inspected source, never patch data."""
from __future__ import annotations

import ast
import tokenize
from pathlib import Path

from .builtin import _iter_text_files, read_file
from .registry import openai_function_tool


def _definitions(root: Path, path: str):
    target = (root / path).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Symbol path must be a file inside the repository")
    with tokenize.open(target) as source:
        tree = ast.parse(source.read(), filename=path)
    records = []

    def visit(node, parents):
        is_symbol = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        if is_symbol:
            names = [*parents, node.name]
            qualified = ".".join(names)
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            signature = "class " + node.name if isinstance(node, ast.ClassDef) else node.name + "(" + ast.unparse(node.args) + ")"
            records.append({"symbol_id": f"{path}::{qualified}", "path": path,
                            "qualified_name": qualified, "name": node.name,
                            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                            "signature": signature, "start_line": start, "end_line": node.end_lineno})
            parents = names
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, [])
    return records


def register_symbol_tools(registry, repo_path):
    root = Path(repo_path).resolve(strict=True)

    def find_symbol(name: str, path: str | None = None, offset: int = 0, limit: int = 20):
        if not name.strip() or offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Use a nonempty name, nonnegative offset and limit between 1 and 100")
        files = [root / path] if path else sorted(_iter_text_files(root))
        matches, errors = [], []
        for file in files:
            if file.suffix not in {".py", ".pyi"}:
                continue
            relative = file.relative_to(root).as_posix()
            try:
                matches.extend(d for d in _definitions(root, relative)
                               if name in {d["name"], d["qualified_name"]})
            except (SyntaxError, UnicodeError) as exc:
                errors.append({"path": relative, "error": str(exc)})
        page = matches[offset:offset + limit]
        return {"symbols": page, "total_matches": len(matches),
                "next_offset": offset + limit if offset + limit < len(matches) else None,
                "parse_errors": errors,
                "note": "Exact Python definitions; use grep/read_file for dynamic or unparseable code."}

    def read_symbol(symbol_id: str, start_line: int | None = None, max_lines: int = 120):
        path, qualified = symbol_id.split("::", 1)
        matches = [d for d in _definitions(root, path) if d["qualified_name"] == qualified]
        if len(matches) != 1:
            raise ValueError("Symbol missing or ambiguous; use find_symbol and read_file line ranges")
        symbol = matches[0]
        start = symbol["start_line"] if start_line is None else start_line
        if not symbol["start_line"] <= start <= symbol["end_line"]:
            raise ValueError("start_line must lie inside the symbol")
        return {"symbol": symbol, **read_file(str(root), path, start, symbol["end_line"], max_lines)}

    registry.register("find_symbol", find_symbol, openai_function_tool(
        name="find_symbol", description="Find Python class/function definitions, including nested functions, by exact short or qualified name. Searching for an inner function name returns its full path::Class.outer.inner ID and line range, not its body.",
        properties={"name": {"type": "string"}, "path": {"type": "string"},
                    "offset": {"type": "integer"}, "limit": {"type": "integer"}}, required=["name"]))
    registry.register("read_symbol", read_symbol, openai_function_tool(
        name="read_symbol", description="Read a Python symbol from find_symbol, including decorators. Follow next_start_line to continue a long body. Module-level context is available via read_file.",
        properties={"symbol_id": {"type": "string"}, "start_line": {"type": "integer"},
                    "max_lines": {"type": "integer"}}, required=["symbol_id"]))
    return registry
