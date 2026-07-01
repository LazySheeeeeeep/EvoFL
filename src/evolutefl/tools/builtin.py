from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from .registry import ToolRegistry, openai_function_tool


SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", "venv", ".venv"}
TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".txt",
    ".md",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".html",
    ".css",
}


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    run_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> ToolRegistry:
    fixed_repo_path = str(Path(repo_path).resolve()) if repo_path is not None else None
    if fixed_repo_path:
        registry.register("grep", _grep_factory(fixed_repo_path), _grep_schema(include_repo_path=False))
        registry.register("read_file", _read_file_factory(fixed_repo_path), _read_file_schema(include_repo_path=False))
    else:
        registry.register("grep", grep, _grep_schema(include_repo_path=True))
        registry.register("read_file", read_file, _read_file_schema(include_repo_path=True))
    registry.register("write", _write_factory(run_dir), _write_schema())
    return registry


def _grep_factory(fixed_repo_path: str):
    def fixed_grep(
        pattern: str,
        glob: str | None = None,
        max_results: int = 50,
        use_regex: bool = False,
        repo_path: str | None = None,
    ) -> dict[str, Any]:
        return grep(
            repo_path=fixed_repo_path,
            pattern=pattern,
            glob=glob,
            max_results=max_results,
            use_regex=use_regex,
        )

    return fixed_grep


def _read_file_factory(fixed_repo_path: str):
    def fixed_read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 200,
        repo_path: str | None = None,
    ) -> dict[str, Any]:
        return read_file(
            repo_path=fixed_repo_path,
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_lines=max_lines,
        )

    return fixed_read_file


def grep(
    repo_path: str,
    pattern: str,
    glob: str | None = None,
    max_results: int = 50,
    use_regex: bool = False,
) -> dict[str, Any]:
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"repo_path does not exist: {root}")
    if not pattern:
        raise ValueError("pattern cannot be empty.")
    max_results = int(max_results or 50)
    matches: list[dict[str, Any]] = []
    regex = re.compile(pattern, re.IGNORECASE) if use_regex else None
    pattern_lower = pattern.lower()
    for file_path in _iter_text_files(root, glob_pattern=glob):
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    matched = bool(regex.search(line)) if regex else pattern_lower in line.lower()
                    if matched:
                        matches.append(
                            {
                                "path": str(file_path.relative_to(root)).replace("\\", "/"),
                                "line": line_no,
                                "text": line.rstrip("\n"),
                            }
                        )
                        if len(matches) >= max_results:
                            return {"pattern": pattern, "use_regex": use_regex, "matches": matches, "truncated": True}
        except OSError:
            continue
    return {"pattern": pattern, "use_regex": use_regex, "matches": matches, "truncated": False}


def read_file(repo_path: str, path: str, start_line: int | None = None, end_line: int | None = None, max_lines: int = 200) -> dict[str, Any]:
    root = Path(repo_path).resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"read_file path escapes repo_path: {path}")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    max_lines = int(max_lines or 200)
    start = max(1, int(start_line or 1))
    end = int(end_line or min(len(lines), start + max_lines - 1))
    end = min(end, start + max_lines - 1, len(lines))
    snippet = [
        {"line": line_no, "text": lines[line_no - 1]}
        for line_no in range(start, end + 1)
    ]
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "content": snippet,
    }


def _write_factory(run_dir: str | Path | None):
    base = Path(run_dir or ".").resolve()

    def write(output_path: str, content: str) -> dict[str, Any]:
        target = (base / output_path).resolve()
        if base != target and base not in target.parents:
            raise ValueError("write can only write under the run_dir.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"output_path": str(target), "bytes": len(content.encode("utf-8"))}

    return write


def _iter_text_files(root: Path, *, glob_pattern: str | None = None):
    for file_path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        if not file_path.is_file():
            continue
        if glob_pattern and not fnmatch.fnmatch(str(file_path.relative_to(root)).replace("\\", "/"), glob_pattern):
            continue
        if file_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield file_path


def _grep_schema(*, include_repo_path: bool = True) -> dict[str, Any]:
    properties = {
        "pattern": {"type": "string", "description": "Literal text pattern to search for."},
        "glob": {"type": "string", "description": "Optional file glob, e.g. **/*.py."},
        "max_results": {"type": "integer", "description": "Maximum number of matches to return."},
        "use_regex": {"type": "boolean", "description": "Set true to interpret pattern as a Python regular expression."},
    }
    required = ["pattern"]
    if include_repo_path:
        properties = {"repo_path": {"type": "string", "description": "Repository root path."}, **properties}
        required = ["repo_path", *required]
    return openai_function_tool(
        name="grep",
        description=(
            "Search repository text files and return matching file lines. "
            "By default pattern is literal text; set use_regex=true for regular expressions like 'foo|bar' or '^def '."
        ),
        properties=properties,
        required=required,
    )


def _read_file_schema(*, include_repo_path: bool = True) -> dict[str, Any]:
    properties = {
        "path": {"type": "string", "description": "File path relative to repo_path."},
        "start_line": {"type": "integer", "description": "Optional 1-based start line."},
        "end_line": {"type": "integer", "description": "Optional 1-based end line."},
        "max_lines": {"type": "integer", "description": "Maximum number of lines to return."},
    }
    required = ["path"]
    if include_repo_path:
        properties = {"repo_path": {"type": "string", "description": "Repository root path."}, **properties}
        required = ["repo_path", *required]
    return openai_function_tool(
        name="read_file",
        description="Read a bounded line range from a repository file.",
        properties=properties,
        required=required,
    )


def _write_schema() -> dict[str, Any]:
    return openai_function_tool(
        name="write",
        description="Write notes or intermediate artifacts under the current run directory only.",
        properties={
            "output_path": {"type": "string", "description": "Path under run_dir."},
            "content": {"type": "string", "description": "Content to write."},
        },
        required=["output_path", "content"],
    )
