from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
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
# A single minified/generated source line must not consume the whole Explorer
# context merely because it contains a requested literal.
MAX_GREP_MATCH_TEXT_CHARS = 2_000


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
    if not use_regex:
        ripgrep_result = _grep_literal_with_ripgrep(root, pattern, glob=glob, max_results=max_results)
        if ripgrep_result is not None:
            return ripgrep_result

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
                                "text": _truncate_grep_match_text(line.rstrip("\n")),
                            }
                        )
                        if len(matches) >= max_results:
                            return {"pattern": pattern, "use_regex": use_regex, "matches": matches, "truncated": True}
        except OSError:
            continue
    return {"pattern": pattern, "use_regex": use_regex, "matches": matches, "truncated": False}


def _grep_literal_with_ripgrep(
    root: Path,
    pattern: str,
    *,
    glob: str | None,
    max_results: int,
) -> dict[str, Any] | None:
    """Use rg for fast literal searches, falling back when it is unavailable.

    Repository copies frequently live under the WSL-mounted Windows drive. A
    Python file-by-file scan there is disproportionately slow, while ripgrep
    performs the traversal natively. Regex requests retain the Python engine
    below so their documented semantics remain unchanged.
    """
    ripgrep = os.environ.get("EVOLUTEFL_RIPGREP") or shutil.which("rg")
    if not ripgrep:
        return None

    command = [
        ripgrep,
        "--no-heading",
        "--color",
        "never",
        "--line-number",
        "--with-filename",
        "--ignore-case",
        "--fixed-strings",
    ]
    if glob:
        command.extend(["--glob", glob])
    command.extend([pattern, "."])

    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    matches: list[dict[str, Any]] = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            relative_path = parts[0][2:] if parts[0].startswith("./") else parts[0]
            relative_path = relative_path.replace("\\", "/")
            if any(part in SKIP_DIRS for part in Path(relative_path).parts):
                continue
            if not glob and Path(relative_path).suffix.lower() not in TEXT_EXTENSIONS:
                continue
            matches.append(
                {
                    "path": relative_path,
                    "line": int(parts[1]),
                    "text": _truncate_grep_match_text(parts[2]),
                }
            )
            if len(matches) >= max_results:
                process.terminate()
                return {"pattern": pattern, "use_regex": False, "matches": matches, "truncated": True}
    finally:
        process.stdout.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    return {"pattern": pattern, "use_regex": False, "matches": matches, "truncated": False}


def _truncate_grep_match_text(text: str) -> str:
    """Keep grep observations bounded while retaining both ends of long lines."""
    if len(text) <= MAX_GREP_MATCH_TEXT_CHARS:
        return text
    head = 1_600
    tail = 300
    omitted = len(text) - head - tail
    return f"{text[:head]} ... [truncated {omitted} characters] ... {text[-tail:]}"


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
    max_lines = min(200, max(1, int(max_lines or 200)))
    start = max(1, int(start_line or 1))
    end = int(end_line or min(len(lines), start + max_lines - 1))
    end = min(end, start + max_lines - 1, len(lines))
    requested_end = min(int(end_line) if end_line is not None else len(lines), len(lines))
    snippet, chars = [], 0
    for line_no in range(start, end + 1):
        text = lines[line_no - 1]
        # Always deliver at least one complete line, even an unusually long one.
        if snippet and chars + len(text) > 16000:
            break
        snippet.append({"line": line_no, "text": text})
        chars += len(text)
    delivered_end = snippet[-1]["line"] if snippet else None
    return {
        "path": path,
        "start_line": start,
        "end_line": delivered_end,
        "content": snippet,
        "total_lines": len(lines),
        "requested_end_line": requested_end,
        "next_start_line": delivered_end + 1 if delivered_end is not None and delivered_end < requested_end else None,
        "truncated": delivered_end is not None and delivered_end < requested_end,
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
    # Prune expensive trees before descending. Path.rglob() filters them only
    # after traversal, which makes a simple search unexpectedly slow in large
    # repositories with vendored dependencies or VCS metadata.
    for current_root, directories, filenames in os.walk(root):
        directories[:] = [directory for directory in directories if directory not in SKIP_DIRS]
        current_path = Path(current_root)
        for filename in filenames:
            file_path = current_path / filename
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            relative_path = str(file_path.relative_to(root)).replace("\\", "/")
            if glob_pattern and not fnmatch.fnmatch(relative_path, glob_pattern):
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
        "start_line": {"type": "integer", "description": "Inclusive 1-based start, default 1. Set explicitly for a known location or use next_start_line to continue a previous page."},
        "end_line": {"type": "integer", "description": "Inclusive end of the requested range. Setting only end_line still starts at line 1. Returned end_line is the actual last delivered line."},
        "max_lines": {"type": "integer", "description": "Lines per page, default and maximum 200. A source-character budget may shorten the page; follow next_start_line for the remainder."},
    }
    required = ["path"]
    if include_repo_path:
        properties = {"repo_path": {"type": "string", "description": "Repository root path."}, **properties}
        required = ["repo_path", *required]
    return openai_function_tool(
        name="read_file",
        description=("Read a source page with complete numbered lines. For lines 380-430, pass start_line=380 and end_line=430. "
                     "Omitting start_line reads from line 1, even if end_line is large. "
                     "Continue with next_start_line; use find_symbol/read_symbol for a known Python function."),
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
