from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def safe_child(root: str | Path, child: str | Path) -> Path:
    root_path = Path(root).resolve()
    target = (root_path / child).resolve()
    if root_path != target and root_path not in target.parents:
        raise ValueError(f"Path escapes root: {target}")
    return target

