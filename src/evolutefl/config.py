from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .json_utils import read_json


DEFAULT_CONFIG_PATH = Path("config/evolutefl.global.json")


def project_root() -> Path:
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return read_json(config_path)


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(root) if root else project_root()) / candidate


def llm_config(config: dict[str, Any], provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    base = deepcopy(config.get("llm", {}))
    providers = base.pop("providers", {})
    if provider:
        selected = deepcopy(providers.get(provider, {}))
        if not selected:
            raise ValueError(f"Unknown provider '{provider}'. Available: {sorted(providers)}")
        base.update(selected)
    if model:
        base["model"] = model
    base_url_env = base.get("base_url_env")
    if base_url_env and os.getenv(base_url_env):
        base["base_url"] = os.environ[base_url_env]
    return base

