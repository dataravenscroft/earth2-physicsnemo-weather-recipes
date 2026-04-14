"""Shared utility helpers for small recipe scripts."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_configured_path(config: dict[str, Any], key: str, repo_root: Path) -> Path:
    value = config.get(key, f"./{key}")
    return (repo_root / value).resolve()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    numpy_spec = importlib.util.find_spec("numpy")
    if numpy_spec is None:
        return

    import numpy as np

    np.random.seed(seed)


def optional_dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
