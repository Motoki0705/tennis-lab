"""Utilities for loading third-party model classes by file path."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_class(module: ModuleType, class_name: str) -> type[nn.Module]:
    cls: Any = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' was not found in module '{module.__name__}'.")
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError(f"'{class_name}' in module '{module.__name__}' is not an nn.Module class.")
    return cls


@lru_cache(maxsize=1)
def load_wasb_hrnet_class() -> type[nn.Module]:
    module_path = _repo_root() / "third_party" / "WASB-SBDT" / "src" / "models" / "hrnet.py"
    module = _load_module_from_file("third_party_wasb_hrnet", module_path)
    return _get_class(module, "HRNet")


@lru_cache(maxsize=1)
def load_tracknetv3_tracknet_class() -> type[nn.Module]:
    module_path = _repo_root() / "third_party" / "TrackNetV3" / "model.py"
    module = _load_module_from_file("third_party_tracknetv3_model", module_path)
    return _get_class(module, "TrackNet")
