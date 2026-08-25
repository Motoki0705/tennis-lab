"""Fail-closed BLCS checkpoint metadata contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.schema.court_normalization import load_and_validate_checkpoint


def load_checkpoint_config(path: Path) -> Any:
    """Load the explicit configuration required to compose a BLCS checkpoint."""
    checkpoint = load_and_validate_checkpoint(path)
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise RuntimeError(
            "BLCS checkpoint is incompatible: hyper_parameters.config is required "
            "to compose its typed model I/O contract."
        )
    return hyper_parameters["config"]


__all__ = ["load_checkpoint_config"]
