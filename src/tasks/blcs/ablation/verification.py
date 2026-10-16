"""Validate saved BLCS predictions, support counts and physical diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import clean_json, recompute, save_json
from .real_observations import recompute_real


def assert_recomputed(expected: Any, actual: Any, key: str = "metrics") -> None:
    """Require the same support/undefined structure and numerical values."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Saved metric keys changed: {key}")
        for name in expected:
            assert_recomputed(expected[name], actual[name], f"{key}/{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Saved support changed: {key}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_recomputed(left, right, f"{key}/{index}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or not np.isclose(expected, actual, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Saved metric mismatch: {key}: {expected} != {actual}")
    elif expected != actual:
        raise ValueError(f"Saved value changed: {key}: {expected} != {actual}")


def validate_saved_metrics(directory: Path) -> None:
    saved = json.loads((directory / "evaluation.json").read_text())
    if saved["split"] not in {"val", "test"}:
        raise ValueError("Saved evaluation has an unknown split")
    restored = recompute(
        directory / f"pred_{saved['split']}.npz", include_diagnostics=True
    )
    assert_recomputed(saved["schema"], restored["schema"])
    assert_recomputed(saved["metrics"], restored["metrics"])
    for mode in ("oracle", "predicted"):
        key = f"integrated_{mode}"
        if key in saved:
            assert_recomputed(saved[key], restored[key], key)
    originals = {row["scene_id"]: row for row in saved["clips"]}
    for row in restored.get("diagnostics", []):
        original = originals[row["scene_id"]]
        for key in row:
            assert_recomputed(original[key], row[key], f"{row['scene_id']}/{key}")
    save_json(directory / "recomputed.json", restored)


def validate_real_metrics(directory: Path, *, gravity: float | None = None) -> None:
    saved = json.loads((directory / "evaluation.json").read_text())
    restored = clean_json(recompute_real(directory, gravity=gravity))
    for key, value in restored.items():
        assert_recomputed(saved[key], value, key)
    save_json(directory / "recomputed.json", restored)
