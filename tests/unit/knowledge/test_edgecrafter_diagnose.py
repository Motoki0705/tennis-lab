"""Test image-slot alignment and raw-head diagnostics on small CPU fixtures."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    runs = Path(__file__).resolve().parents[3] / "knowledge/runs"
    comparison = runs / "run-meiji-player-model-comparison"
    monkeypatch.syspath_prepend(str(runs / "run-person-model-speed-baseline"))
    monkeypatch.syspath_prepend(str(comparison))
    return runpy.run_path(str(comparison / "edgecrafter_diagnose.py"))


def test_comparison_aligns_reverse_slots_and_reports_nonfinite_heads(
    script: dict[str, Any],
) -> None:
    capture = script["Capture"]
    boxes = (np.array([[0.0, 0.0, 12.0, 4.0]]), np.array([[40.0, 8.0, 80.0, 32.0]]))
    scores = (np.array([0.99]), np.array([0.8]))
    first = capture(
        ("a", "b"),
        {
            "logits": np.array([[-8.0, 5.0, 2.0], [-2.0, 9.0, 4.0]]),
            "coordinates": np.array([[0.1, 0.9], [0.5, 0.8]]),
        },
        boxes,
        scores,
    )
    second = capture(
        ("b", "a"),
        {
            "logits": np.array([[-2.0, np.inf, 1.0], [-8.0, 5.0, 2.0]]),
            "coordinates": np.array([[0.5, 0.7], [0.1, 0.9]]),
        },
        boxes[::-1],
        scores[::-1],
    )

    result = script["compare_images"](first, second)
    assert result["a"]["candidate_slot"] == 1
    assert result["a"]["heads"]["logits"]["abs_diff_max"] == 0.0
    assert result["a"]["decoded"]["matches_at_iou_0_50"] == 1
    assert result["b"]["candidate_slot"] == 0
    assert result["b"]["heads"]["logits"]["finite_pairs"] == 2
    assert result["b"]["heads"]["logits"]["finite_mask_difference"] == 1
    assert result["b"]["heads"]["logits"]["abs_diff_max"] == 3.0
    assert result["b"]["heads"]["coordinates"]["abs_diff_max"] == pytest.approx(0.1)
    stats = script["head_stats"](second.heads["logits"][0])
    assert stats == {
        "finite_count": 2,
        "nan_count": 0,
        "posinf_count": 1,
        "neginf_count": 0,
        "finite_min": -2.0,
        "finite_max": 1.0,
    }


def test_comparison_rejects_a_raw_head_with_different_shape(
    script: dict[str, Any],
) -> None:
    capture = script["Capture"]
    first = capture(
        ("same",),
        {"logits": np.zeros((1, 2, 4)), "coordinates": np.zeros((1, 2, 4))},
        (np.empty((0, 4)),),
        (np.empty(0),),
    )
    second = capture(
        ("same",),
        {"logits": np.zeros((1, 1, 4)), "coordinates": np.zeros((1, 2, 4))},
        first.boxes,
        first.scores,
    )

    with pytest.raises(ValueError, match="Raw head shape differs for same/logits"):
        script["compare_images"](first, second)
