"""Tiny deterministic contracts for diagnostic interventions and meter units."""

import importlib.util
from pathlib import Path

import pytest
import torch

from src.utils.schema.court import COURT_COORD_SCALE_XYZ

spec = importlib.util.spec_from_file_location(
    "sensitivity_probe", Path(__file__).with_name("probe.py")
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "condition,keys",
    [("ball_reverse", {"ball_uv", "ball_vis"}), ("dino_reverse", {"dino_tokens"})],
)
def test_reversal_preserves_slots_masks_targets(condition: str, keys: set[str]) -> None:
    batch = {
        "padding_mask": torch.tensor([[False, False, True]]),
        "ball_uv": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]),
        "ball_vis": torch.tensor([[1.0, 0.25, 0.0]]),
        "dino_tokens": torch.tensor([[[1.0], [2.0], [0.0]]]),
        "dino_frame_idx": torch.tensor([[0, 10, 0]]),
        "dino_padding_mask": torch.tensor([[False, False, True]]),
        "target_ball_position": torch.arange(9).reshape(1, 3, 3),
        "target_ball_valid": torch.tensor([[True, False, False]]),
        "target_ball_weight": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    original = {k: v.clone() for k, v in batch.items()}
    changed, mapping = probe.intervene(batch, condition)
    assert mapping == {"destination_slots": [0, 1], "source_slots": [1, 0]}
    for key in keys:
        assert torch.equal(changed[key], original[key][:, [1, 0, 2]])
    for key in set(batch) - keys:
        assert torch.equal(changed[key], original[key])
    for key in batch:
        assert torch.equal(batch[key], original[key])
    restored, _ = probe.intervene(changed, condition)
    assert all(torch.equal(restored[k], original[k]) for k in batch)


def test_metric_is_euclidean_meters_with_separate_teacher_mask() -> None:
    scale = torch.tensor(COURT_COORD_SCALE_XYZ, dtype=torch.float64)
    pred = (
        torch.tensor(
            [[[3.0, 4.0, 0.0], [0.0, 0.0, 12.0], [999.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        / scale
    )
    baseline = torch.zeros_like(pred)
    result = probe.metrics(
        pred,
        baseline,
        baseline,
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, False, False]]),
    )
    assert result["output_difference_m"]["mean_euclidean"] == pytest.approx(8.5)
    assert result["output_difference_m"]["rms_euclidean"] == pytest.approx(
        (169 / 2) ** 0.5
    )
    assert result["output_difference_m"]["max_euclidean"] == pytest.approx(12.0)
    assert result["teacher_error_m_mean"] == pytest.approx(5.0)
