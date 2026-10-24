from typing import Any

import pytest
import torch

from src.tasks.ball_detection.training.dpt_pretrain.objective import (
    heatmap_objective as dpt_objective,
)
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.tasks.ball_detection.training.hybrid.objective import (
    heatmap_objective as hybrid_heatmap_objective,
)
from src.tasks.base.training.losses import FocalBCEWithLogitsLoss
from src.utils.data.heatmaps import generate_gaussian_heatmaps


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_objective_matches_original_dpt_arithmetic_exactly(dtype: torch.dtype) -> None:
    assert dpt_objective is hybrid_heatmap_objective is heatmap_objective
    torch.manual_seed(42)
    logits = torch.randn(2, 32, 9, 13, dtype=dtype, requires_grad=True)
    positive = torch.rand(2, 32) > .6
    valid = positive | (torch.rand(2, 32) > .4)
    uv = torch.rand(2, 32, 2).masked_fill(~positive[..., None], float("nan"))
    batch = dict(uv=uv, position_valid=positive, heatmap_valid=valid)
    # Historical DPT: zero invalid UV, Gaussian on native grid, spatial mean,
    # then average only explicitly supervised frames, in that order.
    target = generate_gaussian_heatmaps((9, 13), torch.where(positive[..., None], uv, 0.), .012, visibility=positive)
    assert not target[~positive].any()
    frames = FocalBCEWithLogitsLoss(2.).elementwise(logits.float(), target).mean((-2, -1))
    expected = torch.where(valid, frames, 0.).sum() / valid.sum()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, actual_frames = heatmap_objective(logits, batch)
    assert actual.dtype == actual_frames.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_frames, frames, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0], rtol=0, atol=0)


def test_central_masks_exclude_context_and_unknown_but_supervise_confirmed_absence() -> None:
    logits = torch.zeros(1, 64, 5, 7, requires_grad=True)
    positive = torch.zeros(1, 64, dtype=torch.bool)
    valid = torch.zeros_like(positive)
    positive[:, 24:28] = True
    valid[:, 24:30] = True  # two confirmed negatives; frame 30 is unknown
    uv = torch.full((1, 64, 2), float("nan"))
    uv[positive] = .4
    loss, frames = heatmap_objective(logits, dict(uv=uv, position_valid=positive, heatmap_valid=valid))
    torch.testing.assert_close(loss, frames[valid].mean())
    loss.backward()
    assert logits.grad[~valid].abs().sum() == 0
    assert (logits.grad[:, 28:30] > 0).all()
    assert logits.grad[positive].abs().sum() > 0
    absent = dict(uv=torch.full((1, 64, 2), float("nan")), position_valid=torch.zeros_like(valid), heatmap_valid=valid)
    assert torch.isfinite(heatmap_objective(logits, absent)[0])


@pytest.mark.parametrize("damage", ["mask_dtype", "mask_shape", "positive_without_valid", "uv_shape", "nan_positive", "out_of_range", "no_supervision"])
def test_invalid_supervision_is_rejected(damage: str) -> None:
    batch: dict[str, Any] = dict(uv=torch.full((1, 3, 2), .5), position_valid=torch.ones(1, 3, dtype=torch.bool), heatmap_valid=torch.ones(1, 3, dtype=torch.bool))
    if damage == "mask_dtype":
        batch["heatmap_valid"] = batch["heatmap_valid"].float()
    elif damage == "mask_shape":
        batch["heatmap_valid"] = batch["heatmap_valid"][:, :1]
    elif damage == "positive_without_valid":
        batch["heatmap_valid"][:, 0] = False
    elif damage == "uv_shape":
        batch["uv"] = batch["uv"][..., :1]
    elif damage == "nan_positive":
        batch["uv"][:, 0] = float("nan")
    elif damage == "out_of_range":
        batch["uv"][:, 0] = 1.1
    else:
        batch["heatmap_valid"].zero_()
        batch["position_valid"].zero_()
    with pytest.raises(ValueError):
        heatmap_objective(torch.zeros(1, 3, 5, 7), batch)
