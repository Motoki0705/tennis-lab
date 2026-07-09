"""Tests for tennis_scene BLCS pipeline component."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

from src.tasks.blcs.models import BLCSModel, BLCSMultiViewAxialModel
from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule, BLCSResult


class _FakeBLCSPredictor:
    def predict(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        del court_vis, denormalize
        assert ball_uv.shape == (1, 1, 4, 2)
        assert court_kp.shape == (1, 1, 4, 20, 2)
        assert ball_vis is not None
        assert ball_vis.shape == (1, 1, 4)
        torch.testing.assert_close(
            ball_vis,
            torch.tensor([[[1.0, 0.0, 1.0, 1.0]]]),
        )
        assert ball_mask is not None
        assert ball_mask.shape == (1, 1, 4)
        torch.testing.assert_close(ball_mask, torch.ones_like(ball_mask))
        return {"position": torch.ones(1, 4, 3)}


def test_process_keeps_detector_visibility_for_invisible_tokens_without_zero_filling() -> None:
    module = BLCSModule(BLCSConfig(checkpoint="dummy.ckpt", device="cpu"))
    module._predictor = cast(Any, _FakeBLCSPredictor())
    result = module.process(
        ball_uv=np.zeros((1, 4, 2), dtype=np.float32),
        court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
        ball_vis=np.array([[True, False, True, True]], dtype=np.bool_),
        court_vis=np.ones((1, 4, 20), dtype=np.float32),
    )

    assert result.ball_3d.shape == (4, 3)
    assert result.visibility is not None
    np.testing.assert_array_equal(
        result.visibility,
        np.array([True, False, True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(result.ball_3d, np.ones((4, 3), dtype=np.float32))


def test_result_validate_allows_inferred_positions_for_invisible_frames() -> None:
    result = BLCSResult(
        ball_3d=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        visibility=np.array([True, False], dtype=np.bool_),
    )

    is_valid, errors = result.validate()

    assert is_valid
    assert errors == []


def test_validate_pipeline_checkpoint_rejects_single_model() -> None:
    module = BLCSModule(BLCSConfig(checkpoint="dummy.ckpt", device="cpu"))
    module._predictor = cast(
        Any,
        SimpleNamespace(model=BLCSModel(hidden_dim=16, num_layers=1, num_heads=4)),
    )

    with pytest.raises(ValueError, match="requires a multiview BLCS checkpoint"):
        module._validate_pipeline_checkpoint_profile()


def test_validate_pipeline_checkpoint_accepts_multiview_model() -> None:
    module = BLCSModule(BLCSConfig(checkpoint="dummy.ckpt", device="cpu"))
    module._predictor = cast(
        Any,
        SimpleNamespace(
            model=BLCSMultiViewAxialModel(
                hidden_dim=16,
                num_layers=1,
                num_heads=4,
                max_num_cameras=1,
                max_seq_len=4,
            )
        ),
    )

    module._validate_pipeline_checkpoint_profile()
