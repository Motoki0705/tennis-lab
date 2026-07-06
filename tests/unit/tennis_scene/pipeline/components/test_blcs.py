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
        del ball_vis, court_vis, denormalize
        assert ball_uv.shape == (1, 1, 2, 2)
        assert court_kp.shape == (1, 1, 2, 20, 2)
        assert ball_mask is not None
        assert ball_mask.shape == (1, 1, 2)
        return {"position": torch.ones(1, 2, 3)}


def test_process_runs_only_contiguous_valid_clip_without_zero_filling() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = cast(Any, _FakeBLCSPredictor())
    result = module.process(
        ball_uv=np.zeros((1, 4, 2), dtype=np.float32),
        court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
        ball_vis=np.array([[False, True, True, False]], dtype=np.bool_),
        court_vis=np.ones((1, 4, 20), dtype=np.float32),
    )

    assert result.ball_3d.shape == (4, 3)
    assert result.visibility is not None
    np.testing.assert_array_equal(
        result.visibility,
        np.array([False, True, True, False], dtype=np.bool_),
    )
    np.testing.assert_array_equal(result.ball_3d[1:3], np.ones((2, 3), dtype=np.float32))
    assert np.isnan(result.ball_3d[0]).all()
    assert np.isnan(result.ball_3d[3]).all()


def test_process_rejects_noncontiguous_visibility_clip() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = cast(Any, _FakeBLCSPredictor())

    with pytest.raises(ValueError, match="one contiguous trajectory clip"):
        module.process(
            ball_uv=np.zeros((1, 4, 2), dtype=np.float32),
            court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
            ball_vis=np.array([[True, False, True, True]], dtype=np.bool_),
            court_vis=np.ones((1, 4, 20), dtype=np.float32),
        )


def test_result_serializes_invalid_frames_as_json_null() -> None:
    result = BLCSResult(
        ball_3d=np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        visibility=np.array([True, False], dtype=np.bool_),
    )

    data = result.to_dict()

    assert data["ball_3d"][1] == [None, None, None]
    loaded = BLCSResult.from_dict(data)
    assert np.isnan(loaded.ball_3d[1]).all()


def test_validate_pipeline_checkpoint_rejects_single_model() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = cast(
        Any,
        SimpleNamespace(model=BLCSModel(hidden_dim=16, num_layers=1, num_heads=4)),
    )

    with pytest.raises(ValueError, match="requires a multiview BLCS checkpoint"):
        module._validate_pipeline_checkpoint_profile()


def test_validate_pipeline_checkpoint_accepts_multiview_model() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
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
