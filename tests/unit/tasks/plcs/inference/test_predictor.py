"""Tests for PLCS inference predictor output conversions."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.data import (
    CourtCoordinateContractMismatchError,
    MissingCourtCoordinateMetadataError,
)
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io import (
    PLCSInputProfile,
    PLCSModelIOAdapter,
    prepare_plcs_checkpoint_config,
    write_plcs_checkpoint_normalization,
)
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


class _FixedRotationModel(nn.Module):
    def __init__(self, rotation: Tensor, *, position_value: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("rotation", rotation)
        self.position_value = position_value

    def forward(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
    ) -> dict[str, Tensor]:
        del human_kp, court_kp, human_vis, padding_mask, court_vis
        rotation = cast(Tensor, self.rotation)
        return {
            "position": torch.full(
                (*rotation.shape[:-1], 3),
                self.position_value,
                device=rotation.device,
                dtype=rotation.dtype,
            ),
            "rotation": rotation,
        }


def test_yaw_radians_round_trips_dataset_cos_sin_encoding() -> None:
    angles = torch.tensor(
        [[0.0, math.pi / 6.0, -math.pi / 2.0], [2.3, -1.2, 3.0]],
        dtype=torch.float32,
    )
    dataset_encoded_rotation = torch.stack(
        [torch.cos(angles), torch.sin(angles)],
        dim=-1,
    )
    predictor = PLCSPredictor(
        model=_FixedRotationModel(dataset_encoded_rotation),
        adapter=PLCSModelIOAdapter(
            model_type=_FixedRotationModel,
            profile=PLCSInputProfile.MULTIVIEW,
            num_court_tokens=20,
            camera_index=0,
            output_rank=3,
            predict_canonical_pose=False,
            predict_auxiliary_position=False,
        ),
        device=torch.device("cpu"),
    )

    human_shape = (2, 1, 3)
    result = predictor.predict(
        human_kp=torch.zeros(*human_shape, 17, 2),
        court_kp=torch.zeros(*human_shape, 20, 2),
        human_vis=torch.ones(*human_shape, 17, dtype=torch.bool),
        padding_mask=torch.zeros(*human_shape, dtype=torch.bool),
        court_vis=torch.ones(*human_shape, 20, dtype=torch.bool),
        denormalize=True,
    )

    assert torch.allclose(result["rotation"], dataset_encoded_rotation)
    assert torch.allclose(result["yaw_radians"], angles, atol=1e-6)

    physical = predictor.predict_multiview_observations(
        human_kp=np.zeros((*human_shape, 17, 2), dtype=np.float32),
        court_kp=np.zeros((1, 3, 20, 2), dtype=np.float32),
        human_vis=np.ones((*human_shape, 17), dtype=np.bool_),
        padding_mask=np.zeros(human_shape, dtype=np.bool_),
        court_vis=np.ones((1, 3, 20), dtype=np.bool_),
    )
    np.testing.assert_allclose(physical.yaw_radians, angles.numpy(), atol=1e-6)
    assert physical.position_meters.shape == (2, 3, 3)
    assert physical.position_meters.dtype == np.float32


def test_v2_predictor_returns_meter_translation_without_rescaling_canonical_pose() -> None:
    rotation = torch.tensor([[[1.0, 0.0]]])
    contract = resolve_court_coordinate_normalization("v2")
    predictor = PLCSPredictor(
        model=_FixedRotationModel(rotation, position_value=1.0),
        adapter=PLCSModelIOAdapter(
            model_type=_FixedRotationModel,
            profile=PLCSInputProfile.MULTIVIEW,
            num_court_tokens=20,
            camera_index=0,
            output_rank=3,
            predict_canonical_pose=False,
            predict_auxiliary_position=False,
        ),
        device=torch.device("cpu"),
        court_coordinate_normalization=contract,
    )
    shape = (1, 1, 1)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        court_kp=torch.zeros(*shape, 20, 2),
        human_vis=torch.ones(*shape, 17, dtype=torch.bool),
        padding_mask=torch.zeros(*shape, dtype=torch.bool),
        court_vis=torch.ones(*shape, 20, dtype=torch.bool),
        denormalize=True,
    )

    torch.testing.assert_close(
        result["position_meters"],
        torch.tensor(contract.scale_xyz).view(1, 1, 3),
    )


def test_plcs_checkpoint_legacy_and_mismatch_matrix() -> None:
    legacy = {"hyper_parameters": {"config": {"model": {"name": "plcs"}}}}
    with pytest.raises(MissingCourtCoordinateMetadataError, match="metadata is absent"):
        prepare_plcs_checkpoint_config(legacy, None)
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        prepare_plcs_checkpoint_config(
            legacy,
            resolve_court_coordinate_normalization("v2"),
        )

    config, contract = prepare_plcs_checkpoint_config(
        legacy,
        resolve_court_coordinate_normalization("v1"),
    )
    assert contract.version == "v1"
    assert config.court_coordinate_normalization.version == "v1"

    versioned: dict[str, object] = {
        "hyper_parameters": {
            "config": {"court_coordinate_normalization": {"version": "v1"}}
        }
    }
    write_plcs_checkpoint_normalization(
        versioned,
        resolve_court_coordinate_normalization("v2"),
    )
    with pytest.raises(CourtCoordinateContractMismatchError, match="saved config"):
        prepare_plcs_checkpoint_config(versioned, None)
