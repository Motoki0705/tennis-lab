"""Tests for typed standard BLCS inference."""

from __future__ import annotations

from typing import TypedDict, cast

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.generate_dataset import (
    MissingCourtKeypointMetadataError,
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    bind_model_io,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io import (
    SingleTrajectoryModelIOAdapter,
    TrajectoryBoundModelIO,
    blcs_trajectory_prediction_to_physical,
)
from src.tasks.blcs.models import BLCSModel


class _BLCSPredictInputs(TypedDict):
    ball_uv: Tensor
    court_kp: Tensor
    ball_vis: Tensor
    padding_mask: Tensor
    court_vis: Tensor
    denormalize: bool


class _FixedTrajectoryModel(BLCSModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        del court_kp, ball_vis, court_vis, padding_mask
        shape = (ball_uv.shape[0], ball_uv.shape[1], 3)
        return {
            "position": torch.ones(shape, device=ball_uv.device),
            "velocity": torch.full(shape, 2.0, device=ball_uv.device),
        }


def test_predict_returns_typed_cpu_trajectory_decode() -> None:
    binding = cast(
        "TrajectoryBoundModelIO",
        bind_model_io(
            _FixedTrajectoryModel(),
            SingleTrajectoryModelIOAdapter(
                num_court_tokens=14,
                max_seq_len=8,
                predict_velocity=True,
                input_profile="single",
                max_num_cameras=None,
            ),
        ),
    )
    predictor = BLCSPredictor(model_io=binding, device=torch.device("cpu"))

    prediction = predictor.predict(
        ball_uv=torch.zeros(1, 3, 2),
        court_kp=torch.zeros(1, 14, 2),
        ball_vis=torch.ones(1, 3, dtype=torch.bool),
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        court_vis=torch.ones(1, 14, dtype=torch.bool),
        denormalize=False,
    )

    assert prediction.position.shape == (1, 3, 3)
    assert prediction.velocity is not None
    torch.testing.assert_close(prediction.velocity, torch.full((1, 3, 3), 2.0))
    assert prediction.position.device.type == "cpu"
    assert prediction.velocity.device.type == "cpu"

    physical = predictor.predict(
        ball_uv=torch.zeros(1, 3, 2),
        court_kp=torch.zeros(1, 14, 2),
        ball_vis=torch.ones(1, 3, dtype=torch.bool),
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        court_vis=torch.ones(1, 14, dtype=torch.bool),
        denormalize=True,
    )
    torch.testing.assert_close(physical.position, torch.full((1, 3, 3), 11.885))
    assert physical.velocity is not None
    torch.testing.assert_close(physical.velocity, torch.full((1, 3, 3), 23.77))


def test_camera_view_direct_input_requires_marker_and_reversible_provenance() -> None:
    binding = cast(
        "TrajectoryBoundModelIO",
        bind_model_io(
            _FixedTrajectoryModel(),
            SingleTrajectoryModelIOAdapter(
                num_court_tokens=14,
                max_seq_len=8,
                predict_velocity=True,
                input_profile="single",
                max_num_cameras=None,
            ),
        ),
    )
    contract = resolve_court_keypoint_contract("camera_view_v2")
    predictor = BLCSPredictor(
        model_io=binding,
        device=torch.device("cpu"),
        court_keypoint_contract=contract,
    )
    kwargs: _BLCSPredictInputs = {
        "ball_uv": torch.zeros(1, 3, 2),
        "court_kp": torch.zeros(1, 14, 2),
        "ball_vis": torch.ones(1, 3, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 3, dtype=torch.bool),
        "court_vis": torch.ones(1, 14, dtype=torch.bool),
        "denormalize": True,
    }
    with pytest.raises(MissingCourtKeypointMetadataError):
        predictor.predict(**kwargs)

    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(document, contract)
    view = build_court_view_record(
        camera_id="cam_positive",
        camera_center_court_m=(1.0, 12.0, 5.0),
        contract=contract,
    )
    provenance = build_reference_frame_provenance(
        (view,),
        reference_camera_id=view.camera_id,
    )
    prediction = predictor.predict(
        **kwargs,
        court_keypoint_document=document,
        court_reference_provenance=(provenance,),
    )
    physical = blcs_trajectory_prediction_to_physical(prediction)
    torch.testing.assert_close(
        physical.position,
        torch.tensor([-11.885, -11.885, 11.885]).expand(1, 3, 3),
    )
