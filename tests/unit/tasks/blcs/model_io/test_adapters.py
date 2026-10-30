"""Boundary tests for BLCS padding and fixed-query model calls."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.base.generate_dataset import (
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    ModelInputContractError,
    TrackQueryReferenceContract,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.model_io import AxialTrajectoryModelIOAdapter
from tests.support.tasks.blcs_physics import physics_sample


def _single_batch() -> dict[str, torch.Tensor]:
    return {
        "ball_uv": torch.zeros(2, 3, 2),
        "court_kp": torch.zeros(2, 14, 2),
        "ball_vis": torch.ones(2, 3, dtype=torch.bool),
        "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        "court_vis": torch.ones(2, 14, dtype=torch.bool),
    }


def _add_reference_fields(batch: dict[str, object]) -> None:
    batch.update(
        {
            "reference_view_index": torch.tensor([0], dtype=torch.int64),
            "view_camera_ids": torch.tensor([[0, 1]], dtype=torch.int64),
            "reference_camera_id": torch.tensor([0], dtype=torch.int64),
            "reference_from_physical": torch.eye(3).unsqueeze(0),
        }
    )
    write_track_query_reference_contract(
        batch,
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
    )


def _positive_side_provenance() -> CourtReferenceFrameProvenance:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id="camera_positive",
        camera_center_court_m=(2.0, 12.0, 5.0),
        contract=contract,
    )
    return build_reference_frame_provenance(
        (view,),
        reference_camera_id=view.camera_id,
    )


def test_multiview_adapter_builds_exact_five_tensor_all_padding_call() -> None:
    adapter = AxialTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=8,
        predict_velocity=False,
        input_profile="multiview",
        max_num_cameras=2,
    )
    batch = {
        "ball_uv": torch.zeros(1, 2, 3, 2),
        "ball_vis": torch.zeros(1, 2, 3, dtype=torch.bool),
        "padding_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 3, 14, 2),
        "court_vis": torch.zeros(1, 2, 3, 14, dtype=torch.bool),
    }

    call = adapter.build_call(batch)

    assert set(call.kwargs) == {
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    }
    assert call.kwargs["padding_mask"] is batch["padding_mask"]
    assert not adapter._loss_mask(batch).any()


def _multiview_scene_adapter() -> AxialTrajectoryModelIOAdapter:
    return AxialTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=8,
        predict_velocity=False,
        input_profile="multiview",
        max_num_cameras=4,
    )


def _scene_camera(frames: int, court_keypoints: int) -> dict[str, np.ndarray]:
    return {
        "ball_uv": np.full((frames, 2), 0.5, dtype=np.float32),
        "ball_vis": np.ones((frames,), dtype=bool),
        "court_kp_uv": np.full((court_keypoints, 2), 0.25, dtype=np.float32),
        "court_kp_vis": np.ones((court_keypoints,), dtype=bool),
    }


def test_scene_batch_slices_court_keypoints_to_the_model_contract() -> None:
    """A 20-point scene camera is truncated to the model's 14 court tokens."""
    scene = {"cameras": [_scene_camera(frames=4, court_keypoints=20) for _ in range(3)]}
    batch = _multiview_scene_adapter().build_inference_batch_from_scene(
        scene,
        [0, 1, 2],
    )
    assert batch["court_kp"].shape == (1, 3, 14, 2)
    assert batch["court_vis"].shape == (1, 3, 14)


def test_scene_batch_rejects_too_few_court_keypoints() -> None:
    scene = {"cameras": [_scene_camera(frames=4, court_keypoints=10)]}
    with pytest.raises(ModelInputContractError, match="court_kp_uv"):
        _multiview_scene_adapter().build_inference_batch_from_scene(scene, [0])


def test_scene_batch_rejects_too_few_court_visibility_entries() -> None:
    camera = _scene_camera(frames=4, court_keypoints=20)
    camera["court_kp_vis"] = np.ones((10,), dtype=bool)
    with pytest.raises(ModelInputContractError, match="court_kp_vis"):
        _multiview_scene_adapter().build_inference_batch_from_scene(
            {"cameras": [camera]},
            [0],
        )


def _training_batch() -> dict[str, torch.Tensor]:
    return {
        name: value.unsqueeze(0).expand(2, *value.shape).clone()
        for name, value in physics_sample(frames=3).items()
        if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize("name", ["camera_f", "camera_w", "camera_h"])
@pytest.mark.parametrize("invalid", [0.0, -1.0])
def test_training_boundary_rejects_nonpositive_active_camera_values(
    name: str, invalid: float
) -> None:
    batch = _training_batch()
    # This view has one real frame and no visible targets. It still needs a camera.
    batch["padding_mask"][1, 1, :-1] = True
    batch["ball_vis_target"][1, 1] = False
    batch[name][1, 1] = invalid
    with pytest.raises(ModelInputContractError, match=f"{name} must be positive"):
        _multiview_scene_adapter().build_training_batch(batch)


@pytest.mark.parametrize(
    "name",
    [
        "camera_R",
        "camera_C",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("padded", [False, True], ids=["active", "padded"])
def test_training_boundary_rejects_nonfinite_camera_values(
    name: str, invalid: float, padded: bool
) -> None:
    batch = _training_batch()
    batch["padding_mask"][1, 1] = padded
    batch["ball_vis_target"][1, 1] = False
    batch[name][1, 1] = invalid
    with pytest.raises(ModelInputContractError, match=f"{name}.*finite"):
        _multiview_scene_adapter().build_training_batch(batch)


def test_training_boundary_allows_explicit_finite_padded_camera_values() -> None:
    batch = _training_batch()
    batch["padding_mask"][1, 1] = True
    batch["ball_vis_target"][1, 1] = False
    batch["camera_f"][1, 1] = 0.0
    batch["camera_w"][1, 1] = 1.0
    batch["camera_h"][1, 1] = 1.0

    prepared = _multiview_scene_adapter().build_training_batch(batch)

    assert prepared.camera_f[1, 1] == 0.0
    assert prepared.camera_w[1, 1] == 1.0
    assert prepared.camera_h[1, 1] == 1.0
    assert prepared.loss_mask.all()


def test_training_boundary_rejects_target_views_detached_from_model_views() -> None:
    batch = _training_batch()
    for name in batch:
        if name.startswith("camera_") or name in {"ball_uv_target", "ball_vis_target"}:
            batch[name] = batch[name][:, :1]
    with pytest.raises(ModelInputContractError, match="camera axis.*model call"):
        _multiview_scene_adapter().build_training_batch(batch)
