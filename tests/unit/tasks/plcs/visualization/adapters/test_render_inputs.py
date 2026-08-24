"""Physical-frame restoration tests for PLCS qualitative rendering."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    build_reference_frame_provenance,
    court_headings_physical_to_target,
    court_points_physical_to_target,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.model_io import PLCSDecodedPrediction
from src.tasks.plcs.visualization.adapters.render_inputs import (
    batch_to_pose_render_scenes,
)
from src.utils.schema.court_normalization import normalize_court_position


def _render_fixture():
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id="camera_1",
        camera_center_court_m=(2.0, 12.0, 5.0),
        contract=contract,
    )
    provenance = build_reference_frame_provenance(
        (view,),
        reference_camera_id=view.camera_id,
    )
    physical_position = torch.tensor([[[1.0, 2.0, 0.5]]])
    target_position_m = court_points_physical_to_target(
        physical_position,
        provenance,
    )
    target_position = normalize_court_position(target_position_m)
    physical_heading = torch.tensor([[[1.0, 0.0]]])
    target_heading = court_headings_physical_to_target(
        physical_heading,
        provenance,
    )
    physical_joints = physical_position[:, :, None].expand(1, 1, 17, 3).clone()
    physical_joints[..., 0] += 0.25
    target_joints = court_points_physical_to_target(physical_joints, provenance)
    batch = {
        "position": target_position,
        "rotation": target_heading,
        "human_kp_3d": target_joints,
        "court_keypoint_metadata": (court_keypoint_contract_document(contract),),
        "court_reference_provenance": (provenance,),
    }
    output = PLCSDecodedPrediction(
        position=target_position,
        rotation=target_heading,
    )
    return batch, output, physical_position, physical_heading


def test_render_adapter_restores_position_heading_and_world_joints() -> None:
    batch, output, physical_position, physical_heading = _render_fixture()

    ground_truth, prediction = batch_to_pose_render_scenes(batch, output)

    expected_position = normalize_court_position(physical_position)[0]
    torch.testing.assert_close(
        torch.from_numpy(ground_truth.position),
        expected_position,
    )
    torch.testing.assert_close(
        torch.from_numpy(prediction.position),
        expected_position,
    )
    torch.testing.assert_close(
        torch.from_numpy(ground_truth.rotation),
        physical_heading[0],
    )
    assert ground_truth.canonical_pose_3d is not None
    torch.testing.assert_close(
        torch.from_numpy(ground_truth.canonical_pose_3d[..., 0]),
        torch.full((1, 17), 0.25),
    )


def test_render_adapter_rejects_missing_provenance() -> None:
    batch, output, _, _ = _render_fixture()
    del batch["court_reference_provenance"]

    with pytest.raises(ValueError, match="provenance"):
        batch_to_pose_render_scenes(batch, output)
