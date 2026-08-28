"""Boundary tests for BLCS padding and fixed-query model calls."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.configuration import UnknownConfigurationKeyError
from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import ModelInputContractError
from src.tasks.blcs.configuration import parse_model_config
from src.tasks.blcs.model_io import (
    MultiViewTrajectoryModelIOAdapter,
    SingleTrajectoryModelIOAdapter,
    TrackQueryModelIOAdapter,
)


def _single_adapter() -> SingleTrajectoryModelIOAdapter:
    return SingleTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=8,
        predict_velocity=False,
        input_profile="single",
        max_num_cameras=None,
    )


def _single_batch() -> dict[str, torch.Tensor]:
    return {
        "ball_uv": torch.zeros(2, 3, 2),
        "court_kp": torch.zeros(2, 14, 2),
        "ball_vis": torch.ones(2, 3, dtype=torch.bool),
        "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        "court_vis": torch.ones(2, 14, dtype=torch.bool),
    }


def _tracking_adapter() -> TrackQueryModelIOAdapter:
    return TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
    )


def _tracking_batch() -> dict[str, torch.Tensor]:
    return {
        "ball_uv": torch.zeros(1, 2, 3, 2, 2),
        "ball_vis": torch.zeros(1, 2, 3, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "padding_mask": torch.tensor([[[False, False, True], [False, True, True]]]),
    }


def _tracking_training_batch() -> dict[str, object]:
    batch: dict[str, object] = dict(_tracking_batch())
    batch.update(
        {
            "target_position": torch.zeros(1, 3, 2, 3),
            "target_velocity": torch.zeros(1, 3, 2, 3),
            "target_presence": torch.zeros(1, 3, 2, dtype=torch.bool),
            "target_instance_id": torch.full((1, 3, 2), -1),
            "target_slot_mask": torch.zeros(1, 2, dtype=torch.bool),
        }
    )
    return batch


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


def test_single_adapter_builds_exact_five_tensor_padding_call() -> None:
    batch = _single_batch()
    batch["padding_mask"][:, -1] = True

    call = _single_adapter().build_call(batch)
    assert set(call.kwargs) == {
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    }
    assert call.kwargs["padding_mask"] is batch["padding_mask"]
    assert _single_adapter()._loss_mask(batch).tolist() == [
        [True, True, False],
        [True, True, False],
    ]


def test_standard_adapters_reject_removed_ball_mask_key() -> None:
    batch = _single_batch()
    batch["ball_mask"] = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ModelInputContractError, match="ball_mask"):
        _single_adapter().build_call(batch)


def test_multiview_adapter_builds_exact_five_tensor_all_padding_call() -> None:
    adapter = MultiViewTrajectoryModelIOAdapter(
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


def test_tracking_adapter_builds_exact_five_tensor_call() -> None:
    call = _tracking_adapter().build_call(_tracking_batch())
    assert set(call.kwargs) == {
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    }


@pytest.mark.parametrize("width", [1, 3])
def test_tracking_adapter_requires_exact_q(width: int) -> None:
    batch = _tracking_batch()
    batch["ball_uv"] = torch.zeros(1, 2, 3, width, 2)
    batch["ball_vis"] = torch.zeros(1, 2, 3, width, dtype=torch.bool)
    with pytest.raises(ModelInputContractError, match="model.num_queries"):
        _tracking_adapter().build_call(batch)


@pytest.mark.parametrize(
    "removed_key", ["ball_visible", "candidate_mask", "frame_mask", "view_mask"]
)
def test_tracking_adapter_rejects_removed_input_keys(removed_key: str) -> None:
    batch = _tracking_batch()
    batch[removed_key] = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(ModelInputContractError, match=removed_key):
        _tracking_adapter().build_call(batch)


def test_tracking_training_batch_derives_frame_valid_from_padding() -> None:
    batch = _tracking_training_batch()

    prepared = _tracking_adapter().build_training_batch(batch)

    assert prepared.frame_valid.tolist() == [[True, True, False]]
    assert prepared.court_reference_provenance == (
        build_physical_court_provenance(),
    )


@pytest.mark.parametrize("missing_value", [None, ()])
def test_v2_tracking_adapter_rejects_missing_provenance(
    missing_value: object,
) -> None:
    batch = _tracking_training_batch()
    if missing_value is not None:
        batch["court_reference_provenance"] = missing_value
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
    )

    with pytest.raises(MissingCourtKeypointMetadataError):
        adapter.build_training_batch(batch)


def test_v2_tracking_adapter_rejects_cardinality_and_contract_mismatch() -> None:
    batch = _tracking_training_batch()
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
    )
    provenance = _positive_side_provenance()
    batch["court_reference_provenance"] = (provenance, provenance)
    with pytest.raises(ValueError, match="one record per batch item"):
        adapter.build_training_batch(batch)

    batch["court_reference_provenance"] = (build_physical_court_provenance(),)
    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        adapter.build_training_batch(batch)


@pytest.mark.parametrize(
    "removed_key",
    ["mask_invisible_observations", "observation_fusion", "point_fusion"],
)
def test_track_query_config_rejects_removed_fusion_keys(removed_key: str) -> None:
    model: dict[str, object] = {
        "name": "blcs_track_query",
        "hidden_dim": 16,
        "num_heads": 4,
        "num_stages": 4,
        "ffn_dim": 32,
        "num_queries": 2,
        "rope_dim": 4,
        "dropout": 0.0,
        "role_rope_enabled": True,
        "invisible_init_std": 0.02,
        "mhc": {
            "coefficient_dim": 8,
            "sinkhorn_iters": 5,
            "eps": 1e-6,
            "residual_identity_bias": 4.0,
            "update_scale_init": 0.0,
        },
        "cswa": {
            "compression_ratio": 2,
            "window_radius": 1,
            "backend": "reference",
        },
    }
    model[removed_key] = True if removed_key == "mask_invisible_observations" else {}
    with pytest.raises(UnknownConfigurationKeyError, match=removed_key):
        parse_model_config({"model": model})
