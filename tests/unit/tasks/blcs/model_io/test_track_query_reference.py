"""BLCS six-input adapter and checkpoint semantic contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    TRACK_QUERY_REFERENCE_METADATA_KEY,
    MissingTrackQueryReferenceMetadataError,
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMismatchError,
    write_checkpoint_track_query_reference_contract,
    write_model_artifact_court_keypoint_contract,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.model_io.adapters import (
    TrackQueryModelIOAdapter,
    TrackQueryReferenceModelIOAdapter,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.utils.schema.court_normalization import add_court_coordinate_normalization


def _reference_adapter() -> TrackQueryReferenceModelIOAdapter:
    return TrackQueryReferenceModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        track_query_reference_contract=TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode.REFERENCE
        ),
    )


def _batch() -> dict[str, object]:
    result: dict[str, object] = {
        "ball_uv": torch.zeros(2, 3, 2, 2, 2),
        "ball_vis": torch.zeros(2, 3, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(2, 3, 2, 14, 2),
        "court_vis": torch.zeros(2, 3, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(2, 3, 2, dtype=torch.bool),
        "reference_view_index": torch.tensor([1, 2], dtype=torch.int64),
        "view_camera_ids": torch.tensor([[10, 11, 12], [20, 21, 22]]),
        "reference_camera_id": torch.tensor([11, 22]),
        "reference_from_physical": torch.eye(3).expand(2, -1, -1).clone(),
    }
    write_track_query_reference_contract(
        result,
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
    )
    return result


def test_reference_adapter_builds_exact_six_tensor_call() -> None:
    batch = _batch()
    call = _reference_adapter().build_call(batch)
    assert set(call.kwargs) == {
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
    }
    assert call.kwargs["reference_view_index"] is batch["reference_view_index"]


@pytest.mark.parametrize(
    "reference_view_index",
    [
        None,
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([[1], [2]], dtype=torch.int64),
        torch.tensor([1, 3], dtype=torch.int64),
    ],
)
def test_reference_adapter_rejects_missing_or_invalid_sixth_tensor(
    reference_view_index: torch.Tensor | None,
) -> None:
    batch = _batch()
    if reference_view_index is None:
        del batch["reference_view_index"]
    else:
        batch["reference_view_index"] = reference_view_index
    with pytest.raises(ValueError):
        _reference_adapter().build_call(batch)


def test_reference_adapter_rejects_missing_metadata_identity_mismatch_and_masked_reference() -> (
    None
):
    adapter = _reference_adapter()
    missing_metadata = _batch()
    del missing_metadata[TRACK_QUERY_REFERENCE_METADATA_KEY]
    with pytest.raises(ValueError, match="metadata is absent"):
        adapter.build_call(missing_metadata)

    mismatched_identity = _batch()
    mismatched_identity["reference_camera_id"] = torch.tensor([10, 22])
    with pytest.raises(ValueError, match="must exactly equal"):
        adapter.build_call(mismatched_identity)

    masked_reference = _batch()
    padding = masked_reference["padding_mask"]
    assert isinstance(padding, torch.Tensor)
    padding[0, 1, 0] = True
    with pytest.raises(ValueError, match="unmasked reference-view"):
        adapter.build_call(masked_reference)


def test_legacy_adapter_retains_five_inputs_and_rejects_camera_view_semantics() -> None:
    legacy_batch = _batch()
    del legacy_batch["reference_view_index"]
    legacy = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
    )
    assert len(legacy.build_call(legacy_batch).kwargs) == 5
    with pytest.raises(CourtKeypointContractMismatchError):
        TrackQueryModelIOAdapter(
            num_court_tokens=14,
            num_queries=2,
            presence_threshold=0.5,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


def _write_checkpoint(
    path: Path,
    *,
    config_selector_mode: str = "reference",
    metadata_selector_mode: ReferenceSelectorMode | None = (
        ReferenceSelectorMode.REFERENCE
    ),
) -> None:
    config = {
        "court_keypoints": {"selector": "camera_view_v2"},
        "model": {
            "name": "blcs_track_query_reference",
            "target_frame_contract": "reference_camera_court_rzpi_v1",
            "track_query_rope_contract": "time_camera_reference_selector_v1",
            "reference_selector_mode": config_selector_mode,
        },
    }
    checkpoint: dict[str, object] = {"hyper_parameters": {"config": config}}
    add_court_coordinate_normalization(
        checkpoint,
        artifact="BLCS reference test checkpoint",
    )
    write_model_artifact_court_keypoint_contract(
        checkpoint,
        resolve_court_keypoint_contract("camera_view_v2"),
    )
    if metadata_selector_mode is not None:
        write_checkpoint_track_query_reference_contract(
            checkpoint,
            TrackQueryReferenceContract.reference_v2(metadata_selector_mode),
        )
    torch.save(checkpoint, path)


def test_v2_checkpoint_requires_and_exactly_matches_independent_markers(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.ckpt"
    _write_checkpoint(valid)
    runtime = load_checkpoint_runtime(valid)
    assert runtime.track_query_reference_contract == (
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE)
    )

    missing = tmp_path / "missing.ckpt"
    _write_checkpoint(missing, metadata_selector_mode=None)
    with pytest.raises(MissingTrackQueryReferenceMetadataError):
        load_checkpoint_runtime(missing)


def test_metadata_free_track_query_semantics_are_legacy_v1_only(
    tmp_path: Path,
) -> None:
    checkpoint: dict[str, object] = {
        "hyper_parameters": {
            "config": {
                "court_keypoints": {"selector": "physical_v1"},
                "model": {"name": "blcs_track_query"},
            }
        }
    }
    add_court_coordinate_normalization(
        checkpoint,
        artifact="BLCS legacy test checkpoint",
    )
    write_model_artifact_court_keypoint_contract(
        checkpoint,
        resolve_court_keypoint_contract("physical_v1"),
    )
    path = tmp_path / "legacy_v1.ckpt"
    torch.save(checkpoint, path)
    assert load_checkpoint_runtime(path).track_query_reference_contract == (
        TrackQueryReferenceContract.legacy_v1()
    )

    mixed: dict[str, object] = {
        "hyper_parameters": {
            "config": {
                "court_keypoints": {"selector": "camera_view_v2"},
                "model": {"name": "blcs_track_query"},
            }
        }
    }
    add_court_coordinate_normalization(
        mixed,
        artifact="BLCS mixed-contract test checkpoint",
    )
    write_model_artifact_court_keypoint_contract(
        mixed,
        resolve_court_keypoint_contract("camera_view_v2"),
    )
    mixed_path = tmp_path / "mixed_v1.ckpt"
    torch.save(mixed, mixed_path)
    with pytest.raises(TrackQueryReferenceContractMismatchError):
        load_checkpoint_runtime(mixed_path)
