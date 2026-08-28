"""Typed BLCS reference metadata boundary tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import TrackQueryReferenceContract
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.model_io import (
    BLCSReferenceMetadata,
    BLCSTrackQueryPrediction,
    BLCSTrajectoryPrediction,
    blcs_reference_metadata_from_batch,
    blcs_track_query_prediction_to_physical,
    blcs_trajectory_prediction_to_physical,
)


def _metadata() -> BLCSReferenceMetadata:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = {
        "camera_a": build_court_view_record(
            camera_id="camera_a",
            camera_center_court_m=(0.0, -10.0, 3.0),
            contract=contract,
        ),
        "camera_b": build_court_view_record(
            camera_id="camera_b",
            camera_center_court_m=(0.0, 10.0, 3.0),
            contract=contract,
        ),
        "camera_c": build_court_view_record(
            camera_id="camera_c",
            camera_center_court_m=(2.0, -11.0, 4.0),
            contract=contract,
        ),
    }
    table = StableCameraIdTable.from_complete_scene_camera_ids(tuple(views))
    selections = (
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=(views["camera_c"], views["camera_a"]),
            reference_camera_id="camera_a",
        ),
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=(
                views["camera_b"],
                views["camera_c"],
                views["camera_a"],
            ),
            reference_camera_id="camera_b",
        ),
    )
    forward = torch.tensor(
        [selection.provenance.reference_from_physical for selection in selections],
        dtype=torch.float32,
    )
    return BLCSReferenceMetadata(
        selections=selections,
        stable_camera_id_tables=(table, table),
        reference_view_index=torch.tensor([1, 0], dtype=torch.int64),
        view_camera_ids=torch.tensor([[2, 0, -1], [1, 2, 0]], dtype=torch.int64),
        reference_camera_id=torch.tensor([0, 1], dtype=torch.int64),
        reference_from_physical=forward,
        physical_from_reference=forward.transpose(-1, -2),
        track_query_contract=TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode.REFERENCE
        ),
    )


def test_reference_metadata_round_trips_complete_table_codes_and_fixed_width_payload() -> None:
    metadata = _metadata()

    parsed = blcs_reference_metadata_from_batch(metadata.to_batch_fields())

    assert parsed is not None
    assert parsed.reference_camera_ids == ("camera_a", "camera_b")
    assert parsed.selected_camera_ids == (
        ("camera_c", "camera_a"),
        ("camera_b", "camera_c", "camera_a"),
    )
    payload = parsed.prediction_payload(max_views=4)
    assert np.asarray(payload["view_camera_ids"]).tolist() == [
        [2, 0, -1, -1],
        [1, 2, 0, -1],
    ]
    assert np.asarray(payload["view_camera_id_strings"]).tolist() == [
        ["camera_c", "camera_a", "", ""],
        ["camera_b", "camera_c", "camera_a", ""],
    ]
    assert np.asarray(payload["reference_camera_id_string"]).tolist() == [
        "camera_a",
        "camera_b",
    ]
    assert np.asarray(payload["target_frame_contract"]).tolist() == [
        "reference_camera_court_rzpi_v1",
        "reference_camera_court_rzpi_v1",
    ]


def test_reference_metadata_rejects_mixed_schema_and_non_trailing_padding() -> None:
    metadata = _metadata()
    missing = metadata.to_batch_fields()
    del missing["physical_from_reference"]
    with pytest.raises(ValueError, match="missing/mixed"):
        blcs_reference_metadata_from_batch(missing)

    with pytest.raises(ValueError, match="padding must be trailing"):
        BLCSReferenceMetadata(
            selections=metadata.selections,
            stable_camera_id_tables=metadata.stable_camera_id_tables,
            reference_view_index=metadata.reference_view_index,
            view_camera_ids=torch.tensor([[2, -1, 0], [1, 2, 0]]),
            reference_camera_id=metadata.reference_camera_id,
            reference_from_physical=metadata.reference_from_physical,
            physical_from_reference=metadata.physical_from_reference,
            track_query_contract=metadata.track_query_contract,
        )


def test_physical_conversion_preserves_reference_metadata() -> None:
    metadata = _metadata()
    provenances = tuple(selection.provenance for selection in metadata.selections)
    trajectory = BLCSTrajectoryPrediction(
        position=torch.ones(2, 1, 3),
        velocity=torch.ones(2, 1, 3),
        court_reference_provenance=provenances,
        coordinates_in_metres=True,
        reference_metadata=metadata,
    )
    logits = torch.ones(2, 1, 1)
    tracking = BLCSTrackQueryPrediction(
        position=torch.ones(2, 1, 1, 3),
        presence_logits=logits,
        presence_probability=logits.sigmoid(),
        presence=torch.ones(2, 1, 1, dtype=torch.bool),
        court_reference_provenance=provenances,
        coordinates_in_metres=True,
        reference_metadata=metadata,
    )

    assert blcs_trajectory_prediction_to_physical(trajectory).reference_metadata is metadata
    assert blcs_track_query_prediction_to_physical(tracking).reference_metadata is metadata
