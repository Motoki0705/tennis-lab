"""Strict PLCS reference metadata contract tests."""

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
from src.tasks.plcs.model_io import (
    PLCSReferenceMetadata,
    plcs_reference_metadata_from_batch,
)


def _reference_metadata() -> PLCSReferenceMetadata:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_left",
            camera_center_court_m=(0.0, -10.0, 3.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_right",
            camera_center_court_m=(0.0, 10.0, 3.0),
            contract=contract,
        ),
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        tuple(view.camera_id for view in views)
    )
    selections = (
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=views,
            reference_camera_id="camera_right",
        ),
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=(views[1],),
            reference_camera_id="camera_right",
        ),
    )
    forward = torch.tensor(
        [selection.provenance.reference_from_physical for selection in selections],
        dtype=torch.float32,
    )
    return PLCSReferenceMetadata(
        selections=selections,
        stable_camera_id_tables=(table, table),
        reference_view_index=torch.tensor([1, 0], dtype=torch.int64),
        view_camera_ids=torch.tensor([[0, 1], [1, -1]], dtype=torch.int64),
        reference_camera_id=torch.tensor([1, 1], dtype=torch.int64),
        reference_from_physical=forward,
        physical_from_reference=forward.transpose(-1, -2),
        track_query_contract=TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode.REFERENCE
        ),
    )


def test_reference_metadata_round_trips_strict_batch_and_prediction_fields() -> None:
    expected = _reference_metadata()

    actual = plcs_reference_metadata_from_batch(expected.to_batch_fields())

    assert actual == expected
    assert actual.reference_camera_ids == ("camera_right", "camera_right")
    assert actual.selected_camera_ids == (
        ("camera_left", "camera_right"),
        ("camera_right",),
    )
    payload = actual.prediction_payload(max_views=3)
    assert np.asarray(payload["reference_view_index"]).tolist() == [1, 0]
    assert np.asarray(payload["view_camera_ids"]).tolist() == [
        [0, 1, -1],
        [1, -1, -1],
    ]
    np.testing.assert_array_equal(
        payload["reference_camera_id_string"],
        np.asarray(["camera_right", "camera_right"]),
    )
    np.testing.assert_array_equal(
        payload["view_camera_id_strings"],
        np.asarray(
            [
                ["camera_left", "camera_right", ""],
                ["camera_right", "", ""],
            ]
        ),
    )
    assert np.asarray(payload["reference_selector_mode"]).tolist() == [
        "reference",
        "reference",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda batch: batch.pop("physical_from_reference"), "missing/mixed"),
        (
            lambda batch: batch.__setitem__(
                "reference_camera_id",
                torch.tensor([0, 1], dtype=torch.int64),
            ),
            "exactly equal",
        ),
    ],
)
def test_reference_metadata_parser_rejects_missing_or_inconsistent_fields(
    mutation: object,
    message: str,
) -> None:
    batch = _reference_metadata().to_batch_fields()
    assert callable(mutation)
    mutation(batch)

    with pytest.raises(ValueError, match=message):
        plcs_reference_metadata_from_batch(batch)


def test_non_track_query_reference_metadata_does_not_infer_a_rope_contract() -> None:
    batch = _reference_metadata().to_batch_fields()
    batch.pop("track_query_reference")

    metadata = plcs_reference_metadata_from_batch(batch)

    assert metadata is not None
    assert metadata.track_query_contract is None
    assert metadata.reference_camera_ids == ("camera_right", "camera_right")


def test_reference_metadata_rejects_transform_not_owned_by_selection() -> None:
    metadata = _reference_metadata()
    wrong = metadata.reference_from_physical.clone()
    wrong[0] = torch.eye(3)

    with pytest.raises(ValueError, match="transform mismatch"):
        PLCSReferenceMetadata(
            selections=metadata.selections,
            stable_camera_id_tables=metadata.stable_camera_id_tables,
            reference_view_index=metadata.reference_view_index,
            view_camera_ids=metadata.view_camera_ids,
            reference_camera_id=metadata.reference_camera_id,
            reference_from_physical=wrong,
            physical_from_reference=wrong.transpose(-1, -2),
            track_query_contract=metadata.track_query_contract,
        )
