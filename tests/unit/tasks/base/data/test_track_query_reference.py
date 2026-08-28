"""Stable camera ID and reference-view data contract tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.tasks.base.data.track_query_reference import (
    CAMERA_ID_PADDING_VALUE,
    ReferenceViewBatchError,
    ReferenceViewSelection,
    ReferenceViewSelectionError,
    StableCameraIdTable,
    StableCameraIdTableError,
    include_evaluation_reference_camera,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
    validate_reference_view_batch,
)
from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    CameraCourtViewError,
    build_court_view_record,
    resolve_court_keypoint_contract,
)


def _views() -> tuple[object, ...]:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    return (
        build_court_view_record(
            camera_id="camera_10",
            camera_center_court_m=(1.0, -4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_2",
            camera_center_court_m=(-1.0, 5.0, 2.0),
            contract=contract,
        ),
    )


def test_complete_scene_table_is_collision_free_and_not_subset_ranked() -> None:
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        ("z_unused", "camera_2", "camera_10")
    )

    assert table.camera_ids == ("camera_10", "camera_2", "z_unused")
    assert table.encode_many(("camera_2", "z_unused")) == (1, 2)
    assert table.decode(1) == "camera_2"
    assert StableCameraIdTable.from_mapping(table.to_dict()) == table
    assert CAMERA_ID_PADDING_VALUE == -1


@pytest.mark.parametrize(
    "camera_ids",
    [(), ("",), ("camera_0", "camera_0")],
)
def test_complete_scene_table_rejects_invalid_canonical_ids(
    camera_ids: tuple[str, ...],
) -> None:
    with pytest.raises(StableCameraIdTableError):
        StableCameraIdTable.from_complete_scene_camera_ids(camera_ids)


def test_table_rejects_unsorted_direct_construction_unknown_and_padding() -> None:
    with pytest.raises(StableCameraIdTableError, match="lexicographic"):
        StableCameraIdTable(("camera_2", "camera_10"))
    table = StableCameraIdTable.from_complete_scene_camera_ids(("a", "b"))
    with pytest.raises(StableCameraIdTableError, match="absent"):
        table.encode("c")
    with pytest.raises(StableCameraIdTableError, match="padding only"):
        table.decode(-1)


def test_table_mapping_is_exact_and_does_not_infer_order() -> None:
    document = StableCameraIdTable.from_complete_scene_camera_ids(("b", "a")).to_dict()
    document["extra"] = True
    with pytest.raises(StableCameraIdTableError, match="exactly"):
        StableCameraIdTable.from_mapping(document)

    unsorted = deepcopy(document)
    del unsorted["extra"]
    unsorted["camera_ids"] = ["b", "a"]
    with pytest.raises(StableCameraIdTableError, match="lexicographic"):
        StableCameraIdTable.from_mapping(unsorted)

    invalid_version = deepcopy(unsorted)
    invalid_version["camera_ids"] = ["a", "b"]
    invalid_version["schema_version"] = True
    with pytest.raises(StableCameraIdTableError, match="schema_version"):
        StableCameraIdTable.from_mapping(invalid_version)


def test_selection_resolves_reference_after_reorder_without_changing_identity() -> None:
    raw_views = _views()
    first, second = raw_views
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        ("camera_2", "unused", "camera_10")
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=(first, second),  # type: ignore[arg-type]
        reference_camera_id="camera_2",
    )
    reordered = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=(second, first),  # type: ignore[arg-type]
        reference_camera_id="camera_2",
    )

    assert selection.reference_camera_id == reordered.reference_camera_id == "camera_2"
    assert selection.reference_camera_id_code == reordered.reference_camera_id_code == 1
    assert selection.reference_view_index == 1
    assert reordered.reference_view_index == 0
    assert selection.provenance.reference_from_physical == (
        reordered.provenance.reference_from_physical
    )
    fields = selection.to_tensor_fields(dtype=torch.float64)
    assert fields["reference_view_index"].dtype == torch.int64
    assert fields["view_camera_ids"].tolist() == [0, 1]
    assert fields["reference_camera_id"].item() == 1
    assert fields["reference_from_physical"].dtype == torch.float64


def test_selection_rejects_reference_missing_from_selected_or_complete_table() -> None:
    first, second = _views()
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        ("camera_10", "camera_2")
    )
    with pytest.raises(ReferenceViewSelectionError, match="not in selected"):
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=(first, second),  # type: ignore[arg-type]
            reference_camera_id="missing",
        )
    incomplete = StableCameraIdTable.from_complete_scene_camera_ids(("camera_10",))
    with pytest.raises(ReferenceViewSelectionError, match="absent"):
        ReferenceViewSelection.create(
            stable_camera_id_table=incomplete,
            selected_views=(first, second),  # type: ignore[arg-type]
            reference_camera_id="camera_10",
        )


def test_mid_plane_camera_is_rejected_by_geometry_authority() -> None:
    with pytest.raises(CameraCourtViewError, match="mid-plane"):
        build_court_view_record(
            camera_id="mid",
            camera_center_court_m=(0.0, 0.0, 2.0),
            contract=resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR),
        )


def test_training_policy_is_seeded_and_independent_of_view_order() -> None:
    first = select_seeded_training_reference_camera_id(
        ("camera_b", "camera_a"),
        rng=np.random.default_rng(19),
    )
    reordered = select_seeded_training_reference_camera_id(
        ("camera_a", "camera_b"),
        rng=np.random.default_rng(19),
    )
    assert first == reordered
    assert first in {"camera_a", "camera_b"}


def test_eval_policy_requires_explicit_multiview_but_resolves_single_view() -> None:
    assert (
        resolve_evaluation_reference_camera_id(
            ("only",),
            requested_camera_id=None,
        )
        == "only"
    )
    assert (
        resolve_evaluation_reference_camera_id(
            ("a", "b"),
            requested_camera_id="b",
        )
        == "b"
    )
    with pytest.raises(ReferenceViewSelectionError, match="explicit"):
        resolve_evaluation_reference_camera_id(
            ("a", "b"),
            requested_camera_id=None,
        )


def test_eval_subset_preserves_order_when_reference_is_already_selected() -> None:
    selected = (2, 0)

    result = include_evaluation_reference_camera(
        ("camera_0", "camera_1", "camera_2"),
        selected,
        requested_camera_id="camera_2",
        rng=np.random.default_rng(0),
    )

    assert result == selected


def test_eval_subset_includes_absent_reference_without_fixing_local_zero() -> None:
    result = include_evaluation_reference_camera(
        ("camera_0", "camera_1", "camera_2"),
        (2, 0),
        requested_camera_id="camera_1",
        rng=np.random.default_rng(0),
    )

    assert result == (2, 1)
    assert result.index(1) == 1


def test_eval_subset_rejects_unknown_reference_and_invalid_indices() -> None:
    with pytest.raises(ReferenceViewSelectionError, match="complete camera domain"):
        include_evaluation_reference_camera(
            ("camera_0", "camera_1"),
            (0,),
            requested_camera_id="missing",
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ReferenceViewSelectionError, match="outside"):
        include_evaluation_reference_camera(
            ("camera_0", "camera_1"),
            (2,),
            requested_camera_id="camera_0",
            rng=np.random.default_rng(0),
        )


def _valid_batch() -> dict[str, torch.Tensor]:
    return {
        "reference_view_index": torch.tensor([1, 0], dtype=torch.int64),
        "view_camera_ids": torch.tensor([[0, 2, -1], [1, 2, 3]], dtype=torch.int64),
        "reference_camera_id": torch.tensor([2, 1], dtype=torch.int64),
        "view_valid_mask": torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        ),
        "reference_from_physical": torch.eye(3).repeat(2, 1, 1),
        "physical_from_reference": torch.eye(3).repeat(2, 1, 1),
    }


def test_batch_contract_accepts_padding_only_outside_stable_id_domain() -> None:
    validate_reference_view_batch(
        **_valid_batch(),
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            ("a", "b", "c", "d")
        ),
    )


def test_batch_transform_validation_is_safe_under_bfloat16_autocast() -> None:
    batch = _valid_batch()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_product = (
            batch["reference_from_physical"].transpose(-1, -2)
            @ batch["reference_from_physical"]
        )
        assert autocast_product.dtype == torch.bfloat16
        validate_reference_view_batch(**batch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "reference_from_physical",
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            "proper rotations",
        ),
        ("physical_from_reference", torch.zeros(2, 3, 3), "must equal"),
    ],
)
def test_batch_transform_validation_fails_closed_under_bfloat16_autocast(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    batch = _valid_batch()
    batch[field] = value

    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(ReferenceViewBatchError, match=message),
    ):
        validate_reference_view_batch(**batch)


def test_batch_contract_supports_one_complete_id_table_per_scene() -> None:
    validate_reference_view_batch(
        **_valid_batch(),
        stable_camera_id_tables=(
            StableCameraIdTable.from_complete_scene_camera_ids(("a", "b", "c")),
            StableCameraIdTable.from_complete_scene_camera_ids(
                ("one", "two", "three", "four")
            ),
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_view_index", torch.tensor([1, 3]), "in \[0, 3\)"),
        ("reference_view_index", torch.tensor([1, -1]), "negative"),
        ("reference_view_index", torch.tensor([1, 0], dtype=torch.int32), "int64"),
        ("reference_camera_id", torch.tensor([0, 1]), "exactly equal"),
        (
            "view_camera_ids",
            torch.tensor([[0, -1, 2], [1, 2, 3]]),
            "trailing",
        ),
        (
            "view_camera_ids",
            torch.tensor([[0, 0, -1], [1, 2, 3]]),
            "duplicate",
        ),
        (
            "view_valid_mask",
            torch.tensor([[True, True, True], [True, True, True]]),
            "exactly mark",
        ),
        (
            "reference_from_physical",
            torch.zeros(2, 3, 3),
            "proper rotations",
        ),
        (
            "reference_from_physical",
            torch.tensor(
                [
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            "identity or Rz",
        ),
        (
            "physical_from_reference",
            torch.zeros(2, 3, 3),
            "must equal",
        ),
    ],
)
def test_batch_contract_rejects_index_identity_padding_mask_and_transform_errors(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    batch = _valid_batch()
    batch[field] = value
    with pytest.raises(ReferenceViewBatchError, match=message):
        validate_reference_view_batch(**batch)
