"""Strict paired-reference counterfactual evaluator tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.base.evaluation.reference_counterfactual import (
    REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
    ReferenceCounterfactualError,
    ReferenceCounterfactualManifest,
    ReferenceCounterfactualPass,
    ReferenceCounterfactualPassRow,
    ReferenceCounterfactualQuantityArrays,
    ReferenceCounterfactualQuantitySchema,
    ReferenceCounterfactualReport,
    ReferenceCounterfactualRunIdentity,
    ReferenceSideSelection,
    array_payload_sha256,
    build_reference_counterfactual_manifest_from_documents,
    canonical_json_sha256,
    evaluate_reference_counterfactual,
    masked_counterfactual_quantity_for_digest,
    read_reference_counterfactual_report,
    write_reference_counterfactual_report,
)
from src.tasks.base.evaluation.track_query_reference import PairedReferenceKey
from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    CourtKeypointArtifactMetadata,
    CourtReferenceFrameProvenance,
    build_court_view_record,
    court_headings_physical_to_target,
    court_points_physical_to_target,
    court_vectors_physical_to_target,
    court_world_joints_physical_to_target,
    resolve_court_keypoint_contract,
)

_SCHEMA_ID = "test_reference_counterfactual_v2"
_DIGEST = "a" * 64


def _documents() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    views = (
        build_court_view_record(
            camera_id="looks_opposite_9",
            camera_center_court_m=(0.0, -4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="same_z",
            camera_center_court_m=(1.0, 4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="same_a",
            camera_center_court_m=(-1.0, 4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="looks_same_0",
            camera_center_court_m=(2.0, -4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="identity_first",
            camera_center_court_m=(-2.0, -4.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="opposite_first",
            camera_center_court_m=(3.0, 4.0, 2.0),
            contract=contract,
        ),
    )
    metadata = CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id=_SCHEMA_ID,
    ).to_dict()
    root: dict[str, object] = {
        "court_keypoints": metadata,
        "scenes": [
            {
                "file": "scene_a",
                "scene_id": "scene_a",
                "num_cameras": 6,
            }
        ],
    }
    scene: dict[str, object] = {
        "scene_id": "scene_a",
        "num_cameras": 6,
        "court_keypoints": metadata,
        "court_keypoint_views": [view.to_dict() for view in views],
    }
    return root, {"scene_a": scene}


def _manifest() -> ReferenceCounterfactualManifest:
    root, scenes = _documents()
    return build_reference_counterfactual_manifest_from_documents(
        root_metadata=root,
        scene_metadata=scenes,
        expected_dataset_schema_id=_SCHEMA_ID,
    )


def _identity(
    manifest: ReferenceCounterfactualManifest,
    **changes: object,
) -> ReferenceCounterfactualRunIdentity:
    values: dict[str, object] = {
        "task": "plcs",
        "seed": 42,
        "selector_mode": "reference",
        "resolved_config": {"data": {"views": 6}, "model": "selector"},
        "checkpoint_sha256": "b" * 64,
        "manifest_digest": manifest.digest,
        "court_keypoint_contract": "camera_view_courtkp20_rzpi_v1",
        "target_frame_contract": "reference_camera_court_rzpi_v1",
        "track_query_rope_contract": "track_query_reference_selector_v2",
    }
    values.update(changes)
    return ReferenceCounterfactualRunIdentity.create(**cast("Any", values))


def _row(
    manifest: ReferenceCounterfactualManifest,
    side: str,
    *,
    digests: dict[str, str] | None = None,
) -> ReferenceCounterfactualPassRow:
    scene = manifest.scenes[0]
    selection = scene.selection(cast("Any", side))
    parity = {
        "frame_digest": canonical_json_sha256({"window": [2, 4]}),
        "lifecycle_digest": canonical_json_sha256({"slots": [1, 7]}),
        "observation_digest": canonical_json_sha256({"uv": [0.25, 0.75]}),
        "target_digest": canonical_json_sha256({"physical_target": [1.0, 2.0]}),
    }
    if digests:
        parity.update(digests)
    return ReferenceCounterfactualPassRow(
        key=scene.key,
        window_start=2,
        window_stop=4,
        reference_camera_id=selection.camera_id,
        reference_view_index=selection.local_index,
        provenance=selection.provenance,
        **parity,
    )


def _pass(
    manifest: ReferenceCounterfactualManifest,
    side: str,
    *,
    identity: ReferenceCounterfactualRunIdentity | None = None,
    row: ReferenceCounterfactualPassRow | None = None,
    nonfinite: bool = False,
    task: str = "plcs",
    include_vector: bool = False,
    include_world_joints: bool = False,
) -> ReferenceCounterfactualPass:
    scene = manifest.scenes[0]
    selection = scene.selection(cast("Any", side))
    physical_target = np.asarray(
        [[[1.0, 2.0, 0.5], [-2.0, -3.0, 1.0]]],
        dtype=np.float64,
    )
    physical_prediction = physical_target + np.asarray(
        [[[0.5, -0.25, 1.0], [-0.5, 0.25, -1.0]]]
    )
    position_target = cast(
        "np.ndarray[Any, Any]",
        court_points_physical_to_target(physical_target, selection.provenance),
    )
    position_prediction = cast(
        "np.ndarray[Any, Any]",
        court_points_physical_to_target(physical_prediction, selection.provenance),
    )
    physical_heading_target = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    physical_heading_prediction = np.asarray(
        [[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64
    )
    heading_target = cast(
        "np.ndarray[Any, Any]",
        court_headings_physical_to_target(
            physical_heading_target,
            selection.provenance,
        ),
    )
    heading_prediction = cast(
        "np.ndarray[Any, Any]",
        court_headings_physical_to_target(
            physical_heading_prediction,
            selection.provenance,
        ),
    )
    physical_vector_target = np.asarray(
        [[[0.5, -1.0, 0.25], [-0.25, 1.0, 0.5]]], dtype=np.float64
    )
    physical_vector_prediction = physical_vector_target + 0.125
    vector_target = cast(
        "np.ndarray[Any, Any]",
        court_vectors_physical_to_target(physical_vector_target, selection.provenance),
    )
    vector_prediction = cast(
        "np.ndarray[Any, Any]",
        court_vectors_physical_to_target(
            physical_vector_prediction, selection.provenance
        ),
    )
    physical_joints_target = np.stack(
        (physical_target, physical_target + 0.25), axis=-2
    )
    physical_joints_prediction = physical_joints_target + 0.1
    joints_target = cast(
        "np.ndarray[Any, Any]",
        court_world_joints_physical_to_target(
            physical_joints_target, selection.provenance
        ),
    )
    joints_prediction = cast(
        "np.ndarray[Any, Any]",
        court_world_joints_physical_to_target(
            physical_joints_prediction, selection.provenance
        ),
    )
    if nonfinite:
        position_prediction[0, 0, 0] = np.nan
    return ReferenceCounterfactualPass(
        schema_version=REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
        side=cast("Any", side),
        identity=identity or _identity(manifest, task=task),
        quantity_schema=ReferenceCounterfactualQuantitySchema.for_task(
            cast("Any", task),
            vector=include_vector,
            world_joints=include_world_joints,
        ),
        rows=(row or _row(manifest, side),),
        valid_mask=np.ones((1, 2), dtype=np.bool_),
        position=ReferenceCounterfactualQuantityArrays(
            prediction=position_prediction,
            target=position_target,
            quantity="point",
        ),
        vector=(
            ReferenceCounterfactualQuantityArrays(
                prediction=vector_prediction,
                target=vector_target,
                quantity="vector",
            )
            if include_vector
            else None
        ),
        heading=(
            ReferenceCounterfactualQuantityArrays(
                prediction=heading_prediction,
                target=heading_target,
                quantity="heading",
            )
            if task == "plcs"
            else None
        ),
        world_joints=(
            ReferenceCounterfactualQuantityArrays(
                prediction=joints_prediction,
                target=joints_target,
                quantity="world_joints",
            )
            if include_world_joints
            else None
        ),
    )


def _report() -> ReferenceCounterfactualReport:
    manifest = _manifest()
    return evaluate_reference_counterfactual(
        manifest,
        _pass(manifest, "same_side"),
        _pass(manifest, "opposite_side"),
    )


def _two_scene_manifest() -> ReferenceCounterfactualManifest:
    single = _manifest()
    second_scene = replace(single.scenes[0], scene_id="scene_b")
    return replace(single, scenes=(single.scenes[0], second_scene))


def _pass_with_empty_scene(
    manifest: ReferenceCounterfactualManifest,
    side: str,
) -> ReferenceCounterfactualPass:
    pass_value = _pass(manifest, side, task="blcs")
    empty_scene = manifest.scenes[1]
    empty_selection = empty_scene.selection(cast("Any", side))
    empty_row = replace(
        pass_value.rows[0],
        key=empty_scene.key,
        reference_camera_id=empty_selection.camera_id,
        reference_view_index=empty_selection.local_index,
        provenance=empty_selection.provenance,
    )
    position = pass_value.position
    return replace(
        pass_value,
        rows=(pass_value.rows[0], empty_row),
        valid_mask=np.asarray([[True, True], [False, False]], dtype=np.bool_),
        position=replace(
            position,
            prediction=np.concatenate(
                (position.prediction, position.prediction + 37.0), axis=0
            ),
            target=np.concatenate((position.target, position.target - 19.0), axis=0),
        ),
    )


def test_manifest_uses_persisted_transform_classes_and_lexicographic_first_ids() -> (
    None
):
    manifest = _manifest()
    scene = manifest.scenes[0]

    assert scene.same_side.camera_id == "identity_first"
    assert scene.same_side.local_index == 4
    assert scene.opposite_side.camera_id == "opposite_first"
    assert scene.opposite_side.local_index == 5
    assert scene.local_ordering[0] == "looks_opposite_9"
    assert scene.view_camera_ids == tuple(sorted(scene.local_ordering))


def test_manifest_digest_is_deterministic_and_round_trips_exactly() -> None:
    first = _manifest()
    second = _manifest()

    assert first.digest == second.digest
    restored = ReferenceCounterfactualManifest.from_dict(first.to_dict())
    assert restored == first
    assert restored.digest == first.digest


def test_named_array_payload_digest_is_stable_and_semantic() -> None:
    little = np.asarray([[1.0, 2.0]], dtype="<f8")
    big = np.asarray([[1.0, 2.0]], dtype=">f8")
    first = array_payload_sha256({"uv": little, "mask": np.asarray([True])})
    reordered = array_payload_sha256({"mask": np.asarray([True]), "uv": big})

    assert first == reordered
    assert first != array_payload_sha256({"uv": little, "other": np.asarray([True])})
    assert first != array_payload_sha256(
        {"uv": little.reshape(2), "mask": np.asarray([True])}
    )
    with pytest.raises(ReferenceCounterfactualError, match="cannot be empty"):
        array_payload_sha256({})
    with pytest.raises(ReferenceCounterfactualError, match="object dtype"):
        array_payload_sha256({"bad": np.asarray([object()], dtype=object)})
    with pytest.raises(ReferenceCounterfactualError, match="non-finite"):
        array_payload_sha256({"bad": np.asarray([np.nan])})


def test_masked_digest_quantity_excludes_inactive_fill_and_signed_zero() -> None:
    value = np.asarray([[-0.0, 1.0], [-1.0, -0.0]], dtype=np.float32)
    masked = masked_counterfactual_quantity_for_digest(
        value,
        np.asarray([True, False], dtype=np.bool_),
    )

    assert masked.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert not np.signbit(masked).any()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("root_missing", "absent from dataset root"),
        ("root_duplicate", "duplicate scene identity"),
        ("root_count", "exactly six cameras"),
        ("scene_count", "exactly six cameras"),
        ("scene_identity", "disagrees"),
        ("five_views", "exactly six"),
    ],
)
def test_manifest_rejects_missing_duplicate_non_six_and_inconsistent_scene(
    mutation: str,
    message: str,
) -> None:
    root, scenes = _documents()
    scene = scenes["scene_a"]
    summaries = cast("list[dict[str, object]]", root["scenes"])
    if mutation == "root_missing":
        summaries.clear()
    elif mutation == "root_duplicate":
        summaries.append(dict(summaries[0]))
    elif mutation == "root_count":
        summaries[0]["num_cameras"] = 5
    elif mutation == "scene_count":
        scene["num_cameras"] = 5
    elif mutation == "scene_identity":
        scene["scene_id"] = "other"
    else:
        cast("list[object]", scene["court_keypoint_views"]).pop()

    with pytest.raises(ReferenceCounterfactualError, match=message):
        build_reference_counterfactual_manifest_from_documents(
            root_metadata=root,
            scene_metadata=scenes,
            expected_dataset_schema_id=_SCHEMA_ID,
        )


def test_side_label_cannot_override_persisted_transform() -> None:
    selection = _manifest().scenes[0].same_side

    with pytest.raises(ReferenceCounterfactualError, match="persisted"):
        ReferenceSideSelection(
            side="opposite_side",
            camera_id=selection.camera_id,
            local_index=selection.local_index,
            provenance=selection.provenance,
        )


def test_join_metrics_are_reference_target_frame_but_consistency_is_physical() -> None:
    report = _report()
    same = report.same_side_pass.position.target
    opposite = report.opposite_side_pass.position.target

    assert not np.array_equal(same, opposite)
    assert report.metrics.same_side.position.y_sign_accuracy == pytest.approx(1.0)
    assert report.metrics.opposite_side.position.y_sign_accuracy == pytest.approx(1.0)
    assert report.metrics.same_side.position.axis_wise_position_error.to_dict() == (
        pytest.approx({"x": 0.5, "y": 0.25, "z": 1.0})
    )
    assert report.metrics.opposite_side.position.axis_wise_position_error.to_dict() == (
        pytest.approx({"x": 0.5, "y": 0.25, "z": 1.0})
    )
    assert report.metrics.same_side.heading_error_deg == pytest.approx(90.0)
    assert report.metrics.opposite_side.heading_error_deg == pytest.approx(90.0)
    assert report.metrics.physical_consistency.position_error_m == pytest.approx(
        0.0, abs=1e-12
    )
    assert report.metrics.physical_consistency.heading_error == pytest.approx(
        0.0, abs=1e-12
    )
    assert report.metrics.physical_consistency.vector_error_m is None
    assert report.metrics.physical_consistency.world_joints_error_m is None


def test_mixed_supervised_and_empty_rows_retain_complete_finite_evidence(
    tmp_path: Path,
) -> None:
    supervised_manifest = _manifest()
    supervised_report = evaluate_reference_counterfactual(
        supervised_manifest,
        _pass(supervised_manifest, "same_side", task="blcs"),
        _pass(supervised_manifest, "opposite_side", task="blcs"),
    )
    manifest = _two_scene_manifest()
    same_side = _pass_with_empty_scene(manifest, "same_side")
    opposite_side = _pass_with_empty_scene(manifest, "opposite_side")

    report = evaluate_reference_counterfactual(manifest, same_side, opposite_side)

    assert report.metrics.to_dict() == supervised_report.metrics.to_dict()
    assert all(np.isfinite(value) for value in report.metrics.flat_dict().values())
    assert [row.key.scene_id for row in report.same_side_pass.rows] == [
        "scene_a",
        "scene_b",
    ]
    changed_empty_evidence = evaluate_reference_counterfactual(
        manifest,
        replace(
            same_side,
            rows=(
                same_side.rows[0],
                replace(same_side.rows[1], frame_digest="c" * 64),
            ),
        ),
        replace(
            opposite_side,
            rows=(
                opposite_side.rows[0],
                replace(opposite_side.rows[1], frame_digest="c" * 64),
            ),
        ),
    )
    assert changed_empty_evidence.parity_digest != report.parity_digest

    paths = write_reference_counterfactual_report(report, tmp_path)
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert [
        row["scene_id"] for row in document["passes"]["same_side"]["rows"]
    ] == ["scene_a", "scene_b"]
    with np.load(paths.npz_path, allow_pickle=False) as archive:
        assert archive["scene_ids"].tolist() == ["scene_a", "scene_b"]
        assert archive["valid_mask"].tolist() == [
            [True, True],
            [False, False],
        ]


def test_pass_rejects_only_aggregate_empty_supervision_and_keeps_mask_strict() -> (
    None
):
    manifest = _two_scene_manifest()
    mixed = _pass_with_empty_scene(manifest, "same_side")

    with pytest.raises(
        ReferenceCounterfactualError,
        match="at least one supervised observation across all rows",
    ):
        replace(mixed, valid_mask=np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(ReferenceCounterfactualError, match="bool numpy array"):
        replace(mixed, valid_mask=np.ones((2, 2), dtype=np.int64))
    with pytest.raises(ReferenceCounterfactualError, match="match position leading axes"):
        replace(mixed, valid_mask=np.ones((2, 1), dtype=np.bool_))


def test_task_specific_vector_and_world_joint_consistency_are_recomputed() -> None:
    manifest = _manifest()
    blcs = evaluate_reference_counterfactual(
        manifest,
        _pass(manifest, "same_side", task="blcs", include_vector=True),
        _pass(manifest, "opposite_side", task="blcs", include_vector=True),
    )
    plcs = evaluate_reference_counterfactual(
        manifest,
        _pass(manifest, "same_side", include_world_joints=True),
        _pass(manifest, "opposite_side", include_world_joints=True),
    )

    assert blcs.metrics.same_side.heading_error_deg is None
    assert blcs.metrics.physical_consistency.vector_error_m == pytest.approx(
        0.0, abs=1e-12
    )
    assert blcs.metrics.physical_consistency.heading_error is None
    assert plcs.metrics.physical_consistency.world_joints_error_m == pytest.approx(
        0.0, abs=1e-12
    )
    assert plcs.metrics.physical_consistency.vector_error_m is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 43, "Seed"),
        ("checkpoint_sha256", "c" * 64, "checkpoint"),
        ("resolved_config", {"different": True}, "resolved config"),
        ("court_keypoint_contract", "wrong_contract", "contracts"),
        ("target_frame_contract", "wrong_frame", "contracts"),
        ("track_query_rope_contract", "wrong_rope", "contracts"),
        ("selector_mode", "selector_zero", "selector mode"),
    ],
)
def test_join_rejects_run_identity_mismatch(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _manifest()
    opposite_identity = _identity(manifest, **{field: value})

    with pytest.raises(ReferenceCounterfactualError, match=message):
        evaluate_reference_counterfactual(
            manifest,
            _pass(manifest, "same_side"),
            _pass(manifest, "opposite_side", identity=opposite_identity),
        )


@pytest.mark.parametrize(
    ("digest_field", "message"),
    [
        ("frame_digest", "frame_digest"),
        ("lifecycle_digest", "lifecycle_digest"),
        ("observation_digest", "observation_digest"),
        ("target_digest", "target_digest"),
    ],
)
def test_join_rejects_frame_lifecycle_uv_observation_and_target_mismatch(
    digest_field: str,
    message: str,
) -> None:
    manifest = _manifest()
    changed = _row(
        manifest,
        "opposite_side",
        digests={digest_field: "f" * 64},
    )

    with pytest.raises(ReferenceCounterfactualError, match=message):
        evaluate_reference_counterfactual(
            manifest,
            _pass(manifest, "same_side"),
            _pass(manifest, "opposite_side", row=changed),
        )


def test_join_rejects_order_window_duplicate_missing_and_padding_identity() -> None:
    manifest = _manifest()
    opposite = _pass(manifest, "opposite_side")
    row = opposite.rows[0]
    reordered = (
        row.key.local_ordering[-1],
        *row.key.local_ordering[1:-1],
        row.key.local_ordering[0],
    )
    bad_key = PairedReferenceKey(
        scene_id=row.key.scene_id,
        view_camera_ids=row.key.view_camera_ids,
        local_ordering=reordered,
    )
    with pytest.raises(
        ReferenceCounterfactualError, match="identity-index|index/identity"
    ):
        replace(row, key=bad_key)
    non_reference_reorder = (
        row.key.local_ordering[1],
        row.key.local_ordering[0],
        *row.key.local_ordering[2:],
    )
    non_reference_key = PairedReferenceKey(
        scene_id=row.key.scene_id,
        view_camera_ids=row.key.view_camera_ids,
        local_ordering=non_reference_reorder,
    )
    bad_order_pass = replace(opposite, rows=(replace(row, key=non_reference_key),))
    with pytest.raises(ReferenceCounterfactualError, match="camera set/order"):
        evaluate_reference_counterfactual(
            manifest,
            _pass(manifest, "same_side"),
            bad_order_pass,
        )
    with pytest.raises(ReferenceCounterfactualError, match="positive"):
        replace(row, window_stop=row.window_start)
    with pytest.raises(ReferenceCounterfactualError, match="six non-padding"):
        replace(row, reference_view_index=-1)
    with pytest.raises(ReferenceCounterfactualError, match="duplicate"):
        replace(opposite, rows=(row, row), valid_mask=np.ones((2, 2), dtype=np.bool_))
    with pytest.raises(ReferenceCounterfactualError, match="pass is empty"):
        replace(opposite, rows=())


def test_nonfinite_and_stale_schema_are_hard_errors() -> None:
    manifest = _manifest()

    with pytest.raises(ReferenceCounterfactualError, match="finite"):
        _pass(manifest, "same_side", nonfinite=True)
    with pytest.raises(ReferenceCounterfactualError, match="stale"):
        replace(_pass(manifest, "same_side"), schema_version=0)


def test_task_quantity_schema_requires_explicit_absence() -> None:
    with pytest.raises(ReferenceCounterfactualError, match="PLCS"):
        ReferenceCounterfactualQuantitySchema.for_task("plcs", vector=True)
    with pytest.raises(ReferenceCounterfactualError, match="BLCS"):
        ReferenceCounterfactualQuantitySchema.for_task("blcs", world_joints=True)
    with pytest.raises(ReferenceCounterfactualError, match="Unknown"):
        ReferenceCounterfactualQuantityArrays(
            prediction=np.zeros((1, 1, 3)),
            target=np.zeros((1, 1, 3)),
            quantity=cast("Any", "not_a_quantity"),
        )


def test_report_round_trip_recomputes_metrics_and_has_deterministic_digest(
    tmp_path: Path,
) -> None:
    report = _report()
    second = _report()
    assert report.parity_digest == second.parity_digest
    assert report.report_digest == second.report_digest

    paths = write_reference_counterfactual_report(report, tmp_path)
    loaded = read_reference_counterfactual_report(tmp_path)

    assert paths.npz_path.name == "pred_test.npz"
    assert paths.metrics_path.name == "metrics.json"
    assert paths.json_path.name == "reference_counterfactual.json"
    assert loaded.report_digest == report.report_digest
    assert loaded.metrics.to_dict() == report.metrics.to_dict()
    flat_metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    assert flat_metrics == report.metrics.flat_dict()
    assert flat_metrics
    assert all(type(value) in (int, float) for value in flat_metrics.values())
    assert "reference_target_same_side_y_sign_accuracy" in flat_metrics
    assert "reference_target_opposite_side_position_error_y_m" in flat_metrics
    assert "reference_target_same_side_heading_error_deg" in flat_metrics
    assert "physical_restored_position_consistency_error_m" in flat_metrics
    with np.load(paths.npz_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(report.npz_arrays())
        assert archive["same_side_reference_view_index"].tolist() == [4]
        assert archive["opposite_side_reference_view_index"].tolist() == [5]
    with pytest.raises(ReferenceCounterfactualError, match="overwrite"):
        write_reference_counterfactual_report(report, tmp_path)


def test_reader_rejects_tampered_metric_and_mixed_pair(tmp_path: Path) -> None:
    report = _report()
    paths = write_reference_counterfactual_report(report, tmp_path)
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    document["metrics"]["reference_target_frame"]["same_side"]["y_sign_accuracy"] = 0.0
    paths.json_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReferenceCounterfactualError, match="report digest|metrics"):
        read_reference_counterfactual_report(tmp_path)

    paths.json_path.unlink()
    with pytest.raises(ReferenceCounterfactualError, match="missing one or more"):
        read_reference_counterfactual_report(tmp_path)


def test_reader_rejects_tampered_npz_content(tmp_path: Path) -> None:
    report = _report()
    paths = write_reference_counterfactual_report(report, tmp_path)
    with np.load(paths.npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["same_side_position_prediction"][0, 0, 0] += 1.0
    np.savez_compressed(paths.npz_path, **arrays)

    with pytest.raises(ReferenceCounterfactualError, match="content digest"):
        read_reference_counterfactual_report(tmp_path)


@pytest.mark.parametrize("missing_field", ["json_path", "npz_path", "metrics_path"])
def test_reader_rejects_partial_standard_queue_bundle(
    tmp_path: Path,
    missing_field: str,
) -> None:
    paths = write_reference_counterfactual_report(_report(), tmp_path)
    cast("Path", getattr(paths, missing_field)).unlink()

    with pytest.raises(ReferenceCounterfactualError, match="missing one or more"):
        read_reference_counterfactual_report(tmp_path)


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {"nested": {"ignored_by_kg_register": 1.0}},
        {"nonfinite": float("nan")},
        {"boolean": True},
    ],
)
def test_reader_rejects_nonflat_or_nonfinite_queue_metrics(
    tmp_path: Path,
    invalid_metrics: dict[str, object],
) -> None:
    paths = write_reference_counterfactual_report(_report(), tmp_path)
    paths.metrics_path.write_text(
        json.dumps(invalid_metrics, allow_nan=True),
        encoding="utf-8",
    )

    with pytest.raises(ReferenceCounterfactualError, match="finite flat numbers"):
        read_reference_counterfactual_report(tmp_path)


def test_reader_rejects_stale_flat_queue_metric(tmp_path: Path) -> None:
    paths = write_reference_counterfactual_report(_report(), tmp_path)
    metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    metrics["reference_target_same_side_y_sign_accuracy"] = 0.0
    paths.metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ReferenceCounterfactualError, match="flat metrics"):
        read_reference_counterfactual_report(tmp_path)


def test_provenance_rejects_unknown_transform_before_evaluation() -> None:
    selection = _manifest().scenes[0].same_side
    mapping = selection.provenance.to_dict()
    mapping["reference_from_physical"] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    mapping["physical_from_reference"] = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    with pytest.raises(ValueError, match="identity or Rz"):
        CourtReferenceFrameProvenance.from_mapping(mapping, location="bad")


def test_manifest_rejects_missing_transform_class() -> None:
    root, scenes = _documents()
    scene = scenes["scene_a"]
    views = cast("list[dict[str, object]]", scene["court_keypoint_views"])
    identity = next(
        view
        for view in views
        if cast("list[list[float]]", view["canonical_from_physical"])[0][0] == 1.0
    )
    replacement = dict(identity)
    replacement["camera_id"] = "replacement"
    replacement["camera_center_court_m"] = [-3.0, -5.0, 2.0]
    scene["court_keypoint_views"] = [
        {**replacement, "camera_id": f"replacement_{index}"} for index in range(6)
    ]

    with pytest.raises(ReferenceCounterfactualError, match="no persisted"):
        build_reference_counterfactual_manifest_from_documents(
            root_metadata=root,
            scene_metadata=scenes,
            expected_dataset_schema_id=_SCHEMA_ID,
        )
