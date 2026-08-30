"""Strict schema-v2 consumer contracts for synthetic Court input."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.court_detection.configuration import SyntheticCourtSourceConfig
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.processing.geometry import CourtProcessingGeometry
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    COURT_KP_NAMES,
    OPPOSITE_COURT_END_INDEX,
)

_IDENTITY_PHYSICAL = tuple(range(14))
_OPPOSITE_PHYSICAL = OPPOSITE_COURT_END_INDEX
_CAMERA_VIEW_PHYSICAL = CAMERA_VIEW_HALF_TURN_INDEX
_FLIP = (1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13)


def _target(court_id: str = "court-b") -> dict[str, object]:
    return {
        "binding": {
            "court_instance_id": court_id,
            "candidate_id": f"candidate-{court_id}",
            "scene_from_court": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "selection_seed": 761,
        },
        "resolution_policy": "nearest_camera",
        "camera_to_court_center_distance_m": 10.0,
    }


def _camera(sample_id: str, sample_index: int) -> dict[str, object]:
    return {
        "camera_id": sample_id,
        "source_frame_index": sample_index,
        "width": 64,
        "height": 48,
        "intrinsics": [50.0, 0.0, 31.5, 0.0, 50.0, 23.5, 0.0, 0.0, 1.0],
        "camera_to_scene": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "image_path": f"generated/{sample_id}.png",
    }


def _court(
    court_id: str,
    *,
    physical_order: tuple[int, ...],
    x_offset: float,
    all_invisible: bool = False,
    camera_view_uv: bool = False,
) -> dict[str, object]:
    classes: list[dict[str, object]] = []
    for class_id, (class_name, physical_index) in enumerate(
        zip(COURT_KP_NAMES[:14], physical_order, strict=True)
    ):
        in_front = not (court_id == "court-a" and class_id == 0)
        in_frame = not all_invisible and not (court_id == "court-a" and class_id == 2)
        renderer_visible = not (court_id == "court-a" and class_id == 1)
        point = {
            "physical_index": physical_index,
            "uv": [
                (
                    float(5 + class_id * 2) + x_offset
                    if camera_view_uv
                    else float(5 + (physical_index % 4) * 10) + x_offset
                ),
                float(5 + (physical_index // 4) * 10),
            ],
            "camera_depth_m": 10.0,
            "scene_xyz_m": [float(physical_index), x_offset, 0.0],
            "in_front": in_front,
            "in_frame": in_frame,
            "renderer_visible": renderer_visible,
        }
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "renderer_visible": renderer_visible,
                "points": [point],
            }
        )
    return {
        "court_instance_id": court_id,
        "coverage_mode": "full",
        "classes": classes,
    }


def _projection(
    sample_id: str,
    *,
    court_order: tuple[str, str] = ("court-a", "court-b"),
    invisible_court_id: str | None = None,
    schema: Literal["v2", "v3", "v4"] = "v2",
) -> dict[str, object]:
    court_specs = {
        "court-a": (_IDENTITY_PHYSICAL, 0.0),
        "court-b": (
            _OPPOSITE_PHYSICAL if schema == "v2" else _CAMERA_VIEW_PHYSICAL,
            2.0,
        ),
    }
    courts = [
        _court(
            court_id,
            physical_order=court_specs[court_id][0],
            x_offset=court_specs[court_id][1],
            all_invisible=court_id == invisible_court_id,
            camera_view_uv=schema in {"v3", "v4"},
        )
        for court_id in court_order
    ]
    visible_names = list(COURT_KP_NAMES[:14])
    visible_count = sum(
        cast(bool, semantic["renderer_visible"])
        for court in courts
        for semantic in cast(list[dict[str, object]], court["classes"])
    )
    return {
        "camera_id": sample_id,
        "resolution": [64, 48],
        "coverage_modes": ["full", "full"],
        "visible_class_names": visible_names,
        "visible_point_count": visible_count,
        "courts": courts,
    }


def _write_v2_dataset(
    root: Path,
    *,
    court_order: tuple[str, str] = ("court-a", "court-b"),
    target_court_id: str = "court-b",
    invisible_court_id: str | None = None,
    schema: Literal["v2", "v3", "v4"] = "v2",
) -> tuple[Path, dict[str, object]]:
    dataset_root = root / "B00" / "datasets" / "court"
    records: list[dict[str, object]] = []
    for sample_index, split in enumerate(("train", "validation", "test")):
        sample_id = f"sample-{split}"
        relative = Path("samples") / sample_id
        sample_root = dataset_root / relative
        sample_root.mkdir(parents=True, exist_ok=True)
        np.save(sample_root / "rgb.npy", np.full((48, 64, 3), 0.5, np.float32))
        np.save(sample_root / "alpha.npy", np.ones((48, 64, 1), np.float32))
        np.save(sample_root / "depth.npy", np.ones((48, 64, 1), np.float32))
        Image.new("RGB", (64, 48)).save(sample_root / "rgb.png")
        Image.new("L", (64, 48)).save(sample_root / "alpha.png")
        projection = _projection(
            sample_id,
            court_order=court_order,
            invisible_court_id=invisible_court_id,
            schema=schema,
        )
        target = _target(target_court_id)
        camera = _camera(sample_id, sample_index)
        metadata = {"fixture": True}
        labels = {
            "schema": f"canonical_court_sample_{schema}",
            "sample_index": sample_index,
            "sample_id": sample_id,
            "trajectory_group_id": f"group-{split}",
            "trajectory_id": f"trajectory-{split}",
            "view_id": "view-a",
            "trajectory_frame_index": 0,
            "split": split,
            "camera": camera,
            "projection": projection,
            "target_court": target,
            "metadata": metadata,
        }
        if schema == "v4":
            labels["safety_support_input_digest"] = "a" * 64
        (sample_root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        record = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "trajectory_group_id": f"group-{split}",
            "trajectory_id": f"trajectory-{split}",
            "view_id": "view-a",
            "trajectory_frame_index": 0,
            "split": split,
            "shard_id": "shard-a",
            "width": 64,
            "height": 48,
            "camera": camera,
            "projection": projection,
            "target_court": target,
            "directory": relative.as_posix(),
            "rgb": (relative / "rgb.npy").as_posix(),
            "rgb_preview": (relative / "rgb.png").as_posix(),
            "alpha": (relative / "alpha.npy").as_posix(),
            "alpha_preview": (relative / "alpha.png").as_posix(),
            "depth": (relative / "depth.npy").as_posix(),
            "depth_coordinate_space": "metric_scene_metres",
            "labels": (relative / "labels.json").as_posix(),
            "metadata": metadata,
        }
        if schema == "v4":
            record["safety_support_input_digest"] = "a" * 64
        records.append(record)
    manifest: dict[str, object] = {
        "schema": f"canonical_court_dataset_{schema}",
        "status": "completed",
        "scene_id": "B00",
        "profile": f"{schema}-fixture",
        "seed": 761,
        "sampling_policy": {},
        "metadata_fields": [],
        "trajectory_groups": [],
        "samples": records,
        "rejected_samples": [],
        "metrics": {},
        "diagnostics": [],
    }
    manifest_path = dataset_root / "dataset.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _input(
    root: Path,
    *,
    schema: Literal["v1", "v2", "v3", "v4"] = "v2",
    keypoint_court_scope: Literal["all_courts", "target_court"] = "all_courts",
) -> SyntheticCourtInput:
    return SyntheticCourtInput(
        SyntheticCourtSourceConfig(
            kind="synthetic_court",
            schema=schema,
            keypoint_court_scope=keypoint_court_scope,
            workspace_root=root,
            scene_ids=("B00",),
        ),
        target_store=CourtDerivedTargetStore(root / "derived"),
    )


def _rewrite(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v2_keeps_semantic_multi_peaks_separate_from_physical_instances(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path)
    input_layer = _input(tmp_path)

    assert input_layer.available_splits == ("train", "val", "test")
    assert input_layer.spec.source_schema == "canonical_court_dataset_v2"
    assert input_layer.spec.keypoint_schema == "synthetic_camera_relative_kp14"
    assert input_layer.spec.keypoint_channel_names == COURT_KP_NAMES[:14]
    assert input_layer.spec.keypoint_flip_permutation == _FLIP

    raw = input_layer.load(input_layer.records("train")[0])
    assert raw.keypoint_channels is not None
    channels = raw.keypoint_channels
    assert channels.points_xy.shape == (14, 2, 2)
    assert channels.points_xy.dtype == torch.float32
    assert channels.physical_indices[:, 0].tolist() == list(range(14))
    assert channels.physical_indices[:, 1].tolist() == list(_OPPOSITE_PHYSICAL)
    assert not bool(channels.point_visible[0, 0])  # in_front=False
    assert not bool(channels.point_visible[1, 0])  # renderer_visible=False
    assert not bool(channels.point_visible[2, 0])  # in_frame=False
    assert bool(channels.point_visible[0, 1])
    assert len(raw.court_instances) == 2
    for instance in raw.court_instances:
        assert instance.physical_indices.tolist() == list(range(14))
        assert instance.points_xy.dtype == torch.float32
    second = raw.court_instances[1]
    torch.testing.assert_close(second.points_xy[2], channels.points_xy[0, 1])

    flipped = CourtProcessingGeometry._transform_channels(
        channels,
        matrix=torch.eye(3, dtype=torch.float64),
        output_size_hw=(48, 64),
        horizontal_flipped=True,
    )
    torch.testing.assert_close(flipped.points_xy[0], channels.points_xy[1])
    torch.testing.assert_close(flipped.points_xy[4], channels.points_xy[6])


def test_v3_uses_distinct_schema_full_half_turn_and_one_flip_only(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path, schema="v3")
    input_layer = _input(tmp_path, schema="v3")

    assert input_layer.spec.source_schema == "canonical_court_dataset_v3"
    assert input_layer.spec.keypoint_schema == "synthetic_camera_view_kp14_v3"
    assert input_layer.spec.keypoint_flip_permutation == _FLIP
    raw = input_layer.load(input_layer.records("train")[0])
    assert raw.keypoint_channels is not None
    assert raw.pose_authority is not None
    assert raw.pose_authority.camera.camera_id == "sample-train"
    assert raw.pose_authority.target_court.court_instance_id == "court-b"
    channels = raw.keypoint_channels
    assert channels.points_xy.shape == (14, 2, 2)
    assert channels.points_xy.dtype == torch.float64
    assert channels.physical_indices[:, 0].tolist() == list(range(14))
    assert channels.physical_indices[:, 1].tolist() == list(_CAMERA_VIEW_PHYSICAL)
    assert all(
        instance.points_xy.dtype == torch.float32 for instance in raw.court_instances
    )

    flipped = CourtProcessingGeometry._transform_channels(
        channels,
        matrix=torch.eye(3, dtype=torch.float64),
        output_size_hw=(48, 64),
        horizontal_flipped=True,
    )
    permutation = torch.tensor(_FLIP, dtype=torch.long)
    torch.testing.assert_close(
        flipped.points_xy,
        channels.points_xy.index_select(0, permutation),
    )
    torch.testing.assert_close(
        flipped.point_visible,
        channels.point_visible.index_select(0, permutation),
    )
    torch.testing.assert_close(
        flipped.physical_indices,
        channels.physical_indices.index_select(0, permutation),
    )


def test_v4_requires_and_preserves_safe_path_digest(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path, schema="v4")
    input_layer = _input(tmp_path, schema="v4")

    assert input_layer.spec.source_schema == "canonical_court_dataset_v4"
    assert input_layer.spec.keypoint_schema == "synthetic_camera_view_kp14_v4"
    raw = input_layer.load(input_layer.records("train")[0])
    assert raw.pose_authority is not None
    assert raw.pose_authority.source_schema == "canonical_court_dataset_v4"

    labels_path = input_layer.records("train")[0].annotation_path

    def remove_safety_digest(labels: dict[str, object]) -> None:
        labels.pop("safety_support_input_digest")

    _rewrite(labels_path, remove_safety_digest)
    with pytest.raises(ValueError, match="schema/fields changed"):
        input_layer.load(input_layer.records("train")[0])


def test_v3_parser_preserves_court_sample_001588_serialized_precision(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path, schema="v3")
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    projection = cast(dict[str, object], record["projection"])
    courts = cast(list[dict[str, object]], projection["courts"])
    classes = cast(list[dict[str, object]], courts[1]["classes"])
    point = cast(list[dict[str, object]], classes[5]["points"])[0]
    serialized_uv = [387122.0463825724, 48177.58336823048]
    point["uv"] = serialized_uv
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["projection"] = deepcopy(projection)
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    input_layer = _input(
        tmp_path,
        schema="v3",
        keypoint_court_scope="target_court",
    )
    raw = input_layer.load(input_layer.records("train")[0])

    assert raw.keypoint_channels is not None
    assert raw.keypoint_channels.points_xy.dtype == torch.float64
    torch.testing.assert_close(
        raw.keypoint_channels.points_xy[5, 0],
        torch.tensor(serialized_uv, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert raw.court_instances[1].points_xy.dtype == torch.float32


def test_v3_target_scope_preserves_distinct_bundle_identity_and_physical_mapping(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path, schema="v3")
    all_input = _input(tmp_path, schema="v3")
    target_input = _input(
        tmp_path,
        schema="v3",
        keypoint_court_scope="target_court",
    )

    all_raw = all_input.load(all_input.records("train")[0])
    target_raw = target_input.load(target_input.records("train")[0])
    assert all_raw.keypoint_channels is not None
    assert target_raw.keypoint_channels is not None
    assert (
        target_input.spec.keypoint_schema
        == "synthetic_camera_view_kp14_v3_target_court"
    )
    assert target_raw.keypoint_channels.points_xy.shape == (14, 1, 2)
    assert target_raw.keypoint_channels.physical_indices[:, 0].tolist() == list(
        _CAMERA_VIEW_PHYSICAL
    )
    assert [instance.court_instance_id for instance in target_raw.court_instances] == [
        "court-a",
        "court-b",
    ]
    assert (
        target_input.records("train")[0].dense_target_refs
        == all_input.records("train")[0].dense_target_refs
    )


@pytest.mark.parametrize(
    ("artifact_schema", "selected_schema"),
    [("v2", "v3"), ("v3", "v2")],
)
def test_v2_v3_artifacts_are_never_cross_accepted(
    tmp_path: Path,
    artifact_schema: Literal["v2", "v3"],
    selected_schema: Literal["v2", "v3"],
) -> None:
    _write_v2_dataset(tmp_path, schema=artifact_schema)

    with pytest.raises(ValueError, match="selected schema"):
        _input(tmp_path, schema=selected_schema)


def test_v3_rejects_legacy_mapping(tmp_path: Path) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path, schema="v3")
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    projection = cast(dict[str, object], record["projection"])
    courts = cast(list[dict[str, object]], projection["courts"])
    classes = cast(list[dict[str, object]], courts[1]["classes"])
    for class_id, physical_index in enumerate(_OPPOSITE_PHYSICAL):
        point = cast(list[dict[str, object]], classes[class_id]["points"])[0]
        point["physical_index"] = physical_index
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["projection"] = deepcopy(projection)
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_layer = _input(tmp_path, schema="v3")

    with pytest.raises(ValueError, match="physical indices"):
        input_layer.load(input_layer.records("train")[0])


def test_v3_accepts_finite_lateral_projected_u_reversal(tmp_path: Path) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path, schema="v3")
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    projection = cast(dict[str, object], record["projection"])
    courts = cast(list[dict[str, object]], projection["courts"])
    classes = cast(list[dict[str, object]], courts[1]["classes"])
    left = cast(list[dict[str, object]], classes[2]["points"])[0]
    right = cast(list[dict[str, object]], classes[3]["points"])[0]
    left_u = cast(list[float], left["uv"])[0]
    cast(list[float], right["uv"])[0] = left_u - 1.0
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["projection"] = deepcopy(projection)
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_layer = _input(tmp_path, schema="v3")

    loaded = input_layer.load(input_layer.records("train")[0])

    assert loaded.keypoint_channels is not None
    assert (
        loaded.keypoint_channels.points_xy[2, 1, 0]
        > (loaded.keypoint_channels.points_xy[3, 1, 0])
    )


def test_v3_rejects_nonfinite_projected_uv(tmp_path: Path) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path, schema="v3")
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    projection = cast(dict[str, object], record["projection"])
    courts = cast(list[dict[str, object]], projection["courts"])
    classes = cast(list[dict[str, object]], courts[1]["classes"])
    point = cast(list[dict[str, object]], classes[0]["points"])[0]
    cast(list[float], point["uv"])[0] = float("nan")
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["projection"] = deepcopy(projection)
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        _input(tmp_path, schema="v3")


@pytest.mark.parametrize(
    ("transform_owner", "mutation", "message"),
    [
        ("camera", "missing", "missing=.*camera_to_scene"),
        ("camera", "nonfinite", "finite JSON"),
        ("court", "missing", "missing=.*scene_from_court"),
        ("court", "nonfinite", "finite JSON"),
    ],
)
def test_v3_rejects_missing_or_nonfinite_camera_and_court_transforms(
    tmp_path: Path,
    transform_owner: Literal["camera", "court"],
    mutation: Literal["missing", "nonfinite"],
    message: str,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path, schema="v3")
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    if transform_owner == "camera":
        owner = cast(dict[str, object], record["camera"])
        field = "camera_to_scene"
    else:
        target = cast(dict[str, object], record["target_court"])
        owner = cast(dict[str, object], target["binding"])
        field = "scene_from_court"
    if mutation == "missing":
        del owner[field]
    else:
        transform = cast(list[float], owner[field])
        transform[3] = float("nan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _input(tmp_path, schema="v3")


def test_v2_target_scope_selects_exact_bound_court_and_keeps_dense_inventory(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path)
    all_input = _input(tmp_path)
    target_input = _input(tmp_path, keypoint_court_scope="target_court")

    all_record = all_input.records("train")[0]
    target_record = target_input.records("train")[0]
    all_raw = all_input.load(all_record)
    target_raw = target_input.load(target_record)
    assert all_raw.keypoint_channels is not None
    assert target_raw.keypoint_channels is not None

    all_channels = all_raw.keypoint_channels
    target_channels = target_raw.keypoint_channels
    assert target_input.spec.source_schema == all_input.spec.source_schema
    assert (
        target_input.spec.keypoint_schema
        == "synthetic_camera_relative_kp14_target_court"
    )
    assert target_channels.points_xy.shape == (14, 1, 2)
    assert target_channels.point_visible.shape == (14, 1)
    assert target_channels.physical_indices.shape == (14, 1)
    torch.testing.assert_close(
        target_channels.points_xy[:, 0], all_channels.points_xy[:, 1]
    )
    torch.testing.assert_close(
        target_channels.point_visible[:, 0], all_channels.point_visible[:, 1]
    )
    torch.testing.assert_close(
        target_channels.physical_indices[:, 0], all_channels.physical_indices[:, 1]
    )
    assert target_record.dense_target_refs == all_record.dense_target_refs
    assert (
        target_record.payload["source_target_sha256"]
        == all_record.payload["source_target_sha256"]
    )
    assert [instance.court_instance_id for instance in target_raw.court_instances] == [
        "court-a",
        "court-b",
    ]
    for target_instance, all_instance in zip(
        target_raw.court_instances,
        all_raw.court_instances,
        strict=True,
    ):
        torch.testing.assert_close(target_instance.points_xy, all_instance.points_xy)
        torch.testing.assert_close(
            target_instance.point_visible, all_instance.point_visible
        )


def test_v2_target_scope_is_independent_of_projection_order(tmp_path: Path) -> None:
    selected_points: list[torch.Tensor] = []
    for name, order in (
        ("target-first", ("court-b", "court-a")),
        ("target-last", ("court-a", "court-b")),
    ):
        root = tmp_path / name
        _write_v2_dataset(root, court_order=order)
        input_layer = _input(root, keypoint_court_scope="target_court")

        raw = input_layer.load(input_layer.records("train")[0])

        assert raw.keypoint_channels is not None
        assert [instance.court_instance_id for instance in raw.court_instances] == list(
            order
        )
        selected_points.append(raw.keypoint_channels.points_xy)

    torch.testing.assert_close(selected_points[0], selected_points[1])


def test_v2_target_scope_keeps_all_invisible_target_without_fallback(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path, invisible_court_id="court-b")
    all_input = _input(tmp_path)
    target_input = _input(tmp_path, keypoint_court_scope="target_court")

    all_raw = all_input.load(all_input.records("train")[0])
    target_raw = target_input.load(target_input.records("train")[0])
    assert all_raw.keypoint_channels is not None
    assert target_raw.keypoint_channels is not None

    target_channels = target_raw.keypoint_channels
    assert target_channels.points_xy.shape == (14, 1, 2)
    assert not bool(target_channels.point_visible.any())
    torch.testing.assert_close(
        target_channels.points_xy[:, 0],
        all_raw.keypoint_channels.points_xy[:, 1],
    )
    assert target_channels.physical_indices[:, 0].tolist() == list(_OPPOSITE_PHYSICAL)


def test_v2_target_scope_flip_preserves_semantic_and_physical_identity(
    tmp_path: Path,
) -> None:
    _write_v2_dataset(tmp_path)
    input_layer = _input(tmp_path, keypoint_court_scope="target_court")
    raw = input_layer.load(input_layer.records("train")[0])
    assert raw.keypoint_channels is not None

    channels = raw.keypoint_channels
    flipped = CourtProcessingGeometry._transform_channels(
        channels,
        matrix=torch.eye(3, dtype=torch.float64),
        output_size_hw=(48, 64),
        horizontal_flipped=True,
    )

    permutation = torch.tensor(_FLIP, dtype=torch.long)
    torch.testing.assert_close(
        flipped.points_xy, channels.points_xy.index_select(0, permutation)
    )
    torch.testing.assert_close(
        flipped.point_visible, channels.point_visible.index_select(0, permutation)
    )
    torch.testing.assert_close(
        flipped.physical_indices,
        channels.physical_indices.index_select(0, permutation),
    )
    assert sorted(flipped.physical_indices[:, 0].tolist()) == list(range(14))
    assert [instance.court_instance_id for instance in raw.court_instances] == [
        "court-a",
        "court-b",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "canonical_court_dataset_v3", "selected schema"),
        ("status", "running", "completed"),
    ],
)
def test_v2_rejects_manifest_schema_and_publication_status(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _input(tmp_path)


def test_selected_v1_never_infers_v2_from_directory(tmp_path: Path) -> None:
    _write_v2_dataset(tmp_path)

    with pytest.raises(ValueError, match="selected schema"):
        _input(tmp_path, schema="v1")


def test_v2_rejects_root_escape_and_symlinked_published_files(tmp_path: Path) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    records[0]["rgb"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative|escapes"):
        _input(tmp_path)

    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    rgb_path = manifest_path.parent / cast(str, records[0]["rgb"])
    rgb_path.unlink()
    external = tmp_path / "external.npy"
    np.save(external, np.zeros((48, 64, 3), np.float32))
    rgb_path.symlink_to(external)
    with pytest.raises(ValueError, match="escapes|ordinary"):
        _input(tmp_path)


def test_v2_rejects_manifest_root_symlink_escape_before_read(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external_workspace = tmp_path / "external"
    _write_v2_dataset(external_workspace)
    (workspace_root / "B00").symlink_to(
        external_workspace / "B00", target_is_directory=True
    )

    with pytest.raises(ValueError, match="manifest root.*workspace_root"):
        _input(workspace_root)


def test_v2_never_uses_preview_when_authoritative_rgb_is_missing(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    (manifest_path.parent / cast(str, record["rgb"])).unlink()

    with pytest.raises(FileNotFoundError, match="rgb"):
        _input(tmp_path)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((48, 64, 3), dtype=np.float64),
        np.full((48, 64, 3), 1.1, dtype=np.float32),
        np.full((48, 64, 3), np.nan, dtype=np.float32),
        np.zeros((47, 64, 3), dtype=np.float32),
    ],
)
def test_v2_rejects_noncanonical_rgb(tmp_path: Path, rgb: np.ndarray) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    np.save(manifest_path.parent / cast(str, record["rgb"]), rgb)
    input_layer = _input(tmp_path)

    with pytest.raises(ValueError, match="RGB"):
        input_layer.load(input_layer.records("train")[0])


@pytest.mark.parametrize(("value", "expected"), [(0.0, 0), (1.0, 255)])
def test_v2_accepts_rgb_unit_interval_boundaries(
    tmp_path: Path,
    value: float,
    expected: int,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    np.save(
        manifest_path.parent / cast(str, record["rgb"]),
        np.full((48, 64, 3), value, dtype=np.float32),
    )
    input_layer = _input(tmp_path)

    raw = input_layer.load(input_layer.records("train")[0])

    assert np.asarray(raw.image).min() == expected
    assert np.asarray(raw.image).max() == expected


def test_v2_rejects_labels_manifest_drift_and_mixed_sample_schema(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    labels_path = manifest_path.parent / cast(str, record["labels"])
    _rewrite(labels_path, lambda labels: labels.__setitem__("view_id", "changed"))
    input_layer = _input(tmp_path)
    with pytest.raises(ValueError, match="view_id"):
        input_layer.load(input_layer.records("train")[0])

    _write_v2_dataset(tmp_path)
    _rewrite(
        labels_path,
        lambda labels: labels.__setitem__("schema", "canonical_court_sample_v1"),
    )
    input_layer = _input(tmp_path)
    with pytest.raises(ValueError, match="schema"):
        input_layer.load(input_layer.records("train")[0])


@pytest.mark.parametrize(
    "failure",
    [
        "absent_target",
        "duplicate_target_instance",
        "non_target_class",
        "non_target_duplicate_physical",
    ],
)
def test_v2_rejects_invalid_target_and_physical_inventory(
    tmp_path: Path,
    failure: str,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    record = records[0]
    if failure == "absent_target":
        target = cast(dict[str, object], record["target_court"])
        binding = cast(dict[str, object], target["binding"])
        binding["court_instance_id"] = "court-missing"
    elif failure == "duplicate_target_instance":
        projection = cast(dict[str, object], record["projection"])
        courts = cast(list[dict[str, object]], projection["courts"])
        courts[0]["court_instance_id"] = "court-b"
    elif failure == "non_target_class":
        projection = cast(dict[str, object], record["projection"])
        courts = cast(list[dict[str, object]], projection["courts"])
        classes = cast(list[dict[str, object]], courts[0]["classes"])
        classes[0]["class_id"] = 99
    else:
        projection = cast(dict[str, object], record["projection"])
        courts = cast(list[dict[str, object]], projection["courts"])
        classes = cast(list[dict[str, object]], courts[0]["classes"])
        first = cast(list[dict[str, object]], classes[0]["points"])[0]
        second = cast(list[dict[str, object]], classes[1]["points"])[0]
        second["physical_index"] = first["physical_index"]
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["target_court"] = deepcopy(record["target_court"])
    labels["projection"] = deepcopy(record["projection"])
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_layer = _input(tmp_path, keypoint_court_scope="target_court")

    with pytest.raises(ValueError, match="target_court|instance|class|physical"):
        input_layer.load(input_layer.records("train")[0])


@pytest.mark.parametrize("failure", ["missing_binding", "invalid_binding_id"])
def test_v2_target_scope_rejects_missing_or_invalid_binding(
    tmp_path: Path,
    failure: str,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    manifest_target = cast(dict[str, object], record["target_court"])
    labels_target = cast(dict[str, object], labels["target_court"])
    if failure == "missing_binding":
        manifest_target.pop("binding")
        labels_target.pop("binding")
    else:
        manifest_binding = cast(dict[str, object], manifest_target["binding"])
        labels_binding = cast(dict[str, object], labels_target["binding"])
        manifest_binding["court_instance_id"] = 7
        labels_binding["court_instance_id"] = 7
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="target_court|binding|court"):
        _input(tmp_path, keypoint_court_scope="target_court")


def test_v2_target_scope_rejects_labels_manifest_binding_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    record = cast(list[dict[str, object]], manifest["samples"])[0]
    labels_path = manifest_path.parent / cast(str, record["labels"])

    def _change_binding(labels: dict[str, object]) -> None:
        target = cast(dict[str, object], labels["target_court"])
        binding = cast(dict[str, object], target["binding"])
        binding["court_instance_id"] = "court-a"

    _rewrite(labels_path, _change_binding)
    input_layer = _input(tmp_path, keypoint_court_scope="target_court")

    with pytest.raises(ValueError, match="target_court disagrees with manifest"):
        input_layer.load(input_layer.records("train")[0])


@pytest.mark.parametrize(
    ("matrix_index", "replacement", "message"),
    [
        (15, 0.0, "bottom row"),
        (0, 2.0, "orthonormal"),
        (0, -1.0, "proper rotation"),
    ],
    ids=("bad_homogeneous_bottom_row", "scaled_rotation", "determinant_minus_one"),
)
def test_v2_rejects_invalid_target_rigid_transform_during_construction(
    tmp_path: Path,
    matrix_index: int,
    replacement: float,
    message: str,
) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    record = records[0]
    target = cast(dict[str, object], record["target_court"])
    binding = cast(dict[str, object], target["binding"])
    matrix = binding["scene_from_court"]
    assert isinstance(matrix, list)
    matrix[matrix_index] = replacement

    labels_path = manifest_path.parent / cast(str, record["labels"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    labels["target_court"] = deepcopy(record["target_court"])
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _input(tmp_path)


def test_v2_rejects_split_leakage_and_empty_split(tmp_path: Path) -> None:
    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    records[1]["trajectory_group_id"] = records[0]["trajectory_group_id"]
    labels_path = manifest_path.parent / cast(str, records[1]["labels"])
    _rewrite(
        labels_path,
        lambda labels: labels.__setitem__(
            "trajectory_group_id", records[0]["trajectory_group_id"]
        ),
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        _input(tmp_path)

    manifest_path, manifest = _write_v2_dataset(tmp_path)
    records = cast(list[dict[str, object]], manifest["samples"])
    manifest["samples"] = records[:2]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        _input(tmp_path)
