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
from src.utils.schema.court import COURT_KP_NAMES

_IDENTITY_PHYSICAL = tuple(range(14))
_OPPOSITE_PHYSICAL = (2, 3, 0, 1, 5, 4, 7, 6, 10, 11, 8, 9, 13, 12)
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
) -> dict[str, object]:
    classes: list[dict[str, object]] = []
    for class_id, (class_name, physical_index) in enumerate(
        zip(COURT_KP_NAMES[:14], physical_order, strict=True)
    ):
        in_front = not (court_id == "court-a" and class_id == 0)
        in_frame = not (court_id == "court-a" and class_id == 2)
        renderer_visible = not (court_id == "court-a" and class_id == 1)
        point = {
            "physical_index": physical_index,
            "uv": [
                float(5 + (physical_index % 4) * 10) + x_offset,
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


def _projection(sample_id: str) -> dict[str, object]:
    courts = [
        _court("court-a", physical_order=_IDENTITY_PHYSICAL, x_offset=0.0),
        _court("court-b", physical_order=_OPPOSITE_PHYSICAL, x_offset=2.0),
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


def _write_v2_dataset(root: Path) -> tuple[Path, dict[str, object]]:
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
        projection = _projection(sample_id)
        target = _target()
        camera = _camera(sample_id, sample_index)
        metadata = {"fixture": True}
        labels = {
            "schema": "canonical_court_sample_v2",
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
        (sample_root / "labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )
        records.append(
            {
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
        )
    manifest: dict[str, object] = {
        "schema": "canonical_court_dataset_v2",
        "status": "completed",
        "scene_id": "B00",
        "profile": "v2-fixture",
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
    root: Path, *, schema: Literal["v1", "v2"] = "v2"
) -> SyntheticCourtInput:
    return SyntheticCourtInput(
        SyntheticCourtSourceConfig(
            kind="synthetic_court",
            schema=schema,
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
    assert input_layer.spec.keypoint_channel_names == COURT_KP_NAMES[:14]
    assert input_layer.spec.keypoint_flip_permutation == _FLIP

    raw = input_layer.load(input_layer.records("train")[0])
    assert raw.keypoint_channels is not None
    channels = raw.keypoint_channels
    assert channels.points_xy.shape == (14, 2, 2)
    assert channels.physical_indices[:, 0].tolist() == list(range(14))
    assert channels.physical_indices[:, 1].tolist() == list(_OPPOSITE_PHYSICAL)
    assert not bool(channels.point_visible[0, 0])  # in_front=False
    assert not bool(channels.point_visible[1, 0])  # renderer_visible=False
    assert not bool(channels.point_visible[2, 0])  # in_frame=False
    assert bool(channels.point_visible[0, 1])
    assert len(raw.court_instances) == 2
    for instance in raw.court_instances:
        assert instance.physical_indices.tolist() == list(range(14))
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
    ["absent_target", "duplicate_target_instance", "duplicate_physical"],
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
    input_layer = _input(tmp_path)

    with pytest.raises(ValueError, match="target_court|instance|physical"):
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
