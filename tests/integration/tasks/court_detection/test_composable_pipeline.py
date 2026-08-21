"""Integration contracts for real Court source/target compositions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image

from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
    COURT_SAMPLE_SCHEMA,
)
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.target_generation.materializer import (
    CourtTargetMaterializer,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.utils.schema.court import (
    COURT_KP_NAMES,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

pytestmark = pytest.mark.integration

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _image_points() -> list[list[float]]:
    metric = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    x_coord = (metric[:, 0] / 12.0 + 0.5) * 63.0
    y_coord = (0.5 - metric[:, 1] / 26.0) * 47.0
    return cast("list[list[float]]", torch.stack((x_coord, y_coord), dim=1).tolist())


def _write_tennis_court_detector(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8)).save(
        root / "images" / "court.png"
    )
    payload = [{"id": "court", "kps": _image_points()}]
    for split in ("train", "val"):
        (root / f"data_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _projection() -> dict[str, object]:
    points = _image_points()
    classes: list[dict[str, object]] = []
    for class_id, (class_name, physical_indices) in enumerate(
        zip(SEMANTIC_CLASS_NAMES, PHYSICAL_INDICES_BY_CLASS, strict=True)
    ):
        class_points = [
            {
                "physical_index": physical_index,
                "uv": points[physical_index],
                "camera_depth_m": 10.0,
                "scene_xyz_m": [0.0, 0.0, 0.0],
                "in_front": True,
                "in_frame": True,
                "renderer_visible": True,
            }
            for physical_index in physical_indices
        ]
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "renderer_visible": True,
                "points": class_points,
            }
        )
    return {
        "camera_id": "camera-0",
        "resolution": [64, 48],
        "coverage_modes": ["full"],
        "visible_class_names": list(SEMANTIC_CLASS_NAMES),
        "visible_point_count": 14,
        "courts": [
            {
                "court_instance_id": "court-0",
                "coverage_mode": "full",
                "classes": classes,
            }
        ],
    }


def _write_synthetic_court(workspace_root: Path) -> None:
    root = workspace_root / "B00" / "datasets" / "court"
    projection = _projection()
    camera = {"camera_id": "camera-0"}
    records: list[dict[str, object]] = []
    for sample_index, split in enumerate(("train", "validation", "test")):
        sample_id = f"sample-{split}"
        relative = Path("samples") / sample_id
        sample_root = root / relative
        sample_root.mkdir(parents=True)
        np.save(
            sample_root / "rgb.npy",
            np.full((48, 64, 3), 0.5, dtype=np.float32),
        )
        metadata = {"fixture": True}
        labels = {
            "schema": COURT_SAMPLE_SCHEMA,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "trajectory_group_id": "group-0",
            "trajectory_id": "trajectory-0",
            "view_id": "view-0",
            "trajectory_frame_index": sample_index,
            "split": split,
            "camera": camera,
            "projection": projection,
            "metadata": metadata,
        }
        (sample_root / "labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )
        records.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "trajectory_group_id": "group-0",
                "trajectory_id": "trajectory-0",
                "view_id": "view-0",
                "trajectory_frame_index": sample_index,
                "split": split,
                "shard_id": "shard-0",
                "width": 64,
                "height": 48,
                "camera": camera,
                "projection": projection,
                "directory": relative.as_posix(),
                "rgb": (relative / "rgb.npy").as_posix(),
                "rgb_preview": (relative / "rgb.png").as_posix(),
                "alpha": (relative / "alpha.npy").as_posix(),
                "alpha_preview": (relative / "alpha.png").as_posix(),
                "depth": (relative / "depth.npy").as_posix(),
                "depth_coordinate_space": "camera",
                "labels": (relative / "labels.json").as_posix(),
                "metadata": metadata,
            }
        )
    manifest = {
        "schema": COURT_DATASET_SCHEMA,
        "status": "completed",
        "scene_id": "B00",
        "profile": "fixture",
        "seed": 714,
        "sampling_policy": {},
        "metadata_fields": [],
        "trajectory_groups": [],
        "samples": records,
        "rejected_samples": [],
        "metrics": {},
        "diagnostics": {},
    }
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _v2_target() -> dict[str, object]:
    return {
        "binding": {
            "court_instance_id": "court-1",
            "candidate_id": "candidate-1",
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
        "camera_to_court_center_distance_m": 20.0,
    }


def _v2_camera(sample_id: str, sample_index: int) -> dict[str, object]:
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


def _v2_projection(sample_id: str) -> dict[str, object]:
    points = _image_points()
    physical_orders = (
        tuple(range(14)),
        (2, 3, 0, 1, 5, 4, 7, 6, 10, 11, 8, 9, 13, 12),
    )
    courts: list[dict[str, object]] = []
    for court_index, physical_order in enumerate(physical_orders):
        classes: list[dict[str, object]] = []
        for class_id, (class_name, physical_index) in enumerate(
            zip(COURT_KP_NAMES[:14], physical_order, strict=True)
        ):
            uv = list(points[physical_index])
            uv[0] += float(court_index)
            classes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "renderer_visible": True,
                    "points": [
                        {
                            "physical_index": physical_index,
                            "uv": uv,
                            "camera_depth_m": 10.0,
                            "scene_xyz_m": [float(physical_index), 0.0, 0.0],
                            "in_front": True,
                            "in_frame": True,
                            "renderer_visible": True,
                        }
                    ],
                }
            )
        courts.append(
            {
                "court_instance_id": f"court-{court_index}",
                "coverage_mode": "full",
                "classes": classes,
            }
        )
    return {
        "camera_id": sample_id,
        "resolution": [64, 48],
        "coverage_modes": ["full", "full"],
        "visible_class_names": list(COURT_KP_NAMES[:14]),
        "visible_point_count": 28,
        "courts": courts,
    }


def _write_synthetic_court_v2(workspace_root: Path) -> None:
    root = workspace_root / "V2" / "datasets" / "court"
    records: list[dict[str, object]] = []
    for sample_index, split in enumerate(("train", "validation", "test")):
        sample_id = f"v2-{split}"
        relative = Path("samples") / sample_id
        sample_root = root / relative
        sample_root.mkdir(parents=True)
        np.save(sample_root / "rgb.npy", np.full((48, 64, 3), 0.5, np.float32))
        np.save(sample_root / "alpha.npy", np.ones((48, 64, 1), np.float32))
        np.save(sample_root / "depth.npy", np.ones((48, 64, 1), np.float32))
        Image.new("RGB", (64, 48)).save(sample_root / "rgb.png")
        Image.new("L", (64, 48)).save(sample_root / "alpha.png")
        projection = _v2_projection(sample_id)
        target = _v2_target()
        camera = _v2_camera(sample_id, sample_index)
        metadata = {"fixture": "v2"}
        labels = {
            "schema": "canonical_court_sample_v2",
            "sample_index": sample_index,
            "sample_id": sample_id,
            "trajectory_group_id": f"v2-group-{split}",
            "trajectory_id": f"v2-trajectory-{split}",
            "view_id": "view-0",
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
                "trajectory_group_id": f"v2-group-{split}",
                "trajectory_id": f"v2-trajectory-{split}",
                "view_id": "view-0",
                "trajectory_frame_index": 0,
                "split": split,
                "shard_id": "shard-0",
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
    manifest = {
        "schema": "canonical_court_dataset_v2",
        "status": "completed",
        "scene_id": "V2",
        "profile": "fixture-v2",
        "seed": 761,
        "sampling_policy": {},
        "metadata_fields": [],
        "trajectory_groups": [],
        "samples": records,
        "rejected_samples": [],
        "metrics": {},
        "diagnostics": [],
    }
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _compose(tmp_path: Path, *, source: str, processing: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                f"data/source={source}",
                f"data/processing={processing}",
            ],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    config.paths.checkpoint_root = "checkpoints"
    config.paths.artifact_root = "artifacts"
    config.paths.output_root = "outputs"
    config.paths.cache_root = "cache"
    config.paths.external_asset_root = "external"
    config.data.batch_size = 1
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.augmentation.train_scales = [32]
    config.data.augmentation.val_short_side = 32
    if source == "synthetic_court":
        config.data.source.scene_ids = ["V2"]
    return config


def _source_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _materialize(config: DictConfig) -> None:
    runtime = CourtTrainingConfig.from_config(config)
    store = CourtDerivedTargetStore(runtime.data.processing.derived_target_root)
    input_layer = build_court_input(runtime.data.source, target_store=store)
    CourtTargetMaterializer(
        input_layer=input_layer,
        target_store=store,
    ).materialize(
        splits=input_layer.available_splits,
        target_kinds=("seg", "line"),
    )


@pytest.fixture
def court_roots(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_tennis_court_detector(data_root / "court")
    _write_synthetic_court(data_root / "synthetic_data_generation" / "scenes")
    _write_synthetic_court_v2(
        data_root / "synthetic_data_generation" / "scenes"
    )
    return tmp_path


@pytest.mark.parametrize(
    ("source", "processing", "channels"),
    [
        ("tennis_court_detector", "kp", 14),
        ("tennis_court_detector", "seg", 7),
        ("tennis_court_detector", "line", 1),
        ("synthetic_court_v1", "kp", 7),
        ("synthetic_court_v1", "seg", 7),
        ("synthetic_court_v1", "line", 1),
        ("synthetic_court", "kp", 14),
        ("synthetic_court", "seg", 7),
        ("synthetic_court", "line", 1),
    ],
)
def test_real_single_target_dataset_dataloader_paths(
    court_roots: Path,
    source: str,
    processing: CourtTargetKind,
    channels: int,
) -> None:
    config = _compose(court_roots, source=source, processing=processing)
    _materialize(config)
    datamodule = CourtDetectionDataModule(config)
    datamodule.setup("validate")

    batch = next(iter(datamodule.val_dataloader()))

    assert set(cast(Mapping[str, object], batch["targets"])) == {processing}
    assert datamodule.target_bundle_spec.targets[processing].output_channels == channels
    assert cast(torch.Tensor, batch["image"]).shape == (1, 3, 32, 48)


@pytest.mark.parametrize(
    ("source", "kp_channels"),
    [
        ("tennis_court_detector", 14),
        ("synthetic_court_v1", 7),
        ("synthetic_court", 14),
    ],
)
def test_real_three_target_dataset_dataloader_contract(
    court_roots: Path,
    source: str,
    kp_channels: int,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    _materialize(config)
    datamodule = CourtDetectionDataModule(config)
    datamodule.setup("validate")

    batch = next(iter(datamodule.val_dataloader()))
    targets = cast(Mapping[str, object], batch["targets"])
    kp = cast(Mapping[str, torch.Tensor], targets["kp"])

    assert tuple(targets) == ("kp", "seg", "line")
    assert kp["heatmap"].shape == (1, kp_channels, 32, 48)
    assert kp["point_visible"].dtype == torch.bool
    assert cast(torch.Tensor, targets["seg"]).shape == (1, 32, 48)
    assert cast(torch.Tensor, targets["seg"]).dtype == torch.long
    assert cast(torch.Tensor, targets["line"]).shape == (1, 1, 32, 48)


@pytest.mark.parametrize(
    ("processing", "expected"),
    [
        ("kp_seg", ("kp", "seg")),
        ("kp_line", ("kp", "line")),
        ("seg_line", ("seg", "line")),
    ],
)
def test_real_two_target_v2_dataset_dataloader_contract(
    court_roots: Path,
    processing: str,
    expected: tuple[CourtTargetKind, CourtTargetKind],
) -> None:
    config = _compose(
        court_roots,
        source="synthetic_court",
        processing=processing,
    )
    _materialize(config)
    datamodule = CourtDetectionDataModule(config)
    datamodule.setup("validate")

    batch = next(iter(datamodule.val_dataloader()))

    assert tuple(cast(Mapping[str, object], batch["targets"])) == expected


@pytest.mark.parametrize("processing", ["seg", "line", "seg_line"])
def test_missing_dense_targets_fail_during_setup_before_worker_start(
    court_roots: Path,
    processing: str,
) -> None:
    config = _compose(
        court_roots,
        source="synthetic_court",
        processing=processing,
    )
    config.data.num_workers = 2
    datamodule = CourtDetectionDataModule(config)

    with pytest.raises(FileNotFoundError, match="Precomputed Court"):
        datamodule.setup("validate")

    assert datamodule.val_dataset is None


@pytest.mark.parametrize(
    "source",
    ["tennis_court_detector", "synthetic_court_v1", "synthetic_court"],
)
def test_materialization_preserves_both_source_trees(
    court_roots: Path,
    source: str,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    source_root = (
        court_roots / "data/court"
        if source == "tennis_court_detector"
        else court_roots / "data/synthetic_data_generation/scenes"
    )
    before = _source_files(source_root)

    _materialize(config)

    assert _source_files(source_root) == before
    derived_kind = (
        "tennis_court_detector"
        if source == "tennis_court_detector"
        else "synthetic_court"
    )
    derived = court_roots / "data/court_detection/derived_targets" / derived_kind
    expected_file_count = 4 if source == "tennis_court_detector" else 6
    assert len(tuple(derived.rglob("*.png"))) == expected_file_count
    assert len(tuple(derived.rglob("*.json"))) == expected_file_count


def test_shared_geometry_keeps_kp_and_line_correspondence(
    court_roots: Path,
) -> None:
    config = _compose(
        court_roots,
        source="tennis_court_detector",
        processing="all",
    )
    _materialize(config)
    datamodule = CourtDetectionDataModule(config)
    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader()))
    targets = cast(
        Mapping[str, object], batch["targets"]
    )
    kp = cast(Mapping[str, torch.Tensor], targets["kp"])
    line = cast(torch.Tensor, targets["line"])[0, 0]
    image_height, image_width = cast(torch.Tensor, batch["image_size"])[0]
    points = kp["points_xy"][0, :, 0] * torch.stack(
        (image_width - 1, image_height - 1)
    ).float()
    visible = kp["point_visible"][0, :, 0]

    for point in points[visible]:
        x_pos, y_pos = (int(round(float(value))) for value in point)
        y_start, y_end = max(0, y_pos - 1), min(line.shape[0], y_pos + 2)
        x_start, x_end = max(0, x_pos - 1), min(line.shape[1], x_pos + 2)
        assert bool(line[y_start:y_end, x_start:x_end].any())


@pytest.mark.parametrize(
    ("source", "kp_channels"),
    [
        ("tennis_court_detector", 14),
        ("synthetic_court_v1", 7),
        ("synthetic_court", 14),
    ],
)
def test_datamodule_bound_three_head_forward_loss_backward(
    court_roots: Path,
    source: str,
    kp_channels: int,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    _materialize(config)
    datamodule = CourtDetectionDataModule(config)
    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader()))
    pair = build_court_detection_pair(
        config,
        target_bundle=datamodule.target_bundle_spec,
    )

    adapter = cast(CourtModelIOAdapter, pair.adapter)
    call = adapter.prepare_training_batch(batch)
    logits = pair.model(*call.model_call.model_args)
    result = adapter.training_result(logits, call)
    result.loss.backward()

    assert {kind: value.shape[1] for kind, value in logits.items()} == {
        "kp": kp_channels,
        "seg": 7,
        "line": 1,
    }
    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in pair.model.parameters())
