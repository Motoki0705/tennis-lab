"""Integration contracts for real Court source/target compositions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image

from src.synthetic_data_generation.dataset.court.schema import (
    COURT_DATASET_SCHEMA_V2,
    COURT_DATASET_SCHEMA_V3,
    COURT_SAMPLE_SCHEMA_V2,
    COURT_SAMPLE_SCHEMA_V3,
)
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.training.runner import (
    resolve_training_config,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    COURT_KP_NAMES,
    OPPOSITE_COURT_END_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)
from tests.unit.tasks.court_detection.data.inputs.fixtures import (
    pack_tennis_fixture,
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
    for split in ("train", "val"):
        sample_id = f"court_{split}"
        Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8)).save(
            root / "images" / f"{sample_id}.png"
        )
        payload = [{"id": sample_id, "kps": _image_points(), "metric": 0.25}]
        (root / f"data_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _singleton_target() -> dict[str, object]:
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


def _singleton_camera(sample_id: str, sample_index: int) -> dict[str, object]:
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


def _singleton_projection(
    sample_id: str,
    *,
    schema: Literal["v2", "v3"],
) -> dict[str, object]:
    points = _image_points()
    physical_orders = (
        tuple(range(14)),
        (OPPOSITE_COURT_END_INDEX if schema == "v2" else CAMERA_VIEW_HALF_TURN_INDEX),
    )
    courts: list[dict[str, object]] = []
    for court_index, physical_order in enumerate(physical_orders):
        classes: list[dict[str, object]] = []
        for class_id, (class_name, physical_index) in enumerate(
            zip(COURT_KP_NAMES[:14], physical_order, strict=True)
        ):
            uv = list(points[physical_index])
            if schema == "v3" and court_index == 1:
                uv[0] = 63.0 - uv[0]
            uv[0] += float(court_index * 8)
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


def _write_synthetic_court_singleton(
    workspace_root: Path,
    *,
    schema: Literal["v2", "v3"],
) -> None:
    scene_id = schema.upper()
    dataset_schema = (
        COURT_DATASET_SCHEMA_V2 if schema == "v2" else COURT_DATASET_SCHEMA_V3
    )
    sample_schema = COURT_SAMPLE_SCHEMA_V2 if schema == "v2" else COURT_SAMPLE_SCHEMA_V3
    root = workspace_root / scene_id / "datasets" / "court"
    records: list[dict[str, object]] = []
    for sample_index, split in enumerate(("train", "validation", "test")):
        sample_id = f"{schema}-{split}"
        relative = Path("samples") / sample_id
        sample_root = root / relative
        sample_root.mkdir(parents=True)
        np.save(sample_root / "rgb.npy", np.full((48, 64, 3), 0.5, np.float32))
        np.save(sample_root / "alpha.npy", np.ones((48, 64, 1), np.float32))
        np.save(sample_root / "depth.npy", np.ones((48, 64, 1), np.float32))
        Image.new("RGB", (64, 48)).save(sample_root / "rgb.png")
        Image.new("L", (64, 48)).save(sample_root / "alpha.png")
        projection = _singleton_projection(sample_id, schema=schema)
        target = _singleton_target()
        camera = _singleton_camera(sample_id, sample_index)
        metadata = {"fixture": schema}
        labels = {
            "schema": sample_schema,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "trajectory_group_id": f"{schema}-group-{split}",
            "trajectory_id": f"{schema}-trajectory-{split}",
            "view_id": "view-0",
            "trajectory_frame_index": 0,
            "split": split,
            "camera": camera,
            "projection": projection,
            "target_court": target,
            "metadata": metadata,
        }
        (sample_root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        records.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "trajectory_group_id": f"{schema}-group-{split}",
                "trajectory_id": f"{schema}-trajectory-{split}",
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
        "schema": dataset_schema,
        "status": "completed",
        "scene_id": scene_id,
        "profile": f"fixture-{schema}",
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


def _compose(
    tmp_path: Path,
    *,
    source: str,
    processing: str,
    court_scope: Literal["all_courts", "target_court"] | None = None,
) -> DictConfig:
    overrides = [
        "loss=default",
        "data/augmentation=default",
        f"data/source={source}",
        f"data/processing={processing}",
    ]
    if court_scope is not None:
        overrides.append(f"data.source.court_scope={court_scope}")
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=overrides,
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
        config.data.source.scene_ids = ["V3"]
    elif source == "tennis_court_detector":
        config.data.source.excluded_sample_ids = []
    return config


def _compose_mixed(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data/augmentation=pose_safe",
                # This fixture tests dense data mixing without a pose model.
                "loss=default",
                "mixed.sources.tennis_court_detector.excluded_sample_ids=[]",
                "run.output_dir=court_detection/mixed-source/integration-test",
            ],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    config.paths.checkpoint_root = "checkpoints"
    config.paths.artifact_root = "artifacts"
    config.paths.output_root = "outputs"
    config.paths.cache_root = "cache"
    config.paths.external_asset_root = "external"
    config.data.source.scene_ids = ["V3"]
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.augmentation.train_scales = [32]
    config.data.augmentation.val_short_side = 32
    config.mixed.train_batch_counts.synthetic_court = 1
    config.mixed.train_batch_counts.tennis_court_detector = 1
    return config


def _source_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def court_roots(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_tennis_court_detector(data_root / "upstream")
    pack_tennis_fixture(
        data_root / "upstream", data_root / "court_detection/tennis_court_detector-v1"
    )
    singleton_root = data_root / "synthetic_data_generation" / "scenes"
    _write_synthetic_court_singleton(singleton_root, schema="v3")
    return tmp_path


def test_mixed_datamodule_uses_both_real_input_pipelines_in_each_batch(
    court_roots: Path,
) -> None:
    config = _compose_mixed(court_roots)
    standard, mixed = resolve_training_config(config)
    datamodule = CourtDetectionDataModule(standard, mixed_config=mixed)
    datamodule.setup(None)

    batch = next(iter(datamodule.train_dataloader()))
    metadata = cast(list[Mapping[str, object]], batch["metadata"])
    targets = cast(Mapping[str, object], batch["targets"])
    kp = cast(Mapping[str, torch.Tensor], targets["kp"])

    assert [item["source_kind"] for item in metadata].count("synthetic_court") == 1
    assert [item["source_kind"] for item in metadata].count(
        "tennis_court_detector"
    ) == 1
    assert tuple(targets) == ("kp", "seg", "line", "semantic_line")
    assert datamodule.target_bundle_spec.targets["kp"].channel_names == tuple(
        COURT_KP_NAMES[:14]
    )
    assert kp["heatmap"].shape == (2, 14, 32, 32)
    assert kp["points_xy"].shape == (2, 14, 1, 2)
    torch.testing.assert_close(
        cast(torch.Tensor, batch["pose_supervision_mask"]),
        torch.tensor([False, False]),
    )

    test_batch = next(iter(datamodule.test_dataloader()))
    test_metadata = cast(list[Mapping[str, object]], test_batch["metadata"])
    assert {item["source_kind"] for item in test_metadata} == {"synthetic_court"}


@pytest.mark.parametrize(
    ("source", "processing", "channels"),
    [
        ("tennis_court_detector", "kp", 14),
        ("tennis_court_detector", "seg", 7),
        ("tennis_court_detector", "line", 1),
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
    batch, bundle = _source_batch(config)

    assert set(cast(Mapping[str, object], batch["targets"])) == {processing}
    assert bundle.targets[processing].output_channels == channels
    assert cast(torch.Tensor, batch["image"]).shape == (1, 3, 32, 48)


@pytest.mark.parametrize(
    ("source", "kp_channels"),
    [
        ("tennis_court_detector", 14),
        ("synthetic_court", 14),
    ],
)
def test_real_three_target_dataset_dataloader_contract(
    court_roots: Path,
    source: str,
    kp_channels: int,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    batch, bundle = _source_batch(config)
    targets = cast(Mapping[str, object], batch["targets"])
    kp = cast(Mapping[str, torch.Tensor], targets["kp"])

    assert tuple(targets) == ("kp", "seg", "line", "semantic_line")
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
def test_real_two_target_v3_dataset_dataloader_contract(
    court_roots: Path,
    processing: str,
    expected: tuple[CourtTargetKind, CourtTargetKind],
) -> None:
    config = _compose(
        court_roots,
        source="synthetic_court",
        processing=processing,
    )
    batch, bundle = _source_batch(config)

    assert tuple(cast(Mapping[str, object], batch["targets"])) == expected


@pytest.mark.parametrize("processing", ["seg", "line", "seg_line"])
def test_dense_targets_are_generated_in_workers_without_disk_targets(
    court_roots: Path,
    processing: str,
) -> None:
    config = _compose(
        court_roots,
        source="synthetic_court",
        processing=processing,
    )
    config.data.num_workers = 2
    batch, _ = _source_batch(config)
    assert set(cast(Mapping[str, object], batch["targets"])) == {
        target.kind
        for target in CourtTrainingConfig.from_config(config).data.processing.targets
    }
    assert not (court_roots / "data/court_detection/derived_targets").exists()


@pytest.mark.parametrize(
    "source",
    [
        "tennis_court_detector",
        "synthetic_court",
    ],
)
def test_online_targets_preserve_source_trees_without_writing_masks(
    court_roots: Path,
    source: str,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    source_root = (
        court_roots / "data/court_detection/tennis_court_detector-v1"
        if source == "tennis_court_detector"
        else court_roots / "data/synthetic_data_generation/scenes"
    )
    before = _source_files(source_root)
    _source_batch(config)
    assert _source_files(source_root) == before
    assert not (court_roots / "data/court_detection/derived_targets").exists()


def test_shared_geometry_keeps_kp_and_line_correspondence(
    court_roots: Path,
) -> None:
    config = _compose(
        court_roots,
        source="tennis_court_detector",
        processing="all",
    )
    batch, bundle = _source_batch(config)
    targets = cast(Mapping[str, object], batch["targets"])
    kp = cast(Mapping[str, torch.Tensor], targets["kp"])
    line = cast(torch.Tensor, targets["line"])[0, 0]
    image_height, image_width = cast(torch.Tensor, batch["image_size"])[0]
    points = (
        kp["points_xy"][0, :, 0]
        * torch.stack((image_width - 1, image_height - 1)).float()
    )
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
        ("synthetic_court", 14),
    ],
)
def test_pipeline_bound_four_head_forward_loss_backward(
    monkeypatch: pytest.MonkeyPatch,
    court_roots: Path,
    source: str,
    kp_channels: int,
) -> None:
    config = _compose(court_roots, source=source, processing="all")
    batch, bundle = _source_batch(config)
    from src.tasks.court_detection.models import dinov3_dpt
    from tests.unit.tasks.court_detection.models.test_dinov3_dpt import (
        FakeDINOv3,
        _encoder,
    )

    monkeypatch.setattr(
        dinov3_dpt, "build_court_encoder", lambda **kwargs: _encoder(FakeDINOv3())
    )
    config.model.transformer_encoder.dim = 8
    config.model.transformer_encoder.depth = 1
    config.model.transformer_encoder.num_heads = 2
    config.model.transformer_encoder.head_dim = 4
    config.model.transformer_encoder.rope_dim = 4
    config.model.transformer_encoder.ffn_dim = 16
    config.model.decoder.size = "tiny"
    config.model.decoder.channels = 64
    config.model.dense_head.normalization_groups = 2
    for kind in ("kp", "seg", "line", "semantic_line"):
        config.model.dense_head[kind].hidden_channels = 8
        config.model.dense_head[kind].depth = 1
    pair = build_court_detection_pair(
        config,
        target_bundle=bundle,
    )

    adapter = cast(CourtModelIOAdapter, pair.adapter)
    call = adapter.prepare_training_batch(batch)
    logits = pair.model(*call.model_call.model_args)
    result = adapter.training_result(logits, call)
    result.loss.backward()

    assert {kind: value.shape[1] for kind, value in logits.dense_logits.items()} == {
        "kp": kp_channels,
        "seg": 7,
        "line": 1,
        "semantic_line": 12,
    }
    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in pair.model.parameters())


def _source_batch(
    config: DictConfig,
) -> tuple[dict[str, object], CourtTargetBundleSpec]:
    from functools import partial

    from torch.utils.data import DataLoader

    from src.tasks.court_detection.data.collate import court_detection_collate
    from src.tasks.court_detection.data.dataset import CourtDetectionDataset
    from src.tasks.court_detection.data.processing.factory import (
        build_court_processing_pipeline,
    )

    runtime = CourtTrainingConfig.from_config(config)
    pipeline = build_court_processing_pipeline(runtime.data, is_train=False)
    dataset = CourtDetectionDataset(
        pipeline.input_layer.records("val"), pipeline=pipeline
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=runtime.data.num_workers,
        collate_fn=partial(court_detection_collate, bundle=pipeline.target_bundle_spec),
    )
    return next(iter(loader)), pipeline.target_bundle_spec


def test_v3_target_binding_is_shared_by_all_dense_teachers(court_roots: Path) -> None:
    config = _compose(court_roots, source="synthetic_court", processing="all")
    runtime = CourtTrainingConfig.from_config(config)
    source = build_court_input(runtime.data.source)
    raw = source.load(source.records("val")[0])
    assert [court.court_instance_id for court in raw.court_instances] == ["court-1"]
    assert raw.keypoint_channels is not None
    assert raw.keypoint_channels.points_xy.shape == (14, 1, 2)
    assert (
        tuple(raw.keypoint_channels.physical_indices[:, 0].tolist())
        == CAMERA_VIEW_HALF_TURN_INDEX
    )
    batch, _ = _source_batch(config)
    targets = cast(Mapping[str, torch.Tensor], batch["targets"])
    assert targets["seg"].any()
    assert targets["line"].any()
    assert targets["semantic_line"].any()
