"""Generate scale publications on CPU and consume them through the training Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import TensorBoardLogger

from src.synthetic_data_generation.dataset.court.sample_store import (
    finish_court_store,
    open_court_store,
)
from src.synthetic_data_generation.dataset.court.scale import publication
from src.synthetic_data_generation.dataset.court.scale.core import (
    canonical_bytes,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.dataset.court.scale.recipes import ProductionRecipe
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.tasks.court_detection.ablation.evaluation import evaluate_checkpoint
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtDatasetVersionSourceConfig,
    SyntheticCourtStoresSourceConfig,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.processing.factory import (
    build_court_processing_pipeline,
)
from src.tasks.court_detection.scaling.provenance import data_condition
from src.tasks.court_detection.scaling.training import (
    build_step_trainer,
    step_checkpoints,
)
from src.tasks.court_detection.training.runner import (
    CourtDetectionTrainingRunner,
    resolve_training_config,
)
from src.utils.data.image_record_store import (
    ImageRecordWriter,
    encode_rgb_jpeg,
    file_sha256,
)
from tests.integration.tasks.court_detection.test_composable_pipeline import (
    _configure_tiny_model,
)
from tests.integration.tasks.court_detection.test_data_scaling import _config, _inputs
from tests.unit.synthetic_data_generation.dataset.court.conftest import (
    captured_cameras as captured_cameras,
)
from tests.unit.synthetic_data_generation.dataset.court.conftest import (
    multi_court_layout as multi_court_layout,
)

pytestmark = pytest.mark.integration
_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


@pytest.fixture
def scale_version(tmp_path, captured_cameras, multi_court_layout, monkeypatch):
    # Reuse the generator's CPU scene fixtures and public publication entrypoints.
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda _: SimpleNamespace(layout=multi_court_layout),
    )
    production = ProductionRecipe.from_dict(
        publication.read_json(
            Path(
                "src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json"
            )
        )
    )
    production = replace(
        production,
        geometry=replace(
            production.geometry,
            width=80,
            height=16,
            base_step_m=8,
        ),
    )
    plan = build_candidate_plan(
        scene_id="fixture",
        method=production.method,
        recipe=production.geometry,
        seed=24,
        trajectory_count=6,
        density_level=0,
        cameras=captured_cameras,
        layout=multi_court_layout,
    )
    root = tmp_path / "data/scale"
    request = publication.freeze_request(
        root,
        plans=[plan],
        assets=[{"scene_id": "fixture", "alignment": {}}],
        renderer={"schema": "fixture_renderer_v1"},
        quality=production.quality,
        production_recipe=production,
    )
    rendered = []

    def render(scene, records):
        for record in records:
            rendered.append(record["sample_id"])
            h, w = record["height"], record["width"]
            yield (
                record["sample_id"],
                NHTRenderArrays(
                    rgb=np.full((h, w, 3), 0.5, dtype=np.float32),
                    alpha=np.full(
                        (h, w, 1), 0 if len(rendered) == 1 else 1, dtype=np.float32
                    ),
                    depth=np.ones((h, w, 1), dtype=np.float32),
                ),
            )

    path = publication.execute_request(root, request, render)
    return path, publication.load_dataset_version(path), rendered


def version_config(tmp_path: Path, path: Path) -> DictConfig:
    config = _inputs(tmp_path)
    config.data.source = OmegaConf.create(
        {
            "kind": "synthetic_court_dataset_version",
            "schema": "v3",
            "court_scope": "target_court",
            "manifest": path.relative_to(tmp_path / "data").as_posix(),
        }
    )
    config.data.augmentation.val_short_side = 80
    config.data.augmentation.train_scales = [80]
    config.data.batch_size = 1
    config.mixed.train_batch_counts.synthetic_court = 1
    config.data.num_workers = 0
    return config


def test_scaling_default_requires_manifest_and_rejects_manual_identity(tmp_path):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_data_scaling")
    assert config.data.source.kind == "synthetic_court_dataset_version"
    assert OmegaConf.is_missing(config.data.source, "manifest")
    config = _config(tmp_path)
    config.data.source = OmegaConf.create(
        {
            "kind": "synthetic_court_dataset_version",
            "schema": "v3",
            "court_scope": "target_court",
            "manifest": "missing.dataset.json",
        }
    )
    for field in ("data_version", "generation_seed", "store_roots"):
        invalid = config.copy()
        with open_dict(invalid.data.source):
            invalid.data.source[field] = "unexpected"
        with pytest.raises(ValueError, match="Unknown configuration key"):
            CourtTrainingConfig.from_config(invalid)


def test_generated_version_dataset_matches_accepted_kp_visibility_pose_and_rgb(
    tmp_path,
    scale_version,
):
    path, version, rendered = scale_version
    config = version_config(tmp_path, path)
    _, mixed = resolve_training_config(config)
    assert mixed.train_batch_counts == {
        "synthetic_court": 1,
        "tennis_court_detector": 0,
    }
    runtime = CourtTrainingConfig.from_config(config)
    assert isinstance(runtime.data.source, SyntheticCourtDatasetVersionSourceConfig)
    pipeline = build_court_processing_pipeline(
        runtime.data, is_train=False, require_pose=True
    )
    source = pipeline.input_layer
    dataset = CourtDetectionDataset(source.records("train"), pipeline=pipeline)
    scene = version["scenes"][0]
    counts = scene["counts"]
    accepted = publication.read_json(path.parent / scene["accepted_ids_path"])[
        "sample_ids"
    ]
    assert counts["geometry_rejected"] > 0 and counts["qa_rejected"] > 0
    assert rendered[0] not in accepted
    assert len(dataset) == counts["qa_accepted"] == len(accepted)
    assert source.available_splits == ("train",)
    assert {r.sample_id for r in dataset.records} == {f"fixture:{s}" for s in accepted}
    stored = {}
    for entry in scene["stores"]:
        store = open_court_store(path.parent / entry["path"])
        for row in range(len(store)):
            raw = store.record(row)
            stored[raw["sample_id"]] = raw, store.rgb(row)
    for index, record in enumerate(dataset.records):
        expected, rgb = stored[record.payload["source_sample_id"]]
        raw = source.load(record)
        assert np.array_equal(np.asarray(raw.image), rgb)
        assert raw.pose_authority is not None and raw.keypoint_channels is not None
        assert raw.pose_authority.camera.to_dict() == expected["camera"]
        assert (
            raw.pose_authority.target_court.to_dict()
            == expected["target_court"]["binding"]
        )
        target_id = expected["target_court"]["binding"]["court_instance_id"]
        target = next(
            c
            for c in expected["projection"]["courts"]
            if c["court_instance_id"] == target_id
        )
        points = [c["points"][0] for c in target["classes"]]
        channels = raw.keypoint_channels
        assert torch.equal(
            channels.points_xy[:, 0],
            torch.tensor([p["uv"] for p in points], dtype=torch.float64),
        )
        assert channels.point_visible[:, 0].tolist() == [
            p["in_front"] and p["in_frame"] and p["renderer_visible"] for p in points
        ]
        assert channels.physical_indices[:, 0].tolist() == [
            p["physical_index"] for p in points
        ]
        item = dataset[index]
        targets = cast(Mapping[str, object], item["targets"])
        assert set(targets) == {"kp", "seg", "line", "semantic_line"}
        kp = cast(Mapping[str, torch.Tensor], targets["kp"])
        dense = [
            cast(torch.Tensor, targets[kind])
            for kind in ("seg", "line", "semantic_line")
        ]
        assert all(
            torch.isfinite(value).all()
            for value in (*dense, kp["heatmap"], kp["points_xy"])
        )
        normalized = (
            channels.points_xy
            / channels.points_xy.new_tensor(
                [expected["width"] - 1, expected["height"] - 1]
            )
        ).float()
        assert torch.equal(kp["points_xy"], normalized)
        # Raw visibility above is unchanged; shared preprocessing also clips
        # supervision to the output pixel-centre bounds, even at identity resize.
        in_image = (normalized >= 0).all(-1) & (normalized <= 1).all(-1)
        assert torch.equal(kp["point_visible"], channels.point_visible & in_image)
        assert torch.equal(kp["physical_indices"], channels.physical_indices)
        seg = cast(torch.Tensor, targets["seg"])
        assert torch.allclose(seg.sum(0), torch.ones_like(seg[0]))
        assert "pose_target" in item
    dm = CourtDetectionDataModule(config)
    dm.setup("fit")
    condition = data_condition(dm)
    assert condition["data_version"] == version["dataset_version"]
    assert condition["generation_seed"] == version["generation_seed"] == 24
    assert condition["accepted_images"] == len(dataset)
    assert condition["scene_images"] == {"fixture": len(dataset)}
    assert condition["dataset_version_manifest"]["sha256"] == file_sha256(path)


def test_cpu_updates_and_checkpoint_evaluation_revalidate_dataset_version(
    tmp_path,
    scale_version,
    monkeypatch,
):
    torch.set_num_threads(1)
    path, version, _ = scale_version
    config = version_config(tmp_path, path)
    _configure_tiny_model(config, monkeypatch)
    config.data.augmentation.train_scales = [32]
    config.data.augmentation.val_short_side = 32
    config.training.trainer.precision = "32-true"
    config.training.trainer.enable_progress_bar = False
    config.training.trainer.enable_model_summary = False
    config.training.compile.enabled = False
    config.training.lr_monitor.enabled = False
    dm = CourtDetectionDataModule(config)
    dm.setup("fit")
    (tmp_path / "outputs/court_detection/train/data-scaling-fixture/s42").mkdir(
        parents=True
    )
    module = CourtDetectionTrainingRunner().build_lightning_module(config, dm)
    logger = TensorBoardLogger(save_dir=str(tmp_path / "training"), name="logs")
    trainer = build_step_trainer(
        config,
        step_checkpoints(config, logger),
        logger,
        accelerator="cpu",
        devices=1,
        artifact_store=None,
    )
    trainer.fit(module, datamodule=dm)
    checkpoint = Path(logger.log_dir) / "checkpoints/update-00000006.ckpt"
    payload = torch.load(checkpoint, weights_only=False)
    assert trainer.global_step == 6
    assert payload["hyper_parameters"]["data_condition"] == data_condition(dm)
    evaluated = evaluate_checkpoint(
        checkpoint,
        tmp_path / "eval-version",
        device="cpu",
        expected_step=6,
        real_split="val",
        series="smoke",
    )
    assert evaluated["data_condition"]["data_version"] == version["dataset_version"]
    assert evaluated["data_manifest_sha256"][
        "synthetic_court/dataset_version"
    ] == file_sha256(path)
    assert set(evaluated["splits"]) == {"tennis_court_detector-val"}


def test_scale_stores_require_the_manifest_source(scale_version):
    path, version, _ = scale_version
    root = path.parent / version["scenes"][0]["stores"][0]["path"]
    legacy = SyntheticCourtStoresSourceConfig(
        "synthetic_court_stores",
        "v3",
        "target_court",
        (root,),
        "manual",
        24,
    )
    with pytest.raises(ValueError, match="fields changed"):
        build_court_input(legacy)


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest",
        "seed",
        "split",
        "membership",
        "descriptor",
        "index",
        "image",
        "sample_split",
    ],
)
def test_training_rejects_tampered_version_and_referenced_content(
    tmp_path,
    scale_version,
    mutation,
):
    path, version, _ = scale_version
    config = version_config(tmp_path, path)
    source = CourtTrainingConfig.from_config(config).data.source
    scene = version["scenes"][0]
    if mutation in {"manifest", "seed", "split"}:
        version.update(
            {
                "manifest": {"dataset_version": "0" * 64},
                "seed": {"generation_seed": 25},
                "split": {"split": "validation"},
            }[mutation]
        )
        if mutation != "manifest":
            version["dataset_version"] = digest(
                {k: v for k, v in version.items() if k != "dataset_version"}
            )
        path.write_bytes(canonical_bytes(version))
    elif mutation == "sample_split":
        entry = scene["stores"][0]
        store = open_court_store(path.parent / entry["path"])
        rewritten = path.parent / "non-train-store"
        with ImageRecordWriter(rewritten / "samples") as writer:
            for row in range(len(store)):
                sample = store.record(row)
                if row == 0:
                    sample["split"] = "test"
                writer.append(encode_rgb_jpeg(store.rgb(row)), sample)
            finish_court_store(rewritten, writer, store.metadata)
        entry.update(
            path=rewritten.relative_to(path.parent).as_posix(),
            dataset_sha256=file_sha256(rewritten / "dataset.json"),
            index_sha256=file_sha256(rewritten / "samples/index.npz"),
        )
        version["dataset_version"] = digest(
            {k: v for k, v in version.items() if k != "dataset_version"}
        )
        path.write_bytes(canonical_bytes(version))
    else:
        root = path.parent / scene["stores"][0]["path"]
        damaged = {
            "membership": path.parent / scene["accepted_ids_path"],
            "descriptor": root / "dataset.json",
            "index": root / "samples/index.npz",
            "image": next((root / "samples/shards").glob("*.bin")),
        }[mutation]
        payload = damaged.read_bytes()
        damaged.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(
        ValueError, match="hash|checksum|train split|frozen request|non-train"
    ):
        build_court_input(source)
