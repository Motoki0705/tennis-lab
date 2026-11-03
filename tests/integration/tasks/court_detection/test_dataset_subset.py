"""Ordered subset, original validation and checkpoint identity integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.court_detection.ablation.evaluation import evaluate_checkpoint
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtDatasetSubsetSourceConfig,
)
from src.tasks.court_detection.data.contracts import CourtSampleRecord
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.scaling.provenance import data_condition
from src.tasks.court_detection.scaling.real_split import (
    make_original_val_registry,
    real_source,
)
from src.tasks.court_detection.scaling.training import (
    build_step_trainer,
    step_checkpoints,
)
from src.tasks.court_detection.training.runner import (
    CourtDetectionTrainingRunner,
    resolve_training_config,
)
from src.utils.data.image_record_store import file_sha256
from tests.integration.tasks.court_detection.test_composable_pipeline import (
    _configure_tiny_model,
)
from tests.integration.tasks.court_detection.test_scale_dataset_version import (
    scale_version as scale_version,
)
from tests.integration.tasks.court_detection.test_scale_dataset_version import (
    version_config,
)
from tests.unit.synthetic_data_generation.dataset.court.conftest import (
    captured_cameras as captured_cameras,
)
from tests.unit.synthetic_data_generation.dataset.court.conftest import (
    multi_court_layout as multi_court_layout,
)

pytestmark = pytest.mark.integration


def _subset_config(tmp_path, scale_version):
    path, _, _ = scale_version
    config = version_config(tmp_path, path)
    pool = build_court_input(CourtTrainingConfig.from_config(config).data.source)
    records = pool.records("train")
    assert len(records) > 2
    groups: dict[str, list[CourtSampleRecord]] = {}
    for record in records:
        groups.setdefault(str(record.payload["trajectory_group_id"]), []).append(record)
    assert len(groups) >= 2
    first, second = list(groups.values())[:2]
    assert len(first) >= 3
    records = (first[1], first[0])
    rows = []
    for record in records:
        scene_id, sample_id = record.sample_id.split(":", 1)
        rows.append({"scene_id": scene_id, "sample_id": sample_id})
    subset = path.parent / "selected.subset.json"
    subset.write_text(
        json.dumps(
            {
                "schema": "court_dataset_subset_v1",
                "source_manifest": path.name,
                "source_manifest_sha256": file_sha256(path),
                "selection_seed": 42,
                "split": "train",
                "samples": rows,
            }
        )
    )
    validation = path.parent / "validation.subset.json"
    document = json.loads(subset.read_text())
    scene_id, sample_id = second[0].sample_id.split(":", 1)
    document.update(
        split="val", samples=[{"scene_id": scene_id, "sample_id": sample_id}]
    )
    validation.write_text(json.dumps(document))
    config.data.source = OmegaConf.create(
        {
            "kind": "synthetic_court_dataset_subset",
            "schema": "v3",
            "court_scope": "target_court",
            "manifest": subset.relative_to(tmp_path / "data").as_posix(),
            "validation_manifest": validation.relative_to(tmp_path / "data").as_posix(),
        }
    )
    root = tmp_path / "data/court_detection/tennis_court_detector-v1"
    registry = make_original_val_registry(real_source(root), version="original-val")
    (tmp_path / "artifacts/real/registry.json").write_text(json.dumps(registry))
    return config, subset, list(records)


def test_subset_preserves_order_and_supervision(tmp_path, scale_version):
    config, _, records = _subset_config(tmp_path, scale_version)
    resolve_training_config(config)
    source = build_court_input(CourtTrainingConfig.from_config(config).data.source)
    assert source.records("train") == tuple(records)
    assert source.available_splits == ("train", "val")
    with pytest.raises(ValueError, match="no.*test"):
        source.records("test")
    assert source.records("val")[0].split == "val"
    source.load(source.records("val")[0])
    raw = source.load(records[0])
    assert raw.sample_id == records[0].sample_id
    dm = CourtDetectionDataModule(config)
    dm.setup("fit")
    condition = data_condition(dm)
    assert condition["accepted_images"] == 2
    assert condition["scene_images"] == {"fixture": 2}
    assert condition["generation_seed"] == 24
    assert condition["dataset_subset"]["selection_seed"] == 42
    assert len(dm.val_dataset) == 3
    assert condition["synthetic_validation"]["images"] == 1
    with pytest.raises(ValueError, match="no 'test'"):
        dm.source_eval_dataloader("tennis_court_detector", "test")
    batch = next(iter(dm.train_dataloader()))
    assert batch["pose_supervision_mask"].all()


@pytest.mark.parametrize(
    "mutation", ["hash", "unknown", "duplicate", "empty", "escape", "schema", "split"]
)
def test_subset_rejects_invalid_membership(tmp_path, scale_version, mutation):
    config, path, _ = _subset_config(tmp_path, scale_version)
    doc = json.loads(path.read_text())
    if mutation == "hash":
        doc["source_manifest_sha256"] = "0" * 64
    elif mutation == "unknown":
        doc["samples"][0]["sample_id"] = "absent"
    elif mutation == "duplicate":
        doc["samples"].append(doc["samples"][0])
    elif mutation == "empty":
        doc["samples"] = []
    elif mutation == "escape":
        doc["source_manifest"] = "../outside.json"
    elif mutation == "schema":
        doc["schema"] = "unknown"
    else:
        doc["split"] = "test"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build_court_input(CourtTrainingConfig.from_config(config).data.source)


def test_subset_cpu_train_and_exact_checkpoint_evaluation(
    tmp_path, scale_version, monkeypatch
):
    torch.set_num_threads(1)
    config, selection, _ = _subset_config(tmp_path, scale_version)
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
    runtime = CourtTrainingConfig.from_config(config)
    runtime.shared.run.output_dir.mkdir(parents=True)
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
    result = evaluate_checkpoint(
        checkpoint,
        tmp_path / "evaluated",
        device="cpu",
        expected_step=6,
        real_split="val",
        series="scene-count",
        include_synthetic_val=True,
    )
    assert result["splits"]["synthetic_court-val"]["count"] == 1
    assert result["splits"]["synthetic_court-val"]["pose_supervised"]
    assert result["global_step"] == 6
    assert result["data_condition"]["accepted_images"] == 2
    assert result["splits"]["tennis_court_detector-val"]["count"] == 2
    assert result["schema"] == "court_data_scaling_evaluation_v1"
    # A second selection from the validation trajectory replaces the saved val,
    # while the checkpoint still authenticates the original train/val manifests.
    source = dm._train_pipelines["synthetic_court"].input_layer
    training_group = source.records("train")[0].payload["trajectory_group_id"]
    candidates = [
        record
        for record in source.source.records("train")
        if record.payload["trajectory_group_id"] != training_group
    ]
    selected = candidates[:3][::-1]
    assert len(selected) == 3
    document = json.loads(selection.read_text())
    document.update(
        split="val",
        samples=[
            dict(
                zip(
                    ("scene_id", "sample_id"),
                    record.sample_id.split(":", 1),
                    strict=True,
                )
            )
            for record in selected
        ],
    )
    manifest = selection.parent / "shared-evaluation.subset.json"
    manifest.write_text(json.dumps(document))
    shared = evaluate_checkpoint(
        checkpoint,
        tmp_path / "shared-evaluation",
        device="cpu",
        expected_step=6,
        real_split="val",
        series="scene-count",
        synthetic_eval_manifest=manifest,
    )
    assert shared["schema"] == "court_data_scaling_evaluation_v2"
    synthetic = shared["splits"]["synthetic_court-val"]
    expected_ids = [record.sample_id for record in selected]
    assert synthetic["sample_ids"] == expected_ids
    assert synthetic["count"] == 3
    assert (
        synthetic["ordered_sample_ids_sha256"]
        == hashlib.sha256(
            json.dumps(expected_ids, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert synthetic["scenes"]["fixture"]["sample_ids"] == expected_ids
    for name, value in synthetic["scene_macro_mean"].items():
        # Lightning serializes logged whole-split scalars through float32.
        assert value == pytest.approx(synthetic["metrics"][name])
    assert shared["data_condition"] == result["data_condition"]
    assert shared["synthetic_evaluation_manifest"]["sha256"] == file_sha256(manifest)
    assert shared["data_manifest_sha256"][
        "synthetic_court/evaluation_subset"
    ] == file_sha256(manifest)
    assert (
        shared["splits"]["tennis_court_detector-val"]
        == result["splits"]["tennis_court_detector-val"]
    )
    assert (
        json.loads((tmp_path / "shared-evaluation/evaluation.json").read_text())
        == shared
    )
    doc = json.loads(selection.read_text())
    doc["samples"].reverse()
    selection.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="changed since checkpoint"):
        evaluate_checkpoint(
            checkpoint,
            tmp_path / "drifted",
            device="cpu",
            expected_step=6,
            real_split="val",
            series="scene-count",
            include_synthetic_val=True,
        )


def test_subset_rejects_disjoint_frames_on_the_same_trajectory(tmp_path, scale_version):
    config, selection, records = _subset_config(tmp_path, scale_version)
    source_config = CourtTrainingConfig.from_config(config).data.source
    assert isinstance(source_config, SyntheticCourtDatasetSubsetSourceConfig)
    selected = build_court_input(source_config)
    ids = {record.sample_id for record in records}
    group = records[0].payload["trajectory_group_id"]
    candidate = next(
        record
        for record in selected.source.records("train")
        if record.payload["trajectory_group_id"] == group
        and record.sample_id not in ids
    )
    scene, sample = candidate.sample_id.split(":", 1)
    path = source_config.validation_manifest
    document = json.loads(path.read_text())
    document["samples"] = [{"scene_id": scene, "sample_id": sample}]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="trajectory leakage"):
        build_court_input(source_config)
