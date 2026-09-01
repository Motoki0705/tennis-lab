"""One-step CPU training/test smoke for the procedural KP14 pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from src.tasks.court_alignment.training.lightning_module import (
    CourtAlignmentLightningModule,
)
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _smoke_config() -> DictConfig:
    root = Path(__file__).resolve().parents[4]
    config_dir = root / "src" / "tasks" / "court_alignment" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="smoke")


def test_one_step_writes_standard_prediction_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_dir = tmp_path / "repro"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    cfg = _smoke_config()
    runner = CourtAlignmentTrainingRunner()
    runner.prepare_config(cfg)
    datamodule = runner.build_datamodule(cfg)
    module = runner.build_lightning_module(cfg, datamodule)
    assert isinstance(module, CourtAlignmentLightningModule)

    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[checkpoint],
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, datamodule=datamodule)
    best_model_path = checkpoint.best_model_path
    assert best_model_path
    assert Path(best_model_path).is_file()
    test_checkpoint = runner.test_checkpoint_path(cfg, trainer)
    assert test_checkpoint == "best"
    with runner.resume_checkpoint_load_env(test_checkpoint):
        trainer.test(module, datamodule=datamodule, ckpt_path=test_checkpoint)
    assert trainer.ckpt_path is not None
    assert Path(trainer.ckpt_path).resolve() == Path(best_model_path).resolve()

    prediction_dir = repro_dir / "predictions"
    with np.load(prediction_dir / "pred_test.npz") as bundle:
        expected = {
            "sample_id",
            "pred_peak_keypoints_px",
            "pred_peak_valid",
            "gt_peak_keypoints_px",
            "gt_peak_valid",
            "pred_instance_keypoints_px",
            "pred_instance_centers_px",
            "pred_instance_semantic_count",
            "pred_instance_aggregate_confidence",
            "pred_instance_geometry_residual_px",
            "pred_num_instances",
            "gt_instance_keypoints_px",
            "gt_instance_centers_px",
            "gt_num_instances",
        }
        assert expected <= set(bundle.files)
        assert bundle["sample_id"].shape == (2,)
        assert bundle["pred_peak_keypoints_px"].shape == (2, 14, 4, 2)
        assert bundle["gt_instance_keypoints_px"].shape == (2, 1, 14, 2)
        assert "heatmap_logits" not in bundle.files

    metrics = json.loads((prediction_dir / "metrics.json").read_text())
    diagnostics = json.loads(
        (prediction_dir / "diagnostic_metrics.json").read_text()
    )
    assert {
        "instance_precision",
        "instance_recall",
        "instance_f1",
        "instance_kp_mean_error_px",
        "instance_kp_pck_2px",
        "instance_count_accuracy",
    } <= set(metrics)
    assert {
        "instance_tp",
        "instance_fp",
        "instance_fn",
        "false_positive_count",
        "loss",
        "loss_heatmap",
        "loss_center_vote",
    } <= set(diagnostics)
