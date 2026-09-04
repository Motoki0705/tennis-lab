"""One-step CPU smoke for the DINO Lightning lifecycle using a fake detector."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, nn

from src.tasks.court_alignment.data.datamodule import GroundCourtDataModule
from src.tasks.court_alignment.training.detr_lightning_module import (
    DinoCourtAlignmentLightningModule,
)
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class _FakeDinoCourtDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[4.0], [-6.0], [-6.0]]))
        self.box_logits = nn.Parameter(
            torch.tensor(
                [[0.0, 0.0, -0.85, -0.85], [1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]]
            )
        )
        self.court = nn.Parameter(
            torch.tensor([[-0.85, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        )

    def forward(
        self,
        image: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Mapping[str, object]:
        del targets
        batch_size = image.shape[0]
        logits = self.logits.unsqueeze(0).expand(batch_size, -1, -1)
        boxes = self.box_logits.sigmoid().unsqueeze(0).expand(batch_size, -1, -1)
        court = self.court.unsqueeze(0).expand(batch_size, -1, -1)
        auxiliary = {
            "pred_logits": logits,
            "pred_boxes": boxes,
            "pred_court_boxes": court,
        }
        return {
            **auxiliary,
            "aux_outputs": [dict(auxiliary)],
        }


def _smoke_config() -> DictConfig:
    root = Path(__file__).resolve().parents[4]
    config_dir = root / "src" / "tasks" / "court_alignment" / "configs"
    overrides = [
        "data.train_samples=1",
        "data.val_samples=1",
        "data.test_samples=1",
        "data.batch_size=1",
        "data.min_courts=1",
        "data.max_courts=1",
        "data.min_scale_px_per_metre=10.0",
        "data.max_scale_px_per_metre=10.0",
        "data.court_margin_px=250.0",
        "training.trainer.max_epochs=1",
        "training.trainer.precision=32-true",
        "training.trainer.enable_progress_bar=false",
        "training.trainer.enable_model_summary=false",
        "training.steps_per_epoch=1",
        "training.warmup_steps=0",
        "training.checkpoint.save_top_k=1",
        "training.lr_monitor.enabled=false",
        "run.gpus=0",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train_dino", overrides=overrides)


def test_fake_dino_one_step_writes_query_prediction_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_dir = tmp_path / "repro"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    config = _smoke_config()
    datamodule = GroundCourtDataModule(**{
        key: value
        for key, value in dict(config.data).items()
        if key != "_target_"
    })
    module = DinoCourtAlignmentLightningModule(
        config,
        model=_FakeDinoCourtDetector(),
    )
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
    assert checkpoint.best_model_path
    test_checkpoint = CourtAlignmentTrainingRunner().test_checkpoint_path(
        config,
        trainer,
    )
    trainer.test(
        module,
        datamodule=datamodule,
        ckpt_path=test_checkpoint,
        weights_only=False,
    )

    prediction_dir = repro_dir / "predictions"
    with np.load(prediction_dir / "pred_test.npz") as bundle:
        assert {
            "sample_id",
            "pred_scores",
            "pred_query_indices",
            "pred_aabb_cxcywh_normalized",
            "pred_centers_px",
            "pred_long_sides_px",
            "pred_axial_vectors",
            "pred_corners_px",
            "pred_rotation_rad",
            "pred_scale_px_per_metre",
            "pred_num_instances",
            "gt_aabb_cxcywh_normalized",
            "gt_court_boxes_normalized",
            "gt_num_instances",
        } <= set(bundle.files)
        assert bundle["pred_scores"].shape == (1, 6)
        assert bundle["pred_corners_px"].shape == (1, 6, 4, 2)
        assert bundle["gt_court_boxes_normalized"].shape == (1, 1, 5)

    metrics: dict[str, Any] = json.loads(
        (prediction_dir / "metrics.json").read_text()
    )
    diagnostics: dict[str, Any] = json.loads(
        (prediction_dir / "diagnostic_metrics.json").read_text()
    )
    assert {
        "instance_f1",
        "instance_count_accuracy",
        "matched_center_mean_error_px",
        "matched_scale_relative_error",
        "matched_axial_angle_mean_error_deg",
        "matched_corner_mean_error_px",
    } <= set(metrics)
    assert {
        "instance_tp",
        "instance_fp",
        "instance_fn",
        "loss_total",
        "loss_class",
        "loss_bbox",
        "loss_giou",
        "loss_scale",
        "loss_axis",
        "loss_class_aux_0",
    } <= set(diagnostics)
