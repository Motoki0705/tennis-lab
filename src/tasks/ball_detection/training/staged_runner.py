"""Training runner for the staged multi-frame schedule (issue #579).

Differences from the base ball-detection runner:

* builds :class:`StagedBallDataModule` and the manual-optimization
  :class:`StagedBallDetectionLightningModule`;
* on a real (non dry-run) GPU start, calibrates ``B(T)`` via an OOM probe and
  injects the resulting batch plan (``EBS = B(1)``) into both;
* loads model weights from the previous phase via ``run.init_weights`` (a
  weights-only transfer, NOT a full-state resume, so each phase starts a fresh
  optimizer/schedule at epoch 0);
* pins the TensorBoard logger version so each phase's ``last.ckpt`` lands at a
  deterministic path the next phase can point ``run.init_weights`` at.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from src.tasks.ball_detection.training.staged_calibration import probe_batch_size_by_t
from src.tasks.ball_detection.training.staged_lightning_module import (
    StagedBallDetectionLightningModule,
)


class StagedBallDetectionTrainingRunner(BallDetectionTrainingRunner):
    """Runner for the 4-phase staged training schedule."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        datamodule = StagedBallDataModule(config)
        if not bool(config.run.dry_run):
            self._calibrate(config, datamodule)
        return datamodule

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        module = StagedBallDetectionLightningModule(config)
        if not isinstance(datamodule, StagedBallDataModule):
            raise TypeError("Staged runner requires StagedBallDataModule.")
        module.set_effective_batch_size(datamodule.effective_batch_size)
        return module

    def build_logger(self, config: Any, output_dir: Any) -> TensorBoardLogger:
        # Fixed version => deterministic checkpoint dir for phase chaining:
        #   <output_dir>/logs/run/checkpoints/last.ckpt
        return TensorBoardLogger(save_dir=str(output_dir), name="logs", version="run")

    # ------------------------------------------------------------------
    def _calibrate(self, config: Any, datamodule: StagedBallDataModule) -> None:
        if not torch.cuda.is_available():
            print("[staged] CUDA unavailable; using config batch_size_by_t as-is.")
            return
        staged_cfg = config.training.staged
        token_budget = int(staged_cfg.calibration_token_budget)
        safety = float(staged_cfg.calibration_safety)
        t_values = sorted(datamodule.t_probs)
        print(
            f"[staged] calibrating B(T) for T={t_values} (token_budget={token_budget})..."
        )
        table = probe_batch_size_by_t(
            config,
            t_values,
            device=torch.device("cuda"),
            token_budget=token_budget,
            safety=safety,
        )
        effective_batch = (
            datamodule.effective_batch_size
            if datamodule.t_distribution == "fixed"
            else table[min(table)]
        )
        datamodule.set_batch_plan(
            {**datamodule.batch_size_by_t, **table},
            effective_batch,
        )
        print(f"[staged] calibrated B(T)={table}; EBS={effective_batch}")


__all__ = ["StagedBallDetectionTrainingRunner"]
