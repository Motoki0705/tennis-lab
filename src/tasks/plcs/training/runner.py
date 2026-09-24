"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.configuration import PLCSTrainingConfig, validate_residual_config
from src.tasks.plcs.model_io import (
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_court_keypoints,
    validate_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.model_io.axial_reference import validate_axial_reference_checkpoint
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)
from src.utils.schema.court_normalization import validate_court_coordinate_normalization


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        if str(config.model.name) == "plcs_view_association":
            from src.tasks.plcs.configuration import validate_association_config

            validate_association_config(config)
        elif config.model.name == "plcs_triangulation_residual":
            validate_residual_config(config)
        else:
            PLCSTrainingConfig.from_config(config)
        super().prepare_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return build_plcs_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return build_plcs_lightning_module(config, steps_per_epoch=steps_per_epoch)

    def maybe_load_init_weights(
        self,
        config: Any,
        lightning_module: pl.LightningModule,
    ) -> None:
        """Validate artifact contracts before weight-only initialization."""
        init_path = config.run.init_weights
        if init_path is not None:
            checkpoint = torch.load(
                init_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid PLCS init_weights checkpoint: {init_path}.")
            if str(lightning_module.config.model.name) in {"plcs_view_association", "plcs_triangulation_residual"}:
                lightning_module.on_load_checkpoint(checkpoint)
                lightning_module.load_state_dict(checkpoint["state_dict"], strict=True)
                return
            runtime = PLCSTrainingConfig.from_config(lightning_module.config)
            validate_court_coordinate_normalization(
                checkpoint,
                artifact="PLCS init_weights checkpoint",
            )
            validate_plcs_checkpoint_court_keypoints(
                checkpoint,
                runtime.court_keypoint_contract,
            )
            validate_axial_reference_checkpoint(
                checkpoint, model_name=runtime.model.name
            )
            if runtime.model.name == "plcs_multiview_axial_reference":
                state_dict = checkpoint.get("state_dict")
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        "Axial reference init_weights requires a complete state_dict."
                    )
                lightning_module.load_state_dict(state_dict, strict=True)
                return
            if runtime.model.name in {
                "plcs_track_query",
                "plcs_track_query_reference",
            }:
                validate_plcs_checkpoint_track_query_reference(
                    checkpoint,
                    resolve_plcs_track_query_reference_contract(
                        runtime.model,
                        runtime.court_keypoint_contract,
                    ),
                )
        super().maybe_load_init_weights(config, lightning_module)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        extras: list[Any] = super().callbacks_extra(config, datamodule, logger)

        if str(config.model.name) in {"plcs_view_association", "plcs_triangulation_residual"}:
            return extras
        runtime = PLCSTrainingConfig.from_config(config)
        if runtime.data.backend != "chunked":
            return extras

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        extras.append(ChunkRotationCallback())
        return extras

    def test_after_fit(
        self,
        trainer: pl.Trainer,
        lightning_module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        callbacks: list[Any],
    ) -> None:
        from src.tasks.plcs.training.residual_lightning_module import (
            ResidualLightningModule,
        )

        if not isinstance(lightning_module, ResidualLightningModule):
            super().test_after_fit(trainer, lightning_module, datamodule, callbacks)
            return
        monitored = [
            c
            for c in callbacks
            if isinstance(c, ModelCheckpoint) and c.monitor == "val/world_mpjpe_m"
        ]
        if (
            len(monitored) != 1
            or not monitored[0].best_model_path
            or monitored[0].best_model_score is None
        ):
            raise RuntimeError(
                "A validation-selected best checkpoint is required for residual testing"
            )
        best = Path(monitored[0].best_model_path)
        if not best.is_file():
            raise FileNotFoundError(best)
        results = trainer.test(
            lightning_module,
            datamodule=datamodule,
            ckpt_path=str(best),
            weights_only=False,
        )
        out = lightning_module.residual_config.runtime.run.output_dir
        (out / "evaluation.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(best),
                    "selection": "minimum val/world_mpjpe_m",
                    "best_epoch_score": float(monitored[0].best_model_score.cpu()),
                    "test": results,
                },
                indent=2,
            )
        )
        print(f"BEST_CHECKPOINT={best}", flush=True)
