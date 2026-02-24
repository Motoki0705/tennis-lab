"""MAE training runner extending BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.experiments.mae.data import MAEDataModule
from src.experiments.mae.data.producer import CacheProducerConfig, PreprocessConfig
from src.experiments.mae.training import MAELightningModule
from src.experiments.mae.training.epoch_cache_callback import (
    EpochCacheCallbackConfig,
    MAEEpochCacheCallback,
)


class MAETrainingRunner(BaseTrainingRunner):
    """Training runner for MAE pre-training task.

    MAE config follows the shared `config.run.*` schema for runtime settings.
    """

    # ---- config access helpers ----
    def _get_seed(self, config: Any) -> int | None:
        run_cfg = config.get("run", {})
        return run_cfg.get("seed", 42)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return MAEDataModule.from_config(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return MAELightningModule.from_config(config)

    def checkpoint_monitor(self, config: Any) -> str:
        data_cfg = config.get("data", {})
        val_split = float(data_cfg.get("val_split", 0.1))
        checkpoint_cfg = config.get("checkpoint", {})
        monitor = checkpoint_cfg.get("monitor", "val/loss")
        if val_split == 0.0 and str(monitor).startswith("val/"):
            return "train/loss"
        return str(monitor)

    def checkpoint_prefix(self, config: Any) -> str:
        return "mae"

    def early_stopping_enabled(self, config: Any) -> bool:
        return bool(config.get("early_stopping", {}).get("enabled", False))

    def early_stopping_patience(self, config: Any) -> int:
        return int(config.get("early_stopping", {}).get("patience", 50))

    def lr_monitor_interval(self, config: Any) -> str:
        return "epoch"

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        data_cfg = config.get("data", {})
        mode = str(data_cfg.get("mode", "video"))
        if mode != "cached_batches":
            return []

        preprocess_cfg = data_cfg.get("preprocess", None)
        preprocess_dict = (
            OmegaConf.to_container(preprocess_cfg, resolve=True)
            if preprocess_cfg is not None
            else {}
        )
        pp_cfg = PreprocessConfig(**(preprocess_dict or {}))

        run_cfg = config.get("run", {})
        producer_cfg = CacheProducerConfig(
            cache_root=str(data_cfg.get("cache_root", "data/mae/cache")),
            video_dir=str(data_cfg.get("video_dir", "data/tennis/raw/videos")),
            use_decord=bool(data_cfg.get("use_decord", True)),
            min_frames=int(data_cfg.get("min_frames", 10)),
            samples_per_video=int(data_cfg.get("samples_per_video", 4)),
            buckets=tuple(
                int(x)
                for x in data_cfg.get(
                    "buckets", [256, 320, 384, 448, 512, 640, 768, 1024]
                )
            ),
            bucket_alpha=float(data_cfg.get("bucket_alpha", 2.0)),
            base_batch_size=int(data_cfg.get("base_batch_size", 32)),
            min_batch_size=int(data_cfg.get("min_batch_size", 1)),
            upsample_limit=float(data_cfg.get("upsample_limit", 1.0)),
            frame_sample_ratio=float(data_cfg.get("frame_sample_ratio", 1.0)),
            seed=int(run_cfg.get("seed", 42)),
            val_split=float(data_cfg.get("val_split", 0.1)),
            static_val=True,
            preprocess=pp_cfg,
        )
        return [MAEEpochCacheCallback(EpochCacheCallbackConfig(producer=producer_cfg))]

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        training_cfg = config.get("training", {})
        kwargs: dict[str, Any] = {
            "precision": training_cfg.get(
                "precision", "16-mixed" if accelerator == "gpu" else 32
            ),
            "accumulate_grad_batches": training_cfg.get("accumulate_grad_batches", 1),
            "log_every_n_steps": training_cfg.get("log_every_n_steps", 50),
            "val_check_interval": training_cfg.get("val_check_interval", 1.0),
            "enable_progress_bar": training_cfg.get("enable_progress_bar", True),
            "strategy": training_cfg.get("strategy", "auto"),
        }
        # MAE uses non-deterministic by default for performance
        if not training_cfg.get("deterministic", False):
            kwargs["deterministic"] = False
        return kwargs

    def skip_test(self, config: Any) -> bool:
        # MAE pre-training doesn't have a test phase
        return True

    def is_dry_run(self, config: Any) -> bool:
        # MAE uses run.fast_dev_run for dry run behavior
        run_cfg = config.get("run", {})
        return bool(run_cfg.get("fast_dev_run", False))

    def seed_everything(self, config: Any) -> None:
        seed = self._get_seed(config)
        if seed is not None:
            pl.seed_everything(int(seed), workers=True)

    def build_trainer(
        self, config: Any, callbacks: list[Any], logger: TensorBoardLogger
    ) -> pl.Trainer:
        """Build trainer with MAE-specific configuration."""
        accelerator, devices = self.select_devices(config)
        run_cfg = config.get("run", {})
        training_cfg = config.get("training", {})

        base_kwargs: dict[str, Any] = {
            "max_epochs": int(training_cfg.get("max_epochs", 400)),
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "logger": logger,
            "gradient_clip_val": training_cfg.get("gradient_clip_val", 1.0),
            "fast_dev_run": bool(run_cfg.get("fast_dev_run", False)),
        }

        extra_kwargs = self.trainer_kwargs(config, accelerator, devices)
        base_kwargs.update(extra_kwargs)
        return pl.Trainer(**base_kwargs)
