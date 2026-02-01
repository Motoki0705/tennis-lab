"""MAE training runner extending BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.base.training.runner import BaseTrainingRunner
from src.mae.data import MAEDataModule
from src.mae.data.producer import CacheProducerConfig, PreprocessConfig
from src.mae.training import MAELightningModule
from src.mae.training.epoch_cache_callback import (
    EpochCacheCallbackConfig,
    MAEEpochCacheCallback,
)


class MAETrainingRunner(BaseTrainingRunner):
    """Training runner for MAE pre-training task.

    **Configuration Exception Policy**:
    MAE uses a flattened config structure (seed, trainer, training at top-level)
    instead of the standard nested config.run.* schema used by other tasks
    (WASB, PLCS, BLCS, Court Detection).

    This design choice aligns with MAE's unique requirements:
    - Hydra-managed working directory (no explicit output_dir)
    - Direct PyTorch Lightning Trainer parameter exposure
    - No test phase (pre-training only)
    - Epoch-based caching callbacks

    See docs/config_architecture.md for the full exception policy and rationale.

    This runner overrides base methods to translate MAE's config layout to
    the base runner interface.
    """

    # ---- config access helpers ----
    def _get_seed(self, config: Any) -> int | None:
        return config.get("seed", 42)

    def _get_gpus(self, config: Any) -> int:
        trainer_cfg = config.get("trainer", {})
        devices = trainer_cfg.get("devices", "auto")
        if devices == "auto" or devices is None:
            return 1
        return int(devices) if isinstance(devices, (int, str)) and str(devices).isdigit() else 1

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
            seed=int(config.get("seed", 42)),
            val_split=float(data_cfg.get("val_split", 0.1)),
            static_val=True,
            preprocess=pp_cfg,
        )
        return [MAEEpochCacheCallback(EpochCacheCallbackConfig(producer=producer_cfg))]

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        trainer_cfg = config.get("trainer", {})
        kwargs: dict[str, Any] = {
            "precision": trainer_cfg.get(
                "precision", "16-mixed" if accelerator == "gpu" else 32
            ),
            "accumulate_grad_batches": trainer_cfg.get("accumulate_grad_batches", 1),
            "log_every_n_steps": trainer_cfg.get("log_every_n_steps", 50),
            "val_check_interval": trainer_cfg.get("val_check_interval", 1.0),
            "enable_progress_bar": trainer_cfg.get("enable_progress_bar", True),
            "strategy": trainer_cfg.get("strategy", "auto"),
        }
        # MAE uses non-deterministic by default for performance
        if not trainer_cfg.get("deterministic", False):
            kwargs["deterministic"] = False
        return kwargs

    def skip_test(self, config: Any) -> bool:
        # MAE pre-training doesn't have a test phase
        return True

    def prepare_output_dir(self, config: Any) -> Path:
        # MAE uses Hydra-managed output directory (cwd)
        return Path.cwd()

    def is_dry_run(self, config: Any) -> bool:
        # MAE uses trainer.fast_dev_run for dry run behavior
        trainer_cfg = config.get("trainer", {})
        return bool(trainer_cfg.get("fast_dev_run", False))

    def seed_everything(self, config: Any) -> None:
        seed = self._get_seed(config)
        if seed is not None:
            pl.seed_everything(int(seed), workers=True)

    def select_devices(self, config: Any) -> tuple[str, int]:
        trainer_cfg = config.get("trainer", {})
        accelerator = str(trainer_cfg.get("accelerator", "auto"))
        devices = trainer_cfg.get("devices", "auto")

        if accelerator == "auto":
            import torch
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        if devices == "auto" or devices is None:
            devices = 1
        elif isinstance(devices, str) and devices.isdigit():
            devices = int(devices)
        elif not isinstance(devices, int):
            devices = 1

        return accelerator, devices

    def build_trainer(
        self, config: Any, callbacks: list[Any], logger: TensorBoardLogger
    ) -> pl.Trainer:
        """Build trainer with MAE-specific configuration."""
        accelerator, devices = self.select_devices(config)
        trainer_cfg = config.get("trainer", {})
        training_cfg = config.get("training", {})

        base_kwargs: dict[str, Any] = {
            "max_epochs": int(training_cfg.get("max_epochs", 400)),
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "logger": logger,
            "gradient_clip_val": trainer_cfg.get("gradient_clip_val", 1.0),
            "fast_dev_run": bool(trainer_cfg.get("fast_dev_run", False)),
        }

        extra_kwargs = self.trainer_kwargs(config, accelerator, devices)
        base_kwargs.update(extra_kwargs)
        return pl.Trainer(**base_kwargs)
