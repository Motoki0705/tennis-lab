"""Base training runner with overridable hooks."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class BaseTrainingRunner:
    """Base training runner with overridable hooks for task-specific behavior."""

    def run(self, config: Any) -> None:
        """Run training with the provided config."""
        self.seed_everything(config)

        output_dir = self.prepare_output_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_config(config, output_dir)

        if self.is_dry_run(config):
            self.run_dry_run(config, output_dir)
            return

        datamodule = self.build_datamodule(config)
        steps_per_epoch = self.resolve_steps_per_epoch(
            config, datamodule, train_loader=None
        )
        lightning_module = self.build_lightning_module(
            config, datamodule, steps_per_epoch=steps_per_epoch
        )

        logger = self.build_logger(config, output_dir)
        callbacks = self.build_callbacks(config, datamodule, logger)
        trainer = self.build_trainer(config, callbacks, logger)

        trainer.fit(
            lightning_module,
            datamodule=datamodule,
            ckpt_path=self.resolve_resume(config, output_dir),
        )

        if not self.skip_test(config):
            trainer.test(lightning_module, datamodule=datamodule)

        print(f"Training complete. Outputs saved to {output_dir}")

    # ---- template methods (override as needed) ----
    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        raise NotImplementedError

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        raise NotImplementedError

    def prepare_output_dir(self, config: Any) -> Path:
        return Path(self._ensure_absolute(str(config.run.output_dir)))

    def resolve_resume(self, config: Any, output_dir: Path) -> str | None:
        resume = getattr(config.run, "resume", None)
        if not resume:
            return None
        return self._ensure_absolute(str(resume))

    def checkpoint_monitor(self, config: Any) -> str:
        return "val/loss"

    def checkpoint_mode(self, config: Any) -> str:
        return "min"

    def checkpoint_prefix(self, config: Any) -> str:
        return "model"

    def checkpoint_save_top_k(self, config: Any) -> int:
        return 3

    def checkpoint_save_last(self, config: Any) -> bool:
        return True

    def early_stopping_enabled(self, config: Any) -> bool:
        return True

    def early_stopping_monitor(self, config: Any) -> str:
        return self.checkpoint_monitor(config)

    def early_stopping_mode(self, config: Any) -> str:
        return "min"

    def early_stopping_patience(self, config: Any) -> int:
        return 10

    def early_stopping_min_delta(self, config: Any) -> float | None:
        return None

    def lr_monitor_enabled(self, config: Any) -> bool:
        return True

    def lr_monitor_interval(self, config: Any) -> str:
        return "step"

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        return []

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        return {}

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        return None

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int | None:
        return None

    # ---- shared helpers ----
    def seed_everything(self, config: Any) -> None:
        seed = getattr(config.run, "seed", None)
        if seed is not None:
            pl.seed_everything(int(seed))

    def is_dry_run(self, config: Any) -> bool:
        return bool(getattr(config.run, "dry_run", False))

    def skip_test(self, config: Any) -> bool:
        return bool(getattr(config.run, "fast_dev_run", False))

    def save_config(self, config: Any, output_dir: Path) -> None:
        OmegaConf.save(config, output_dir / "config.yaml")

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        return TensorBoardLogger(save_dir=str(output_dir), name="logs")

    def build_callbacks(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Build callbacks for training.

        Callback settings can be provided via `config.training.*` and take precedence
        over runner hook methods. For backward compatibility, missing values fall back
        to the hook defaults.
        """
        missing = object()

        def _select(path: str, *, default: Any = missing) -> Any:
            if OmegaConf.is_config(config):
                value = OmegaConf.select(config, path, default=missing)
                if value is not missing:
                    return value
                return default

            current: Any = config
            for key in path.split("."):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current

        checkpoint_dir = Path(logger.log_dir) / "checkpoints"
        callbacks: list[Any] = []

        ckpt_enabled = bool(_select("training.checkpoint.enabled", default=True))
        if ckpt_enabled:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=str(
                        _select(
                            "training.checkpoint.filename",
                            default=f"{self.checkpoint_prefix(config)}-{{epoch:02d}}",
                        )
                    ),
                    monitor=str(
                        _select(
                            "training.checkpoint.monitor",
                            default=self.checkpoint_monitor(config),
                        )
                    ),
                    mode=str(
                        _select(
                            "training.checkpoint.mode",
                            default=self.checkpoint_mode(config),
                        )
                    ),
                    save_top_k=int(
                        _select(
                            "training.checkpoint.save_top_k",
                            default=self.checkpoint_save_top_k(config),
                        )
                    ),
                    save_last=bool(
                        _select(
                            "training.checkpoint.save_last",
                            default=self.checkpoint_save_last(config),
                        )
                    ),
                )
            )

        early_enabled = bool(
            _select(
                "training.early_stopping.enabled",
                default=self.early_stopping_enabled(config),
            )
        )
        if early_enabled:
            kwargs: dict[str, Any] = {
                "monitor": str(
                    _select(
                        "training.early_stopping.monitor",
                        default=self.early_stopping_monitor(config),
                    )
                ),
                "patience": int(
                    _select(
                        "training.early_stopping.patience",
                        default=self.early_stopping_patience(config),
                    )
                ),
                "mode": str(
                    _select(
                        "training.early_stopping.mode",
                        default=self.early_stopping_mode(config),
                    )
                ),
            }
            min_delta = _select(
                "training.early_stopping.min_delta",
                default=self.early_stopping_min_delta(config),
            )
            if min_delta is not None:
                kwargs["min_delta"] = min_delta
            callbacks.append(EarlyStopping(**kwargs))

        lr_enabled = bool(
            _select("training.lr_monitor.enabled", default=self.lr_monitor_enabled(config))
        )
        if lr_enabled:
            callbacks.append(
                LearningRateMonitor(
                    logging_interval=str(
                        _select(
                            "training.lr_monitor.interval",
                            default=self.lr_monitor_interval(config),
                        )
                    )
                )
            )

        callbacks.extend(self.callbacks_extra(config, datamodule, logger))
        return callbacks

    def build_trainer(
        self, config: Any, callbacks: list[Any], logger: TensorBoardLogger
    ) -> pl.Trainer:
        accelerator, devices = self.select_devices(config)
        missing = object()

        def _select(path: str, *, default: Any = missing) -> Any:
            if OmegaConf.is_config(config):
                value = OmegaConf.select(config, path, default=missing)
                if value is not missing:
                    return value
                return default

            current: Any = config
            for key in path.split("."):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current

        base_kwargs: dict[str, Any] = {
            "max_epochs": int(getattr(config.training, "max_epochs", 1)),
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "logger": logger,
            "gradient_clip_val": getattr(config.training, "gradient_clip_val", None),
            "fast_dev_run": bool(getattr(config.run, "fast_dev_run", False)),
            "deterministic": True,
        }

        # Prefer flattened `training.*` keys for Trainer configuration.
        # Temporarily accept legacy `training.trainer.*` during migration.
        trainer_keys = (
            "precision",
            "log_every_n_steps",
            "deterministic",
            "accumulate_grad_batches",
            "val_check_interval",
            "check_val_every_n_epoch",
            "enable_progress_bar",
            "strategy",
        )
        for key in trainer_keys:
            value = _select(f"training.{key}", default=missing)
            if value is missing:
                value = _select(f"training.trainer.{key}", default=missing)
            if value is not missing and value is not None:
                base_kwargs[key] = value

        extra_kwargs = self.trainer_kwargs(config, accelerator, devices)
        for key, value in extra_kwargs.items():
            base_kwargs.setdefault(key, value)
        return pl.Trainer(**base_kwargs)

    def select_devices(self, config: Any) -> tuple[str, int]:
        gpus = int(getattr(config.run, "gpus", 0))
        if gpus > 0 and torch.cuda.is_available():
            return "gpu", gpus
        return "cpu", 1

    def run_dry_run(self, config: Any, output_dir: Path) -> None:
        print("Running dry run (no training)...")
        self._force_cpu_for_dry_run()

        datamodule = self.build_datamodule(config)
        if hasattr(datamodule, "num_workers"):
            datamodule.num_workers = 0
        if hasattr(datamodule, "pin_memory"):
            datamodule.pin_memory = False
        if hasattr(datamodule, "setup"):
            datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        self._print_batch_shapes(batch)
        self.dry_run_postprocess(batch, output_dir)

        steps_per_epoch = self.resolve_steps_per_epoch(
            config, datamodule, train_loader=train_loader
        )
        lightning_module = self.build_lightning_module(
            config, datamodule, steps_per_epoch=steps_per_epoch
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(lightning_module, datamodule=datamodule)
        print(f"Dry run complete. Outputs saved to {output_dir}")

    def _ensure_absolute(self, path: str) -> str:
        return str(to_absolute_path(path))

    def _force_cpu_for_dry_run(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        torch.cuda.is_available = types.MethodType(lambda *_a, **_k: False, torch.cuda)
        torch.cuda.device_count = types.MethodType(lambda *_a, **_k: 0, torch.cuda)
        torch.cuda.current_device = types.MethodType(lambda *_a, **_k: 0, torch.cuda)

    def _print_batch_shapes(self, batch: Any) -> None:
        if isinstance(batch, dict):
            print("Loaded batch:")
            for key, value in batch.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: {tuple(value.shape)}")
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            if hasattr(inputs, "shape"):
                print(f"Loaded batch: inputs {tuple(inputs.shape)}")
            if hasattr(targets, "shape"):
                print(f"  targets {tuple(targets.shape)}")
