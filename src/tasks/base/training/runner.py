"""Base training runner with strict config-driven behavior.

All training infrastructure settings (trainer, checkpoint, early_stopping, lr_monitor)
must be defined in config. The runner does NOT provide fallback values - missing keys
cause immediate errors to catch misconfigurations early.
"""

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
    """Base training runner with strict config-driven behavior.

    Design principles:
    - Config is the single source of truth for training infrastructure settings.
    - No fallback values in runner methods - missing config keys cause errors.
    - Subclasses only implement build_datamodule() and build_lightning_module().
    - callbacks_extra() is the only extension point for additional callbacks.
    """

    def run(self, config: Any) -> None:
        """Run training with the provided config."""
        self.seed_everything(config)
        self.apply_runtime_settings(config)

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

    # ---- abstract methods (must be implemented by subclasses) ----
    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build the data module. Must be implemented by subclasses."""
        raise NotImplementedError

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build the lightning module. Must be implemented by subclasses."""
        raise NotImplementedError

    # ---- overridable hooks ----
    def prepare_output_dir(self, config: Any) -> Path:
        """Prepare output directory path."""
        return Path(self._ensure_absolute(str(config.run.output_dir)))

    def resolve_resume(self, config: Any, output_dir: Path) -> str | None:
        """Resolve resume checkpoint path."""
        resume = config.run.resume
        if not resume:
            return None
        return self._ensure_absolute(str(resume))

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Return additional callbacks. Override in subclasses for task-specific callbacks."""
        return []

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Post-process after dry run batch loading. Override in subclasses if needed."""
        return None

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int | None:
        """Resolve steps per epoch for scheduler warmup. Override if needed."""
        return None

    # ---- shared helpers (config-driven) ----
    def seed_everything(self, config: Any) -> None:
        """Seed all random number generators."""
        seed = config.run.seed
        if seed is not None:
            pl.seed_everything(int(seed))

    def is_dry_run(self, config: Any) -> bool:
        """Check if dry run mode is enabled."""
        return bool(config.run.dry_run)

    def skip_test(self, config: Any) -> bool:
        """Check if test phase should be skipped."""
        return bool(config.run.fast_dev_run)

    def save_config(self, config: Any, output_dir: Path) -> None:
        """Save resolved config to output directory."""
        OmegaConf.save(config, output_dir / "config.yaml")

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        """Build TensorBoard logger."""
        return TensorBoardLogger(save_dir=str(output_dir), name="logs")

    def build_callbacks(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Build all callbacks from config."""
        callbacks: list[Any] = []

        # Checkpoint callback (required)
        checkpoint_cfg = config.training.checkpoint
        if checkpoint_cfg.enabled:
            checkpoint_dir = Path(logger.log_dir) / "checkpoints"
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=checkpoint_cfg.filename,
                    monitor=checkpoint_cfg.monitor,
                    mode=checkpoint_cfg.mode,
                    save_top_k=checkpoint_cfg.save_top_k,
                    save_last=checkpoint_cfg.save_last,
                )
            )

        # Early stopping callback (optional)
        early_cfg = config.training.early_stopping
        if early_cfg.enabled:
            kwargs: dict[str, Any] = {
                "monitor": early_cfg.monitor,
                "patience": early_cfg.patience,
                "mode": early_cfg.mode,
            }
            if early_cfg.min_delta is not None:
                kwargs["min_delta"] = early_cfg.min_delta
            callbacks.append(EarlyStopping(**kwargs))

        # LR monitor callback (optional)
        lr_cfg = config.training.lr_monitor
        if lr_cfg.enabled:
            callbacks.append(LearningRateMonitor(logging_interval=lr_cfg.interval))

        # Add task-specific callbacks
        callbacks.extend(self.callbacks_extra(config, datamodule, logger))
        return callbacks

    def build_trainer(
        self, config: Any, callbacks: list[Any], logger: TensorBoardLogger
    ) -> pl.Trainer:
        """Build PyTorch Lightning Trainer from config."""
        accelerator, devices = self.select_devices(config)
        trainer_cfg = config.training.trainer

        kwargs: dict[str, Any] = {
            "max_epochs": trainer_cfg.max_epochs,
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "logger": logger,
            "deterministic": trainer_cfg.deterministic,
            "log_every_n_steps": trainer_cfg.log_every_n_steps,
            "check_val_every_n_epoch": trainer_cfg.check_val_every_n_epoch,
            "fast_dev_run": bool(config.run.fast_dev_run),
        }
        if hasattr(trainer_cfg, "benchmark"):
            kwargs["benchmark"] = trainer_cfg.benchmark

        # Optional parameters
        if trainer_cfg.gradient_clip_val is not None:
            kwargs["gradient_clip_val"] = trainer_cfg.gradient_clip_val

        if trainer_cfg.precision is not None:
            precision = trainer_cfg.precision
            if str(precision) == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                precision = "16-mixed"
                print("bf16-mixed is not supported on this GPU. Falling back to 16-mixed.")
            kwargs["precision"] = precision

        return pl.Trainer(**kwargs)

    def select_devices(self, config: Any) -> tuple[str, int]:
        """Select accelerator and device count."""
        gpus = int(config.run.gpus)
        if gpus > 0 and torch.cuda.is_available():
            return "gpu", gpus
        return "cpu", 1

    def apply_runtime_settings(self, config: Any) -> None:
        """Apply backend/runtime settings from training config."""
        train_cfg = config.get("training", {})
        trainer_cfg = train_cfg.get("trainer", {})

        matmul_precision = str(train_cfg.get("matmul_precision", "high"))
        torch.set_float32_matmul_precision(matmul_precision)

        allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            deterministic = bool(trainer_cfg.get("deterministic", False))
            benchmark_cfg = trainer_cfg.get("benchmark")
            if benchmark_cfg is None:
                torch.backends.cudnn.benchmark = not deterministic
            else:
                torch.backends.cudnn.benchmark = bool(benchmark_cfg)

    def run_dry_run(self, config: Any, output_dir: Path) -> None:
        """Run dry run mode without full training."""
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
        """Convert path to absolute path."""
        return str(to_absolute_path(path))

    def _force_cpu_for_dry_run(self) -> None:
        """Force CPU usage during dry run."""
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        torch.cuda.is_available = types.MethodType(lambda *_a, **_k: False, torch.cuda)
        torch.cuda.device_count = types.MethodType(lambda *_a, **_k: 0, torch.cuda)
        torch.cuda.current_device = types.MethodType(lambda *_a, **_k: 0, torch.cuda)

    def _print_batch_shapes(self, batch: Any) -> None:
        """Print batch tensor shapes for debugging."""
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
