"""Base training runner with strict config-driven behavior.

All training infrastructure settings (trainer, checkpoint, early_stopping, lr_monitor)
must be defined in config. The runner does NOT provide fallback values - missing keys
cause immediate errors to catch misconfigurations early.
"""

from __future__ import annotations

import os
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.utils.device import select_accelerator


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
        self.prepare_config(config)
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
        self.maybe_load_init_weights(config, lightning_module)

        logger = self.build_logger(config, output_dir)
        callbacks = self.build_callbacks(config, datamodule, logger)
        trainer = self.build_trainer(config, callbacks, logger)
        resume_ckpt = self.resolve_resume(config, output_dir)

        with self.resume_checkpoint_load_env(resume_ckpt):
            trainer.fit(
                lightning_module,
                datamodule=datamodule,
                ckpt_path=resume_ckpt,
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

    def _gan_enabled(self, config: Any) -> bool:
        train_cfg = config.get("training", {}) or {}
        return bool((train_cfg.get("gan", {}) or {}).get("enabled", False))

    def _apply_gan_runtime_config(self, config: Any) -> None:
        if not self._gan_enabled(config):
            raise RuntimeError(
                "GAN runtime config should only be applied when GAN is enabled."
            )
        config.training.early_stopping.enabled = False
        config.training.trainer.gradient_clip_val = None

    def prepare_config(self, config: Any) -> None:
        """Apply task-specific config mutations before the run starts."""
        if self._gan_enabled(config):
            self._apply_gan_runtime_config(config)
        return None

    def resolve_resume(self, config: Any, output_dir: Path) -> str | None:
        """Resolve resume checkpoint path."""
        resume = config.run.resume
        if not resume:
            return None
        return self._ensure_absolute(str(resume))

    def maybe_load_init_weights(
        self, config: Any, lightning_module: pl.LightningModule
    ) -> None:
        """Load model weights only (no optimizer/epoch state) for fine-tuning.

        Unlike ``run.resume`` (full-state resume that continues the same
        schedule), ``run.init_weights`` copies the checkpoint's weights into a
        *fresh* trainer: epoch 0, new optimizer, new LR schedule. This is the
        fine-tune-from-pretrained path (e.g. refine a converged model with an
        added loss term without the from-scratch dynamics).
        """
        run_cfg = config.run
        init = run_cfg.get("init_weights") if hasattr(run_cfg, "get") else None
        if not init:
            return
        if config.run.resume:
            raise ValueError(
                "run.init_weights (weight-only fine-tune) and run.resume "
                "(full-state resume) are mutually exclusive; set only one."
            )
        init_path = self._ensure_absolute(str(init))
        # Trusted local checkpoint: Lightning ckpts carry non-tensor payloads,
        # so weights_only=False is required.
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        result = lightning_module.load_state_dict(state_dict, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        loaded = len(state_dict) - len(unexpected)
        print(
            f"[init_weights] loaded {loaded}/{len(state_dict)} tensors from "
            f"{init_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
        # A near-empty load means the checkpoint does not match this model:
        # fail loudly rather than silently fine-tuning from random weights.
        if loaded < 0.5 * len(state_dict):
            raise RuntimeError(
                f"init_weights loaded only {loaded}/{len(state_dict)} tensors; "
                f"checkpoint likely does not match the model. missing[:5]={missing[:5]} "
                f"unexpected[:5]={unexpected[:5]}"
            )

    @contextmanager
    def resume_checkpoint_load_env(self, resume_ckpt: str | None) -> Iterator[None]:
        """Temporarily allow full-state loading for trusted local resume checkpoints."""
        if not resume_ckpt:
            yield
            return

        env_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        previous_value = os.environ.get(env_key)
        if previous_value is None:
            os.environ[env_key] = "1"
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop(env_key, None)

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Return additional callbacks. Override in subclasses for task-specific callbacks."""
        _ = datamodule
        _ = logger
        extras: list[Any] = []
        if self._gan_enabled(config):
            from src.tasks.base.training.gan_transition_callback import (
                GANTransitionCallback,
            )

            extras.append(GANTransitionCallback(config))
        return extras

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
        return bool(config.run.fast_dev_run) or not bool(config.run.test_after_fit)

    def save_config(self, config: Any, output_dir: Path) -> None:
        """Save resolved config to output directory."""
        # Some callers invoke runner methods directly without going through run(),
        # so prepare task-specific runtime config here as well before persisting it.
        self.prepare_config(config)
        OmegaConf.save(config, output_dir / "config.yaml")

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        """Build TensorBoard logger."""
        return TensorBoardLogger(save_dir=str(output_dir), name="logs")

    def _record_ckpt_dir_pointer(self, checkpoint_dir: Path) -> None:
        """Record this run's checkpoint dir into the repro bundle (issue #533).

        When the run is launched via the training queue, ``TENNIS_REPRO_DIR``
        points at the job's gitignored ``.training_queue/repro/<jobid>/`` staging
        dir (where the test-split ``pred_test.npz`` also lands). Writing the
        checkpoint dir there gives the optional post-run ckpt pruner a
        deterministic repro -> ckpt link, so it can delete *only this run's*
        checkpoints once the predictions are saved. Best-effort: never raise, so
        a bookkeeping hiccup cannot abort training.
        """
        repro_dir = os.environ.get("TENNIS_REPRO_DIR")
        if not repro_dir:
            return
        try:
            target = Path(repro_dir)
            target.mkdir(parents=True, exist_ok=True)
            resolved_checkpoint_dir = checkpoint_dir.resolve()
            (target / "output_dir.txt").write_text(
                f"{resolved_checkpoint_dir}\n", encoding="utf-8"
            )
        except OSError:
            pass

    def build_callbacks(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Build all callbacks from config."""
        callbacks: list[Any] = []

        # Checkpoint callback (required)
        checkpoint_cfg = config.training.checkpoint
        if checkpoint_cfg.enabled:
            checkpoint_dir = Path(logger.log_dir) / "checkpoints"
            self._record_ckpt_dir_pointer(checkpoint_dir)
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

        # Qualitative validation logging callback (optional)
        qual_cfg = config.training.get("qualitative_logging", {})
        if qual_cfg.get("enabled", False):
            selected_indices_cfg = qual_cfg.get("selected_indices")
            selected_indices = (
                [int(idx) for idx in selected_indices_cfg]
                if selected_indices_cfg is not None
                else None
            )
            callbacks.append(
                QualitativeLoggingCallback(
                    every_n_epochs=int(qual_cfg.get("every_n_epochs", 1)),
                    num_samples=int(qual_cfg.get("num_samples", 4)),
                    enabled=True,
                    selection_mode=str(qual_cfg.get("selection_mode", "random")),
                    selected_indices=selected_indices,
                )
            )

        # Add task-specific callbacks
        callbacks.extend(self.callbacks_extra(config, datamodule, logger))

        # Lightning 2.6 defaults to RichProgressBar when no explicit progress
        # callback is provided, which is noisy in notebook environments.
        if config.training.trainer.get("enable_progress_bar", True) and not any(
            isinstance(callback, ProgressBar) for callback in callbacks
        ):
            callbacks.append(TQDMProgressBar())

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
            if (
                str(precision) == "bf16-mixed"
                and torch.cuda.is_available()
                and not torch.cuda.is_bf16_supported()
            ):
                precision = "16-mixed"
                print(
                    "bf16-mixed is not supported on this GPU. Falling back to 16-mixed."
                )
            kwargs["precision"] = precision

        optional_trainer_keys = (
            "max_steps",
            "accumulate_grad_batches",
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
            "num_sanity_val_steps",
            "enable_progress_bar",
            "enable_model_summary",
        )
        for key in optional_trainer_keys:
            if hasattr(trainer_cfg, key):
                value = getattr(trainer_cfg, key)
                if value is not None:
                    kwargs[key] = value

        return pl.Trainer(**kwargs)

    def select_devices(self, config: Any) -> tuple[str, int]:
        """Select accelerator and device count."""
        accelerator_and_devices: tuple[str, int] = select_accelerator(
            int(config.run.gpus)
        )
        return accelerator_and_devices

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
