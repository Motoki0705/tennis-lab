"""Base training runner with strict config-driven behavior.

All training infrastructure settings (trainer, checkpoint, early_stopping, lr_monitor)
must be defined in config. The runner does NOT provide fallback values - missing keys
cause immediate errors to catch misconfigurations early.
"""

from __future__ import annotations

import os
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.compilation import compile_modules
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback
from src.utils.configuration import PathResolver, PathRole
from src.utils.device import select_accelerator
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import load_and_validate_checkpoint


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
        runtime = self.validate_runtime_config(config)
        self.prepare_config(config)
        self.seed_everything(runtime)
        self.apply_runtime_settings(runtime)

        output_dir = self.prepare_output_dir(runtime)
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
        self.maybe_load_init_weights(runtime, lightning_module)
        self.maybe_compile_models(runtime, lightning_module)

        logger = self.build_logger(config, output_dir)
        callbacks = self.build_callbacks(config, datamodule, logger)
        trainer = self.build_trainer(config, callbacks, logger)
        resume_ckpt = self.resolve_resume(runtime, output_dir)

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
    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        """Build the shared strict contract before any processing or writes."""
        return TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)

    def prepare_output_dir(self, config: TrainingRuntimeConfig) -> Path:
        """Prepare output directory path."""
        output_dir: Path = config.run.output_dir
        return output_dir

    def _gan_enabled(self, config: Any) -> bool:
        runtime = self.validate_runtime_config(config)
        enabled: bool = runtime.training.gan.enabled
        return enabled

    def prepare_config(self, config: Any) -> None:
        """Validate task-specific configuration before the run starts.

        Subclasses may override this hook to construct their task-owned typed
        runtime contract. Configuration mutation is intentionally unsupported:
        all shared values must be explicit and valid at the entry boundary.
        """
        _ = config
        return None

    def resolve_resume(
        self, config: TrainingRuntimeConfig, output_dir: Path
    ) -> str | None:
        """Resolve resume checkpoint path."""
        _ = output_dir
        return str(config.run.resume) if config.run.resume is not None else None

    def maybe_load_init_weights(
        self, config: TrainingRuntimeConfig, lightning_module: pl.LightningModule
    ) -> None:
        """Load model weights only (no optimizer/epoch state) for fine-tuning.

        Unlike ``run.resume`` (full-state resume that continues the same
        schedule), ``run.init_weights`` copies the checkpoint's weights into a
        *fresh* trainer: epoch 0, new optimizer, new LR schedule. This is the
        fine-tune-from-pretrained path (e.g. refine a converged model with an
        added loss term without the from-scratch dynamics).
        """
        init_path = config.run.init_weights
        if init_path is None:
            return
        checkpoint = load_and_validate_checkpoint(init_path)
        if "state_dict" not in checkpoint:
            raise ValueError(
                f"init_weights checkpoint {init_path} has no required 'state_dict'."
            )
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"init_weights checkpoint {init_path} 'state_dict' must be a mapping."
            )
        result = lightning_module.load_state_dict(state_dict, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
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

    def maybe_compile_models(
        self,
        config: TrainingRuntimeConfig,
        lightning_module: pl.LightningModule,
    ) -> tuple[str, ...]:
        """Compile all explicit training model targets without replacing them."""
        compile_config = config.training.compile
        if not compile_config.enabled:
            return ()
        if not isinstance(lightning_module, BaseLightningModule):
            raise TypeError(
                "training.compile.enabled=true requires a BaseLightningModule "
                "with explicit compilation_targets()."
            )
        return cast(
            tuple[str, ...],
            compile_modules(
                lightning_module.compilation_targets(),
                compile_config,
            ),
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
    def seed_everything(self, config: TrainingRuntimeConfig) -> None:
        """Seed all random number generators."""
        pl.seed_everything(config.run.seed)

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
        runtime = self.validate_runtime_config(config)
        self.prepare_config(config)
        config_path = runtime.resolver.resolve_beneath(
            PathRole.OUTPUT,
            output_dir,
            "config.yaml",
        )
        OmegaConf.save(config, config_path)

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        """Build TensorBoard logger."""
        runtime = self.validate_runtime_config(config)
        validated_output_dir = runtime.resolver.validate(PathRole.OUTPUT, output_dir)
        return TensorBoardLogger(save_dir=str(validated_output_dir), name="logs")

    def _record_ckpt_dir_pointer(
        self, checkpoint_dir: Path, resolver: PathResolver
    ) -> None:
        """Record this run's checkpoint dir into the repro bundle (issue #533).

        When the run is launched via the training queue, ``TENNIS_REPRO_DIR``
        points at the job's gitignored ``.training_queue/repro/<jobid>/`` staging
        dir (where the test-split ``pred_test.npz`` also lands). Writing the
        checkpoint dir there gives the optional post-run ckpt pruner a
        deterministic repro -> ckpt link, so it can delete *only this run's*
        checkpoints once the predictions are saved. Best-effort: never raise, so
        a bookkeeping hiccup cannot abort training.
        """
        target = resolver.resolve(PathRole.ARTIFACT, "repro")
        target.mkdir(parents=True, exist_ok=True)
        resolved_checkpoint_dir = checkpoint_dir.resolve()
        (target / "output_dir.txt").write_text(
            f"{resolved_checkpoint_dir}\n", encoding="utf-8"
        )

    def build_callbacks(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Build all callbacks from config."""
        callbacks: list[Any] = []

        # Checkpoint callback (required)
        runtime = self.validate_runtime_config(config)
        checkpoint_cfg = runtime.training.checkpoint
        if checkpoint_cfg.enabled:
            validated_log_dir = runtime.resolver.validate(
                PathRole.OUTPUT, Path(logger.log_dir)
            )
            # ``checkpoints`` is an immutable Lightning artifact-layout child,
            # not a configured path fragment. The central child resolver rejects
            # this reserved root alias, so validate both its typed parent and the
            # resulting fixed child explicitly.
            checkpoint_dir = runtime.resolver.validate(
                PathRole.OUTPUT,
                validated_log_dir / "checkpoints",
            )
            self._record_ckpt_dir_pointer(checkpoint_dir, runtime.resolver)
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
        early_cfg = runtime.training.early_stopping
        if early_cfg.enabled:
            kwargs: dict[str, Any] = {
                "monitor": early_cfg.monitor,
                "patience": early_cfg.patience,
                "mode": early_cfg.mode,
                "min_delta": early_cfg.min_delta,
                "check_on_train_epoch_end": early_cfg.check_on_train_epoch_end,
            }
            callbacks.append(EarlyStopping(**kwargs))

        # LR monitor callback (optional)
        lr_cfg = runtime.training.lr_monitor
        if lr_cfg.enabled:
            callbacks.append(LearningRateMonitor(logging_interval=lr_cfg.interval))

        # Qualitative validation logging callback (optional)
        qual_cfg = runtime.training.qualitative_logging
        if qual_cfg.enabled:
            callbacks.append(
                QualitativeLoggingCallback(
                    every_n_epochs=qual_cfg.every_n_epochs,
                    num_samples=qual_cfg.num_samples,
                    enabled=True,
                    selection_mode=qual_cfg.selection_mode,
                    selected_indices=(
                        list(qual_cfg.selected_indices)
                        if qual_cfg.selected_indices is not None
                        else None
                    ),
                )
            )

        # Add task-specific callbacks
        callbacks.extend(self.callbacks_extra(config, datamodule, logger))

        # Lightning 2.6 defaults to RichProgressBar when no explicit progress
        # callback is provided, which is noisy in notebook environments.
        if runtime.training.trainer.enable_progress_bar and not any(
            isinstance(callback, ProgressBar) for callback in callbacks
        ):
            callbacks.append(TQDMProgressBar())

        return callbacks

    def build_trainer(
        self, config: Any, callbacks: list[Any], logger: TensorBoardLogger
    ) -> pl.Trainer:
        """Build PyTorch Lightning Trainer from config."""
        accelerator, devices = self.select_devices(config)
        runtime = self.validate_runtime_config(config)
        trainer_cfg = runtime.training.trainer

        kwargs: dict[str, Any] = {
            "max_epochs": trainer_cfg.max_epochs,
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "logger": logger,
            "deterministic": trainer_cfg.deterministic,
            "log_every_n_steps": trainer_cfg.log_every_n_steps,
            "check_val_every_n_epoch": trainer_cfg.check_val_every_n_epoch,
            "fast_dev_run": runtime.run.fast_dev_run,
        }
        kwargs["benchmark"] = trainer_cfg.benchmark

        # Optional parameters
        if trainer_cfg.gradient_clip_val is not None:
            kwargs["gradient_clip_val"] = trainer_cfg.gradient_clip_val

        precision = trainer_cfg.precision
        if (
            precision == "bf16-mixed"
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "training.trainer.precision='bf16-mixed' is not supported "
                "by the selected GPU; select an explicit supported precision."
            )
        kwargs["precision"] = precision

        kwargs["accumulate_grad_batches"] = trainer_cfg.accumulate_grad_batches
        kwargs["reload_dataloaders_every_n_epochs"] = (
            trainer_cfg.reload_dataloaders_every_n_epochs
        )
        kwargs["enable_progress_bar"] = trainer_cfg.enable_progress_bar
        kwargs["enable_model_summary"] = trainer_cfg.enable_model_summary

        return pl.Trainer(**kwargs)

    def select_devices(self, config: Any) -> tuple[str, int]:
        """Select accelerator and device count."""
        accelerator_and_devices: tuple[str, int] = select_accelerator(
            self.validate_runtime_config(config).run.gpus
        )
        return accelerator_and_devices

    def apply_runtime_settings(self, config: TrainingRuntimeConfig) -> None:
        """Apply backend/runtime settings from training config."""
        training = config.training
        torch.set_float32_matmul_precision(training.matmul_precision)

        allow_tf32 = training.allow_tf32
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = training.trainer.benchmark

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

    def _force_cpu_for_dry_run(self) -> None:
        """Force CPU usage during dry run."""
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
