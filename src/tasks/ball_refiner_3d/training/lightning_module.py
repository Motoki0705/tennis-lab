"""Lightning optimization with shared GAN strategy and full-rally validation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.tasks.ball_refiner_3d.configuration.training import RefinerTrainingSettings
from src.tasks.ball_refiner_3d.data.augmentation.audit import corruption_audit
from src.tasks.ball_refiner_3d.data.datamodule import RefinerDataModule
from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.evaluation.evaluator import (
    evaluate,
    summarize_predictions,
)
from src.tasks.ball_refiner_3d.model_io.checkpoint import checkpoint_metadata
from src.tasks.ball_refiner_3d.model_io.factory import bind_refiner, build_refiner
from src.tasks.ball_refiner_3d.models.discriminators import build_refiner_discriminator
from src.tasks.ball_refiner_3d.models.generators.flow import FlowRefiner
from src.tasks.ball_refiner_3d.training.losses.composite import generator_objective
from src.tasks.ball_refiner_3d.training.losses.events import event_loss
from src.tasks.ball_refiner_3d.training.losses.flow_matching import flow_matching_loss
from src.tasks.ball_refiner_3d.training.losses.physics import (
    physics_losses,
    weighted_physics,
)
from src.tasks.ball_refiner_3d.training.losses.position import position_loss
from src.tasks.ball_refiner_3d.training.schedules import reconstruction_weight_at
from src.tasks.base.training.gan_loss import LSGANLoss
from src.tasks.base.training.gan_schedule import gan_weight_at
from src.tasks.base.training.gan_training import ManualGANTrainingStrategy
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.utils.io import write_json_atomic


class RefinerLightningModule(BaseLightningModule):
    def __init__(self, settings: RefinerTrainingSettings) -> None:
        super().__init__(settings.raw)
        self.settings = settings
        self.model = build_refiner(settings.model)
        self.binding = bind_refiner(self.model)
        self.discriminator = (
            build_refiner_discriminator(3, settings.discriminator)
            if settings.gan.enabled
            else None
        )
        self.gan_loss_fn = LSGANLoss() if settings.gan.enabled else None
        self.gan_training = (
            ManualGANTrainingStrategy(
                generator_gradient_clip_val=settings.updates.gradient_clip,
                discriminator_gradient_clip_val=settings.updates.gradient_clip,
                scheduler_interval="step",
            )
            if settings.gan.enabled
            else None
        )
        self.automatic_optimization = not settings.gan.enabled
        self.generator_updates = 0
        self.gan_weight = 0.0
        self.reconstruction_weight = settings.reconstruction.initial_weight
        self.validation_rmse = math.inf
        self.best_step = 0
        self.best_rmse = math.inf
        self.flow_rng: torch.Generator | None = None
        self._restored_state: dict[str, Any] | None = None
        self._rows: list[dict[str, float]] = []
        self._step_metrics: dict[str, float] = {}
        self._validation_arrays: list[dict[str, np.ndarray]] = []
        self._validation_seconds = 0.0
        self._validation_digest = hashlib.sha256()
        self.start_time = 0.0

    @property
    def data_module(self) -> RefinerDataModule:
        module = self.trainer.datamodule
        if not isinstance(module, RefinerDataModule):
            raise TypeError("Refiner training requires RefinerDataModule")
        return module

    def additional_compilation_targets(self) -> dict[str, nn.Module]:
        return (
            {"discriminator": self.discriminator}
            if self.discriminator is not None
            else {}
        )

    def _estimate_total_steps(self) -> int:
        return int(self.settings.updates.steps)

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        return [{"params": self.model.parameters()}]

    def configure_optimizers(self) -> Any:
        configured = super().configure_optimizers()
        if self.discriminator is None:
            return configured
        discriminator = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.optimizer_betas,
        )
        return [configured["optimizer"], discriminator], [configured["lr_scheduler"]]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.settings.updates.gradient_clip,
            error_if_nonfinite=True,
        )

    def on_train_start(self) -> None:
        self.start_time = time.monotonic()
        self.flow_rng = torch.Generator(device=self.device).manual_seed(
            self.settings.runtime.run.seed + 200
        )
        if self._restored_state is None:
            torch.manual_seed(self.settings.runtime.run.seed + 100)
        else:
            torch.set_rng_state(self._restored_state["torch_rng"])
            self.flow_rng.set_state(self._restored_state["flow_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._restored_state["cuda_rng"], self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_start(self) -> None:
        # The first iterator access prepares the epoch's immutable augmentation block.
        self.model.train()

    def _compute_supervised_result(
        self, batch: dict[str, Tensor], stage: str
    ) -> dict[str, Any]:
        if stage != "train" or self.flow_rng is None:
            raise RuntimeError(
                "Training objective requires the initialized training lifecycle"
            )
        valid = ~batch["padding"]
        physics: dict[str, Tensor] = {}
        if isinstance(self.model, FlowRefiner):
            reconstruction, logits = flow_matching_loss(
                self.model,
                batch["coordinates"],
                batch["missing"],
                batch["padding"],
                batch["target"],
                self.flow_rng,
            )
            fake = None
        else:
            inputs = {key: batch[key] for key in ("coordinates", "missing", "padding")}
            if self.settings.model.physics_heads:
                inputs["segment"] = batch["segment"]
            result = self.binding.run(inputs)
            reconstruction = position_loss(result.coordinates, batch["target"], valid)
            logits, fake = result.event_logits, result.coordinates
            if self.settings.physics.enabled:
                physics = physics_losses(
                    result, batch, self.data_module.clock, self.settings.physics
                )
        events = event_loss(logits, batch["event_target"], valid)
        supervised = generator_objective(
            reconstruction,
            reconstruction.new_zeros(()),
            reconstruction_weight=self.reconstruction_weight,
            gan_weight=0.0,
        )
        loss = supervised + self.settings.event.weight * events
        weighted = (
            weighted_physics(physics, self.settings.physics)
            if physics
            else loss.new_zeros(())
        )
        loss = loss + weighted
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite objective at generator update {self.generator_updates}"
            )
        metrics = {
            "event_loss": float(events.detach()),
            "reconstruction": float(reconstruction.detach()),
            "weighted_physics": float(weighted.detach()),
            **{
                f"physics_{name}": float(value.detach())
                for name, value in physics.items()
            },
        }
        return {
            "loss": loss,
            "metrics": metrics,
            "gan_fake": fake,
            "gan_real": batch["target"],
            "gan_padding_mask": batch["padding"],
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        self.reconstruction_weight = reconstruction_weight_at(
            self.generator_updates, self.settings.reconstruction
        )
        gan = self.settings.gan
        self.gan_weight = (
            (
                gan_weight_at(
                    self.generator_updates,
                    start=gan.start_step,
                    warmup=gan.warmup_steps,
                    target=gan.target_weight,
                )
                if gan.schedule_enabled
                else gan.target_weight
            )
            if gan.enabled
            else 0.0
        )
        if self.gan_training is None:
            result = self._compute_supervised_result(batch, "train")
            loss, metrics = result["loss"], result["metrics"]
        else:
            self.gan_training.phase_active = self.gan_weight > 0
            self.gan_training.set_weight(self.gan_weight)
            loss, metrics = self.gan_training.shared_step(self, batch, "train")
        generator_gan = (
            float(metrics["loss_gan_generator"])
            if "loss_gan_generator" in metrics
            else 0.0
        )
        discriminator = (
            float(metrics["loss_gan_discriminator"])
            if "loss_gan_discriminator" in metrics
            else 0.0
        )
        self._step_metrics = {
            "event_loss": float(metrics["event_loss"]),
            "reconstruction": float(metrics["reconstruction"]),
            "weighted_event": self.settings.event.weight * float(metrics["event_loss"]),
            "weighted_reconstruction": self.reconstruction_weight
            * float(metrics["reconstruction"]),
            "reconstruction_weight": self.reconstruction_weight,
            "gan_weight": self.gan_weight,
            "generator_gan": generator_gan,
            "discriminator": discriminator,
            "weighted_gan": self.gan_weight * generator_gan,
            # GAN steps never carry physics objectives (rejected by the config).
            "weighted_physics": float(metrics.get("weighted_physics", 0.0)),
            "total": float(loss.detach()),
            **{
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("physics_")
            },
        }
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch["coordinates"]),
        )
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.generator_updates += 1
        self._rows.append(self._step_metrics)
        if batch_idx == 0:
            audit = corruption_audit(self.data_module.training_data)
            write_json_atomic(
                self.settings.runtime.run.output_dir
                / f"corruption-block-{self.current_epoch:03d}.json",
                audit,
            )
            for key, value in audit.items():
                self.log(f"corruption/{key}", value, batch_size=1)
        if (
            self.generator_updates % self.settings.updates.log_every == 0
            or self.generator_updates == self.settings.updates.steps
        ):
            row = {
                key: sum(item[key] for item in self._rows) / len(self._rows)
                for key in self._step_metrics
            }
            row.update(
                step=self.generator_updates,
                seconds=time.monotonic() - self.start_time,
                gan_weight_current=self.gan_weight,
                reconstruction_weight_current=self.reconstruction_weight,
            )
            with (self.settings.runtime.run.output_dir / "learning_curve.jsonl").open(
                "a"
            ) as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            for key, value in row.items():
                self.log(
                    f"train/{key}", value, on_step=True, on_epoch=False, batch_size=1
                )
            self._rows.clear()

    def on_validation_epoch_start(self) -> None:
        self._validation_arrays.clear()
        self._validation_seconds = 0.0
        self._validation_digest = hashlib.sha256()

    def validation_step(self, rally: PreparedRally, batch_idx: int) -> None:
        report, arrays = evaluate(
            self.model,
            [rally],
            self.device,
            seed=self.settings.raw["data"]["evaluation_seed"],
            batch_size=self.settings.updates.batch_size,
            physics=False,
        )
        self._validation_arrays.append(arrays)
        self._validation_seconds += report["inference_seconds"]
        self._validation_digest.update(rally.source.name.encode())
        self._validation_digest.update(rally.coordinates.tobytes())
        self._validation_digest.update(rally.missing.tobytes())

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._validation_arrays.clear()
            return
        if not self._validation_arrays:
            raise ValueError("Validation requires the nonempty complete rally split")
        arrays = {
            key: np.concatenate([part[key] for part in self._validation_arrays])
            for key in self._validation_arrays[0]
        }
        report = summarize_predictions(
            arrays,
            elapsed=self._validation_seconds,
            input_sha256=self._validation_digest.hexdigest(),
            method=self.settings.model.architecture,
        )
        self.validation_rmse = float(report["all"]["rmse"])
        if self.validation_rmse < self.best_rmse:
            self.best_rmse, self.best_step = (
                self.validation_rmse,
                self.generator_updates,
            )
        self.log("val/rmse", self.validation_rmse, on_epoch=True, batch_size=1)
        self.log(
            "val/event_brier",
            report["event_probability"]["brier"],
            on_epoch=True,
            batch_size=1,
        )
        for name in ("observed", "missing", "event"):
            if report[name]["rmse"] is not None:
                self.log(
                    f"val/{name}_rmse",
                    report[name]["rmse"],
                    on_epoch=True,
                    batch_size=1,
                )
        write_json_atomic(
            self.settings.runtime.run.output_dir
            / f"validation-{self.generator_updates:06d}.json",
            report,
        )
        write_json_atomic(
            self.settings.runtime.run.output_dir / "state.json",
            {
                "status": "training",
                "step": self.generator_updates,
                "best_step": self.best_step,
                "best_val_rmse": self.best_rmse,
            },
        )
        self._validation_arrays.clear()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.flow_rng is None:
            raise RuntimeError("Cannot checkpoint before training RNG initialization")
        checkpoint.update(
            checkpoint_metadata(
                self.model,
                event_sigma_frames=self.settings.event.sigma_frames,
                clock=self.data_module.clock,
            )
        )
        checkpoint.update(
            schema="ball_refiner_3d.events.v2",
            step=self.generator_updates,
            seed=self.settings.runtime.run.seed,
            fps=self.data_module.dataset.fps,
            manifest_sha256=self.data_module.dataset.manifest_hash,
            validation_rmse=self.validation_rmse,
            best_step=self.best_step,
            best_rmse=self.best_rmse,
            gan_weight=self.gan_weight,
            reconstruction_weight=self.reconstruction_weight,
            gan_config=self.settings.raw["training"]["gan"],
            reconstruction_config=asdict(self.settings.reconstruction),
            torch_rng=torch.get_rng_state(),
            flow_rng=self.flow_rng.get_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            pending_log_rows=self._rows,
            training_contract={
                key: self.settings.raw[key]
                for key in ("model", "data", "augmentation", "loss", "training")
            },
        )

    @staticmethod
    def validate_resume_checkpoint(
        settings: RefinerTrainingSettings, checkpoint: dict[str, Any]
    ) -> None:
        if (
            checkpoint.get("schema") != "ball_refiner_3d.events.v2"
            or "loops" not in checkpoint
        ):
            raise ValueError(
                "Legacy event-head weights support inference/init_weights, not Lightning resume"
            )
        if (checkpoint["cuda_rng"] is not None) != (settings.runtime.run.gpus > 0):
            raise ValueError("Resume requires the same CPU/CUDA device family")
        expected = {
            key: settings.raw[key]
            for key in ("model", "data", "augmentation", "loss", "training")
        }
        if (
            checkpoint["training_contract"] != expected
            or checkpoint["seed"] != settings.runtime.run.seed
        ):
            raise ValueError(
                "Resume requires the same model/data/loss/training contract and seed"
            )
        end = min(
            (checkpoint["epoch"] + 1) * settings.updates.evaluate_every,
            settings.updates.steps,
        )
        if checkpoint["step"] != end:
            raise ValueError(
                "Resume requires a completed augmentation-block checkpoint"
            )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.validate_resume_checkpoint(self.settings, checkpoint)
        self.generator_updates = checkpoint["step"]
        self.validation_rmse = checkpoint["validation_rmse"]
        self.best_step, self.best_rmse = (
            checkpoint["best_step"],
            checkpoint["best_rmse"],
        )
        self._rows = checkpoint["pending_log_rows"]
        self._restored_state = checkpoint

    def on_train_end(self) -> None:
        write_json_atomic(
            self.settings.runtime.run.output_dir / "state.json",
            {
                "status": "trained",
                "step": self.generator_updates,
                "best_step": self.best_step,
                "best_val_rmse": self.best_rmse,
            },
        )
