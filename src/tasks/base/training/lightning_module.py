"""Shared Lightning training utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from src.tasks.base.configuration import (
    BaseTrainingConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT
from src.utils.tensor_utils import to_numpy


class _SavezCompressed(Protocol):
    def __call__(self, file: str | Path, **arrays: np.ndarray) -> None: ...


def _concat_padded(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-batch arrays along axis 0, padding the time axis (1).

    Test batches can have different sequence lengths, so ``(b, T_i, ...)`` arrays
    are right-padded with zeros to the global max ``T`` before stacking. Arrays
    with fewer than 2 dims (one value per sample) are concatenated as-is.
    """
    if len(chunks) == 1:
        return chunks[0]
    if all(c.ndim >= 2 for c in chunks):
        max_t = max(c.shape[1] for c in chunks)
        padded = []
        for c in chunks:
            if c.shape[1] < max_t:
                pad_width = [(0, 0)] * c.ndim
                pad_width[1] = (0, max_t - c.shape[1])
                c = np.pad(c, pad_width)
            padded.append(c)
        chunks = padded
    concatenated: np.ndarray = np.asarray(np.concatenate(chunks, axis=0))
    return concatenated


class BaseLightningModule(pl.LightningModule):
    """Base Lightning module with shared optimizer/scheduler logic.

    This class expects training settings under `config.training` and optional
    dataset sizing under `config.data`.
    """

    steps_per_epoch: int | None = None

    def __init__(self, config: Any) -> None:
        super().__init__()
        # Be explicit: Lightning otherwise walks the concrete subclass's
        # ``__init__`` frame and captures runtime-only dependencies such as a
        # frozen BoundModelIO dataclass.  TensorBoard cannot serialize those
        # objects, while the Hydra config is the only hyperparameter source we
        # intend to persist here.
        self.save_hyperparameters("config")

        root = as_config_mapping(config, path="configuration")
        self.config = config
        self.training_config = BaseTrainingConfig.from_validated_task_mapping(
            require_config_mapping(root, "training", path="configuration")
        )
        self.path_resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                require_config_mapping(root, "paths", path="configuration"),
                repository_root=PROJECT_ROOT,
            )
        )

        optimizer = self.training_config.optimizer
        self.learning_rate = optimizer.learning_rate
        self.weight_decay = optimizer.weight_decay
        self.warmup_steps = optimizer.warmup_steps
        self.warmup_epochs = optimizer.warmup_epochs
        self.max_epochs = optimizer.max_epochs
        self.min_lr = optimizer.min_lr
        self.optimizer_betas = optimizer.betas

    def additional_compilation_targets(self) -> dict[str, nn.Module]:
        """Return task-owned models invoked outside the primary model forward.

        The primary model is supplied by :meth:`compilation_targets`.  GAN and
        future teacher/student modules extend this hook instead of relying on
        recursive child-module discovery.
        """
        return {}

    def compilation_targets(self) -> dict[str, nn.Module]:
        """Return the explicit, named modules compiled by the shared runner."""
        model = getattr(self, "model", None)
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                f"{type(self).__name__} must expose its primary nn.Module as "
                "self.model when training.compile.enabled=true."
            )
        targets = {"model": model}
        additional = self.additional_compilation_targets()
        overlap = sorted(set(targets) & set(additional))
        if overlap:
            rendered = ", ".join(overlap)
            raise RuntimeError(
                f"Additional compile target names overlap primary targets: {rendered}."
            )
        targets.update(additional)
        return targets

    def _mark_compiled_iteration(self) -> None:
        """Declare a batch boundary for Inductor modes backed by CUDA Graphs.

        Lightning can invoke a compiled model from sanity validation, training,
        validation, and testing.  Graph breaks inside a model make PyTorch's
        automatic iteration heuristic ambiguous, so mark the outer batch once.
        Keeping this at the Lightning boundary also lets GAN generator and
        discriminator calls within one training batch share the same iteration.
        """
        compile_config = self.training_config.compile
        if (
            compile_config.enabled
            and compile_config.backend == "inductor"
            and compile_config.mode in {"reduce-overhead", "max-autotune"}
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self._mark_compiled_iteration()

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        self._mark_compiled_iteration()

    def on_test_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        self._mark_compiled_iteration()

    def on_predict_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        self._mark_compiled_iteration()

    # ------------------------------------------------------------------
    # Qualitative validation logging hook
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render qualitative validation samples.

        Override in task-specific subclasses to produce visualizations.
        The default implementation is a no-op.

        Args:
            batches: Collected validation batch dicts (CPU tensors).
            outputs: Corresponding validation_step outputs (CPU tensors).
            artifact_dir: Directory to save artifact images/files.
            tb_writer: TensorBoard SummaryWriter (may be ``None``).
            global_step: Current global training step.
            epoch: Current epoch number.
        """

    # ------------------------------------------------------------------
    # Test-split inference saving (issue #533)
    #
    # Instead of keeping bulky checkpoints, we persist the model's predictions
    # on the test split (scenes from ``test.txt``). New metrics can then be
    # recomputed from these arrays without re-running the model, and the
    # checkpoint can be deleted. Task modules override
    # :meth:`test_prediction_payload` to declare which arrays to save.
    # ------------------------------------------------------------------

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Return per-sample test arrays to persist, keyed by name.

        Each array has shape ``(B, ...)`` (batch first). Override in task
        subclasses. The default returns ``{}`` (nothing saved).
        """
        _ = (batch, result)
        return {}

    def _safe_trainer(self) -> Any:
        """Return the attached ``Trainer`` or ``None`` (the property raises if
        the module is not attached, e.g. in unit tests)."""
        try:
            return self.trainer
        except (RuntimeError, AttributeError):
            return None

    def _reset_test_prediction_buffer(self) -> None:
        self._test_pred_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        self._test_pred_scene_ids: list[str] = []
        self._test_pred_cursor: int = 0

    def _scene_ids_for_test_batch(self, batch_size: int) -> list[str]:
        """Map the current test batch to scene ids from ``test.txt`` order.

        The test loader uses ``shuffle=False, drop_last=False`` (a sequential
        sampler), so a running cursor over ``test_dataset.scenes`` recovers the
        scene id per sample without threading it through the collate fn.
        """
        trainer = self._safe_trainer()
        dm = trainer.datamodule if trainer is not None else None
        dataset = dm.test_dataset if dm is not None else None
        scenes = (
            dataset.scenes
            if dataset is not None and hasattr(dataset, "scenes")
            else None
        )
        start = self._test_pred_cursor
        ids: list[str] = []
        for i in range(batch_size):
            gi = start + i
            if scenes is not None and gi < len(scenes):
                ids.append(Path(scenes[gi]).stem)
            else:
                ids.append(f"sample_{gi:06d}")
        self._test_pred_cursor = start + batch_size
        return ids

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        array: np.ndarray = to_numpy(value)
        return array

    def collect_test_predictions(self, batch: Any, result: dict[str, Any]) -> None:
        """Accumulate one test batch's prediction arrays into the buffer."""
        payload = self.test_prediction_payload(batch, result)
        if not payload:
            return
        if not hasattr(self, "_test_pred_arrays"):
            self._reset_test_prediction_buffer()
        arrays = {k: self._to_numpy(v) for k, v in payload.items()}
        batch_size = int(next(iter(arrays.values())).shape[0])
        self._test_pred_scene_ids.extend(self._scene_ids_for_test_batch(batch_size))
        for key, arr in arrays.items():
            self._test_pred_arrays[key].append(arr)

    def _test_predictions_dir(self) -> Path:
        resolved: Path = self.path_resolver.resolve(
            PathRole.ARTIFACT, "test_predictions"
        )
        return resolved

    def save_test_predictions(
        self, metrics: dict[str, Any] | None = None
    ) -> Path | None:
        """Write accumulated test predictions to ``pred_test.npz`` (+ metrics.json).

        Returns the npz path, or ``None`` if there is nothing to save / no target
        directory could be resolved.
        """
        if not hasattr(self, "_test_pred_arrays") or not self._test_pred_arrays:
            return None
        out_dir = self._test_predictions_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            key: _concat_padded(chunks)
            for key, chunks in self._test_pred_arrays.items()
        }
        # Fixed-width unicode (not object) so np.load works without allow_pickle.
        arrays["scene_ids"] = np.asarray(self._test_pred_scene_ids)
        npz_path = out_dir / "pred_test.npz"
        cast("_SavezCompressed", np.savez_compressed)(npz_path, **arrays)
        if metrics is not None:
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return npz_path

    def _estimate_total_steps(self) -> int:
        max_epochs = int(self.max_epochs)
        if hasattr(self, "steps_per_epoch") and self.steps_per_epoch is not None:
            total_steps = int(self.steps_per_epoch) * max_epochs
            if total_steps <= 0:
                raise RuntimeError("Resolved total training steps must be > 0.")
            return total_steps

        estimated_steps = None
        if hasattr(self, "_trainer") and self._trainer is not None:
            estimated_steps = self._trainer.estimated_stepping_batches
        if estimated_steps is not None:
            total_steps = int(estimated_steps)
            if total_steps <= 0:
                raise RuntimeError("Resolved total training steps must be > 0.")
            return total_steps

        steps_per_epoch = self.training_config.optimizer.steps_per_epoch
        if steps_per_epoch is not None:
            total_steps = int(steps_per_epoch) * max_epochs
            if total_steps <= 0:
                raise RuntimeError("Resolved total training steps must be > 0.")
            return total_steps
        raise RuntimeError(
            "Training step count is unresolved: set training.steps_per_epoch or "
            "attach the module to a Trainer before configuring the scheduler."
        )

    def optimizer_param_groups(self) -> list[dict[str, Any]] | None:
        """Optional parameter groups for the optimizer."""
        return None

    def _build_optimizer(self) -> AdamW:
        params = self.optimizer_param_groups()
        if params is None:
            kwargs: dict[str, Any] = {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": self.optimizer_betas,
            }
            return AdamW(self.parameters(), **kwargs)
        completed_groups: list[dict[str, Any]] = []
        for raw_group in params:
            group = dict(raw_group)
            if "lr" not in group:
                group["lr"] = self.learning_rate
            if "weight_decay" not in group:
                group["weight_decay"] = self.weight_decay
            if "betas" not in group:
                group["betas"] = self.optimizer_betas
            completed_groups.append(group)
        return AdamW(completed_groups)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = self._build_optimizer()
        scheduler: LRScheduler

        if self.warmup_epochs is not None:
            warmup_epochs = int(self.warmup_epochs)
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=int(self.max_epochs) - warmup_epochs,
                    eta_min=self.min_lr,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=int(self.max_epochs),
                    eta_min=self.min_lr,
                )
            interval = "epoch"
        else:
            if self.warmup_steps is None:
                raise RuntimeError(
                    "Validated optimizer config has no explicit warmup_steps."
                )
            warmup_steps = self.warmup_steps
            total_steps = self._estimate_total_steps()
            if warmup_steps and warmup_steps >= total_steps:
                raise RuntimeError(
                    "training.warmup_steps must be less than the resolved total "
                    "training steps."
                )
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=self.min_lr,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=self.min_lr,
                )
            interval = "step"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }
