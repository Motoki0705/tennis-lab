"""Shared Lightning training utilities."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from omegaconf import DictConfig


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
    return np.concatenate(chunks, axis=0)  # type: ignore[no-any-return]


class BaseLightningModule(pl.LightningModule):
    """Base Lightning module with shared optimizer/scheduler logic.

    This class expects training settings under `config.training` and optional
    dataset sizing under `config.data`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config or {}

        train_cfg = self.config.get("training", {})
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.warmup_steps = train_cfg.get("warmup_steps")
        self.warmup_epochs = train_cfg.get("warmup_epochs")
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.min_lr = train_cfg.get("min_lr", 1e-6)
        optimizer_cfg = train_cfg.get("optimizer", {}) or {}
        betas = optimizer_cfg.get("betas")
        self.optimizer_betas = tuple(betas) if betas is not None else None

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
        dm = getattr(self._safe_trainer(), "datamodule", None)
        scenes = getattr(getattr(dm, "test_dataset", None), "scenes", None)
        start = getattr(self, "_test_pred_cursor", 0)
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
        if isinstance(value, Tensor):
            tensor = value.detach().cpu()
            # numpy has no bfloat16 (and to keep things uniform, half too):
            # upcast to float32 before converting, otherwise .numpy() raises
            # "Got unsupported ScalarType BFloat16" under bf16-mixed precision.
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return tensor.numpy()  # type: ignore[no-any-return]
        return np.asarray(value)  # type: ignore[no-any-return]

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

    def _test_predictions_dir(self) -> Path | None:
        base = os.environ.get("TENNIS_REPRO_DIR")
        if base:
            return Path(base) / "predictions"
        log_dir = getattr(self._safe_trainer(), "log_dir", None)
        if log_dir:
            return Path(log_dir) / "predictions"
        return None

    def save_test_predictions(
        self, metrics: dict[str, Any] | None = None
    ) -> Path | None:
        """Write accumulated test predictions to ``pred_test.npz`` (+ metrics.json).

        Returns the npz path, or ``None`` if there is nothing to save / no target
        directory could be resolved.
        """
        if not getattr(self, "_test_pred_arrays", None):
            return None
        out_dir = self._test_predictions_dir()
        if out_dir is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            key: _concat_padded(chunks)
            for key, chunks in self._test_pred_arrays.items()
        }
        # Fixed-width unicode (not object) so np.load works without allow_pickle.
        arrays["scene_ids"] = np.asarray(self._test_pred_scene_ids)
        npz_path = out_dir / "pred_test.npz"
        np.savez_compressed(npz_path, **arrays)  # type: ignore[arg-type]
        if metrics is not None:
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return npz_path

    def _estimate_total_steps(self) -> int:
        steps_per_epoch_attr = getattr(self, "steps_per_epoch", None)
        if steps_per_epoch_attr:
            return int(steps_per_epoch_attr) * int(self.max_epochs)

        estimated_steps = None
        if getattr(self, "_trainer", None) is not None:
            estimated_steps = getattr(self._trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None:
            return int(estimated_steps)

        data_cfg = self.config.get("data", {})
        sim_cfg = self.config.get("simulation", {})
        train_cfg = self.config.get("training", {})
        steps_per_epoch = train_cfg.get("steps_per_epoch")
        if steps_per_epoch is not None:
            return int(steps_per_epoch) * int(self.max_epochs)

        num_samples = data_cfg.get("num_scenes_per_epoch")
        if num_samples is None:
            num_samples = data_cfg.get("num_samples_per_epoch")
        if num_samples is None:
            num_samples = train_cfg.get("num_samples_per_epoch")
        if num_samples is None:
            num_samples = sim_cfg.get("num_train_scenes")
        if num_samples is None:
            num_samples = 10000
        batch_size = data_cfg.get("batch_size", 64)
        steps_per_epoch = max(num_samples // batch_size, 1)
        return steps_per_epoch * self.max_epochs

    def optimizer_param_groups(self) -> list[dict[str, Any]] | None:
        """Optional parameter groups for the optimizer."""
        return None

    def _build_optimizer(self) -> AdamW:
        params = self.optimizer_param_groups()
        if params is None:
            kwargs = {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
            if self.optimizer_betas is not None:
                kwargs["betas"] = self.optimizer_betas
            return AdamW(self.parameters(), **kwargs)
        for group in params:
            group.setdefault("lr", self.learning_rate)
            group.setdefault("weight_decay", self.weight_decay)
            if self.optimizer_betas is not None:
                group.setdefault("betas", self.optimizer_betas)
        return AdamW(params)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = self._build_optimizer()

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
                    T_max=max(int(self.max_epochs) - warmup_epochs, 1),
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
                    T_max=max(int(self.max_epochs), 1),
                    eta_min=self.min_lr,
                )
            interval = "epoch"
        else:
            warmup_steps = int(self.warmup_steps or 0)
            total_steps = self._estimate_total_steps()
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_steps - warmup_steps, 1),
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
                    T_max=max(total_steps, 1),
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
