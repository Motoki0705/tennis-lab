"""Shared qualitative validation logging callback.

Collects a bounded random subset of validation samples during ``validation_step``
and delegates task-specific rendering to the LightningModule at epoch end.
Outputs are written to both TensorBoard and versioned artifact directories.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger


class QualitativeLoggingCallback(pl.Callback):
    """Collect validation samples and save qualitative visualizations.

    The callback randomly selects ``num_samples`` batches each validation epoch,
    then calls ``render_qualitative_samples`` on the LightningModule to produce
    task-specific visualizations.

    Args:
        every_n_epochs: Run qualitative logging every *n* validation epochs.
        num_samples: Number of validation batches to collect per epoch.
        enabled: Master switch; when ``False`` the callback is a no-op.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        num_samples: int = 4,
        enabled: bool = True,
        selection_mode: str = "random",
        selected_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(every_n_epochs, 1)
        self.num_samples = max(num_samples, 1)
        self.enabled = enabled
        self.selection_mode = selection_mode
        self.selected_indices = selected_indices

        # Populated during validation
        self._collected_batches: list[dict[str, Any]] = []
        self._collected_outputs: list[dict[str, Any]] = []
        self._selected_indices: set[int] = set()
        self._total_val_batches: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Determine which batch indices to collect for this epoch."""
        self._collected_batches.clear()
        self._collected_outputs.clear()
        self._selected_indices.clear()

        if not self._should_log(trainer):
            return

        # Estimate total validation batches
        total = self._estimate_total_batches(trainer)
        self._total_val_batches = total

        self._selected_indices = self._select_batch_indices(total)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect batch data if this index was selected."""
        if not self._should_log(trainer):
            return
        if batch_idx not in self._selected_indices:
            return

        # Detach and move to CPU to avoid GPU memory accumulation
        self._collected_batches.append(_detach_to_cpu(batch))
        if outputs is not None:
            self._collected_outputs.append(_detach_to_cpu(outputs))
        else:
            self._collected_outputs.append({})

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Render and save qualitative outputs."""
        if not self._should_log(trainer):
            self._collected_batches.clear()
            self._collected_outputs.clear()
            return

        if not self._collected_batches:
            return

        # Only rank-zero writes artifacts
        if trainer.global_rank != 0:
            self._collected_batches.clear()
            self._collected_outputs.clear()
            return

        # Resolve output directory
        epoch = trainer.current_epoch
        artifact_dir = self._resolve_artifact_dir(trainer, epoch)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Get TensorBoard SummaryWriter
        tb_writer = self._get_tb_writer(trainer)
        global_step = trainer.global_step

        # Delegate rendering to the task LightningModule
        if hasattr(pl_module, "render_qualitative_samples"):
            pl_module.render_qualitative_samples(
                batches=self._collected_batches,
                outputs=self._collected_outputs,
                artifact_dir=artifact_dir,
                tb_writer=tb_writer,
                global_step=global_step,
                epoch=epoch,
            )

        self._collected_batches.clear()
        self._collected_outputs.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_batch_indices(self, total: int) -> set[int]:
        if total <= 0:
            return set()
        if self.selection_mode == "random":
            n = min(self.num_samples, total)
            return set(random.sample(range(total), n))
        if self.selection_mode == "fixed_indices":
            if not self.selected_indices:
                raise ValueError(
                    "qualitative_logging.selection_mode='fixed_indices' requires "
                    "a non-empty selected_indices list."
                )
            indices = set(self.selected_indices)
            invalid = sorted(idx for idx in indices if idx < 0 or idx >= total)
            if invalid:
                raise ValueError(
                    "qualitative_logging.selected_indices contains out-of-range "
                    f"batch indices {invalid} for total validation batches={total}."
                )
            return indices
        raise ValueError(
            "qualitative_logging.selection_mode must be 'random' or "
            f"'fixed_indices', got {self.selection_mode!r}."
        )

    def _should_log(self, trainer: pl.Trainer) -> bool:
        """Check if qualitative logging should run this epoch."""
        if not self.enabled:
            return False
        if trainer.sanity_checking:
            return False
        return (trainer.current_epoch % self.every_n_epochs) == 0

    def _estimate_total_batches(self, trainer: pl.Trainer) -> int:
        """Estimate the total number of validation batches."""
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None:
            return self.num_samples

        if isinstance(val_dataloaders, list):
            dl = val_dataloaders[0] if val_dataloaders else None
        else:
            dl = val_dataloaders

        if dl is None:
            return self.num_samples

        try:
            return len(dl)
        except (TypeError, NotImplementedError):
            return self.num_samples

    def _resolve_artifact_dir(self, trainer: pl.Trainer, epoch: int) -> Path:
        """Build ``<log_dir>/qualitative/epoch_XXXX`` path."""
        log_dir = _get_log_dir(trainer)
        return log_dir / "qualitative" / f"epoch_{epoch:04d}"

    @staticmethod
    def _get_tb_writer(trainer: pl.Trainer) -> Any | None:
        """Extract the TensorBoard SummaryWriter from the trainer logger."""
        logger = trainer.logger
        if isinstance(logger, TensorBoardLogger):
            return logger.experiment
        return None


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _detach_to_cpu(data: Any) -> Any:
    """Recursively detach tensors and move to CPU."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: _detach_to_cpu(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        cls = type(data)
        return cls(_detach_to_cpu(v) for v in data)
    return data


def _get_log_dir(trainer: pl.Trainer) -> Path:
    """Get the log directory from the trainer's logger."""
    logger = trainer.logger
    if isinstance(logger, TensorBoardLogger):
        return Path(logger.log_dir)
    return Path("outputs")


def save_image_to_tensorboard(
    tb_writer: Any,
    tag: str,
    image: np.ndarray,
    global_step: int,
) -> None:
    """Write a BGR/RGB numpy image to TensorBoard.

    Args:
        tb_writer: TensorBoard SummaryWriter.
        tag: Tag name for the image.
        image: ``(H, W, 3)`` uint8 image (BGR from cv2 — will be converted to RGB).
        global_step: Global step for TensorBoard.
    """
    if tb_writer is None:
        return
    import cv2

    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    # TensorBoard expects (C, H, W) for add_image
    if image_rgb.ndim == 3:
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
    tb_writer.add_image(tag, image_rgb, global_step)
