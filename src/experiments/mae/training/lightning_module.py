"""PyTorch Lightning module for MAE training.

Handles training loop, logging, and checkpointing for MAE pre-training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import torch
import torch.nn as nn
from torch import Tensor
from src.base.training.lightning_module import BaseLightningModule
from src.mae.models import MAEModel


if TYPE_CHECKING:
    from omegaconf import DictConfig


class MAELightningModule(BaseLightningModule):
    """Lightning module for MAE pre-training.

    Handles:
    - Forward pass with masking
    - Loss computation
    - Optimizer and scheduler setup
    - Logging and visualization
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """Initialize MAE Lightning module.

        Args:
            config: Hydra configuration.

        """
        super().__init__(config)
        self.save_hyperparameters()

        self.config = config
        self.model = MAEModel.from_config(config)
        training_cfg = config.get("training", {}) or {}
        model_cfg = config.get("model", {}) or {}
        self.mask_ratio = model_cfg.get("mask_ratio", 0.75)
        self.log_reconstruction_every_n_epochs = training_cfg.get(
            "log_reconstruction_every_n_epochs", 10
        )

    def forward(
        self,
        images: Tensor,
        mask_ratio: Optional[float] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            images: Input images, shape (B, C, H, W).
            mask_ratio: Optional mask ratio override.

        Returns:
            Tuple of (loss, pred, mask).

        """
        return self.model(images, mask_ratio=mask_ratio or self.mask_ratio)

    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Training step.

        Args:
            batch: Dictionary with 'image' tensor.
            batch_idx: Batch index.

        Returns:
            Loss tensor.

        """
        images = batch["image"]
        loss, pred, mask = self(images)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mask_ratio", mask.mean(), sync_dist=True)

        # Log reconstruction visualization periodically
        if (
            self.current_epoch % self.log_reconstruction_every_n_epochs == 0
            and batch_idx == 0
            and self.logger is not None
        ):
            self._log_reconstruction(images, pred, mask, "train")

        return loss

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Validation step.

        Args:
            batch: Dictionary with 'image' tensor.
            batch_idx: Batch index.

        Returns:
            Loss tensor.

        """
        images = batch["image"]
        loss, pred, mask = self(images)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Log reconstruction for first batch
        if batch_idx == 0 and self.logger is not None:
            self._log_reconstruction(images, pred, mask, "val")

        return loss

    def _log_reconstruction(
        self,
        images: Tensor,
        pred: Tensor,
        mask: Tensor,
        prefix: str,
    ) -> None:
        """Log reconstruction visualization.

        Args:
            images: Original images, shape (B, C, H, W).
            pred: Predicted patches, shape (B, N, P*P*C).
            mask: Binary mask, shape (B, N).
            prefix: Log prefix ('train' or 'val').

        """
        try:
            # Only log first few images
            num_log = min(4, images.shape[0])

            # Unpatchify predictions
            B, C, H, W = images.shape
            P = self.model.patch_size
            h, w = H // P, W // P

            reconstructed = self.model.unpatchify(pred[:num_log], h, w)

            # Denormalize if needed
            if self.model.norm_pix_loss:
                # Approximate denormalization (not exact without storing stats)
                target = self.model.patchify(images[:num_log])
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                pred_denorm = pred[:num_log] * (var + 1e-6).sqrt() + mean
                reconstructed = self.model.unpatchify(pred_denorm, h, w)

            # Clamp to valid range
            reconstructed = torch.clamp(reconstructed, 0, 1)

            # Create visualization grid
            # Original | Masked | Reconstructed
            mask_vis = mask[:num_log].unsqueeze(-1).expand(-1, -1, P * P * C)
            mask_vis = self.model.unpatchify(mask_vis.float(), h, w)

            masked_images = images[:num_log] * (1 - mask_vis)

            # Log images
            if hasattr(self.logger, "experiment"):
                try:
                    import torchvision.utils as vutils

                    grid_orig = vutils.make_grid(images[:num_log], nrow=num_log)
                    grid_masked = vutils.make_grid(masked_images, nrow=num_log)
                    grid_recon = vutils.make_grid(reconstructed, nrow=num_log)

                    self.logger.experiment.add_image(
                        f"{prefix}/original", grid_orig, self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"{prefix}/masked", grid_masked, self.global_step
                    )
                    self.logger.experiment.add_image(
                        f"{prefix}/reconstructed", grid_recon, self.global_step
                    )
                except Exception:
                    pass
        except Exception:
            # Silently ignore visualization errors
            pass

    def optimizer_param_groups(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        betas = self.optimizer_betas if self.optimizer_betas is not None else (0.9, 0.95)
        return [
            {"params": decay_params, "weight_decay": self.weight_decay, "betas": betas},
            {"params": no_decay_params, "weight_decay": 0.0, "betas": betas},
        ]

    def get_encoder(self) -> nn.Module:
        """Get the pre-trained encoder for downstream tasks.

        Returns:
            ViT encoder module.

        """
        return self.model.get_encoder()

    @classmethod
    def from_config(cls, config) -> "MAELightningModule":
        """Create module from Hydra config.

        Args:
            config: Hydra configuration.

        Returns:
            Initialized module.

        """
        return cls(config)
