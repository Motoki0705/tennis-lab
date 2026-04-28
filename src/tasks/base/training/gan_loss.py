"""Least-squares GAN loss helpers shared across tasks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LSGANLoss(nn.Module):
    """Least-squares GAN losses for generator and discriminator."""

    def __init__(
        self,
        *,
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.real_label = float(real_label)
        self.fake_label = float(fake_label)

    def generator_loss(self, fake_logits: Tensor) -> Tensor:
        """Encourage generated samples to be classified as real."""
        target = torch.full_like(fake_logits, self.real_label)
        return 0.5 * ((fake_logits - target) ** 2).mean()

    def discriminator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Classify real samples as 1 and fake samples as 0."""
        real_target = torch.full_like(real_logits, self.real_label)
        fake_target = torch.full_like(fake_logits, self.fake_label)
        real_loss = (real_logits - real_target) ** 2
        fake_loss = (fake_logits - fake_target) ** 2
        return 0.5 * (real_loss.mean() + fake_loss.mean())