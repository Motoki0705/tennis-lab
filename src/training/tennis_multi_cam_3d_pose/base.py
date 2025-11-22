"""Shared LightningModule utilities for TennisDETR variants."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class BaseTennisLightningModule(LightningModule):
    """Base module providing optimizer and scheduler helpers.

    Subclasses are expected to define the following attributes in __init__:

    - self._lr: float
    - self._weight_decay: float
    - self._max_steps: int
    - self._scheduler_cfg: dict[str, Any]
    """

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler shared across TennisDETR modules."""
        optimizer = AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _build_scheduler(
        self,
        optimizer: AdamW,
    ) -> CosineAnnealingLR | LambdaLR | None:
        """Return the configured LR scheduler or ``None`` if disabled."""
        if self._max_steps <= 0:
            return None
        scheduler_name = str(self._scheduler_cfg.get("name") or "").lower()
        if scheduler_name == "cosine_with_warmup":
            warmup_steps = int(self._scheduler_cfg.get("warmup_steps", 0))
            min_lr_ratio = float(self._scheduler_cfg.get("min_lr_ratio", 0.0))
            lr_lambda = self._build_warmup_cosine_lambda(warmup_steps, min_lr_ratio)
            return LambdaLR(optimizer, lr_lambda=lr_lambda)
        return CosineAnnealingLR(optimizer, T_max=self._max_steps)

    def _build_warmup_cosine_lambda(
        self,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> Callable[[int], float]:
        """Construct a lambda function implementing warmup + cosine decay."""
        warmup = max(0, int(warmup_steps))
        base_min_ratio = float(min_lr_ratio)
        max_steps = max(1, self._max_steps)

        def _lr_lambda(step: int) -> float:
            step_f = float(step)
            if warmup > 0 and step_f < warmup:
                return step_f / float(max(1, warmup))
            progress_steps = max(1, max_steps - warmup)
            progress = min(max((step_f - warmup) / progress_steps, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_min_ratio + (1.0 - base_min_ratio) * cos

        return _lr_lambda
