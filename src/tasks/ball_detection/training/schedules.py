import math
from dataclasses import dataclass

import torch

"""Learning-rate schedules shared by pretraining and fine-tuning."""


def learning_rate(step: int, *, peak: float, total: int, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1.))))


@dataclass(frozen=True)
class FineTuneSchedule:
    peak: float
    total: int
    warmup: int
    freeze_epochs: int
    encoder_lr_ratio: float

    def apply(self, optimizer: torch.optim.Optimizer, *, step: int, frozen: bool) -> None:
        if [group.get("role") for group in optimizer.param_groups] != ["encoder", "decoder"]:
            raise ValueError("Fine-tuning requires declared encoder and decoder parameter groups")
        lr = learning_rate(step, peak=self.peak, total=self.total, warmup=self.warmup)
        optimizer.param_groups[0]["lr"] = 0. if frozen else lr * self.encoder_lr_ratio
        optimizer.param_groups[1]["lr"] = lr
