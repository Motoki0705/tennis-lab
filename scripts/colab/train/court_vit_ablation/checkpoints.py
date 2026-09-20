"""Full-state resume checks; no weight-only fallback is allowed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf


def validate_checkpoint(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any], *, initial: bool
) -> None:
    for key in (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "hyper_parameters",
    ):
        if not checkpoint.get(key):
            raise ValueError(f"Resume checkpoint is missing full training state: {key}")
    epoch = checkpoint.get("epoch")
    step = checkpoint.get("global_step")
    if not isinstance(epoch, int) or not isinstance(step, int) or step < 1:
        raise ValueError("Invalid checkpoint epoch/global_step")
    if initial and (epoch != 17 or step != 29844):
        raise ValueError(
            f"Expected baseline epoch=17, step=29844; got {epoch=}, {step=}"
        )
    saved = OmegaConf.to_container(
        OmegaConf.create(checkpoint["hyper_parameters"]["config"]), resolve=True
    )
    current = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    assert isinstance(saved, dict) and isinstance(current, dict)
    for key in ("model", "loss", "data"):
        if saved[key] != current[key]:
            raise ValueError(f"Resume {key} differs from the fixed experiment recipe")
    for key in (
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "warmup_steps",
        "min_lr",
        "steps_per_epoch",
        "optimizer",
    ):
        if saved["training"][key] != current["training"][key]:
            raise ValueError(
                f"Resume training.{key} differs from the fixed experiment recipe"
            )
    for key in ("max_epochs", "accumulate_grad_batches", "precision"):
        if saved["training"]["trainer"][key] != current["training"]["trainer"][key]:
            raise ValueError(f"Resume training.trainer.{key} changed")
    if saved["run"]["seed"] != current["run"]["seed"]:
        raise ValueError("Resume seed changed")
