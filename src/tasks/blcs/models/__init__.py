"""BLCS model modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_blcs_model(config: DictConfig) -> nn.Module:
    """Build BLCS model from config `model.name`."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "blcs"))
    if model_name == "blcs":
        return BLCSModel.from_config(config)
    if model_name == "blcs_multiview":
        return BLCSMultiViewModel.from_config(config)
    raise ValueError(
        "Unknown BLCS model.name="
        f"'{model_name}'. Supported: ['blcs', 'blcs_multiview']"
    )


__all__ = [
    "BLCSModel",
    "BLCSMultiViewModel",
    "build_blcs_model",
]
