"""BLCS model modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.blcs.models.blcs_early_fusion_model import BLCSEarlyFusionModel
from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_early_fusion_model import (
    BLCSMultiViewEarlyFusionModel,
)
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.tasks.blcs.models.blcs_query_model import BLCSQueryModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_blcs_model(config: DictConfig) -> nn.Module:
    """Build BLCS model from config `model.name`."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "blcs"))
    if model_name == "blcs":
        return BLCSModel.from_config(config)
    if model_name == "blcs_early_fusion":
        return BLCSEarlyFusionModel.from_config(config)
    if model_name == "blcs_multiview":
        return BLCSMultiViewModel.from_config(config)
    if model_name == "blcs_multiview_early_fusion":
        return BLCSMultiViewEarlyFusionModel.from_config(config)
    if model_name == "blcs_query":
        return BLCSQueryModel.from_config(config)
    raise ValueError(
        "Unknown BLCS model.name="
        f"'{model_name}'. Supported: ['blcs', 'blcs_early_fusion', "
        "'blcs_multiview', 'blcs_multiview_early_fusion', 'blcs_query']"
    )


__all__ = [
    "BLCSModel",
    "BLCSEarlyFusionModel",
    "BLCSMultiViewModel",
    "BLCSMultiViewEarlyFusionModel",
    "BLCSQueryModel",
    "build_blcs_model",
]
