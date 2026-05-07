"""PLCS model architectures and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.plcs.models.components import CanonicalPoseHead, PositionHead, RotationHead
from src.tasks.plcs.models.discriminators import build_plcs_discriminator
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_plcs_model(config: DictConfig) -> nn.Module:
    """Build a PLCS model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "plcs"))
    if model_name == "plcs":
        return PLCSModel.from_config(config)
    if model_name == "plcs_multiview":
        return PLCSMultiViewModel.from_config(config)
    if model_name == "plcs_multiview_axial":
        return PLCSMultiViewAxialModel.from_config(config)
    raise ValueError(
        "Unknown PLCS model.name="
        f"'{model_name}'. Supported: ['plcs', 'plcs_multiview', 'plcs_multiview_axial']"
    )

__all__ = [
    "PLCSModel",
    "build_plcs_discriminator",
    "build_plcs_model",
    "PLCSMultiViewModel",
    "PLCSMultiViewAxialModel",
    "CanonicalPoseHead",
    "PositionHead",
    "RotationHead",
]
