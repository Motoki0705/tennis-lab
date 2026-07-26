"""PLCS model architectures and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.plcs.models.components import (
    CanonicalPoseHead,
    PositionHead,
    RotationHead,
)
from src.tasks.plcs.models.discriminators import build_plcs_discriminator
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_camtoken_model import (
    PLCSMultiViewAxialCamTokenModel,
)
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel

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
    if model_name == "plcs_multiview_axial_split":
        return PLCSMultiViewAxialSplitModel.from_config(config)
    if model_name == "plcs_multiview_axial_camtoken":
        return PLCSMultiViewAxialCamTokenModel.from_config(config)
    if model_name == "plcs_track_query":
        return PLCSTrackQueryModel(model_cfg)
    raise ValueError(
        "Unknown PLCS model.name="
        f"'{model_name}'. Supported: ['plcs', 'plcs_multiview', "
        "'plcs_multiview_axial', 'plcs_multiview_axial_split', "
        "'plcs_multiview_axial_camtoken', 'plcs_track_query']"
    )


__all__ = [
    "PLCSModel",
    "PLCSTrackQueryModel",
    "build_plcs_discriminator",
    "build_plcs_model",
    "PLCSMultiViewModel",
    "PLCSMultiViewAxialModel",
    "PLCSMultiViewAxialSplitModel",
    "PLCSMultiViewAxialCamTokenModel",
    "CanonicalPoseHead",
    "PositionHead",
    "RotationHead",
]
