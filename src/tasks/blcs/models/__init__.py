"""BLCS model modules."""

from __future__ import annotations

from torch import nn

from src.tasks.blcs.configuration import (
    AxialModelConfig,
    MultiViewModelConfig,
    SingleModelConfig,
    TrackQueryModelConfig,
    parse_model_config,
)
from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel
from src.tasks.blcs.models.discriminators import build_blcs_discriminator


def build_blcs_model(config: object) -> nn.Module:
    """Build BLCS model from config `model.name`."""
    model_cfg = parse_model_config(config)
    model_name = model_cfg.name
    if isinstance(model_cfg, SingleModelConfig):
        return BLCSModel.from_config(model_cfg)
    if isinstance(model_cfg, MultiViewModelConfig):
        return BLCSMultiViewModel.from_config(model_cfg)
    if isinstance(model_cfg, AxialModelConfig):
        return BLCSMultiViewAxialModel.from_config(model_cfg)
    if isinstance(model_cfg, TrackQueryModelConfig):
        return BLCSTrackQueryModel(model_cfg)
    raise ValueError(
        "Unknown BLCS model.name="
        f"'{model_name}'. Supported: ['blcs', 'blcs_multiview', "
        "'blcs_multiview_axial', 'blcs_track_query']"
    )


__all__ = [
    "BLCSModel",
    "BLCSTrackQueryModel",
    "BLCSMultiViewModel",
    "BLCSMultiViewAxialModel",
    "build_blcs_discriminator",
    "build_blcs_model",
]
