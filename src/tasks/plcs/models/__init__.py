"""PLCS model architectures and factory."""

from __future__ import annotations

from torch import nn

from src.tasks.plcs.configuration import PLCSTrainingConfig
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


def build_plcs_model(config: PLCSTrainingConfig) -> nn.Module:
    """Build a PLCS model from ``config.model.name``."""
    model_cfg = config.model
    model_name = model_cfg.name
    num_court_tokens = (
        14 if model_name == "plcs_track_query" else config.data.num_court_tokens
    )
    if num_court_tokens is None:
        raise AssertionError("Non-tracking PLCS model requires num_court_tokens.")
    if model_name == "plcs":
        return PLCSModel.from_config(model_cfg, num_court_tokens=num_court_tokens)
    if model_name == "plcs_multiview":
        return PLCSMultiViewModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
    if model_name == "plcs_multiview_axial":
        return PLCSMultiViewAxialModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
    if model_name == "plcs_multiview_axial_split":
        return PLCSMultiViewAxialSplitModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
    if model_name == "plcs_multiview_axial_camtoken":
        return PLCSMultiViewAxialCamTokenModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
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
