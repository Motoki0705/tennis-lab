"""PLCS model architectures and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.plcs.models.components import PositionHead, RotationHead
from src.plcs.models.plcs_kp3d_model import PLCSKeypoint3DModel
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.models.plcs_query_sequence_model import PLCSQuerySequenceModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_plcs_model(config: DictConfig) -> nn.Module:
    """Build a PLCS model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "plcs"))
    if model_name == "plcs":
        return PLCSModel.from_config(config)
    if model_name == "plcs_query_sequence":
        return PLCSQuerySequenceModel.from_config(config)
    if model_name == "plcs_multiview":
        return PLCSMultiViewModel.from_config(config)
    if model_name == "plcs_kp3d":
        return PLCSKeypoint3DModel.from_config(config)
    raise ValueError(
        "Unknown PLCS model.name="
        f"'{model_name}'. Supported: "
        "['plcs', 'plcs_query_sequence', 'plcs_multiview', 'plcs_kp3d']"
    )

__all__ = [
    "PLCSModel",
    "PLCSQuerySequenceModel",
    "build_plcs_model",
    "PLCSMultiViewModel",
    "PLCSKeypoint3DModel",
    "PerTokenKeypoint3DHead",
    "PositionHead",
    "RotationHead",
]
