"""SLCS model factory."""

from __future__ import annotations

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig, SLCSModelConfig
from src.tasks.slcs.models.slcs_model import SLCSFusionModel


def build_slcs_model(
    model: SLCSModelConfig, data: SLCSDataRuntimeConfig
) -> SLCSFusionModel:
    """Build the sole model variant from validated typed configuration."""
    return SLCSFusionModel.from_config(model, data)


__all__ = ["SLCSFusionModel", "build_slcs_model"]
