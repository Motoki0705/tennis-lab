"""SLCS model factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.tasks.slcs.models.slcs_model import SLCSFusionModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

_MODEL_REGISTRY = {
    "slcs_fusion": SLCSFusionModel,
}


def build_slcs_model(config: DictConfig) -> SLCSFusionModel:
    """Build an SLCS model from a Hydra config (``model.name`` dispatch)."""
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "slcs_fusion"))
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.name={name!r}. Available: {sorted(_MODEL_REGISTRY)}."
        )
    return _MODEL_REGISTRY[name].from_config(config)


__all__ = ["SLCSFusionModel", "build_slcs_model"]
