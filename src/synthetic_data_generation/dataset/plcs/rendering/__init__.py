"""Public NHT-backed PLCS frame rendering contracts."""

from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
)
from src.synthetic_data_generation.dataset.plcs.rendering.nht import NHTPLCSRenderer

__all__ = ["NHTPLCSRenderer", "PLCSForegroundCompositor"]
