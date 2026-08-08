"""Public file/subprocess boundary for the independently managed NHT renderer."""

from src.synthetic_data_generation.rendering.nht.client import NHTRenderClient
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHT_RENDER_COMMAND,
    NHTRenderArrays,
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderEvidence,
    NHTRenderRecord,
    NHTRenderRequest,
    NHTRenderResult,
)
from src.synthetic_data_generation.rendering.nht.depth import nht_depth_to_metric

__all__ = [
    "NHT_RENDER_COMMAND",
    "NHTRenderArrays",
    "NHTRenderCamera",
    "NHTRenderClient",
    "NHTRenderCommandRequest",
    "NHTRenderEvidence",
    "NHTRenderRecord",
    "NHTRenderRequest",
    "NHTRenderResult",
    "nht_depth_to_metric",
]
