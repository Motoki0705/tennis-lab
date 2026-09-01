"""Canonical 3DGS-backed generated-dataset visualization API."""

from src.synthetic_data_generation.visualization.contracts import (
    DEFAULT_COURT_OVERLAY_CONFIGURATION,
    VISUALIZATION_METADATA_SCHEMA,
    VISUALIZATION_METADATA_SCHEMA_V2,
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtAABBWireframeTopology,
    CourtOverlayConfiguration,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    DatasetVisualizationResult,
)
from src.synthetic_data_generation.visualization.renderer import visualize_dataset

__all__ = [
    "DatasetVisualizationDomain",
    "DatasetVisualizationRequest",
    "DatasetVisualizationResult",
    "DEFAULT_COURT_OVERLAY_CONFIGURATION",
    "VISUALIZATION_METADATA_SCHEMA",
    "VISUALIZATION_METADATA_SCHEMA_V2",
    "CourtOverlayConfiguration",
    "CourtOverlayMode",
    "CourtAABBRenderStyle",
    "CourtAABBTrajectoryFilterRadiusMode",
    "CourtAABBTrajectoryFilterScope",
    "CourtAABBWireframeTopology",
    "visualize_dataset",
]
