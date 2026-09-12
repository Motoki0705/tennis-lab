"""Explicit automatic/manual baseline loading into one metric comparison frame."""

from dataclasses import dataclass, replace
from pathlib import Path

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentResult,
    GroundPlaneFrame,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    validate_line_heatmaps,
)
from src.synthetic_data_generation.alignment.manual.artifacts import load_manual_source
from src.synthetic_data_generation.alignment.validation import (
    load_alignment_evidence,
    validate_alignment_outputs,
)


@dataclass(frozen=True)
class ComparisonBaseline:
    result: AlignmentResult
    plane: GroundPlaneFrame
    heatmaps: AlignmentLineHeatmaps


def load_comparison_baseline(root: Path) -> ComparisonBaseline:
    """Validate owner first; manual heatmaps retain their pre-edit metric scale."""
    result = validate_alignment_outputs(root)
    heatmaps = validate_line_heatmaps(root / "line-heatmaps")
    if (root / "manual-confirmation.json").exists():
        source = load_manual_source(root)
        factor = (
            result.metric_adapter.nht_scene_units_per_metre
            / source.initial.metric_adapter.nht_scene_units_per_metre
        )
        plane = replace(
            source.plane,
            origin_metric_scene=tuple(
                value / factor for value in source.plane.origin_metric_scene
            ),
            bounds_uv_metres=tuple(
                value / factor for value in source.plane.bounds_uv_metres
            ),
        )
        heatmaps = replace(
            heatmaps,
            bounds_uv=plane.bounds_uv_metres,
            grid_spacing=heatmaps.grid_spacing / factor,
            proximity_scale=heatmaps.proximity_scale / factor,
            views=tuple(
                replace(view, points_uv=view.points_uv / factor)
                for view in heatmaps.views
            ),
        )
    else:
        evidence = load_alignment_evidence(root / "ground-line-map.npz")
        plane = evidence.ground_plane_frame
    if heatmaps.bounds_uv != plane.bounds_uv_metres:
        raise ValueError("Comparison heatmaps disagree with the metric ground plane.")
    return ComparisonBaseline(result, plane, heatmaps)
