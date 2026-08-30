"""Tests for strict raw and weighted alignment line heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
    aggregate_line_heatmaps,
    rasterize_weighted_view,
    validate_line_heatmaps,
    write_line_heatmaps,
)


def test_weighted_projection_uses_view_cell_max_then_global_sum() -> None:
    heatmaps = _heatmaps()

    first, _first_weight = rasterize_weighted_view(heatmaps, heatmaps.views[0])
    aggregate = aggregate_line_heatmaps(heatmaps)

    assert heatmaps.raster_shape == (3, 3)
    assert first[0, 0] == pytest.approx(0.25)
    assert first[1, 1] == pytest.approx(0.4)
    assert aggregate.evidence_sum[0, 0] == pytest.approx(0.5)
    assert aggregate.weight_sum[0, 0] == pytest.approx(0.75)
    assert aggregate.view_count[0, 0] == 2
    assert aggregate.mean_probability[0, 0] == pytest.approx(2.0 / 3.0)
    assert aggregate.evidence_sum[2, 2] == 0.0


def test_line_heatmaps_round_trip_numeric_and_png_inventory(tmp_path: Path) -> None:
    output = tmp_path / "line-heatmaps"
    source = _heatmaps()

    write_line_heatmaps(output, heatmaps=source)
    loaded = validate_line_heatmaps(output)

    assert loaded.camera_ids == ("camera-a", "camera-b", "camera-excluded")
    assert loaded.aggregate_camera_ids == ("camera-a", "camera-b")
    assert {path.name for path in output.iterdir()} == {
        "heatmaps.npz",
        "manifest.json",
        "weighted-projection.png",
        "views",
    }
    assert len(tuple((output / "views").iterdir())) == 6
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weight_model"] == (
        "1/(1+(camera_range/proximity_scale)^power)"
    )
    assert manifest["raster_reducer"] == (
        "per-view cell max then weighted global sum"
    )
    assert manifest["view_count"] == 3
    assert manifest["aggregate_view_count"] == 2
    assert manifest["coordinate_units"] == "metres"
    assert manifest["coordinate_convention"].startswith(
        "right_handed_metric_scene_ground_plane_uv"
    )
    with Image.open(output / "weighted-projection.png") as image:
        assert image.mode == "RGB"
        assert image.size == (3, 3)


def test_line_heatmap_validation_rejects_render_tampering(tmp_path: Path) -> None:
    output = tmp_path / "line-heatmaps"
    write_line_heatmaps(output, heatmaps=_heatmaps())
    tampered = output / "views/view-000-weighted-heatmap.png"
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8), mode="RGB").save(tampered)

    with pytest.raises(ValueError, match="disagrees with numeric evidence"):
        validate_line_heatmaps(output)


def test_line_heatmap_validation_rejects_foreign_coordinate_frame(
    tmp_path: Path,
) -> None:
    output = tmp_path / "line-heatmaps"
    write_line_heatmaps(output, heatmaps=_heatmaps())
    archive = output / "heatmaps.npz"
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["coordinate_units"] = np.asarray("normalized_scene_units")
    np.savez_compressed(archive, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="coordinate_units"):
        validate_line_heatmaps(output)


def test_line_heatmap_view_rejects_non_positive_proximity_weight() -> None:
    with pytest.raises(ValueError, match="proximity_weights"):
        AlignmentLineHeatmapView(
            camera_id="camera-a",
            probability=np.ones((2, 2), dtype=np.float32),
            points_uv=np.asarray(((0.0, 0.0),), dtype=np.float64),
            projected_probabilities=np.asarray((0.8,), dtype=np.float32),
            proximity_weights=np.asarray((0.0,), dtype=np.float64),
            included_in_aggregate=True,
        )


def _heatmaps() -> AlignmentLineHeatmaps:
    return AlignmentLineHeatmaps(
        bounds_uv=(0.0, 2.0, 0.0, 2.0),
        grid_spacing=1.0,
        proximity_scale=0.35,
        proximity_power=2.0,
        views=(
            AlignmentLineHeatmapView(
                camera_id="camera-a",
                probability=np.asarray(((0.0, 0.5), (0.75, 1.0)), dtype=np.float32),
                points_uv=np.asarray(
                    ((0.1, 0.1), (0.2, 0.2), (1.1, 1.1)), dtype=np.float64
                ),
                projected_probabilities=np.asarray(
                    (0.5, 0.9, 0.8), dtype=np.float32
                ),
                proximity_weights=np.asarray((0.5, 0.25, 0.5), dtype=np.float64),
                included_in_aggregate=True,
            ),
            AlignmentLineHeatmapView(
                camera_id="camera-b",
                probability=np.asarray(
                    ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), dtype=np.float32
                ),
                points_uv=np.asarray(((0.4, 0.4),), dtype=np.float64),
                projected_probabilities=np.asarray((1.0,), dtype=np.float32),
                proximity_weights=np.asarray((0.25,), dtype=np.float64),
                included_in_aggregate=True,
            ),
            AlignmentLineHeatmapView(
                camera_id="camera-excluded",
                probability=np.asarray(((0.0, 0.0), (0.0, 1.0)), dtype=np.float32),
                points_uv=np.asarray(((2.0, 2.0),), dtype=np.float64),
                projected_probabilities=np.asarray((1.0,), dtype=np.float32),
                proximity_weights=np.asarray((1.0,), dtype=np.float64),
                included_in_aggregate=False,
            ),
        ),
    )
