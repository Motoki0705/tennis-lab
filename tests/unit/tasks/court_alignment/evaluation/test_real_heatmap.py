"""Pure coordinate and archive tests for measured alignment evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.base.training.repro import QueueReproDirError
from src.tasks.court_alignment.evaluation.real_heatmap import (
    GROUND_PLANE_CONVENTION,
    AcceptedCourtAlignment,
    AlignmentGroundPlane,
    PixelUVTransform,
    PreprocessOptions,
    RealHeatmapArchive,
    letterbox_heatmap,
    project_accepted_reference,
    write_evaluation_artifacts,
)


def _transform() -> PixelUVTransform:
    return PixelUVTransform(
        bounds_uv=(-4.0, 5.0, -3.0, 6.0),
        grid_spacing_m=1.0,
        source_shape=(10, 10),
        content_shape=(8, 8),
        output_shape=(10, 12),
        padding_xy=(2, 1),
    )


def test_uv_pixel_round_trip_and_vertical_axis_convention() -> None:
    transform = _transform()
    points_uv = np.asarray(((-4.0, -3.0), (1.25, 4.5)), dtype=np.float64)

    points_px = transform.uv_to_model_px(points_uv)
    recovered = transform.model_px_to_uv(points_px)

    np.testing.assert_allclose(recovered, points_uv, atol=1.0e-12, rtol=0.0)
    assert points_px[1, 1] < points_px[0, 1]


def test_letterbox_performs_required_vertical_flip() -> None:
    source = np.asarray(((0.1, 0.2), (0.8, 0.9)), dtype=np.float32)
    options = PreprocessOptions(method="nearest", output_size=(2, 2))

    image, transform = letterbox_heatmap(
        source,
        bounds_uv=(0.0, 1.0, 0.0, 1.0),
        grid_spacing_m=1.0,
        options=options,
    )

    np.testing.assert_array_equal(image, np.flipud(source))
    assert transform.vertical_flip is True


def test_letterbox_preserves_aspect_and_centers_content() -> None:
    source: NDArray[np.float32] = np.ones((4, 2), dtype=np.float32)

    image, transform = letterbox_heatmap(
        source,
        bounds_uv=(0.0, 1.0, 0.0, 3.0),
        grid_spacing_m=1.0,
        options=PreprocessOptions(method="area", output_size=(4, 4)),
    )

    assert transform.content_shape == (4, 2)
    assert transform.padding_xy == (1, 0)
    np.testing.assert_array_equal(image[:, 0], 0.0)
    np.testing.assert_array_equal(image[:, 1:3], 1.0)
    np.testing.assert_array_equal(image[:, 3], 0.0)


def test_content_fraction_scales_content_and_keeps_it_centered() -> None:
    source: NDArray[np.float32] = np.ones((4, 2), dtype=np.float32)

    image, transform = letterbox_heatmap(
        source,
        bounds_uv=(0.0, 1.0, 0.0, 3.0),
        grid_spacing_m=1.0,
        options=PreprocessOptions(
            method="area",
            output_size=(8, 8),
            content_fraction=0.5,
        ),
    )

    assert transform.content_shape == (4, 2)
    assert transform.padding_xy == (3, 2)
    assert int(np.count_nonzero(image)) == 8
    assert transform.pixels_per_metre == pytest.approx(1.0)


def test_max_reducer_preserves_peak_in_each_target_cell() -> None:
    source: NDArray[np.float32] = np.zeros((4, 4), dtype=np.float32)
    source[0, 1] = 0.75
    source[3, 3] = 0.9

    image, _transform_value = letterbox_heatmap(
        source,
        bounds_uv=(0.0, 3.0, 0.0, 3.0),
        grid_spacing_m=1.0,
        options=PreprocessOptions(method="max", output_size=(2, 2)),
    )

    # Vertical orientation swaps the two source row groups before reduction.
    assert image[0, 1] == pytest.approx(0.9)
    assert image[1, 0] == pytest.approx(0.75)
    assert float(image.max()) == pytest.approx(0.9)


def test_alignment_reference_projection_is_proper_model_similarity() -> None:
    matrix = np.eye(4, dtype=np.float64)
    alignment = AcceptedCourtAlignment(
        court_instance_id="court-000",
        candidate_id="candidate-000",
        scene_from_court=matrix,
    )
    plane = AlignmentGroundPlane.from_alignments((alignment,))
    transform = PixelUVTransform(
        bounds_uv=(-20.0, 20.0, -20.0, 20.0),
        grid_spacing_m=1.0,
        source_shape=(41, 41),
        content_shape=(41, 41),
        output_shape=(41, 41),
        padding_xy=(0, 0),
    )

    reference = project_accepted_reference(
        (alignment,), ground_plane=plane, transform=transform
    )

    assert reference.keypoints_px.shape == (1, 14, 2)
    assert bool(reference.valid.all())
    np.testing.assert_allclose(reference.centers_px, ((20.0, 20.0),), atol=1.0e-6)
    assert reference.poses[0]["scale_px_per_metre"] == pytest.approx(1.0)
    assert reference.poses[0]["rotation_deg"] == pytest.approx(0.0, abs=1.0e-6)


def test_pixel_error_conversion_uses_effective_letterbox_scale() -> None:
    transform = PixelUVTransform(
        bounds_uv=(0.0, 9.0, 0.0, 9.0),
        grid_spacing_m=0.5,
        source_shape=(10, 10),
        content_shape=(5, 5),
        output_shape=(8, 8),
        padding_xy=(1, 1),
    )

    assert transform.pixels_per_metre == pytest.approx(1.0)
    assert transform.pixels_to_metres(3.25) == pytest.approx(3.25)


def test_malformed_archive_fails_before_manifest_use(tmp_path: Path) -> None:
    archive_path = tmp_path / "heatmaps.npz"
    manifest_path = tmp_path / "manifest.json"
    np.savez_compressed(
        archive_path,
        schema=np.asarray("alignment_line_heatmaps_v2"),
        mean_probability=np.zeros((2, 2), dtype=np.float32),
    )
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="archive keys"):
        RealHeatmapArchive.load(archive_path, manifest_path)


def test_manifest_mismatch_is_an_explicit_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "heatmaps.npz"
    manifest_path = tmp_path / "manifest.json"
    camera_ids = np.asarray(("camera-0",))
    included = np.asarray((True,), dtype=np.bool_)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("alignment_line_heatmaps_v2"),
        "coordinate_convention": np.asarray(GROUND_PLANE_CONVENTION),
        "coordinate_units": np.asarray("metres"),
        "camera_ids": camera_ids,
        "included_in_aggregate": included,
        "bounds_uv": np.asarray((0.0, 1.0, 0.0, 1.0), dtype=np.float64),
        "grid_spacing": np.asarray(1.0, dtype=np.float64),
        "proximity_scale": np.asarray(1.0, dtype=np.float64),
        "proximity_power": np.asarray(2.0, dtype=np.float64),
        "probability_shapes": np.asarray(((2, 2),), dtype=np.int64),
        "probability_offsets": np.asarray((0, 4), dtype=np.int64),
        "probability_values": np.zeros(4, dtype=np.float32),
        "projected_offsets": np.asarray((0, 0), dtype=np.int64),
        "projected_points_uv": np.empty((0, 2), dtype=np.float64),
        "projected_probabilities": np.empty(0, dtype=np.float32),
        "proximity_weights": np.empty(0, dtype=np.float64),
        "evidence_sum": np.zeros((2, 2), dtype=np.float32),
        "weight_sum": np.zeros((2, 2), dtype=np.float32),
        "view_count": np.zeros((2, 2), dtype=np.uint16),
        "mean_probability": np.zeros((2, 2), dtype=np.float32),
    }
    cast(Any, np.savez_compressed)(archive_path, **arrays)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "wrong-schema",
                "coordinate_convention": GROUND_PLANE_CONVENTION,
                "coordinate_units": "metres",
                "archive": "heatmaps.npz",
                "ground_png_orientation": "top row is maximum v (vertical flip)",
                "bounds_uv": [0.0, 1.0, 0.0, 1.0],
                "grid_spacing": 1.0,
                "raster_shape": [2, 2],
                "aggregate_view_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest field 'schema'"):
        RealHeatmapArchive.load(archive_path, manifest_path)


def test_invalid_preprocessor_name_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown real-heatmap preprocess"):
        PreprocessOptions(method="implicit-fallback", output_size=(256, 256))


@pytest.mark.parametrize("fraction", (0.0, -0.1, 1.01, float("inf")))
def test_invalid_content_fraction_fails_explicitly(fraction: float) -> None:
    with pytest.raises(ValueError, match="content_fraction"):
        PreprocessOptions(
            method="max", output_size=(256, 256), content_fraction=fraction
        )


def test_queue_repro_artifacts_are_atomic_byte_identical_mirrors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "normal-output"
    repro_dir = tmp_path / "queue-repro"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
    arrays: dict[str, NDArray[Any]] = {
        "input": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    }

    write_evaluation_artifacts(
        output_dir,
        metrics={"instance_f1": 0.5},
        diagnostic={"reference_type": "accepted_alignment"},
        arrays=arrays,
    )

    queue_output = repro_dir / "predictions"
    for name in ("metrics.json", "diagnostic_metrics.json", "pred_test.npz"):
        assert (output_dir / name).read_bytes() == (queue_output / name).read_bytes()
    assert not tuple(output_dir.glob(".*.tmp"))
    assert not tuple(queue_output.glob(".*.tmp"))


def test_invalid_queue_repro_dir_fails_before_normal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "normal-output"
    monkeypatch.setenv("TENNIS_REPRO_DIR", "relative/repro")

    with pytest.raises(QueueReproDirError, match="must be absolute"):
        write_evaluation_artifacts(
            output_dir,
            metrics={},
            diagnostic={},
            arrays={"input": np.zeros((1, 1, 2, 2), dtype=np.float32)},
        )

    assert not output_dir.exists()
