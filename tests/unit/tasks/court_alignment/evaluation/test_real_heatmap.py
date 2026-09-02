"""Pure coordinate and archive tests for measured alignment evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.base.training.repro import QueueReproDirError
from src.tasks.court_alignment.evaluation.real_heatmap import (
    GROUND_PLANE_CONVENTION,
    AcceptedCourtAlignment,
    AlignmentGroundPlane,
    PixelUVTransform,
    PreprocessOptions,
    ProjectedReference,
    RealHeatmapArchive,
    compute_pose_alignment_diagnostics,
    letterbox_heatmap,
    project_accepted_reference,
    write_evaluation_artifacts,
)
from src.tasks.court_alignment.geometry.court import canonical_court_keypoints
from src.tasks.court_alignment.inference.decoder import CourtInstanceBatch
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX


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


def _similarity(
    points: torch.Tensor,
    *,
    scale: float,
    angle: float,
    translation: tuple[float, float],
) -> torch.Tensor:
    cosine = torch.cos(torch.tensor(angle, dtype=points.dtype))
    sine = torch.sin(torch.tensor(angle, dtype=points.dtype))
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        )
    )
    return points @ (scale * rotation).T + points.new_tensor(translation)


def _pose_diagnostic_fixture() -> tuple[
    CourtInstanceBatch,
    ProjectedReference,
    PixelUVTransform,
]:
    canonical = canonical_court_keypoints(dtype=torch.float32)
    references = torch.stack(
        (
            _similarity(
                canonical,
                scale=2.2,
                angle=0.23,
                translation=(72.0, 96.0),
            ),
            _similarity(
                canonical,
                scale=1.7,
                angle=-0.41,
                translation=(181.0, 137.0),
            ),
        )
    )
    half_turn = torch.as_tensor(CAMERA_VIEW_HALF_TURN_INDEX, dtype=torch.long)
    direct_indices = torch.tensor((0, 1, 2, 4, 5, 7, 9, 11, 13))
    half_indices = torch.tensor((0, 1, 3, 4, 6, 8, 10, 12, 13))
    keypoints = torch.zeros((3, 14, 2), dtype=torch.float32)
    valid = torch.zeros((3, 14), dtype=torch.bool)
    direct_noise = torch.tensor(
        (
            (0.12, -0.08),
            (-0.05, 0.16),
            (0.09, 0.04),
            (-0.11, -0.03),
            (0.02, 0.13),
            (0.07, -0.12),
            (-0.04, 0.06),
            (0.15, 0.01),
            (-0.08, -0.09),
        )
    )
    half_noise = direct_noise.flip(0) * 0.8
    # Prediction order is deliberately opposite the reference order.  The
    # first prediction also uses half-turn semantic correspondence.
    keypoints[0, half_indices] = references[1, half_turn[half_indices]] + half_noise
    valid[0, half_indices] = True
    keypoints[1, direct_indices] = references[0, direct_indices] + direct_noise
    valid[1, direct_indices] = True
    keypoints[2, :3] = references[0, :3]
    valid[2, :3] = True
    sample = CourtInstanceBatch(
        keypoints_px=keypoints,
        scores=valid.float(),
        valid=valid,
        centers_px=torch.stack(
            (references[1].mean(0), references[0].mean(0), references[0].mean(0))
        ),
    )
    reference = ProjectedReference(
        keypoints_px=references.numpy(),
        valid=np.ones((2, 14), dtype=np.bool_),
        centers_px=references.mean(1).numpy(),
        court_instance_ids=("court-a", "court-b"),
        candidate_ids=("candidate-a", "candidate-b"),
        poses=({"court_instance_id": "court-a"}, {"court_instance_id": "court-b"}),
    )
    transform = PixelUVTransform(
        bounds_uv=(0.0, 127.5, 0.0, 127.5),
        grid_spacing_m=0.5,
        source_shape=(256, 256),
        content_shape=(256, 256),
        output_shape=(256, 256),
        padding_xy=(0, 0),
    )
    return sample, reference, transform


def test_pose_diagnostic_matches_partial_half_turn_courts_one_to_one() -> None:
    sample, reference, transform = _pose_diagnostic_fixture()

    summary, diagnostic = compute_pose_alignment_diagnostics(
        sample,
        reference,
        transform,
        match_max_error_px=8.0,
    )

    assert summary["pose_diagnostic_status"] == "partial"
    assert summary["pose_prediction_instance_count"] == 3
    assert summary["pose_reference_instance_count"] == 2
    assert summary["pose_fit_available_prediction_count"] == 2
    assert summary["pose_fit_unavailable_prediction_count"] == 1
    assert summary["pose_matched_instance_count"] == 2
    assert summary["pose_rejected_over_gate_pair_count"] == 0
    assert summary["pose_unmatched_prediction_count"] == 1
    assert summary["pose_unmatched_reference_count"] == 0
    assert summary["pose_half_turn_match_count"] == 1
    assert summary["pose_raw_kp_count"] == 18
    assert summary["pose_raw_kp_coverage"] == pytest.approx(18 / 28)
    assert summary["pose_reconstructed_kp_count"] == 28
    assert cast(float, summary["pose_raw_kp_error_mean_px"]) > 0.0
    assert cast(float, summary["pose_reconstructed_kp_error_q95_px"]) < 0.25
    assert summary["pose_raw_kp_error_mean_m"] == pytest.approx(
        cast(float, summary["pose_raw_kp_error_mean_px"]) / 2.0
    )
    assert summary["pose_reconstructed_kp_error_q95_m"] == pytest.approx(
        cast(float, summary["pose_reconstructed_kp_error_q95_px"]) / 2.0
    )
    matches = cast(list[dict[str, object]], diagnostic["matches"])
    assert {
        (item["prediction_index"], item["reference_index"]) for item in matches
    } == {(1, 0), (0, 1)}
    half_match = next(item for item in matches if item["prediction_index"] == 0)
    assert half_match["half_turn_selected"] is True
    assert half_match["correspondence"] == "half_turn"
    assert diagnostic["unmatched_prediction_indices"] == [2]
    prediction_fits = cast(list[dict[str, object]], diagnostic["prediction_fits"])
    assert prediction_fits[2]["status"] == "unavailable"
    assert "at least 4" in cast(str, prediction_fits[2]["reason"])
    assert diagnostic["summary"] == summary


def test_pose_diagnostic_rejects_far_four_keypoint_false_cluster() -> None:
    _sample_value, reference, transform = _pose_diagnostic_fixture()
    single_reference = ProjectedReference(
        keypoints_px=reference.keypoints_px[:1],
        valid=reference.valid[:1],
        centers_px=reference.centers_px[:1],
        court_instance_ids=reference.court_instance_ids[:1],
        candidate_ids=reference.candidate_ids[:1],
        poses=reference.poses[:1],
    )
    canonical = canonical_court_keypoints(dtype=torch.float32)
    far_court = _similarity(
        canonical,
        scale=2.0,
        angle=0.2,
        translation=(500.0, 500.0),
    )
    valid = torch.zeros((1, 14), dtype=torch.bool)
    valid[0, :4] = True
    keypoints = torch.zeros((1, 14, 2), dtype=torch.float32)
    keypoints[0, :4] = far_court[:4]
    false_cluster = CourtInstanceBatch(
        keypoints_px=keypoints,
        scores=valid.float(),
        valid=valid,
        centers_px=torch.tensor(((500.0, 500.0),), dtype=torch.float32),
    )

    summary, diagnostic = compute_pose_alignment_diagnostics(
        false_cluster,
        single_reference,
        transform,
        match_max_error_px=8.0,
    )

    assert summary["pose_diagnostic_status"] == "unavailable"
    assert summary["pose_fit_available_prediction_count"] == 1
    assert summary["pose_fit_unavailable_prediction_count"] == 0
    assert summary["pose_matched_instance_count"] == 0
    assert summary["pose_rejected_over_gate_pair_count"] == 1
    assert summary["pose_unmatched_prediction_count"] == 1
    assert summary["pose_unmatched_reference_count"] == 1
    assert summary["pose_raw_kp_count"] == 0
    assert summary["pose_reconstructed_kp_count"] == 0
    assert summary["pose_raw_kp_error_mean_px"] is None
    assert summary["pose_reconstructed_kp_error_q95_px"] is None
    rejected = cast(list[dict[str, object]], diagnostic["rejected_over_gate_pairs"])
    assert len(rejected) == 1
    assert rejected[0]["status"] == "unavailable"
    assert cast(float, rejected[0]["pair_cost_px"]) > 8.0
    assert "exceeds match_max_error_px=8.0" in cast(str, rejected[0]["reason"])


def test_pose_metrics_and_diagnostic_artifacts_share_identical_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    sample, reference, transform = _pose_diagnostic_fixture()
    summary, pose_diagnostic = compute_pose_alignment_diagnostics(
        sample,
        reference,
        transform,
        match_max_error_px=8.0,
    )

    write_evaluation_artifacts(
        tmp_path,
        metrics={
            **summary,
            "reference_type": "accepted_alignment",
            "reference_is_independent_ground_truth": False,
        },
        diagnostic={
            "reference": {
                "type": "accepted_alignment",
                "is_independent_ground_truth": False,
            },
            "pose_alignment": pose_diagnostic,
        },
        arrays={"input": np.zeros((1, 1, 2, 2), dtype=np.float32)},
    )

    metrics_json = json.loads((tmp_path / "metrics.json").read_text())
    diagnostic_json = json.loads((tmp_path / "diagnostic_metrics.json").read_text())
    assert diagnostic_json["pose_alignment"]["summary"] == {
        key: metrics_json[key] for key in summary
    }
    assert metrics_json["reference_is_independent_ground_truth"] is False
    assert diagnostic_json["reference"]["is_independent_ground_truth"] is False


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
