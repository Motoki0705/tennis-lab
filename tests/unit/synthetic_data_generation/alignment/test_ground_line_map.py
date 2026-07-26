"""Tests for ray-plane projection and per-view line-map accumulation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.ground_line_map import (
    GROUND_LINE_MAP_SCHEMA,
    GroundLineAccumulator,
    GroundLineMapSettings,
    ProjectedLinePixels,
    load_ground_line_map_artifact,
    project_line_pixels_to_ground,
    publish_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.synthetic_data_generation.scene_contract import SceneCamera


def _plane() -> GroundPlaneEstimate:
    return GroundPlaneEstimate(
        normal=(0.0, 0.0, 1.0),
        offset=0.0,
        origin=(0.0, 0.0, 0.0),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
        metrics={},
    )


def _camera() -> SceneCamera:
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = (0.0, 0.0, 1.0)
    return SceneCamera(
        camera_id="camera-0",
        source_camera_id="synthetic",
        image_uri="images/camera-0.png",
        source_frame_index=0,
        group_id=0,
        width=201,
        height=201,
        intrinsics=(100.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def test_projection_intersects_ground_and_downweights_distant_pixels() -> None:
    settings = GroundLineMapSettings(
        proximity_scale=1.0,
        max_ray_distance=5.0,
    )
    projection = project_line_pixels_to_ground(
        _camera(),
        np.asarray([[100.0, 100.0], [200.0, 100.0]]),
        np.asarray([0.9, 0.8]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=settings,
    )

    np.testing.assert_allclose(
        projection.points_scene,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-8,
    )
    assert projection.camera_ranges[0] == pytest.approx(1.0)
    assert projection.camera_ranges[1] == pytest.approx(np.sqrt(2.0))
    assert projection.proximity_weights[0] > projection.proximity_weights[1]


def test_projection_rejects_rays_behind_camera() -> None:
    camera = _camera()
    pose = np.asarray(camera.camera_to_scene).reshape(4, 4).copy()
    pose[:3, :3] = np.eye(3)
    upward_camera = replace(
        camera,
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )

    projection = project_line_pixels_to_ground(
        upward_camera,
        np.asarray([[100.0, 100.0]]),
        np.asarray([0.9]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=GroundLineMapSettings(),
    )

    assert len(projection.points_uv) == 0
    assert projection.invalid_behind_count == 1


def test_projection_rejects_parallel_rays_without_nonfinite_intersections() -> None:
    camera = _camera()
    pose = np.asarray(camera.camera_to_scene).reshape(4, 4).copy()
    pose[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    horizontal_camera = replace(
        camera,
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )

    projection = project_line_pixels_to_ground(
        horizontal_camera,
        np.asarray([[100.0, 100.0]]),
        np.asarray([0.9]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=GroundLineMapSettings(),
    )

    assert len(projection.points_uv) == 0
    assert projection.invalid_parallel_count == 1


def test_accumulator_limits_duplicate_pixels_to_one_view_contribution() -> None:
    projection = ProjectedLinePixels(
        points_scene=np.asarray([[0.0, 0.0, 0.0], [0.01, 0.01, 0.0]]),
        points_uv=np.asarray([[0.0, 0.0], [0.01, 0.01]]),
        probabilities=np.asarray([0.6, 0.9], dtype=np.float32),
        camera_ranges=np.asarray([1.0, 1.0]),
        proximity_weights=np.asarray([0.5, 0.5]),
        input_count=2,
        invalid_parallel_count=0,
        invalid_behind_count=0,
        invalid_range_count=0,
        invalid_bounds_count=0,
    )
    accumulator = GroundLineAccumulator(
        bounds=(-1.0, 1.0, -1.0, 1.0),
        grid_spacing=0.1,
    )

    assert accumulator.add_view(projection) == 1
    arrays = accumulator.arrays()
    assert arrays["view_count"].max() == 1
    assert arrays["evidence_sum"].max() == pytest.approx(0.45)
    assert arrays["mean_probability"].max() == pytest.approx(0.9)


def test_artifact_round_trip_refuses_overwrite_and_detects_tampering(
    tmp_path: Path,
) -> None:
    accumulator = GroundLineAccumulator(
        bounds=(-1.0, 1.0, -1.0, 1.0),
        grid_spacing=0.1,
    )
    payload = {
        "schema": GROUND_LINE_MAP_SCHEMA,
        "artifact_id": "synthetic-ground-lines-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "provider": {},
        "split": {
            "fit_camera_ids": ["fit-0"],
            "holdout_camera_ids": ["holdout-0"],
            "holdout_inference_status": "not_run",
        },
        "detector": {},
        "ground_plane": {},
        "projection": {},
        "records": [{"camera_id": "fit-0"}],
        "summary": {},
        "provenance": {},
    }

    path = publish_ground_line_map_artifact(
        payload,
        arrays=accumulator.arrays(),
        output_dir=tmp_path,
    )
    manifest, arrays = load_ground_line_map_artifact(path)

    assert manifest["split"]["holdout_inference_status"] == "not_run"
    assert arrays["evidence_sum"].shape == accumulator.evidence_sum.shape
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_ground_line_map_artifact(
            payload,
            arrays=accumulator.arrays(),
            output_dir=tmp_path,
        )

    manifest_path = path / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["summary"]["changed"] = True
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_ground_line_map_artifact(path)
