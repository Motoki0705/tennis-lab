"""Integration tests for strict Court V4 occupancy visualization publication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.visualization.sources as sources_module
from src.synthetic_data_generation.dataset.court.contracts import (
    SupportModelSummary,
    TrajectorySupportPolicy,
)
from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    build_court_v4_support_occupancy_snapshot,
    write_court_v4_support_occupancy,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization import (
    VISUALIZATION_METADATA_SCHEMA_V2,
    CourtOverlayConfiguration,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    visualize_dataset,
)
from src.utils.video.reader import probe_video_info

pytestmark = pytest.mark.integration


def _support_policy() -> TrajectorySupportPolicy:
    return TrajectorySupportPolicy(
        decision_id="integration-support-v1",
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
        maximum_camera_link_distance_m=1.5,
        maximum_source_frame_gap=1,
        occupancy_voxel_size_m=0.5,
        minimum_points_per_voxel=1,
        obstacle_inflation_m=0.5,
        camera_ball_clearance_m=0.05,
        camera_capsule_clearance_m=0.04,
        sweep_step_m=0.1,
        boundary_epsilon_m=1.0e-6,
        minimum_captured_cameras=2,
        minimum_public_points=1,
        maximum_capsule_index_cells=10_000,
        maximum_occupancy_cells=10_000,
        minimum_cycle_frame_span=8,
        maximum_cycle_frame_span=16,
        maximum_cycle_closure_distance_m=1.1,
        maximum_constructive_cycle_count=24,
        cycle_smoothing_distance_m=0.03,
    )


def _write_minimal_v4_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    policy = _support_policy()
    support_digest = "c" * 64
    summary = SupportModelSummary(
        input_digest=support_digest,
        coordinate_space="metric_scene_metres",
        captured_camera_count=2,
        public_point_count=1,
        density_qualified_voxel_count=1,
        raw_inflated_occupancy_cell_count=2,
        inflated_occupancy_cell_count=1,
        camera_ball_carved_cell_count=1,
        camera_capsule_carved_cell_count=0,
        captured_camera_occupied_count=0,
        endpoint_ball_count=2,
        capsule_count=1,
        skipped_gap_link_count=0,
        skipped_obstacle_link_count=0,
        capsule_index_cell_count=1,
    )
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=48,
        height=32,
        intrinsics=(40.0, 0.0, 23.5, 0.0, 40.0, 15.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="request-only",
    )
    projection: dict[str, object] = {"courts": []}
    records: list[dict[str, object]] = []
    for index, depth_m in enumerate((10.0, 1.0)):
        directory = root / "samples" / f"sample-{index}"
        directory.mkdir(parents=True)
        np.save(
            directory / "rgb.npy",
            np.zeros((32, 48, 3), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            directory / "alpha.npy",
            np.ones((32, 48, 1), dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            directory / "depth.npy",
            np.full((32, 48, 1), depth_m, dtype=np.float32),
            allow_pickle=False,
        )
        relative = f"samples/sample-{index}"
        label = {
            "schema": "canonical_court_sample_v4",
            "sample_id": f"sample-{index}",
            "view_id": "view-0",
            "trajectory_frame_index": index,
            "camera": camera.to_dict(),
            "projection": projection,
            "safety_support_input_digest": support_digest,
        }
        (directory / "labels.json").write_text(
            json.dumps(label),
            encoding="utf-8",
        )
        records.append(
            {
                "sample_id": f"sample-{index}",
                "trajectory_id": "orbit-0",
                "view_id": "view-0",
                "trajectory_frame_index": index,
                "width": 48,
                "height": 32,
                "camera": camera.to_dict(),
                "projection": projection,
                "rgb": f"{relative}/rgb.npy",
                "alpha": f"{relative}/alpha.npy",
                "depth": f"{relative}/depth.npy",
                "depth_coordinate_space": "metric_scene_metres",
                "labels": f"{relative}/labels.json",
                "safety_support_input_digest": support_digest,
            }
        )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "canonical_court_dataset_v4",
                "scene_id": "scene-0",
                "profile": "integration",
                "metrics": {"support_input_digest": support_digest},
                "trajectory_groups": [
                    {
                        "trajectory": {"trajectory_id": "orbit-0"},
                        "views": [{"view_id": "view-0"}],
                        "sample_count": 2,
                    }
                ],
                "samples": records,
            }
        ),
        encoding="utf-8",
    )
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    snapshot = build_court_v4_support_occupancy_snapshot(
        np.asarray(((0, 0, 4),), dtype=np.int64),
        voxel_size_m=0.5,
        support_input_digest=support_digest,
        policy_decision_id=policy.decision_id,
    )
    (diagnostics / "trajectory-plan.json").write_text(
        json.dumps(
            {
                "schema": "canonical_court_safe_path_plan_v4",
                "support_policy": policy.to_dict(),
                "support_summary": summary.to_dict(),
                "support_occupancy_identity": snapshot.identity.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    write_court_v4_support_occupancy(
        diagnostics,
        snapshot=snapshot,
        scene_id="scene-0",
        profile="integration",
    )


def test_v4_source_rasterizer_and_sidecar_publish_visible_and_occluded_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "scenes" / "scene-0" / "datasets" / "court"
    _write_minimal_v4_dataset(dataset_root)
    monkeypatch.setattr(
        sources_module,
        "validate_court_dataset",
        lambda *args, **kwargs: None,
    )
    output = tmp_path / "court-aabb.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=dataset_root,
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=4.0,
        crf=20,
        history_frames=0,
        court_overlay=CourtOverlayConfiguration(
            mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
            color_rgb=(255, 0, 0),
            background_color_rgb=(0, 0, 0),
            opacity=1.0,
            depth_epsilon_m=0.02,
            near_plane_m=0.05,
            maximum_cells=1,
            maximum_surface_faces=6,
            maximum_projected_pixels=100_000,
        ),
    )

    result = visualize_dataset(request)

    assert probe_video_info(result.video_path).frame_count == 2
    sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == VISUALIZATION_METADATA_SCHEMA_V2
    overlay = sidecar["court_overlay"]
    assert overlay["mode"] == "trajectory_support_aabb"
    assert overlay["artifact"]["cell_count"] == 1
    assert overlay["artifact"]["coordinate_space"] == "metric_scene_metres"
    assert overlay["drawing_statistics"]["geometry"] == {
        "cell_count": 1,
        "source_triangle_count": 12,
        "surface_face_count": 6,
    }
    totals = overlay["drawing_statistics"]["totals"]
    assert totals["drawn_pixel_count"] > 0
    assert totals["occluded_pixel_count"] > 0
