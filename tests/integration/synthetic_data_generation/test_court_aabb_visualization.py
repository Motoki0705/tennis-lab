"""Integration tests for strict Court V4 occupancy visualization publication."""

from __future__ import annotations

import json
from dataclasses import replace
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
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtAABBWireframeTopology,
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


def _write_minimal_v4_dataset(
    root: Path,
    *,
    centers_scene_m: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    occupancy_cells: tuple[tuple[int, int, int], ...] = (
        (0, 0, 1),
        (1, 0, 1),
        (20, 20, 20),
    ),
    depths_m: tuple[float, ...] = (10.0, 0.2),
) -> None:
    root.mkdir(parents=True)
    if len(centers_scene_m) != len(depths_m):
        raise ValueError("Test centers and depths must have equal length.")
    policy = _support_policy()
    support_digest = "c" * 64
    summary = SupportModelSummary(
        input_digest=support_digest,
        coordinate_space="metric_scene_metres",
        captured_camera_count=2,
        public_point_count=1,
        density_qualified_voxel_count=1,
        raw_inflated_occupancy_cell_count=len(occupancy_cells),
        inflated_occupancy_cell_count=len(occupancy_cells),
        camera_ball_carved_cell_count=0,
        camera_capsule_carved_cell_count=0,
        captured_camera_occupied_count=0,
        endpoint_ball_count=2,
        capsule_count=1,
        skipped_gap_link_count=0,
        skipped_obstacle_link_count=0,
        capsule_index_cell_count=1,
    )
    projection: dict[str, object] = {"courts": []}
    records: list[dict[str, object]] = []
    cameras: list[SceneCamera] = []
    for index, (center, depth_m) in enumerate(
        zip(centers_scene_m, depths_m, strict=True)
    ):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = center
        camera = SceneCamera(
            camera_id="camera-0",
            source_frame_index=index,
            width=48,
            height=32,
            intrinsics=(40.0, 0.0, 23.5, 0.0, 40.0, 15.5, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(transform),
            image_path="request-only",
        )
        cameras.append(camera)
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
                        "sample_count": len(centers_scene_m),
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
        np.asarray(occupancy_cells, dtype=np.int64),
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
                "samples": [
                    {
                        "sample_id": record["sample_id"],
                        "trajectory_id": record["trajectory_id"],
                        "view_id": record["view_id"],
                        "trajectory_frame_index": record[
                            "trajectory_frame_index"
                        ],
                        "camera_center_scene_m": list(centers_scene_m[index]),
                        "camera": cameras[index].to_dict(),
                    }
                    for index, record in enumerate(records)
                ],
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
            render_style=CourtAABBRenderStyle.WIREFRAME,
            wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
            trajectory_filter_scope=(
                CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY
            ),
            trajectory_filter_radius_mode=(
                CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS
            ),
            trajectory_filter_radius_m=None,
            color_rgb=(255, 0, 0),
            background_color_rgb=(0, 0, 0),
            opacity=1.0,
            edge_opacity=1.0,
            edge_width_px=1,
            depth_epsilon_m=0.02,
            near_plane_m=0.05,
            maximum_cells=3,
            maximum_surface_faces=10,
            maximum_edge_segments=16,
            maximum_projected_pixels=100_000,
        ),
    )

    result = visualize_dataset(request)

    assert probe_video_info(result.video_path).frame_count == 2
    sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == VISUALIZATION_METADATA_SCHEMA_V2
    overlay = sidecar["court_overlay"]
    assert overlay["mode"] == "trajectory_support_aabb"
    assert overlay["wireframe_topology"] == "boundary"
    assert overlay["config"]["render_style"] == "wireframe"
    assert overlay["config"]["wireframe_topology"] == "boundary"
    assert overlay["config"]["edge_opacity"] == 1.0
    assert overlay["config"]["edge_width_px"] == 1
    assert overlay["artifact"]["cell_count"] == 3
    assert overlay["artifact"]["coordinate_space"] == "metric_scene_metres"
    assert overlay["trajectory_filter"] == {
        "affects_collision_authority": False,
        "distance_metric": "trajectory_segment_to_closed_cell_aabb",
        "scope": "selected_trajectory",
        "radius_mode": "support_radius",
        "original_cell_count": 3,
        "removed_cell_count": 1,
        "resolved_radius_m": 0.8,
        "retained_cell_count": 2,
        "trajectory_center_count": 2,
        "trajectory_segment_count": 2,
        "filter_segment_count": 2,
        "closed_trajectory": True,
    }
    assert overlay["drawing_statistics"]["geometry"] == {
        "candidate_edge_segment_count": 20,
        "cell_count": 2,
        "edge_segment_count": 16,
        "source_triangle_count": 20,
        "suppressed_seam_segment_count": 4,
        "surface_face_count": 10,
    }
    totals = overlay["drawing_statistics"]["totals"]
    assert totals["drawn_pixel_count"] > 0
    assert totals["occluded_pixel_count"] > 0
    assert totals["edge_pixel_count"] == totals["surface_pixel_count"]
    assert totals["drawn_edge_pixel_count"] == totals["drawn_pixel_count"]
    assert totals["occluded_edge_pixel_count"] == totals["occluded_pixel_count"]

    repeated = visualize_dataset(
        replace(request, output_video=tmp_path / "court-aabb-repeated.mp4")
    )
    repeated_sidecar = json.loads(
        repeated.metadata_path.read_text(encoding="utf-8")
    )
    assert repeated_sidecar["court_overlay"] == overlay

    solid_overlay = replace(
        request.court_overlay,
        render_style=CourtAABBRenderStyle.SOLID,
        maximum_edge_segments=1,
    )
    solid = visualize_dataset(
        replace(
            request,
            output_video=tmp_path / "court-aabb-solid.mp4",
            court_overlay=solid_overlay,
        )
    )
    solid_sidecar = json.loads(solid.metadata_path.read_text(encoding="utf-8"))
    solid_metadata = solid_sidecar["court_overlay"]
    assert solid_metadata["config"]["render_style"] == "solid"
    assert solid_metadata["wireframe_topology"] is None
    assert solid_metadata["config"]["maximum_edge_segments"] == 1
    assert solid_metadata["drawing_statistics"]["geometry"] == {
        "candidate_edge_segment_count": 0,
        "cell_count": 2,
        "edge_segment_count": 0,
        "source_triangle_count": 20,
        "suppressed_seam_segment_count": 0,
        "surface_face_count": 10,
    }


def test_frame_local_scope_publishes_variable_geometry_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "scenes" / "scene-local" / "datasets" / "court"
    _write_minimal_v4_dataset(
        dataset_root,
        centers_scene_m=(
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 4.0, 0.0),
            (0.0, 4.0, 0.0),
        ),
        occupancy_cells=(
            (0, 4, 1),
            (3, 0, 1),
            (3, 8, 1),
            (4, 0, 1),
            (4, 8, 1),
            (5, 8, 1),
            (8, 4, 1),
            (20, 20, 20),
        ),
        depths_m=(10.0, 10.0, 10.0, 10.0),
    )
    monkeypatch.setattr(
        sources_module,
        "validate_court_dataset",
        lambda *args, **kwargs: None,
    )
    overlay = CourtOverlayConfiguration(
        mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        render_style=CourtAABBRenderStyle.WIREFRAME,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        trajectory_filter_scope=(
            CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
        ),
        trajectory_filter_radius_mode=(
            CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
        ),
        trajectory_filter_radius_m=1.5,
        color_rgb=(255, 0, 0),
        background_color_rgb=(0, 0, 0),
        opacity=1.0,
        edge_opacity=1.0,
        edge_width_px=1,
        depth_epsilon_m=0.02,
        near_plane_m=0.05,
        maximum_cells=8,
        maximum_surface_faces=100,
        maximum_edge_segments=200,
        maximum_projected_pixels=100_000,
    )
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=dataset_root,
        output_video=tmp_path / "court-aabb-local.mp4",
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=4.0,
        crf=20,
        history_frames=0,
        court_overlay=overlay,
    )

    result = visualize_dataset(request)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))[
        "court_overlay"
    ]
    trajectory_filter = metadata["trajectory_filter"]

    assert result.frame_count == 4
    assert trajectory_filter["scope"] == "local_swept_segments"
    assert trajectory_filter["radius_mode"] == "explicit_radius"
    assert trajectory_filter["resolved_radius_m"] == 1.5
    assert trajectory_filter["affects_collision_authority"] is False
    assert metadata["config"]["trajectory_filter_radius_mode"] == "explicit_radius"
    assert metadata["config"]["trajectory_filter_radius_m"] == 1.5
    assert trajectory_filter["original_cell_count"] == 8
    assert trajectory_filter["trajectory_segment_count"] == 4
    assert trajectory_filter["frame_count"] == 4
    assert trajectory_filter["frame_local_counts"] == {
        "candidate_edge_segment_count": {"minimum": 32, "maximum": 44, "total": 156},
        "edge_segment_count": {"minimum": 28, "maximum": 40, "total": 132},
        "filter_segment_count": {"minimum": 2, "maximum": 2, "total": 8},
        "removed_cell_count": {"minimum": 4, "maximum": 5, "total": 17},
        "retained_cell_count": {"minimum": 3, "maximum": 4, "total": 15},
        "suppressed_seam_segment_count": {"minimum": 4, "maximum": 8, "total": 24},
        "surface_face_count": {"minimum": 16, "maximum": 22, "total": 78},
    }
    assert trajectory_filter["count_sequence_digest_algorithm"] == "sha256-chain-v1"
    assert len(trajectory_filter["count_sequence_digest"]) == 64
    assert metadata["drawing_statistics"]["geometry"] == {
        "candidate_edge_segment_count": {"minimum": 32, "maximum": 44, "total": 156},
        "cell_count": {"minimum": 3, "maximum": 4, "total": 15},
        "edge_segment_count": {"minimum": 28, "maximum": 40, "total": 132},
        "source_triangle_count": {"minimum": 32, "maximum": 44, "total": 156},
        "suppressed_seam_segment_count": {"minimum": 4, "maximum": 8, "total": 24},
        "surface_face_count": {"minimum": 16, "maximum": 22, "total": 78},
    }
    assert metadata["drawing_statistics"]["totals"]["projected_pixel_count"] >= 0

    repeated = visualize_dataset(
        replace(request, output_video=tmp_path / "court-aabb-local-repeated.mp4")
    )
    repeated_metadata = json.loads(
        repeated.metadata_path.read_text(encoding="utf-8")
    )["court_overlay"]
    assert repeated_metadata == metadata
