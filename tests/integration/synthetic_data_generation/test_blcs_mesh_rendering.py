"""Integration coverage for GLB balls over an NHT background boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallGaussianSettings,
    BLCSBallMeshAsset,
    BLCSBallRendering,
    BLCSCompositionAssets,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.rendering import BLCSMeshNHTRenderer
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.synthetic_data_generation.dataset.camera_profiles import CameraProfileConfig
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderArrays,
    NHTRenderClient,
    NHTRenderRecord,
    NHTRenderResult,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCommandRequest,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)
from src.utils.paths import PROJECT_ROOT


class _StaticBackgroundNHTClient(NHTRenderClient):
    """Publish one deterministic far-depth background per arbitrary camera."""

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> NHTRenderResult:
        del environment, timeout_seconds
        cameras = request.arbitrary_cameras
        camera_path = request.arbitrary_request_path
        if cameras is None or camera_path is None:
            raise ValueError("Mesh integration fake requires arbitrary cameras.")
        cameras.write(camera_path)
        request.output_directory.mkdir(parents=True, exist_ok=False)
        records = []
        for camera in cameras.cameras:
            root = request.output_directory / camera.camera_id
            root.mkdir()
            rgb = np.full((camera.height, camera.width, 3), 0.1, dtype=np.float32)
            alpha = np.ones((camera.height, camera.width, 1), dtype=np.float32)
            depth = np.full((camera.height, camera.width, 1), 100.0, dtype=np.float32)
            rgb_path = root / "rgb.npy"
            alpha_path = root / "alpha.npy"
            depth_path = root / "depth.npy"
            np.save(rgb_path, rgb, allow_pickle=False)
            np.save(alpha_path, alpha, allow_pickle=False)
            np.save(depth_path, depth, allow_pickle=False)
            record = NHTRenderRecord(
                camera_id=camera.camera_id,
                request_source="arbitrary",
                width=camera.width,
                height=camera.height,
                rgb_path=rgb_path,
                rgb_preview_path=root / "rgb.png",
                alpha_path=alpha_path,
                alpha_preview_path=root / "alpha.png",
                depth_path=depth_path,
            )
            record._bind_arrays(NHTRenderArrays(rgb=rgb, alpha=alpha, depth=depth))
            records.append(record)
        return NHTRenderResult(
            scene_id="B00",
            output_directory=request.output_directory,
            records=tuple(records),
        )


def test_glb_mesh_path_produces_compact_blcs_pixels_depth_and_metadata(
    tmp_path: Path,
) -> None:
    asset_path = _resource_repository_root() / (
        "data/synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
    )
    assets = _mesh_assets(asset_path)
    plan = build_blcs_plans(
        (_trajectory(),),
        dataset_scene_id="B00",
        layout=_layout(),
        camera_config=_camera(),
        assets=assets,
        seed=13,
        chunk_size_frames=1,
    )[0]
    scene_path = tmp_path / "reconstruction/export/scene.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}\n", encoding="utf-8")
    renderer = BLCSMeshNHTRenderer(
        assets=assets,
        client=_StaticBackgroundNHTClient(),
        executable="nht-render",
        environment={},
        timeout_seconds=60.0,
        execution_device="cpu",
        maximum_batch_frames=1,
    )

    result = renderer.render(
        plans=(plan,),
        scene_path=scene_path.resolve(),
        samples_directory=tmp_path / "samples",
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
        attempt_token="B00-blcs-mesh-test",
    )

    assert result.nht_invocations == 1
    assert result.background_cache_misses == 2
    trajectory = result.trajectories[0]
    assert trajectory.rendered_visible_object_views == 2
    delta = trajectory.chunk_readers[0].deltas()[0]
    metadata = trajectory.chunk_readers[0].metadata()[0]
    assert len(delta.pixel_indices) > 0
    assert np.all(delta.depth > 0.0)
    assert np.all(delta.depth < 100.0)
    np.testing.assert_array_equal(delta.instance_ids, 1)
    assert metadata["semantic_arrays"]["rendered_visible"] == [True]
    composition = plan.to_dict()["composition"]
    assert isinstance(composition, Mapping)
    asset = composition["asset"]
    assert isinstance(asset, Mapping)
    source = asset["source"]
    assert isinstance(source, Mapping)
    assert asset["rendering"] == "mesh"
    assert source["data_root_relative_path"] == (
        "synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
    )
    assert source["maximum_file_bytes"] == 33554432
    assert source["maximum_source_vertices"] == 500000
    assert source["maximum_source_faces"] == 1000000
    assert source["maximum_faces"] == 4096


def _resource_repository_root() -> Path:
    for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        if (
            candidate
            / "data/synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
        ).is_file():
            return Path(candidate)
    raise FileNotFoundError("BLCS tennis-ball GLB test asset is unavailable.")


def _mesh_assets(path: Path) -> BLCSCompositionAssets:
    return BLCSCompositionAssets(
        ball=GaussianAsset(
            asset_id="regulation-tennis-ball",
            asset_class="ball",
            role=GaussianAssetRole.MOVABLE,
            coordinates=GaussianCoordinates.asset_local_metres(),
            gaussian_count=256,
            feature_dim=3,
            floating_dtype="float32",
            appearance_model="rgb",
            appearance_space="linear_rgb",
        ),
        settings=BLCSBallGaussianSettings(
            radius_m=0.0335,
            radial_scale_m=0.0018,
            tangential_scale_m=0.0048,
            opacity=0.94,
            base_color_linear_rgb=(0.72, 0.92, 0.08),
            seam_color_linear_rgb=(0.92, 0.95, 0.80),
            seam_width_radians=0.08,
            visibility_threshold=0.0001,
        ),
        rendering=BLCSBallRendering.MESH,
        mesh=BLCSBallMeshAsset(
            path=path,
            data_root_relative_path=(
                "synthetic_data_generation/assets/blcs/tennis ball 3d model.glb"
            ),
            maximum_file_bytes=33554432,
            maximum_source_vertices=500000,
            maximum_source_faces=1000000,
            maximum_faces=4096,
        ),
    )


def _trajectory() -> BLCSTrajectory:
    positions = np.asarray((((0.0, 0.0, 1.5),),), dtype=np.float64)
    return BLCSTrajectory(
        trajectory_id="trajectory-0",
        split="train",
        fps=30.0,
        positions_court_m=positions,
        velocities_court_mps=np.zeros_like(positions),
        present=np.ones((1, 1), dtype=np.bool_),
        tracks=(
            BLCSTrack(
                object_id="ball-001",
                source_trajectory_id="trajectory-0",
                source_frame_indices=(0,),
            ),
        ),
        source_metadata={"physics": "mesh-integration"},
    )


def _layout() -> MultiCourtLayout:
    identity = RigidTransform.identity()
    court = CourtInstance(
        court_instance_id="court-0",
        candidate_id="candidate-0",
        scene_from_court=identity,
        court_from_scene=identity,
        fit_status="accepted",
        fit_metrics={"error": 0.1},
        holdout_status="accepted",
        holdout_metrics={"error": 0.2},
    )
    return MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-20.0, -30.0, -2.0, 20.0, 30.0, 20.0),
        primary_court_instance_id="court-0",
    )


def _camera() -> CameraProfileConfig:
    return CameraProfileConfig.from_mapping(
        {
            "profile": "broadcast",
            "image_size": [640, 360],
            "expected_camera_count": 2,
            "slots": [
                {
                    "slot_id": "left",
                    "position_x_m": [-3.0, -3.0],
                    "position_y_m": [-20.0, -20.0],
                    "height_m": [5.0, 5.0],
                    "look_at_x_m": [0.0, 0.0],
                    "look_at_y_m": [0.0, 0.0],
                    "look_at_height_m": [0.5, 0.5],
                    "hfov_degrees": [45.0, 45.0],
                },
                {
                    "slot_id": "right",
                    "position_x_m": [3.0, 3.0],
                    "position_y_m": [-20.0, -20.0],
                    "height_m": [5.0, 5.0],
                    "look_at_x_m": [0.0, 0.0],
                    "look_at_y_m": [0.0, 0.0],
                    "look_at_height_m": [0.5, 0.5],
                    "hfov_degrees": [45.0, 45.0],
                },
            ],
        }
    )
