"""CPU integration for the canonical full-timeline BLCS dataset stage."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.configuration import BLCSDatasetConfiguration
from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSCompactDatasetReader,
    validate_blcs_dataset,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallGaussianSettings,
    BLCSCompositionAssets,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.handler import BLCSDatasetStageHandler
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSNHTRenderer,
)
from src.synthetic_data_generation.dataset.camera_profiles import CameraProfileConfig
from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    DatasetTarget,
    ScenePipelineRequest,
    SceneWorkspace,
    StageDefinition,
    StageExecutionSummary,
    StageName,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.contracts import StageExecutionContext
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)
from tests.support.synthetic_data_generation.composed_nht import (
    FakeComposedNHTClient,
)


@dataclass(frozen=True)
class _Context:
    request: ScenePipelineRequest
    stage: StageDefinition[StageExecutionSummary]
    owner_path: Path
    staging_path: Path


@dataclass(frozen=True)
class _NoOpHandler:
    """Typed unique placeholder for stages outside this focused integration."""

    stage: StageName

    def preflight(self, context: StageExecutionContext) -> None:
        pass

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        return StageExecutionSummary({"stage": self.stage.value})

    def validate(self, context: StageExecutionContext) -> None:
        pass


@dataclass(frozen=True)
class _StaticProvider:
    trajectories: tuple[BLCSTrajectory, ...]

    def preflight(self, *, scene_id: str, seed: int) -> None:
        del scene_id, seed
        if not self.trajectories:
            raise ValueError("Test trajectories must not be empty.")

    def load(self, *, scene_id: str, seed: int) -> tuple[BLCSTrajectory, ...]:
        del scene_id, seed
        return self.trajectories


def _layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((0.0, 30.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        transform = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=transform,
                court_from_scene=transform.inverse(),
                fit_status="accepted",
                fit_metrics={"fit": 0.1},
                holdout_status="accepted",
                holdout_metrics={"holdout": 0.2},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -20.0, -2.0, 50.0, 20.0, 20.0),
        primary_court_instance_id="court-0",
    )


def _metric_adapter(*, scale: float = 0.25) -> MetricSceneAdapter:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = (3.0, -2.0, 1.0)
    return MetricSceneAdapter.from_nht_scene_from_metric_scene(matrix)


def _alignment(
    *,
    layout: MultiCourtLayout | None = None,
    adapter: MetricSceneAdapter | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        layout=_layout() if layout is None else layout,
        metric_adapter=_metric_adapter() if adapter is None else adapter,
    )


def _camera_profile() -> CameraProfileConfig:
    positions = (
        (-8.0, -12.0, 5.0),
        (8.0, -12.0, 5.0),
        (-8.0, 12.0, 5.0),
        (8.0, 12.0, 5.0),
        (0.0, -14.0, 7.0),
        (0.0, 14.0, 7.0),
    )
    return CameraProfileConfig.from_mapping(
        {
            "profile": "default",
            "image_size": [32, 24],
            "expected_camera_count": 6,
            "slots": [
                {
                    "slot_id": f"slot-{index}",
                    "position_x_m": [x, x],
                    "position_y_m": [y, y],
                    "height_m": [z, z],
                    "look_at_x_m": [0.0, 0.0],
                    "look_at_y_m": [0.0, 0.0],
                    "look_at_height_m": [0.5, 0.5],
                    "hfov_degrees": [60.0, 60.0],
                }
                for index, (x, y, z) in enumerate(positions)
            ],
        }
    )


def _assets() -> BLCSCompositionAssets:
    return BLCSCompositionAssets(
        ball=GaussianAsset(
            asset_id="ball-surface",
            asset_class="ball",
            role=GaussianAssetRole.MOVABLE,
            coordinates=GaussianCoordinates.asset_local_metres(),
            gaussian_count=64,
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
    )


def _trajectory(index: int) -> BLCSTrajectory:
    positions: NDArray[np.float64] = np.zeros((3, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = (-0.5, 0.0, 0.5)
    positions[:, 0, 2] = 1.5
    return BLCSTrajectory(
        trajectory_id=f"trajectory-{index}",
        split="train",
        fps=30.0,
        positions_court_m=positions,
        velocities_court_mps=np.gradient(positions, axis=0),
        present=np.ones((3, 1), dtype=np.bool_),
        tracks=(
            BLCSTrack(
                object_id="ball-001",
                source_trajectory_id=f"trajectory-{index}",
                source_frame_indices=(0, 1, 2),
            ),
        ),
        source_metadata={"physics": "integration"},
    )


def test_blcs_stage_carries_all_frames_through_chunks_and_balanced_courts(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = SceneWorkspace(scene_id="B00", root=tmp_path / "B00")
    scene_path = workspace.root / "reconstruction" / "export" / "scene.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}\n", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="B00",
        source_video=source_video.resolve(),
        targets=frozenset({DatasetTarget.BLCS}),
        from_stage=StageName.BLCS_DATASET,
        config_schema="canonical_scene_pipeline_v1",
    )
    handler_module = importlib.import_module(
        "src.synthetic_data_generation.dataset.blcs.handler"
    )
    alignment = _alignment()
    monkeypatch.setattr(
        handler_module,
        "validate_alignment_outputs",
        lambda _path: alignment,
    )
    config = OmegaConf.load(
        Path("src/synthetic_data_generation/configs/dataset/blcs/production.yaml")
    )
    config.timeline.chunk_size_frames = 2
    config.generator.targeted_velocity.gravity = 9.81
    config.render_timeout_seconds = 60.0
    config.performance.maximum_batch_frames = 1
    configuration = BLCSDatasetConfiguration.from_mapping(config)
    client = FakeComposedNHTClient()
    renderer = BLCSNHTRenderer(
        assets=_assets(),
        client=client,
        executable="nht-render",
        environment={},
        timeout_seconds=60.0,
        execution_device="cuda:0",
        maximum_batch_frames=1,
    )
    handler = BLCSDatasetStageHandler(
        workspace=workspace,
        configuration=configuration,
        camera_configuration=_camera_profile(),
        seed=695,
        assets=_assets(),
        trajectory_provider=_StaticProvider(
            tuple(_trajectory(index) for index in range(3))
        ),
        renderer=renderer,
    )
    registry = canonical_registry(
        CanonicalStageHandlers(
            ingest=_NoOpHandler(StageName.INGEST),
            reconstruction=_NoOpHandler(StageName.RECONSTRUCTION),
            alignment=_NoOpHandler(StageName.ALIGNMENT),
            court_dataset=_NoOpHandler(StageName.COURT_DATASET),
            blcs_dataset=handler,
            plcs_dataset=_NoOpHandler(StageName.PLCS_DATASET),
            report=_NoOpHandler(StageName.REPORT),
        )
    )
    definition = registry.definition(StageName.BLCS_DATASET)
    owner = workspace.owner_path(definition)
    staging = workspace.staging_path(definition)
    staging.mkdir(parents=True)
    context = _Context(
        request=request,
        stage=definition,
        owner_path=owner,
        staging_path=staging,
    )

    handler.preflight(context)
    summary = handler.execute(context)
    handler.validate(context)
    result = validate_blcs_dataset(staging)

    assert summary.values["source_frame_count"] == 9
    assert summary.values["planned_frame_count"] == 9
    assert summary.values["rendered_frame_count"] == 9
    assert summary.values["labelled_frame_count"] == 9
    assert summary.values["sample_count"] == 54
    assert result.manifest.frame_inventory.to_dict() == {
        "source": 9,
        "planned": 9,
        "rendered": 9,
        "labelled": 9,
        "first_frame": 0,
        "last_frame": 8,
    }
    assert len(result.manifest.target_courts) == 2
    assert not (staging / "attempt-shards").exists()
    assert (staging / "diagnostics/metrics.json").is_file()
    assert result.metric_adapter == alignment.metric_adapter

    first_plan_path = staging / "samples/trajectory-0/plan.json"
    first_plan = json.loads(first_plan_path.read_text(encoding="utf-8"))
    metric_camera = first_plan["cameras"][0]["camera"]
    arbitrary_cameras = client.requests[0].base.arbitrary_cameras
    assert arbitrary_cameras is not None
    request_camera = arbitrary_cameras.cameras[0]
    assert request_camera.camera_id == metric_camera["camera_id"]
    assert request_camera.width == metric_camera["width"]
    assert request_camera.height == metric_camera["height"]
    assert request_camera.intrinsics == tuple(metric_camera["intrinsics"])
    metric_pose = RigidTransform(tuple(metric_camera["camera_to_scene"]))
    diagnostics = json.loads(
        (staging / "diagnostics/metrics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["trajectories"][0]["camera_poses_metric"][
        metric_camera["camera_id"]
    ] == list(metric_pose.values)
    assert (
        diagnostics["trajectories"][0]["target_court_transform_metric"]
        == first_plan["target_court"]["scene_from_court"]
    )
    np.testing.assert_allclose(
        alignment.metric_adapter.metric_from_nht_camera(
            request_camera.camera_to_scene
        ).matrix(),
        metric_pose.matrix(),
        atol=1.0e-8,
        rtol=0.0,
    )
    assert not np.allclose(
        request_camera.camera_to_scene.matrix(),
        metric_pose.matrix(),
        atol=1.0e-8,
        rtol=0.0,
    )

    first_sample = result.sample_records[0]
    logical = BLCSCompactDatasetReader(staging).materialize(
        trajectory_id=first_sample.trajectory_id,
        source_frame_index=first_sample.source_frame_index,
        camera_id=first_sample.camera_id,
    )
    assert float(logical.render.depth[0, 0, 0]) == pytest.approx(32.0)
    np.testing.assert_allclose(logical.render.rgb[0, 0], (0.72, 0.92, 0.08))
    assert logical.render.instance_ids[0, 0] == 1
    camera_parameters = logical.metadata["camera_parameters"]
    assert isinstance(camera_parameters, Mapping)
    camera_payload = camera_parameters["camera"]
    assert isinstance(camera_payload, Mapping)
    assert camera_payload["camera_to_scene"] == list(metric_pose.values)
    chunk = staging / first_sample.foreground_chunk
    metadata_path = chunk / "metadata.json"
    marker_path = chunk / "chunk.json"
    original_metadata = metadata_path.read_text(encoding="utf-8")
    original_marker = marker_path.read_text(encoding="utf-8")
    corrupted = json.loads(original_metadata)
    corrupted["records"][first_sample.chunk_sample_index]["target_court"] = (
        "court-mismatch"
    )
    metadata_path.write_text(json.dumps(corrupted) + "\n", encoding="utf-8")
    marker = json.loads(original_marker)
    marker["byte_count"] = (
        chunk / "foreground.npz"
    ).stat().st_size + metadata_path.stat().st_size
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample metadata is inconsistent"):
        validate_blcs_dataset(staging)
    metadata_path.write_text(original_metadata, encoding="utf-8")
    marker_path.write_text(original_marker, encoding="utf-8")

    original_plan = first_plan_path.read_text(encoding="utf-8")
    corrupted_plan = json.loads(original_plan)
    corrupted_plan["composition"]["objects"][0]["deformation_kind"] = "articulated"
    first_plan_path.write_text(json.dumps(corrupted_plan) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rigid deformation"):
        validate_blcs_dataset(staging)
    first_plan_path.write_text(original_plan, encoding="utf-8")

    monkeypatch.setattr(
        handler_module,
        "validate_alignment_outputs",
        lambda _path: _alignment(adapter=_metric_adapter(scale=0.5)),
    )
    with pytest.raises(ValueError, match="metric adapter differs"):
        handler.validate(context)

    shifted_layout = _layout()
    only_first_court = MultiCourtLayout(
        courts=(shifted_layout.courts[0],),
        complex_bounds_scene=shifted_layout.complex_bounds_scene,
        primary_court_instance_id="court-0",
    )
    monkeypatch.setattr(
        handler_module,
        "validate_alignment_outputs",
        lambda _path: _alignment(layout=only_first_court),
    )
    with pytest.raises(ValueError, match="accepted alignment inventory"):
        handler.validate(context)
