"""CPU integration for the canonical full-timeline BLCS dataset stage."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
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
    BLCSCompositionAssets,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.handler import BLCSDatasetStageHandler
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSNHTRenderer,
    build_blcs_sample_metadata,
)
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ForegroundDeltaBatch,
    RenderSampleKey,
    sparse_delta_from_composite,
)
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
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderArrays,
    NHTRenderCommandRequest,
    NHTRenderRecord,
    NHTRenderResult,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig


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


class _FakeNHTClient:
    def __init__(self) -> None:
        self.requests: list[NHTRenderCommandRequest] = []

    def validate_scene(self, scene_path: Path) -> SimpleNamespace:
        del scene_path
        return SimpleNamespace(scene_id="B00")

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment=None,
        timeout_seconds=None,
    ) -> NHTRenderResult:
        del environment, timeout_seconds
        self.requests.append(request)
        assert request.arbitrary_cameras is not None
        assert request.arbitrary_request_path is not None
        request.arbitrary_cameras.write(request.arbitrary_request_path)
        request.output_directory.mkdir(parents=True, exist_ok=False)
        records = []
        for camera in request.arbitrary_cameras.cameras:
            camera_root = request.output_directory / camera.camera_id
            camera_root.mkdir()
            rgb: NDArray[np.float32] = np.zeros(
                (camera.height, camera.width, 3), dtype=np.float32
            )
            alpha: NDArray[np.float32] = np.ones(
                (camera.height, camera.width, 1), dtype=np.float32
            )
            depth: NDArray[np.float32] = np.full(
                (camera.height, camera.width, 1), 100.0, dtype=np.float32
            )
            rgb_path = camera_root / "rgb.npy"
            alpha_path = camera_root / "alpha.npy"
            depth_path = camera_root / "depth.npy"
            np.save(rgb_path, rgb, allow_pickle=False)
            np.save(alpha_path, alpha, allow_pickle=False)
            np.save(depth_path, depth, allow_pickle=False)
            preview = camera_root / "unused.png"
            record = NHTRenderRecord(
                camera_id=camera.camera_id,
                request_source="arbitrary",
                width=camera.width,
                height=camera.height,
                rgb_path=rgb_path,
                rgb_preview_path=preview,
                alpha_path=alpha_path,
                alpha_preview_path=preview,
                depth_path=depth_path,
            )
            record._bind_arrays(NHTRenderArrays(rgb=rgb, alpha=alpha, depth=depth))
            records.append(record)
        return NHTRenderResult(
            scene_id="B00",
            output_directory=request.output_directory,
            records=tuple(records),
        )


@dataclass
class _ExplicitCPUOracle:
    """Independent dense oracle selected only by this CPU integration test."""

    execution_device: str = "test-cpu-oracle"
    cuda_peak_bytes: int = 0

    def compose(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        backgrounds: Mapping[str, BackgroundArrays],
        ball_radius_m: float,
    ) -> Iterator[ForegroundDeltaBatch]:
        for chunk in plan.chunks:
            deltas = []
            metadata = []
            for frame_index in chunk.frame_indices:
                for camera_index, sampled in enumerate(plan.camera_rig.cameras):
                    camera = sampled.scene_camera
                    background = backgrounds[camera.camera_id]
                    rgb = np.array(background.rgb, copy=True)
                    alpha = np.array(background.alpha, copy=True)
                    depth = np.array(background.depth, copy=True)
                    labels: NDArray[np.int32] = np.zeros(
                        (camera.height, camera.width), dtype=np.int32
                    )
                    focal = float(camera.intrinsics[0])
                    for object_index in range(plan.source.object_count):
                        if not plan.geometric_visible[
                            frame_index, camera_index, object_index
                        ]:
                            continue
                        centre = plan.camera_uv[frame_index, camera_index, object_index]
                        object_depth = float(
                            plan.camera_depth[frame_index, camera_index, object_index]
                        )
                        radius = max(
                            1, int(round(focal * ball_radius_m / object_depth))
                        )
                        x_min = max(0, int(math.floor(centre[0] - radius)))
                        x_max = min(
                            camera.width, int(math.ceil(centre[0] + radius + 1))
                        )
                        y_min = max(0, int(math.floor(centre[1] - radius)))
                        y_max = min(
                            camera.height, int(math.ceil(centre[1] + radius + 1))
                        )
                        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                        disc = (xx - centre[0]) ** 2 + (
                            yy - centre[1]
                        ) ** 2 <= radius**2
                        local_depth = depth[y_min:y_max, x_min:x_max, 0]
                        visible = disc & (
                            (local_depth <= 0.0) | (object_depth <= local_depth)
                        )
                        rgb[y_min:y_max, x_min:x_max][visible] = (1.0, 0.85, 0.0)
                        alpha[y_min:y_max, x_min:x_max, 0][visible] = 1.0
                        local_depth[visible] = object_depth
                        labels[y_min:y_max, x_min:x_max][visible] = object_index + 1
                    delta = sparse_delta_from_composite(
                        key=RenderSampleKey(frame_index, camera.camera_id),
                        background=background,
                        rgb=rgb,
                        alpha=alpha,
                        depth=depth,
                        instance_ids=labels,
                    )
                    deltas.append(delta)
                    metadata.append(
                        build_blcs_sample_metadata(
                            plan=plan,
                            source_frame_index=frame_index,
                            camera_index=camera_index,
                            chunk_index=chunk.chunk_index,
                            delta=delta,
                        )
                    )
            yield ForegroundDeltaBatch(
                chunk_id=f"chunk-{chunk.chunk_index:06d}",
                deltas=tuple(deltas),
                metadata=tuple(metadata),
            )


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
        background=GaussianAsset(
            asset_id="background",
            asset_class="court",
            role=GaussianAssetRole.BACKGROUND,
            coordinates=GaussianCoordinates.scene(),
            gaussian_count=100,
            feature_dim=8,
            floating_dtype="float32",
            appearance_model="nht-deferred",
            appearance_space="test-space",
        ),
        ball=GaussianAsset(
            asset_id="ball-surface",
            asset_class="ball",
            role=GaussianAssetRole.MOVABLE,
            coordinates=GaussianCoordinates.asset_local_metres(),
            gaussian_count=12,
            feature_dim=8,
            floating_dtype="float32",
            appearance_model="nht-deferred",
            appearance_space="test-space",
        ),
        ball_radius_m=0.0335,
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
    config.performance.maximum_batch_frames = 2
    configuration = BLCSDatasetConfiguration.from_mapping(config)
    configuration = replace(
        configuration,
        performance=replace(
            configuration.performance,
            execution_device="test-cpu-oracle",
            require_cuda=False,
        ),
    )
    client = _FakeNHTClient()
    renderer = BLCSNHTRenderer(
        assets=_assets(),
        client=client,  # type: ignore[arg-type]
        executable="nht-render",
        environment={},
        timeout_seconds=60.0,
        execution_device="test-cpu-oracle",
        maximum_batch_frames=2,
        test_cpu_oracle=_ExplicitCPUOracle(),
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
    request_camera = client.requests[0].arbitrary_cameras.cameras[0]  # type: ignore[union-attr]
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
    assert float(logical.render.depth[0, 0, 0]) == pytest.approx(400.0)
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
