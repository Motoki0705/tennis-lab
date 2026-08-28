"""CPU integration of the real canonical scene-pipeline domain handlers."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvaluationDiagnostics,
    AlignmentEvaluationOutcome,
    AlignmentEvaluationPolicy,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    CameraLineDiagnostics,
    CameraOwnershipRule,
    CameraSelectionPolicy,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    EvaluatedAlignment,
    FixedCameraSelectionDiagnostics,
    LineInferenceDeterminismDiagnostics,
    MeasuredCameraLines,
    MetricSceneAdapter,
    PartitionThresholds,
    ProposalScoreModel,
    ProposalSearchDiagnostics,
    ProposalSearchStopReason,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
)
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.configuration import (
    BLCSDatasetConfiguration,
    CourtDatasetConfiguration,
    PLCSDatasetConfiguration,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCS_DATASET_SCHEMA,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.blcs.handler import (
    BLCSDatasetStageHandler,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSNHTRenderer,
)
from src.synthetic_data_generation.dataset.camera_profiles import CameraProfileConfig
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
)
from src.synthetic_data_generation.dataset.court.handler import (
    CourtDatasetStageHandler,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    CourtNHTRenderer,
)
from src.synthetic_data_generation.dataset.plcs.articulation import (
    MotionArticulationReport,
)
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.components.avatar_asset import (
    AvatarGaussianAsset,
)
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    PLCSAvatarFrameTensors,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.execution import PLCSPreparedAvatar
from src.synthetic_data_generation.dataset.plcs.handler import PLCSStageHandler
from src.synthetic_data_generation.dataset.plcs.rendering import (
    NHTPLCSRenderer,
)
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ForegroundDelta,
    RenderSampleKey,
    sparse_delta_from_composite,
)
from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    DatasetTarget,
    IngestStageHandler,
    ReportStageHandler,
    ScenePipelineRequest,
    ScenePipelineRunner,
    SceneWorkspace,
    StageName,
    StageStatus,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.synthetic_data_generation.reconstruction import (
    NHTPipelineConfig,
    NHTReconstructionHandler,
    NHTTrainingRuntime,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTComposedRenderClient,
    NHTRenderClient,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    ACCADMotionLibrary,
    MotionCategory,
    PLCSMotionClip,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


@dataclass(frozen=True)
class _EvidenceSource:
    evidence: AlignmentEvidence
    policy: AlignmentAcceptancePolicy

    def preflight(self, scene: StandardSceneExport) -> None:
        if scene.scene_id != "B00":
            raise ValueError("Alignment fixture received the wrong scene.")

    def collect_evaluated(self, scene: StandardSceneExport) -> EvaluatedAlignment:
        return EvaluatedAlignment(
            evidence=self.evidence,
            result=fit_alignment(self.evidence, policy=self.policy),
            heatmaps=_line_heatmaps(self.evidence),
        )


@dataclass(frozen=True)
class _StaticTrajectoryProvider:
    trajectories: tuple[BLCSTrajectory, ...]

    def preflight(self, *, scene_id: str, seed: int) -> None:
        if scene_id != "B00" or seed != 695 or not self.trajectories:
            raise ValueError("BLCS fixture request is inconsistent.")

    def load(self, *, scene_id: str, seed: int) -> tuple[BLCSTrajectory, ...]:
        self.preflight(scene_id=scene_id, seed=seed)
        return self.trajectories


@dataclass
class _CPUAvatar:
    clip: PLCSMotionClip
    surface_asset: AvatarGaussianAsset
    semantic_asset: GaussianAsset
    articulation: MotionArticulationReport
    appearance: AvatarAppearance

    def frame_tensors_batch(
        self,
        source_frame_indices: tuple[int, ...],
    ) -> dict[int, PLCSAvatarFrameTensors]:
        result: dict[int, PLCSAvatarFrameTensors] = {}
        count = self.semantic_asset.gaussian_count
        for frame_index in source_frame_indices:
            means = torch.zeros((count, 3), dtype=torch.float32)
            means[:, 2] = 1.0
            means[-1, 0] = 0.08 + 0.04 * frame_index
            gaussians = GaussianTensorSet(
                means=means,
                quaternions_wxyz=torch.tensor(
                    ((1.0, 0.0, 0.0, 0.0),) * count,
                    dtype=torch.float32,
                ),
                log_scales=torch.full((count, 3), math.log(0.05), dtype=torch.float32),
                opacity_logits=torch.full((count,), 5.0, dtype=torch.float32),
                features=self.appearance.features.clone(),
                instance_ids=torch.zeros(count, dtype=torch.int64),
                coordinates=GaussianCoordinates.asset_local_metres(),
                appearance_model=self.semantic_asset.appearance_model,
                appearance_space=self.semantic_asset.appearance_space,
            )
            joints = torch.zeros((52, 3), dtype=torch.float32)
            joints[:, 0] = torch.linspace(-0.2, 0.2, 52)
            joints[:, 1] = torch.linspace(-0.1, 0.1, 52)
            joints[:, 2] = 1.0
            joints[1:, 0] += 0.01 * frame_index
            result[frame_index] = PLCSAvatarFrameTensors(
                gaussians=gaussians,
                joints_m=joints,
            )
        return result


@dataclass
class _PLCSCPUOracle:
    """Constructor-only CPU backend that keeps the real PLCS stage loop intact."""

    execution_device: str = "test-cpu-oracle"
    torch_device: torch.device = torch.device("cpu")
    cuda_peak_bytes: int = 0
    _backgrounds: dict[str, BackgroundArrays] | None = None

    @property
    def background_upload_count(self) -> int:
        return len(self._backgrounds or {})

    def reset_stage(self, *, configured_device: str, compositor: object) -> None:
        if not configured_device.startswith("cuda"):
            raise ValueError("Production parameters must remain CUDA-owned.")
        reset = getattr(compositor, "reset_stage", None)
        if not callable(reset):
            raise TypeError("PLCS compositor does not expose its stage reset.")
        reset()
        self._backgrounds = {}

    def load_model(self, *, model_root: Path, gender: str) -> object:
        if not model_root.is_dir() or gender != "male":
            raise ValueError("Unexpected licensed-free PLCS model fixture.")
        return {"gender": gender}

    def prepare_source(self, *, clip: PLCSMotionClip, model: object) -> None:
        if model != {"gender": clip.gender}:
            raise ValueError("PLCS CPU source/model fixture mismatch.")

    def initial_support_plane(
        self,
        *,
        clip: PLCSMotionClip,
        model: object,
    ) -> PLCSSourceSupportPlane:
        self.prepare_source(clip=clip, model=model)
        return PLCSSourceSupportPlane.from_surface_minimum(
            initial_root_translation_z_m=float(clip.root_translation_m[0, 2]),
            support_local_z_m=1.0,
        )

    def prepare_avatar(
        self,
        *,
        asset_id: str,
        clip: PLCSMotionClip,
        model: object,
        appearance: AvatarAppearance,
        gaussian_count: int,
        seed: int,
    ) -> PLCSPreparedAvatar:
        del seed
        self.prepare_source(clip=clip, model=model)
        surface = _avatar_surface(gaussian_count)
        return _CPUAvatar(
            clip=clip,
            surface_asset=surface,
            semantic_asset=GaussianAsset(
                asset_id=asset_id,
                asset_class="smplh-avatar",
                role=GaussianAssetRole.MOVABLE,
                coordinates=GaussianCoordinates.asset_local_metres(),
                gaussian_count=gaussian_count,
                feature_dim=3,
                floating_dtype="float32",
                appearance_model="rgb",
                appearance_space="linear_rgb",
            ),
            articulation=MotionArticulationReport(
                frame_count=clip.frame_count,
                category=clip.category,
                non_root_pose_range_radians=0.2,
                gaussian_nonrigid_residual_m=0.04,
                region_displacement_m={
                    "legs": 0.04,
                    "arms": 0.03,
                    "torso": 0.02,
                },
                deformed_frame_indices=(1,),
            ),
            appearance=appearance,
        )

    def prepare_background(
        self,
        *,
        compositor: object,
        background: BackgroundArrays,
    ) -> None:
        del compositor
        if self._backgrounds is None:
            raise RuntimeError("PLCS CPU backend was not reset.")
        if background.camera_id in self._backgrounds:
            raise ValueError("PLCS background was prepared twice.")
        self._backgrounds[background.camera_id] = background

    def compose_delta(
        self,
        *,
        compositor: object,
        frame_index: int,
        camera: object,
        gaussians_scene: GaussianTensorSet,
        expected_instance_ids: tuple[int, ...],
    ) -> tuple[ForegroundDelta, dict[int, int]]:
        del compositor
        if gaussians_scene.means.device.type != "cpu":
            raise ValueError("The explicit PLCS oracle received non-CPU tensors.")
        actual_ids = {
            int(value) for value in torch.unique(gaussians_scene.instance_ids).tolist()
        }
        if actual_ids != set(expected_instance_ids):
            raise ValueError("The real PLCS composition changed object identity.")
        camera_id = getattr(camera, "camera_id", None)
        if not isinstance(camera_id, str) or self._backgrounds is None:
            raise TypeError("PLCS CPU oracle received an invalid camera.")
        background = self._backgrounds[camera_id]
        rgb = np.array(background.rgb, copy=True)
        alpha = np.array(background.alpha, copy=True)
        depth = np.array(background.depth, copy=True)
        labels: NDArray[np.int32] = np.zeros(
            (background.height, background.width), dtype=np.int32
        )
        for offset, instance_id in enumerate(expected_instance_ids):
            labels.reshape(-1)[offset] = instance_id
            rgb.reshape(-1, 3)[offset] = (0.9, 0.1 * instance_id, 0.1)
            alpha.reshape(-1)[offset] = 1.0
            depth.reshape(-1)[offset] = 1.0 + 0.1 * offset
        delta = sparse_delta_from_composite(
            key=RenderSampleKey(frame_index, camera_id),
            background=background,
            rgb=rgb,
            alpha=alpha,
            depth=depth,
            instance_ids=labels,
        )
        return delta, {instance_id: 1 for instance_id in expected_instance_ids}


def test_real_domain_handlers_publish_and_recover_through_fake_nht_only(
    tmp_path: Path,
) -> None:
    fixture = _Fixture.create(tmp_path)

    first = fixture.runner(rgb_value=0.1)
    _assert_real_handler_composition(first)
    first_manifest = first.run(fixture.request(from_stage=StageName.INGEST))

    assert all(
        record.status is StageStatus.COMPLETED
        for record in first_manifest.stages.values()
    )
    assert tuple(first.workspace.root.glob("run.json")) == (
        first.workspace.run_manifest_path,
    )
    _assert_published_domains(first.workspace, rgb_value=0.1)
    assert _reconstruction_generation(first.workspace) == 1
    assert first_manifest.stages[StageName.RECONSTRUCTION].summary[
        "pipeline_config"
    ] == {
        "path": str(fixture.pipeline_config.path),
        "schema": "nht_pipeline_config_v1",
    }

    alignment_payload = _json(first.workspace.root / "alignment/alignment.json")
    alignment_payload["stale_attempt"] = True
    (first.workspace.root / "alignment/alignment.json").write_text(
        json.dumps(alignment_payload), encoding="utf-8"
    )
    for target in DatasetTarget:
        dataset_path = first.workspace.root / "datasets" / target.value / "dataset.json"
        dataset_payload = _json(dataset_path)
        dataset_payload["stale_attempt"] = True
        dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

    alignment_rerun = fixture.runner(rgb_value=0.2)
    alignment_manifest = alignment_rerun.run(
        fixture.request(from_stage=StageName.ALIGNMENT)
    )

    assert _reconstruction_generation(first.workspace) == 1
    assert "stale_attempt" not in _json(
        first.workspace.root / "alignment/alignment.json"
    )
    _assert_published_domains(first.workspace, rgb_value=0.2)
    assert alignment_manifest.stages[StageName.RECONSTRUCTION].attempt == 1
    assert alignment_manifest.stages[StageName.ALIGNMENT].attempt == 2
    assert all(
        alignment_manifest.stages[target.stage].attempt == 2 for target in DatasetTarget
    )

    guarded_rerun = fixture.runner(rgb_value=0.25)
    fixture.pipeline_config.path.write_text("schema: invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        guarded_rerun.run(fixture.request(from_stage=StageName.RECONSTRUCTION))
    fixture.pipeline_config.path.write_text(
        "schema: nht_pipeline_config_v1\n",
        encoding="utf-8",
    )
    assert _reconstruction_generation(first.workspace) == 1
    _assert_published_domains(first.workspace, rgb_value=0.2)

    reconstruction_run = first.workspace.root / "reconstruction/run.json"
    reconstruction_payload = _json(reconstruction_run)
    reconstruction_payload["stale_attempt"] = True
    reconstruction_run.write_text(json.dumps(reconstruction_payload), encoding="utf-8")
    reconstruction_rerun = fixture.runner(rgb_value=0.3)
    reconstruction_manifest = reconstruction_rerun.run(
        fixture.request(from_stage=StageName.RECONSTRUCTION)
    )

    assert _reconstruction_generation(first.workspace) == 2
    assert "stale_attempt" not in _json(reconstruction_run)
    _assert_published_domains(first.workspace, rgb_value=0.3)
    assert reconstruction_manifest.stages[StageName.INGEST].attempt == 1
    assert reconstruction_manifest.stages[StageName.RECONSTRUCTION].attempt == 2
    assert reconstruction_manifest.stages[StageName.ALIGNMENT].attempt == 3
    assert all(
        reconstruction_manifest.stages[target.stage].attempt == 3
        for target in DatasetTarget
    )

    failing = fixture.runner(rgb_value=0.4, fail_domain="plcs")
    with pytest.raises(subprocess.CalledProcessError):
        failing.run(fixture.request(from_stage=StageName.PLCS_DATASET))

    failed_manifest = _json(first.workspace.run_manifest_path)
    failed_stages = cast(dict[str, dict[str, object]], failed_manifest["stages"])
    assert failed_stages["court_dataset"]["status"] == "completed"
    assert failed_stages["blcs_dataset"]["status"] == "completed"
    assert failed_stages["plcs_dataset"]["status"] == "failed"
    assert failed_stages["report"]["status"] == "invalidated"
    plcs_root = first.workspace.root / "datasets/plcs"
    assert (plcs_root / "dataset.json").is_file()
    assert (plcs_root / "backgrounds").is_dir()
    assert (plcs_root / "scenes").is_dir()
    assert _first_rgb_value(plcs_root) == pytest.approx(0.3)
    assert not (plcs_root / "chunks").exists()
    assert not (plcs_root / "staging").exists()
    assert not tuple(plcs_root.rglob("partial.tmp"))
    assert not tuple(first.workspace.transaction_root.rglob("partial.tmp"))
    assert not (first.workspace.root / "report/report.json").exists()

    retry = fixture.runner(rgb_value=0.4)
    retry_manifest = retry.run(fixture.request(from_stage=StageName.PLCS_DATASET))

    assert retry_manifest.stages[StageName.PLCS_DATASET].status is StageStatus.COMPLETED
    assert retry_manifest.stages[StageName.REPORT].status is StageStatus.COMPLETED
    assert _first_rgb_value(first.workspace.root / "datasets/plcs") == pytest.approx(
        0.4
    )
    assert _first_rgb_value(first.workspace.root / "datasets/court") == pytest.approx(
        0.3
    )
    assert _first_rgb_value(first.workspace.root / "datasets/blcs") == pytest.approx(
        0.3
    )
    assert not tuple(first.workspace.root.rglob("staging"))
    assert not tuple(first.workspace.root.rglob(".publication-backup"))
    assert not tuple(first.workspace.root.rglob("partial.tmp"))

    interrupted = MutableRunManifest.load(first.workspace.run_manifest_path)
    interrupted.stages[StageName.BLCS_DATASET].attempt = 2
    interrupted.invalidate(StageName.PLCS_DATASET)
    interrupted.stages[StageName.PLCS_DATASET].attempt = 0
    interrupted.invalidate(StageName.REPORT)
    interrupted.stages[StageName.REPORT].attempt = 1
    interrupted.save(first.workspace.run_manifest_path)
    first.workspace.invalidate_outputs(
        retry.registry.definition(StageName.PLCS_DATASET)
    )
    first.workspace.invalidate_outputs(retry.registry.definition(StageName.REPORT))

    blcs_cursor_repair = fixture.runner(rgb_value=0.45)
    repaired_manifest = blcs_cursor_repair.run(
        fixture.request(from_stage=StageName.BLCS_DATASET)
    )

    assert repaired_manifest.stages[StageName.INGEST].attempt == 1
    assert repaired_manifest.stages[StageName.RECONSTRUCTION].attempt == 2
    assert repaired_manifest.stages[StageName.ALIGNMENT].attempt == 3
    assert repaired_manifest.stages[StageName.COURT_DATASET].attempt == 3
    assert repaired_manifest.stages[StageName.BLCS_DATASET].attempt == 3
    assert repaired_manifest.stages[StageName.PLCS_DATASET].attempt == 1
    assert repaired_manifest.stages[StageName.REPORT].attempt == 2
    assert all(
        repaired_manifest.stages[stage].status is StageStatus.COMPLETED
        for stage in StageName
    )
    assert _first_rgb_value(first.workspace.root / "datasets/court") == pytest.approx(
        0.3
    )
    assert _first_rgb_value(first.workspace.root / "datasets/blcs") == pytest.approx(
        0.45
    )
    assert _first_rgb_value(first.workspace.root / "datasets/plcs") == pytest.approx(
        0.45
    )
    assert not tuple(first.workspace.root.rglob("partial.tmp"))

    render_calls = [
        json.loads(line)
        for line in fixture.render_log.read_text(encoding="utf-8").splitlines()
    ]
    assert {call["domain"] for call in render_calls} == {
        "court",
        "blcs",
        "plcs",
    }
    assert all(call["command"] == "nht-render" for call in render_calls)
    reconstruct_calls = [
        json.loads(line)
        for line in fixture.reconstruct_log.read_text(encoding="utf-8").splitlines()
    ]
    assert reconstruct_calls == [
        {
            "command": "nht-reconstruct",
            "generation": 1,
            "scene_id": "B00",
            "trainer": str(fixture.training_runtime.trainer),
            "training_python": str(fixture.training_runtime.python),
        },
        {
            "command": "nht-reconstruct",
            "generation": 2,
            "scene_id": "B00",
            "trainer": str(fixture.training_runtime.trainer),
            "training_python": str(fixture.training_runtime.python),
        },
    ]


@dataclass(frozen=True)
class _Fixture:
    workspace: SceneWorkspace
    source_video: Path
    pipeline_config: NHTPipelineConfig
    training_runtime: NHTTrainingRuntime
    reconstruct_executable: Path
    render_executable: Path
    reconstruct_log: Path
    render_log: Path
    alignment: AlignmentStageHandler
    court_configuration: CourtDatasetConfiguration
    blcs_configuration: BLCSDatasetConfiguration
    plcs_configuration: PLCSDatasetConfiguration
    camera_configuration: CameraProfileConfig
    motion_library: ACCADMotionLibrary
    plcs_backend: _PLCSCPUOracle

    @classmethod
    def create(cls, tmp_path: Path) -> _Fixture:
        resolver = _resolver(tmp_path)
        workspace = SceneWorkspace.resolve(resolver, "B00")
        source_video = tmp_path / "B00.mp4"
        _write_video(source_video)
        pipeline_config_path = tmp_path / "nht-pipeline.yaml"
        pipeline_config_path.write_text(
            "schema: nht_pipeline_config_v1\n",
            encoding="utf-8",
        )
        pipeline_config = NHTPipelineConfig.load(pipeline_config_path.resolve())
        training_python = tmp_path / "trainer/bin/python"
        training_python.parent.mkdir(parents=True)
        training_python.write_text("#!/bin/sh\n", encoding="utf-8")
        training_python.chmod(0o755)
        trainer = tmp_path / "trainer/simple_trainer_nht.py"
        trainer.write_text("# test trainer\n", encoding="utf-8")
        training_runtime = NHTTrainingRuntime(
            python=training_python.resolve(),
            trainer=trainer.resolve(),
        )
        reconstruct_log = tmp_path / "nht-reconstruct.jsonl"
        render_log = tmp_path / "nht-render.jsonl"
        reconstruct, render = _write_fake_nht_commands(
            tmp_path / "bin",
            reconstruct_log=reconstruct_log,
            render_log=render_log,
        )
        motion_library, accad_root = _write_motion_library(tmp_path / "motion")
        smplh_root = tmp_path / "smplh"
        smplh_root.mkdir()
        return cls(
            workspace=workspace,
            source_video=source_video.resolve(),
            pipeline_config=pipeline_config,
            training_runtime=training_runtime,
            reconstruct_executable=reconstruct,
            render_executable=render,
            reconstruct_log=reconstruct_log,
            render_log=render_log,
            alignment=AlignmentStageHandler(
                evidence_source=_EvidenceSource(
                    _alignment_evidence(),
                    _alignment_policy(),
                ),
                policy=_alignment_policy(),
            ),
            court_configuration=_court_configuration(),
            blcs_configuration=_blcs_configuration(),
            plcs_configuration=_plcs_configuration(
                resolver,
                accad_root=accad_root,
                smplh_root=smplh_root,
            ),
            camera_configuration=_camera_configuration(),
            motion_library=motion_library,
            plcs_backend=_PLCSCPUOracle(),
        )

    def request(self, *, from_stage: StageName) -> ScenePipelineRequest:
        return ScenePipelineRequest(
            scene_id="B00",
            source_video=self.source_video,
            targets=frozenset(DatasetTarget),
            from_stage=from_stage,
            config_schema="canonical_scene_pipeline_v1",
        )

    def runner(
        self,
        *,
        rgb_value: float,
        fail_domain: str | None = None,
    ) -> ScenePipelineRunner:
        environment = {
            "FAKE_NHT_RGB_VALUE": str(rgb_value),
            "FAKE_NHT_FAIL_DOMAIN": fail_domain or "",
        }
        nht_client = NHTRenderClient()
        composed_nht_client = NHTComposedRenderClient()
        assets = self.blcs_configuration.assets
        plcs_handler = PLCSStageHandler(
            configuration=self.plcs_configuration,
            camera_configuration=self.camera_configuration,
            motion_library=self.motion_library,
            avatar_appearance_source=self.plcs_configuration.appearance,
            renderer=NHTPLCSRenderer(
                client=nht_client,
                compositor=self.plcs_configuration.foreground_compositor,
                executable=self.render_executable,
                environment=environment,
                timeout_seconds=180.0,
            ),
            parameters=self.plcs_configuration.build_stage_parameters(seed=695),
            execution_backend=self.plcs_backend,
        )
        handlers = CanonicalStageHandlers(
            ingest=IngestStageHandler(),
            reconstruction=NHTReconstructionHandler(
                executable=self.reconstruct_executable,
                pipeline_config=self.pipeline_config,
                training_runtime=self.training_runtime,
                environment={},
                timeout_seconds=180.0,
            ),
            alignment=self.alignment,
            court_dataset=CourtDatasetStageHandler(
                configuration=self.court_configuration,
                profile="train",
                renderer=CourtNHTRenderer(
                    executable=self.render_executable,
                    client=nht_client,
                    environment=environment,
                    timeout_seconds=180.0,
                ),
            ),
            blcs_dataset=BLCSDatasetStageHandler(
                workspace=self.workspace,
                configuration=self.blcs_configuration,
                camera_configuration=self.camera_configuration,
                seed=695,
                assets=assets,
                trajectory_provider=_StaticTrajectoryProvider(
                    (_blcs_trajectory(0), _blcs_trajectory(1))
                ),
                renderer=BLCSNHTRenderer(
                    assets=assets,
                    client=composed_nht_client,
                    executable=self.render_executable,
                    environment=environment,
                    timeout_seconds=180.0,
                    execution_device="cuda:0",
                    maximum_batch_frames=1,
                ),
            ),
            plcs_dataset=plcs_handler,
            report=ReportStageHandler(
                alignment_directory=self.workspace.root / "alignment",
                dataset_manifests={
                    target: self.workspace.root
                    / "datasets"
                    / target.value
                    / "dataset.json"
                    for target in DatasetTarget
                },
            ),
        )
        return ScenePipelineRunner(
            workspace=self.workspace,
            registry=canonical_registry(handlers),
            resolved_config_yaml=(
                "schema: canonical_scene_pipeline_v1\n"
                "request:\n"
                "  from_stage: ingest\n"
                "fixture: real-domain-cpu\n"
            ),
        )


def _assert_real_handler_composition(runner: ScenePipelineRunner) -> None:
    ingest = runner.registry.definition(StageName.INGEST).handler
    reconstruction = runner.registry.definition(StageName.RECONSTRUCTION).handler
    alignment = runner.registry.definition(StageName.ALIGNMENT).handler
    court = runner.registry.definition(StageName.COURT_DATASET).handler
    blcs = runner.registry.definition(StageName.BLCS_DATASET).handler
    plcs = runner.registry.definition(StageName.PLCS_DATASET).handler
    report = runner.registry.definition(StageName.REPORT).handler
    assert isinstance(ingest, IngestStageHandler)
    assert isinstance(reconstruction, NHTReconstructionHandler)
    assert isinstance(alignment, AlignmentStageHandler)
    assert isinstance(court, CourtDatasetStageHandler)
    assert isinstance(blcs, BLCSDatasetStageHandler)
    assert isinstance(plcs, PLCSStageHandler)
    assert isinstance(report, ReportStageHandler)
    assert isinstance(court.renderer, CourtNHTRenderer)
    assert isinstance(blcs.renderer, BLCSNHTRenderer)
    assert isinstance(plcs.renderer, NHTPLCSRenderer)


def _assert_published_domains(
    workspace: SceneWorkspace,
    *,
    rgb_value: float,
) -> None:
    expected = {
        DatasetTarget.COURT: (
            COURT_DATASET_SCHEMA,
            "diagnostics/semantic-manifest.json",
        ),
        DatasetTarget.BLCS: (
            BLCS_DATASET_SCHEMA,
            "diagnostics/metrics.json",
        ),
        DatasetTarget.PLCS: (
            PLCS_DATASET_SCHEMA,
            "diagnostics/motion-camera-court.json",
        ),
    }
    for target, (schema, required_diagnostic) in expected.items():
        root = workspace.root / "datasets" / target.value
        payload = _json(root / "dataset.json")
        assert payload["schema"] == schema
        if target is DatasetTarget.COURT:
            metrics = cast(dict[str, object], payload["metrics"])
            court_group_counts = cast(dict[str, object], metrics["court_group_counts"])
            assert set(court_group_counts) == {"court-0", "court-1"}
        else:
            assert payload["domain"] == target.value
            bindings = cast(list[dict[str, object]], payload["target_courts"])
            assert {value["court_instance_id"] for value in bindings} == {
                "court-0",
                "court-1",
            }
        diagnostics = cast(list[str], payload["diagnostics"])
        assert required_diagnostic in diagnostics
        assert all((root / relative).is_file() for relative in diagnostics)
        assert _first_rgb_value(root) == pytest.approx(rgb_value)
        assert "stale_attempt" not in payload
        assert not (root / "staging").exists()
    report = _json(workspace.root / "report/report.json")
    assert report["schema"] == "canonical_scene_report_v1"
    assert set(cast(dict[str, object], report["datasets"])) == {
        "court",
        "blcs",
        "plcs",
    }
    assert (
        _json(workspace.root / "datasets/blcs/diagnostics/performance.json")[
            "execution_device"
        ]
        == "cuda:0"
    )
    assert (
        _json(workspace.root / "datasets/plcs/diagnostics/performance.json")[
            "execution_device"
        ]
        == "test-cpu-oracle"
    )


def _first_rgb_value(root: Path) -> float:
    path = next(iter(sorted(root.rglob("rgb.npy"))))
    return float(np.load(path, allow_pickle=False).reshape(-1)[0])


def _reconstruction_generation(workspace: SceneWorkspace) -> int:
    value = _json(workspace.root / "reconstruction/run.json")["generation"]
    assert isinstance(value, int)
    return value


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return value


def _resolver(tmp_path: Path) -> PathResolver:
    roots = RuntimePathRoots(
        project_root=tmp_path.resolve(),
        data_root=(tmp_path / "data").resolve(),
        checkpoint_root=(tmp_path / "ckpt").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        cache_root=(tmp_path / "cache").resolve(),
        external_asset_root=(tmp_path / "external").resolve(),
    )
    return PathResolver(roots)


def _court_configuration() -> CourtDatasetConfiguration:
    return CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(
                Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
            ),
            resolve=True,
        )
    )


def _blcs_configuration() -> BLCSDatasetConfiguration:
    config = OmegaConf.load(
        Path("src/synthetic_data_generation/configs/dataset/blcs/production.yaml")
    )
    config.generator.targeted_velocity.gravity = 9.81
    config.render_timeout_seconds = 180.0
    result = BLCSDatasetConfiguration.from_mapping(config)
    return replace(
        result,
        timeline=replace(result.timeline, chunk_size_frames=2),
        trajectory_source=replace(
            result.trajectory_source,
            scene_count=2,
            split_scene_counts={"train": 2, "validation": 0, "test": 0},
        ),
        performance=replace(
            result.performance,
            maximum_published_fraction_of_dense_reference=1.0,
            maximum_nht_invocations=2,
            maximum_background_cache_misses=12,
            maximum_batch_frames=1,
            execution_device="cuda:0",
            require_cuda=True,
        ),
    )


def _plcs_configuration(
    resolver: PathResolver,
    *,
    accad_root: Path,
    smplh_root: Path,
) -> PLCSDatasetConfiguration:
    config = OmegaConf.load(
        Path("src/synthetic_data_generation/configs/dataset/plcs/production.yaml")
    )
    config.render_timeout_seconds = 180.0
    result = PLCSDatasetConfiguration.from_mapping(config, resolver=resolver)
    objects = tuple(replace(value, start_frame=0) for value in result.objects)
    return replace(
        result,
        timeline=replace(result.timeline, chunk_size_frames=2),
        accad_root=accad_root,
        scene_splits={"B00": "train", "B00-plcs-002": "train"},
        objects=objects,
        smplh_model_root=smplh_root,
        gaussian_count=2,
        smplh_batch_size=2,
        performance=replace(
            result.performance,
            maximum_published_fraction_of_dense_reference=1.0,
            maximum_background_cache_misses=12,
            maximum_batch_frames=2,
            execution_device="test-cpu-oracle",
            require_cuda=False,
        ),
    )


def _camera_configuration() -> CameraProfileConfig:
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
            "image_size": [64, 48],
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


def _blcs_trajectory(index: int) -> BLCSTrajectory:
    positions: NDArray[np.float64] = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = (-0.5, 0.0, 0.5)
    positions[:, 1, 0] = (0.5, 0.0, -0.5)
    positions[:, :, 2] = 1.5
    return BLCSTrajectory(
        trajectory_id=f"trajectory-{index}",
        split="train",
        fps=30.0,
        positions_court_m=positions,
        velocities_court_mps=np.gradient(positions, axis=0),
        present=np.ones((3, 2), dtype=np.bool_),
        tracks=tuple(
            BLCSTrack(
                object_id=f"ball-{object_index + 1:03d}",
                source_trajectory_id=f"trajectory-{index}",
                source_frame_indices=(0, 1, 2),
            )
            for object_index in range(2)
        ),
        source_metadata={"physics": "licensed-free-integration-fixture"},
    )


def _avatar_surface(count: int) -> AvatarGaussianAsset:
    means: NDArray[np.float64] = np.zeros((count, 3), dtype=np.float64)
    means[:, 2] = 1.0
    means[-1, 0] = 0.08
    return AvatarGaussianAsset(
        means_m=means,
        quaternions_wxyz=np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
            (count, 1),
        ),
        log_scales_m=np.full((count, 3), math.log(0.05), dtype=np.float64),
        opacity_logits=np.full(count, 5.0, dtype=np.float64),
        point_joint_weights=np.ones((count, 1), dtype=np.float64),
        face_indices=np.zeros(count, dtype=np.int64),
        barycentric_coordinates=np.tile(
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            (count, 1),
        ),
    )


def _write_motion_library(root: Path) -> tuple[ACCADMotionLibrary, Path]:
    root.mkdir(parents=True)
    paths: dict[MotionCategory | str, tuple[Path, ...]] = {}
    for index, category in enumerate(MotionCategory):
        path = root / f"{category.value}_poses.npz"
        poses: NDArray[np.float32] = np.zeros((2, 156), dtype=np.float32)
        poses[1, 3 + index] = 0.2 + 0.1 * index
        translations: NDArray[np.float32] = np.zeros((2, 3), dtype=np.float32)
        translations[1, 0] = 0.1 * (index + 1)
        np.savez(
            path,
            poses=poses,
            trans=translations,
            betas=np.zeros(16, dtype=np.float32),
            gender=np.asarray("male"),
            mocap_framerate=np.asarray(30.0, dtype=np.float32),
        )
        paths[category] = (path,)
    return ACCADMotionLibrary.from_category_paths(paths), root


def _alignment_policy() -> AlignmentAcceptancePolicy:
    thresholds = PartitionThresholds(
        minimum_camera_count=2,
        minimum_correspondence_count=6,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    return AlignmentAcceptancePolicy(fit=thresholds, holdout=thresholds)


def _alignment_candidate(index: int) -> CandidateEvidence:
    points = _identifiable_alignment_points()
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = (-8.0, 8.0)[index]
    transform = RigidTransform.from_matrix(matrix)
    repeated = np.concatenate((points, points))
    scene_points = transform.apply(repeated)
    return CandidateEvidence(
        court_instance_id=f"court-{index}",
        candidate_id=f"candidate-{index}",
        fit=CorrespondenceSet(
            points_court=repeated,
            points_scene=scene_points,
            camera_ids=("captured-0",) * len(points) + ("captured-1",) * len(points),
        ),
        holdout=CorrespondenceSet(
            points_court=repeated,
            points_scene=scene_points,
            camera_ids=("captured-2",) * len(points) + ("captured-3",) * len(points),
        ),
    )


def _alignment_evidence() -> AlignmentEvidence:
    camera_ids = tuple(f"captured-{index}" for index in range(4))
    whole_court_settings = _alignment_whole_court_settings()
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=("captured-0", "captured-1"),
            holdout_camera_ids=("captured-2", "captured-3"),
        ),
        candidates=(_alignment_candidate(0), _alignment_candidate(1)),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=np.asarray(
                    ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)), dtype=np.float64
                ),
            )
            for camera_id in camera_ids
        ),
        complex_points_scene=np.asarray(
            ((-20.0, -25.0, -1.0), (20.0, 25.0, 12.0)), dtype=np.float64
        ),
        primary_candidate_id="candidate-0",
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
        diagnostics=AlignmentEvidenceDiagnostics(
            cameras=tuple(
                CameraLineDiagnostics(
                    camera_id=camera_id,
                    selected_line_pixel_count=10,
                    projected_line_point_count=2,
                )
                for camera_id in camera_ids
            ),
            candidate_scales=tuple(
                CandidateScaleDiagnostics(
                    candidate_id=f"candidate-{index}",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.9 - 0.1 * index,
                    common_scale_refit_center_displacement_metres=0.0,
                    maximum_common_scale_refit_center_displacement_metres=(
                        whole_court_settings.maximum_center_refit_displacement_metres
                    ),
                    proposal_orientation_band_minimum_radians=-0.5,
                    proposal_orientation_band_maximum_radians=0.5,
                    proposal_residual_point_count_before_suppression=100,
                    proposal_residual_point_count_after_suppression=50,
                    native_center_uv=(float(index * 30), 0.0),
                    native_orientation_radians=0.0,
                )
                for index in range(2)
            ),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
            selection=FixedCameraSelectionDiagnostics(
                policy=CameraSelectionPolicy.NESTED_UNIFORM_PREFIX_V1,
                ownership_rule=(CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1),
                requested_camera_count=4,
                available_camera_count=4,
                partition_unit_count=2,
                fit_cameras_per_unit=1,
                holdout_cameras_per_unit=1,
                camera_prefix_ids=(
                    "captured-0",
                    "captured-2",
                    "captured-1",
                    "captured-3",
                ),
                fit_camera_ids=("captured-0", "captured-1"),
                holdout_camera_ids=("captured-2", "captured-3"),
                observed_camera_ids=(
                    "captured-0",
                    "captured-2",
                    "captured-1",
                    "captured-3",
                ),
                excluded_cameras=(),
            ),
            evaluation=AlignmentEvaluationDiagnostics(
                policy=(
                    AlignmentEvaluationPolicy.FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1
                ),
                evaluation_index=0,
                fit_camera_ids=("captured-0", "captured-1"),
                holdout_camera_ids=("captured-2", "captured-3"),
                candidate_ids=("candidate-0", "candidate-1"),
                fit_evaluation_count=1,
                holdout_evaluation_count=1,
                outcome=AlignmentEvaluationOutcome.FULL_VALIDATION_PASS,
            ),
            determinism=LineInferenceDeterminismDiagnostics(
                seed=42,
                device="cpu",
                model_eval=True,
                inference_mode=True,
                deterministic_algorithms=True,
                deterministic_warn_only=False,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
                cublas_workspace_config=None,
                torch_version="test",
                cuda_version=None,
                device_name="cpu",
                cross_hardware_bit_identity_claimed=False,
            ),
            proposal_search=ProposalSearchDiagnostics(
                score_model=(ProposalScoreModel.WEIGHTED_COVERAGE_FLOOR_GAUSSIAN_V1),
                orientation_band_count=1,
                center_tile_count=1,
                maximum_center_tile_width_scene_units=1.0,
                maximum_candidate_count=2,
                maximum_retained_state_count=1,
                maximum_tile_state_count=2,
                maximum_residual_state_count=2,
                residual_state_count=2,
                residual_tree_build_count=2,
                explored_tile_state_count=2,
                geometrically_impossible_tile_state_count=0,
                feasible_proposal_count_before_deduplication=2,
                duplicate_proposal_count=0,
                retained_proposal_count=2,
                expanded_state_count=2,
                pruned_state_count=0,
                feasible_complete_state_count=1,
                frontier_state_counts=(1, 1),
                feasible_complete_state_counts=(0, 1),
                refinement_attempt_count=1,
                refinement_rejected_state_count=0,
                selected_complete_state_rank=0,
                selected_complete_state_candidate_count=2,
                inferred_candidate_count=2,
                stopping_reason=(
                    ProposalSearchStopReason.RESIDUAL_EVIDENCE_BELOW_MINIMUM
                ),
                minimum_explained_evidence_fraction=0.3,
                selected_orientation_band_indices=(0, 0),
                selected_center_tile_indices=(0, 0),
                selected_candidate_explained_evidence_fractions=(0.4, 0.35),
                original_point_count=100,
                selected_residual_point_count=25,
                selected_explained_point_count=75,
                original_evidence_sum=100.0,
                selected_residual_evidence_sum=25.0,
                selected_explained_evidence_sum=75.0,
                selected_explained_evidence_fraction=0.75,
                selected_native_score_sum=1.7,
            ),
            excluded_cameras=(),
        ),
        whole_court_settings=whole_court_settings,
    )


def _identifiable_alignment_points() -> NDArray[np.float64]:
    longitudinal = np.asarray(
        [
            (offset, tangential, 0.0)
            for offset in (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)
            for tangential in np.linspace(-10.0, 10.0, 41)
        ],
        dtype=np.float64,
    )
    transverse = np.asarray(
        [
            (tangential, offset, 0.0)
            for offset in (-HALF_LENGTH, HALF_LENGTH)
            for tangential in np.linspace(-4.63, 4.63, 31)
        ],
        dtype=np.float64,
    )
    points: NDArray[np.float64] = np.asarray(
        np.concatenate((longitudinal, transverse)), dtype=np.float64
    )
    return points


def _line_heatmaps(evidence: AlignmentEvidence) -> AlignmentLineHeatmaps:
    selection = evidence.diagnostics.selection
    counts = {
        item.camera_id: item.projected_line_point_count
        for item in evidence.diagnostics.cameras
    }
    fit_ids = set(evidence.diagnostics.evaluation.fit_camera_ids)
    return AlignmentLineHeatmaps(
        bounds_uv=(-1.0, 1.0, -1.0, 1.0),
        grid_spacing=0.25,
        proximity_scale=0.35,
        proximity_power=2.0,
        views=tuple(
            AlignmentLineHeatmapView(
                camera_id=camera_id,
                probability=np.asarray([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32),
                points_uv=np.asarray(((-0.5, 0.5), (0.5, -0.5)), dtype=np.float64)[
                    : counts[camera_id]
                ],
                projected_probabilities=np.full(
                    counts[camera_id], 0.75, dtype=np.float32
                ),
                proximity_weights=np.full(counts[camera_id], 0.8, dtype=np.float64),
                included_in_aggregate=camera_id in fit_ids,
            )
            for camera_id in selection.camera_prefix_ids
        ),
    )


def _alignment_whole_court_settings() -> WholeCourtEvidenceSettings:
    scale_tolerance = 0.07290400972053462
    localization_tolerance = 0.3
    return WholeCourtEvidenceSettings(
        required_court_count=2,
        maximum_common_scale_relative_deviation=scale_tolerance,
        maximum_center_refit_displacement_metres=(
            scale_tolerance * np.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH)
            + localization_tolerance
        ),
        minimum_distinct_offset_levels=2,
        minimum_matches_per_offset_level=3,
        minimum_level_camera_count=2,
        minimum_secondary_tangential_span_metres=0.6,
        minimum_longitudinal_offset_span_metres=8.23,
        minimum_longitudinal_tangential_span_metres=12.8,
        minimum_transverse_offset_span_metres=12.8,
        minimum_transverse_tangential_span_metres=8.23,
        samples_per_metre=3.0,
        inlier_distance_metres=localization_tolerance,
        minimum_inlier_fraction=0.9,
        maximum_q95_error_metres=0.1,
        minimum_semantic_segment_inlier_fraction=0.8,
        minimum_center_separation_metres=10.97,
        maximum_footprint_overlap_fraction=1.0e-9,
    )


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        10.0,
        (16, 12),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the integration video fixture.")
    writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
    writer.release()


def _write_fake_nht_commands(
    binary_root: Path,
    *,
    reconstruct_log: Path,
    render_log: Path,
) -> tuple[Path, Path]:
    binary_root.mkdir(parents=True)
    interpreter = Path(sys.executable)
    reconstruct = binary_root / "nht-reconstruct"
    reconstruct.write_text(
        _fake_reconstruct_source(interpreter, reconstruct_log), encoding="utf-8"
    )
    reconstruct.chmod(reconstruct.stat().st_mode | 0o111)
    render = binary_root / "nht-render"
    render.write_text(_fake_render_source(interpreter, render_log), encoding="utf-8")
    render.chmod(render.stat().st_mode | 0o111)
    return reconstruct.resolve(), render.resolve()


def _fake_reconstruct_source(interpreter: Path, log_path: Path) -> str:
    return f"""#!{interpreter}
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene-id", required=True)
parser.add_argument("--input-video", required=True)
parser.add_argument("--workspace", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()
workspace = Path(args.workspace)
pipeline_config = Path(args.config)
effective_config = yaml.safe_load(pipeline_config.read_text(encoding="utf-8"))
if effective_config.get("schema") != "nht_pipeline_config_v1":
    raise ValueError("fake NHT received an invalid pipeline config")
training_runtime = effective_config.get("nht_training", {{}})
workspace.mkdir(parents=True, exist_ok=True)
log_path = Path({str(log_path)!r})
generation = len(log_path.read_text(encoding="utf-8").splitlines()) + 1 if log_path.exists() else 1
export = workspace / "export"
if export.exists():
    shutil.rmtree(export)
(export / "images").mkdir(parents=True)
(export / "model/ckpts").mkdir(parents=True)
cameras = []
identity = np.eye(4, dtype=np.float64)
for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)):
    camera_id = f"captured-{{index}}"
    image_name = f"{{camera_id}}.png"
    Image.new("RGB", (16, 12)).save(export / "images" / image_name)
    pose = identity.copy()
    pose[:3, 3] = (24.0 * math.cos(angle), 30.0 * math.sin(angle), 6.0 + 2.0 * math.sin(angle))
    cameras.append({{
        "camera_id": camera_id,
        "source_frame_index": index,
        "time_seconds": float(index),
        "split": "train" if index < 8 else "validation",
        "image": f"images/{{image_name}}",
        "width": 16,
        "height": 12,
        "intrinsics": {{
            "model": "PINHOLE",
            "distortion_model": "NONE",
            "params": [25.0, 25.0, 7.5, 5.5],
            "matrix": [[25.0, 0.0, 7.5], [0.0, 25.0, 5.5], [0.0, 0.0, 1.0]],
        }},
        "camera_to_scene": pose.tolist(),
        "source_image_processing": {{
            "source_resolution": [16, 12],
            "crop_xywh": [0, 0, 16, 12],
            "undistorted": True,
            "data_factor": 1,
        }},
        "diagnostics": {{"sfm_camera_id": index + 1, "sfm_camera_to_world": pose.tolist()}},
        "group": "default",
    }})
(export / "cameras.json").write_text(json.dumps({{
    "schema": "nht_standard_cameras_v1",
    "camera_coordinate_convention": "x-right, y-down, z-forward",
    "transform_semantics": "camera_to_scene maps homogeneous camera coordinates to scene coordinates",
    "cameras": cameras,
}}), encoding="utf-8")
np.save(export / "points_scene.npy", np.asarray([[0.0, 0.0, 0.0, 1.0, 0.5, 0.0]], dtype=np.float32))
(export / "model/ckpts/model.pt").write_bytes(f"public-model-{{generation}}".encode())
(export / "model/runtime-config.json").write_text("opaque-public-runtime", encoding="utf-8")
identity_list = identity.tolist()
(export / "scene.json").write_text(json.dumps({{
    "schema": "nht_standard_scene_v1",
    "scene_id": args.scene_id,
    "camera_coordinate_convention": "x-right, y-down, z-forward",
    "scene_coordinate_convention": "NHT parser normalized world coordinates; right-handed; identical to checkpoint Gaussian means",
    "pixel_coordinate_convention": "origin at top-left; x-right, y-down; pixel centers",
    "image_resolution_semantics": "width and height describe the undistorted, cropped training image at the configured data factor",
    "camera_count": len(cameras),
    "cameras": "cameras.json",
    "point_cloud": {{
        "path": "points_scene.npy",
        "shape": [1, 6],
        "dtype": "float32",
        "columns": ["x", "y", "z", "red", "green", "blue"],
        "color_range": [0.0, 1.0],
    }},
    "image_root": "images",
    "model_root": "model",
    "scene_from_sfm": identity_list,
    "sfm_from_scene": identity_list,
    "normalization": {{
        "applied": True,
        "camera_similarity": identity_list,
        "principal_axis_alignment": identity_list,
        "upside_down_correction": identity_list,
    }},
    "renderer": {{
        "command": "nht-render",
        "model": "model",
        "runtime_config": "model/runtime-config.json",
        "checkpoint": "model/ckpts/model.pt",
        "outputs": {{
            "rgb": "float32 HxWx3 in [0,1] plus PNG preview",
            "alpha": "float32 HxWx1 in [0,1] plus PNG preview",
            "depth": "float32 HxWx1 in canonical scene units",
        }},
    }},
    "sfm_summary": {{}},
    "nht_training_summary": {{"generation": generation}},
    "capabilities": ["nht_rendering_model"],
}}), encoding="utf-8")
(workspace / "run.json").write_text(json.dumps({{"command": "nht-reconstruct", "generation": generation}}), encoding="utf-8")
(workspace / "input-config.yaml").write_text("schema: public-fake-nht-v1\\n", encoding="utf-8")
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"command": "nht-reconstruct", "generation": generation, "scene_id": args.scene_id, "trainer": training_runtime.get("trainer"), "training_python": training_runtime.get("python")}}, sort_keys=True) + "\\n")
"""


def _fake_render_source(interpreter: Path, log_path: Path) -> str:
    return f"""#!{interpreter}
import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--camera-id", action="append", default=[])
parser.add_argument("--cameras")
parser.add_argument("--composition")
parser.add_argument("--output", required=True)
args = parser.parse_args()
scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
request = json.loads(Path(args.cameras).read_text(encoding="utf-8")) if args.cameras else {{"cameras": []}}
composition = json.loads(Path(args.composition).read_text(encoding="utf-8")) if args.composition else None
output = Path(args.output)
output_path = output.as_posix()
domain = next((parts for parts in ("court", "blcs", "plcs") if f"/datasets/{{parts}}/" in output_path or f"/.transactions/{{parts}}_dataset/" in output_path), "unknown")
failure = os.environ.get("FAKE_NHT_FAIL_DOMAIN", "") == domain
with Path({str(log_path)!r}).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"command": "nht-render", "domain": domain, "failure": failure}}, sort_keys=True) + "\\n")
output.mkdir(parents=True, exist_ok=False)
if failure:
    (output / "partial.tmp").write_text("injected public NHT failure", encoding="utf-8")
    sys.exit(7)
rgb_value = float(os.environ["FAKE_NHT_RGB_VALUE"])
render_root = output / "background" if composition is not None else output
if composition is not None:
    render_root.mkdir()
records = []
previews = {{}}
for camera in request["cameras"]:
    camera_id = camera["camera_id"]
    width = camera["width"]
    height = camera["height"]
    root = render_root / camera_id
    root.mkdir()
    np.save(root / "rgb.npy", np.full((height, width, 3), rgb_value, dtype=np.float32))
    np.save(root / "alpha.npy", np.ones((height, width, 1), dtype=np.float32))
    np.save(root / "depth.npy", np.full((height, width, 1), 100.0, dtype=np.float32))
    key = (width, height)
    if key not in previews:
        rgb_buffer = io.BytesIO()
        alpha_buffer = io.BytesIO()
        Image.new("RGB", key).save(rgb_buffer, format="PNG")
        Image.new("L", key, color=255).save(alpha_buffer, format="PNG")
        previews[key] = (rgb_buffer.getvalue(), alpha_buffer.getvalue())
    (root / "rgb.png").write_bytes(previews[key][0])
    (root / "alpha.png").write_bytes(previews[key][1])
    records.append({{
        "camera_id": camera_id,
        "request_source": "arbitrary",
        "width": width,
        "height": height,
        "rgb": f"{{camera_id}}/rgb.npy",
        "rgb_preview": f"{{camera_id}}/rgb.png",
        "alpha": f"{{camera_id}}/alpha.npy",
        "alpha_preview": f"{{camera_id}}/alpha.png",
        "depth": f"{{camera_id}}/depth.npy",
    }})
(render_root / "render.json").write_text(json.dumps({{
    "schema": "nht_render_result_v1",
    "scene_schema": "nht_standard_scene_v1",
    "scene_id": scene["scene_id"],
    "coordinate_space": "canonical NHT scene space",
    "export_validation": {{}},
    "renders": records,
}}), encoding="utf-8")
if composition is not None:
    timeline_path = Path(args.composition).parent / composition["timeline"]["tensors"]
    with np.load(timeline_path, allow_pickle=False) as timeline:
        present = np.array(timeline["present"], copy=True)
    chunks_root = output / "chunks"
    chunks_root.mkdir()
    chunk_records = []
    camera_count = len(request["cameras"])
    for chunk in composition["timeline"]["chunks"]:
        chunk_id = f"chunk-{{int(chunk['chunk_index']):06d}}"
        chunk_root = chunks_root / chunk_id
        chunk_root.mkdir()
        frame_values = []
        camera_values = []
        pixels = []
        rgbs = []
        alphas = []
        depths = []
        instance_ids = []
        offsets = [0]
        for frame_index in chunk["frame_indices"]:
            active = np.flatnonzero(present[frame_index])
            for camera_index, camera in enumerate(request["cameras"]):
                frame_values.append(frame_index)
                camera_values.append(camera_index)
                if len(active):
                    pixels.append((frame_index + camera_index) % (camera["width"] * camera["height"]))
                    rgbs.append((rgb_value, rgb_value, rgb_value))
                    alphas.append(1.0)
                    depths.append(8.0 + frame_index)
                    instance_ids.append(int(active[0]) + 1)
                offsets.append(len(pixels))
        np.savez(
            chunk_root / "composed.npz",
            frame_indices=np.asarray(frame_values, dtype=np.int64),
            camera_indices=np.asarray(camera_values, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            pixel_indices=np.asarray(pixels, dtype=np.int32),
            rgb=np.asarray(rgbs, dtype=np.float32).reshape(-1, 3),
            alpha=np.asarray(alphas, dtype=np.float32),
            depth=np.asarray(depths, dtype=np.float32),
            instance_ids=np.asarray(instance_ids, dtype=np.int32),
        )
        chunk_records.append({{
            "chunk_id": chunk_id,
            "frame_indices": chunk["frame_indices"],
            "camera_ids": [camera["camera_id"] for camera in request["cameras"]],
            "sample_count": len(frame_values),
            "pixel_count": len(pixels),
            "arrays": f"chunks/{{chunk_id}}/composed.npz",
        }})
    (output / "render.json").write_text(json.dumps({{
        "schema": "nht_composed_render_result_v1",
        "scene_schema": "nht_standard_scene_v1",
        "scene_id": scene["scene_id"],
        "coordinate_space": "canonical NHT scene space",
        "background": "background/render.json",
        "composition": {{
            "request_schema": composition["schema"],
            "frame_count": composition["timeline"]["frame_count"],
            "object_count": composition["timeline"]["object_count"],
            "asset_gaussian_count": composition["asset"]["gaussian_count"],
            "appearance_model": "direct_linear_rgb",
            "rasterization": "joint_3dgs_eval3d_transmittance_v1",
            "visibility_threshold": composition["visibility_threshold"],
        }},
        "chunks": chunk_records,
        "cuda_peak_bytes": 4096,
    }}), encoding="utf-8")
"""
