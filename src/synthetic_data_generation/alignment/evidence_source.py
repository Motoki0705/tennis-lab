"""Measured production alignment evidence from public NHT scene files only."""

from __future__ import annotations

import heapq
import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvaluationDiagnostics,
    AlignmentEvaluationOutcome,
    AlignmentEvaluationPolicy,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    CameraEvidencePartition,
    CameraExclusionReason,
    CameraLineDiagnostics,
    CameraOwnershipRule,
    CameraSelectionPolicy,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    EvaluatedAlignment,
    ExcludedCameraDiagnostics,
    FixedCameraSelectionDiagnostics,
    LineInferenceDeterminismDiagnostics,
    MeasuredCameraLines,
    MetricSceneAdapter,
    ProposalSearchDiagnostics,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.settings import (
    AlignmentEvidenceSettings,
    CourtCandidateFitSettings,
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.alignment.whole_court import (
    COURT_LINE_SEGMENTS,
    evaluate_court_topology,
    sample_court_line_template,
    transform_template_2d,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import (
    CourtDecoderConfig,
    CourtEncoderConfig,
    CourtLoRAConfig,
    CourtLossConfig,
    CourtModelConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference import CourtLinePredictor
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.utils.configuration import PathResolver, PathRole
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

_MAXIMUM_PROPOSAL_CANDIDATE_COUNT = 2
_MAXIMUM_ORIENTATION_BAND_COUNT = 2
_REFERENCE_CENTER_TILE_COUNT = 64
_MAXIMUM_BRANCH_FACTOR = (
    _MAXIMUM_ORIENTATION_BAND_COUNT * _REFERENCE_CENTER_TILE_COUNT
)
_MAXIMUM_CENTER_TILE_COUNT = _MAXIMUM_BRANCH_FACTOR
_MAXIMUM_COMPLETE_BRANCH_COUNT = (
    _MAXIMUM_BRANCH_FACTOR**_MAXIMUM_PROPOSAL_CANDIDATE_COUNT
)
_MAXIMUM_TILE_STATE_COUNT = sum(
    _MAXIMUM_BRANCH_FACTOR**depth
    for depth in range(1, _MAXIMUM_PROPOSAL_CANDIDATE_COUNT + 1)
)
_MAXIMUM_RESIDUAL_STATE_COUNT = sum(
    _MAXIMUM_BRANCH_FACTOR**depth
    for depth in range(_MAXIMUM_PROPOSAL_CANDIDATE_COUNT)
)
_MAXIMUM_TILE_OPTIMIZER_WORKERS = 8


class LineProbabilityDetector(Protocol):
    """Explicit trained detector boundary used by measured evidence collection."""

    def preflight(self) -> None:
        """Validate and load the configured detector without writing outputs."""

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Return finite court-line probabilities for one real image."""

    def determinism_diagnostics(self) -> LineInferenceDeterminismDiagnostics:
        """Return the strict inference policy/environment after preflight."""


class ProductionCourtLineDetector:
    """Court-line predictor loaded from explicit repository/checkpoint authority."""

    def __init__(
        self, settings: CourtLineModelSettings, resolver: PathResolver, *, seed: int
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._predictor: CourtLinePredictor | None = None
        self._seed = seed
        self._determinism: LineInferenceDeterminismDiagnostics | None = None

    def preflight(self) -> None:
        """Load the exact configured model so incompatibility precedes invalidation."""
        if self._predictor is not None:
            return
        settings = self._settings
        if not settings.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Court-line checkpoint does not exist: {settings.checkpoint_path}"
            )
        if not settings.backbone_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Court-line backbone checkpoint does not exist: "
                f"{settings.backbone_checkpoint_path}"
            )
        if not settings.backbone_repository_path.is_dir():
            raise FileNotFoundError(
                "Court-line backbone repository does not exist: "
                f"{settings.backbone_repository_path}"
            )
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {settings.device!r} was requested for alignment, "
                "but CUDA is unavailable."
            )
        cublas_workspace_config: str | None = None
        if settings.device.startswith("cuda"):
            configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if configured_workspace not in {None, ":4096:8", ":16:8"}:
                raise RuntimeError(
                    "CUBLAS_WORKSPACE_CONFIG conflicts with strict deterministic "
                    f"alignment inference: {configured_workspace!r}."
                )
            cublas_workspace_config = configured_workspace or ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        torch.manual_seed(self._seed)
        if settings.device.startswith("cuda"):
            torch.cuda.manual_seed_all(self._seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        raw: Any = torch.load(
            settings.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(raw, dict):
            raise ValueError("Court-line checkpoint payload must be a mapping.")
        self._resolver.validate(PathRole.CHECKPOINT, settings.checkpoint_path)
        self._resolver.validate(
            PathRole.EXTERNAL_ASSET,
            settings.backbone_repository_path,
        )
        self._resolver.validate(
            PathRole.EXTERNAL_ASSET,
            settings.backbone_checkpoint_path,
        )
        hyper_parameters = raw.get("hyper_parameters")
        if not isinstance(hyper_parameters, Mapping):
            raise ValueError("Court-line checkpoint has no hyper_parameters mapping.")
        embedded = hyper_parameters.get("config")
        if not isinstance(embedded, Mapping):
            raise ValueError("Court-line checkpoint has no embedded config mapping.")
        config = _plain_mapping(embedded)
        model_mapping = _required_mapping(config, "model")
        encoder = _required_mapping(model_mapping, "encoder")
        embedded_checkpoint = encoder.get("checkpoint_path")
        if not isinstance(embedded_checkpoint, str) or not embedded_checkpoint:
            raise ValueError("Court-line checkpoint has no encoder checkpoint_path.")
        if Path(embedded_checkpoint).name != settings.backbone_checkpoint_path.name:
            raise ValueError(
                "Configured court-line backbone disagrees with the trained checkpoint: "
                f"{settings.backbone_checkpoint_path.name!r} != "
                f"{Path(embedded_checkpoint).name!r}."
            )
        _validate_embedded_architecture(settings, model_mapping)
        architecture = settings.architecture
        encoder_config = CourtEncoderConfig(
            name="dinov3",
            repository_path=settings.backbone_repository_path,
            checkpoint_path=settings.backbone_checkpoint_path,
            backbone_name=architecture.backbone_name,
            strict=architecture.backbone_strict,
            train_mode=architecture.backbone_train_mode,
            last_n_blocks=architecture.backbone_last_n_blocks,
            out_indices=architecture.backbone_out_indices,
            layer_mode=architecture.backbone_layer_mode,
            lora=CourtLoRAConfig(
                enabled=architecture.lora_enabled,
                rank=architecture.lora_rank,
                alpha=architecture.lora_alpha,
                dropout=architecture.lora_dropout,
                target_modules=architecture.lora_target_modules,
            ),
        )
        model_config = CourtModelConfig(
            name="court_hierarchical",
            in_channels=3,
            encoder=encoder_config,
            decoder=CourtDecoderConfig(
                name="dpt",
                channels=architecture.decoder_channels,
                reassemble_factors=architecture.decoder_reassemble_factors,
            ),
        )
        target_bundle = CourtTargetBundleSpec(
            {
                "line": CourtTargetSpec(
                    kind="line",
                    schema="court_line_binary_v1",
                    output_channels=1,
                    channel_names=("court_line",),
                    target_dtype=torch.float32,
                    precomputed=True,
                )
            }
        )
        model = CourtHierarchicalModel.from_config(model_config, target_bundle)
        raw_state = raw.get("state_dict")
        if not isinstance(raw_state, Mapping):
            raise ValueError("Court-line checkpoint has no state_dict mapping.")
        model_state = _line_bundle_model_state(raw_state)
        model.load_state_dict(model_state, strict=True)
        model.eval()
        spec = CourtModelSpec(
            target_bundle=target_bundle,
            in_channels=3,
            short_side=settings.expected_short_side,
            encoder_kind="dinov3",
        )
        adapter = CourtModelIOAdapter(
            spec,
            loss_config=CourtLossConfig(
                seg_ce_weight=1.0,
                seg_dice_weight=1.0,
                kp_focal_gamma=2.0,
                line_bce_weight=architecture.line_bce_weight,
                line_dice_weight=architecture.line_dice_weight,
                line_pos_weight=architecture.line_positive_weight,
            ),
            execution_boundary=CourtDINOv3ExecutionBoundary(
                frozen_backbone=(
                    architecture.backbone_train_mode == "frozen"
                    and not architecture.lora_enabled
                )
            ),
        )
        predictor = CourtLinePredictor(
            bind_model_io(model, adapter),
            torch.device(settings.device),
        )
        if predictor.adapter.spec.short_side != settings.expected_short_side:
            raise AssertionError(
                "Constructed court-line adapter lost the configured preprocessing size."
            )
        if settings.device.startswith("cuda") and predictor.device.type != "cuda":
            raise RuntimeError(
                "Court-line predictor silently changed the requested CUDA device."
            )
        self._predictor = predictor
        is_cuda = settings.device.startswith("cuda")
        self._determinism = LineInferenceDeterminismDiagnostics(
            seed=self._seed,
            device=settings.device,
            model_eval=not model.training,
            inference_mode=True,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cublas_workspace_config=cublas_workspace_config,
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda) if is_cuda else None,
            device_name=(
                torch.cuda.get_device_name(torch.device(settings.device))
                if is_cuda
                else "cpu"
            ),
            cross_hardware_bit_identity_claimed=False,
        )

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Run the trained model; no heuristic or identity fallback is available."""
        self.preflight()
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("Court-line predictor was not loaded by preflight.")
        with torch.inference_mode():
            probability = np.asarray(
                predictor.predict(image_rgb).probability.numpy(),
                dtype=np.float32,
            )
        if probability.ndim != 2 or min(probability.shape) < 2:
            raise ValueError(
                "Court-line predictor returned an invalid probability grid."
            )
        if not np.isfinite(probability).all():
            raise ValueError("Court-line predictor returned non-finite probabilities.")
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError("Court-line predictor probabilities must lie in [0, 1].")
        return probability

    def determinism_diagnostics(self) -> LineInferenceDeterminismDiagnostics:
        """Return strict policy/environment diagnostics after loading the model."""
        self.preflight()
        diagnostics = self._determinism
        if diagnostics is None:
            raise RuntimeError("Line-detector preflight did not record determinism.")
        return diagnostics


@dataclass(frozen=True, slots=True)
class _GroundPlane:
    normal: NDArray[np.float64]
    offset: float
    origin: NDArray[np.float64]
    basis_u: NDArray[np.float64]
    basis_v: NDArray[np.float64]
    support_uv_bounds: tuple[float, float, float, float]

    def to_uv(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        basis = np.stack((self.basis_u, self.basis_v), axis=1)
        return np.asarray((points - self.origin) @ basis, dtype=np.float64)

    def from_uv(self, points_uv: NDArray[np.float64]) -> NDArray[np.float64]:
        basis = np.stack((self.basis_u, self.basis_v), axis=0)
        return np.asarray(points_uv @ basis + self.origin, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ProjectedLineEvidence:
    points_nht_scene: NDArray[np.float64]
    points_uv: NDArray[np.float64]
    selected_line_pixel_count: int


@dataclass(frozen=True, slots=True)
class _CourtHypothesis:
    candidate_id: str
    center_uv: tuple[float, float]
    orientation_radians: float
    nht_scene_units_per_metre: float
    template_score: float
    native_nht_scene_units_per_metre: float
    native_template_score: float
    native_center_uv: tuple[float, float]
    native_orientation_radians: float
    common_scale_refit_center_displacement_metres: float
    maximum_common_scale_refit_center_displacement_metres: float
    proposal_orientation_band_radians: tuple[float, float]
    proposal_residual_point_count_before_suppression: int
    proposal_residual_point_count_after_suppression: int


@dataclass(frozen=True, slots=True)
class _NativeProposal:
    """One orientation-band optimum over the current residual evidence."""

    parameters: NDArray[np.float64]
    measured_score: float
    orientation_band_radians: tuple[float, float]
    residual_point_count: int


@dataclass(frozen=True, slots=True)
class _FixedCameraSelection:
    """One evidence-independent nested-uniform prefix."""

    ordered_cameras: tuple[SceneCamera, ...]


@dataclass(frozen=True, slots=True)
class _ProposalSearchState:
    """One bounded search branch with its own residual and topology."""

    selected: tuple[_CourtHypothesis, ...]
    residual: NDArray[np.float64]
    orientation_band_indices: tuple[int, ...]
    center_tile_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ResidualEvidenceContext:
    """Read-only residual points and their one reusable nearest-neighbour tree."""

    points: NDArray[np.float64]
    nearest_tree: cKDTree


@dataclass(frozen=True, slots=True)
class _ProposalResourceBounds:
    """Exact combinatorial bounds for one configured tiled proposal search."""

    branch_factor: int
    maximum_complete_branch_count: int
    maximum_tile_state_count: int
    maximum_residual_state_count: int


@dataclass(frozen=True, slots=True)
class _CenterTile:
    """One exact half-open 2-D center tile; final-axis tiles own the upper edge."""

    flat_index: int
    u_index: int
    v_index: int
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    logical_u_upper: float
    logical_v_upper: float


@dataclass(frozen=True, slots=True)
class _TiledProposal:
    """One feasible native optimum with deterministic basin ownership."""

    proposal: _NativeProposal
    orientation_band_index: int
    center_tile_index: int


@dataclass(frozen=True, slots=True)
class _ObservableCameraSelection:
    """Measured retained/excluded cameras without evidence-driven replacement."""

    fit: tuple[SceneCamera, ...]
    holdout: tuple[SceneCamera, ...]
    exclusions: tuple[ExcludedCameraDiagnostics, ...]


class MeasuredAlignmentEvidenceSource:
    """Deterministic fit/holdout evidence over public images, cameras, and points."""

    def __init__(
        self,
        settings: AlignmentEvidenceSettings,
        detector: LineProbabilityDetector,
        policy: AlignmentAcceptancePolicy,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._policy = policy

    def preflight(self, scene: StandardSceneExport) -> None:
        """Validate all real evidence and model requirements before mutation."""
        selected = _fixed_camera_selection(
            scene.cameras,
            settings=self._settings,
        ).ordered_cameras
        _partition_cameras(selected, settings=self._settings)
        minimum_points = max(
            self._settings.ground_plane.minimum_candidate_points,
            self._settings.ground_plane.minimum_support_points,
        )
        if scene.point_count < minimum_points:
            raise ValueError(
                "NHT scene has too few public sparse points for ground fitting: "
                f"{scene.point_count} < {minimum_points}."
            )
        for camera in selected:
            image_path = Path(camera.image_path)
            if (
                not image_path.is_absolute()
                or not image_path.is_file()
                or image_path.is_symlink()
            ):
                raise FileNotFoundError(
                    f"Exported camera {camera.camera_id!r} has no ordinary public image: "
                    f"{image_path}."
                )
            try:
                with Image.open(image_path) as image:
                    if image.size != (camera.width, camera.height):
                        raise ValueError(
                            f"Exported image dimensions disagree for {camera.camera_id!r}."
                        )
                    image.verify()
            except UnidentifiedImageError as error:
                raise ValueError(
                    f"Exported image cannot be decoded: {image_path}."
                ) from error
        self._detector.preflight()

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        """Measure lines and build disjoint metric correspondences without fallback."""
        return self.collect_evaluated(scene).evidence

    def collect_evaluated(self, scene: StandardSceneExport) -> EvaluatedAlignment:
        """Measure and evaluate the fixed evidence exactly once."""
        self.preflight(scene)
        return self._collect_after_preflight(scene)

    def _collect_after_preflight(
        self,
        scene: StandardSceneExport,
    ) -> EvaluatedAlignment:
        """Collect after input/model checks; production preflight caches this result."""
        fixed = _fixed_camera_selection(scene.cameras, settings=self._settings)
        selected = fixed.ordered_cameras
        probabilities = {
            camera.camera_id: self._detector.predict_probability(
                _load_rgb_image(camera)
            )
            for camera in selected
        }
        fit_assigned, holdout_assigned = _partition_cameras(
            selected,
            settings=self._settings,
        )
        plane = _estimate_ground_plane(
            np.asarray(scene.points_scene[:, :3], dtype=np.float64),
            fit_assigned,
            seed=self._settings.seed,
            settings=self._settings.ground_plane,
        )
        projected_by_camera = {
            camera.camera_id: _project_probability_to_ground(
                probabilities[camera.camera_id],
                camera=camera,
                plane=plane,
                model_settings=self._settings.line_model,
                projection_settings=self._settings.projection,
            )
            for camera in fit_assigned + holdout_assigned
        }
        observable = _classify_observable_cameras(
            fit_assigned,
            holdout_assigned,
            projected_by_camera=projected_by_camera,
            settings=self._settings,
        )
        _require_observable_camera_minima(observable, settings=self._settings)
        excluded_ids = {item.camera_id for item in observable.exclusions}
        selection = FixedCameraSelectionDiagnostics(
            policy=CameraSelectionPolicy.NESTED_UNIFORM_PREFIX_V1,
            ownership_rule=CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1,
            requested_camera_count=self._settings.camera_prefix_count,
            available_camera_count=len(scene.cameras),
            candidate_count=self._settings.candidate_fit.candidate_count,
            orientation_family_count=(
                self._settings.candidate_fit.orientation_family_count()
            ),
            fit_cameras_per_unit=self._settings.minimum_fit_cameras,
            holdout_cameras_per_unit=self._settings.minimum_holdout_cameras,
            camera_prefix_ids=tuple(camera.camera_id for camera in selected),
            fit_camera_ids=tuple(camera.camera_id for camera in fit_assigned),
            holdout_camera_ids=tuple(camera.camera_id for camera in holdout_assigned),
            observed_camera_ids=tuple(
                camera.camera_id
                for camera in selected
                if camera.camera_id not in excluded_ids
            ),
            excluded_cameras=observable.exclusions,
        )
        try:
            evidence = _alignment_evidence_for_fixed_selection(
                scene=scene,
                plane=plane,
                fit_cameras=observable.fit,
                holdout_cameras=observable.holdout,
                projected_by_camera=projected_by_camera,
                settings=self._settings,
                selection=selection,
                determinism=self._detector.determinism_diagnostics(),
            )
            result = fit_alignment(evidence, policy=self._policy)
        except ValueError as error:
            raise ValueError(
                "Fixed 48-camera alignment evaluation failed without holdout-driven "
                f"reselection or refit: {type(error).__name__}: {error}"
            ) from error
        return EvaluatedAlignment(evidence=evidence, result=result)


def _alignment_evidence_for_fixed_selection(
    *,
    scene: StandardSceneExport,
    plane: _GroundPlane,
    fit_cameras: tuple[SceneCamera, ...],
    holdout_cameras: tuple[SceneCamera, ...],
    projected_by_camera: Mapping[str, _ProjectedLineEvidence],
    settings: AlignmentEvidenceSettings,
    selection: FixedCameraSelectionDiagnostics,
    determinism: LineInferenceDeterminismDiagnostics,
) -> AlignmentEvidence:
    """Build evidence from the one already measured immutable prefix."""
    excluded_cameras = selection.excluded_cameras
    camera_order = fit_cameras + holdout_cameras
    observable_projected_by_camera = {
        camera.camera_id: projected_by_camera[camera.camera_id]
        for camera in camera_order
    }
    fit_points_uv = np.concatenate(
        [
            observable_projected_by_camera[camera.camera_id].points_uv
            for camera in fit_cameras
        ]
    )
    hypotheses, common_scale, maximum_deviation, proposal_search = (
        _fit_court_hypotheses(
        fit_points_uv,
        bounds=plane.support_uv_bounds,
        seed=settings.seed,
        settings=settings.candidate_fit,
        )
    )
    nht_from_metric = np.eye(4, dtype=np.float64)
    nht_from_metric[:3, :3] *= common_scale
    metric_adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(
        nht_from_metric
    )
    assigned_by_candidate = _assign_candidate_evidence(
        hypotheses,
        plane=plane,
        projected_by_camera=observable_projected_by_camera,
        settings=settings.candidate_fit,
    )
    candidates = tuple(
        _candidate_evidence(
            hypothesis,
            plane=plane,
            metric_adapter=metric_adapter,
            fit_cameras=fit_cameras,
            holdout_cameras=holdout_cameras,
            assigned_by_camera=assigned_by_candidate[hypothesis.candidate_id],
            settings=settings,
        )
        for hypothesis in hypotheses
    )
    diagnostics = AlignmentEvidenceDiagnostics(
        cameras=tuple(
            CameraLineDiagnostics(
                camera_id=camera.camera_id,
                selected_line_pixel_count=projected_by_camera[
                    camera.camera_id
                ].selected_line_pixel_count,
                projected_line_point_count=len(
                    projected_by_camera[camera.camera_id].points_nht_scene
                ),
            )
            for camera in camera_order
        ),
        candidate_scales=tuple(
            CandidateScaleDiagnostics(
                candidate_id=hypothesis.candidate_id,
                nht_scene_units_per_metre=(hypothesis.native_nht_scene_units_per_metre),
                template_score=hypothesis.native_template_score,
                common_scale_refit_center_displacement_metres=(
                    hypothesis.common_scale_refit_center_displacement_metres
                ),
                maximum_common_scale_refit_center_displacement_metres=(
                    hypothesis.maximum_common_scale_refit_center_displacement_metres
                ),
                proposal_orientation_band_minimum_radians=(
                    hypothesis.proposal_orientation_band_radians[0]
                ),
                proposal_orientation_band_maximum_radians=(
                    hypothesis.proposal_orientation_band_radians[1]
                ),
                proposal_residual_point_count_before_suppression=(
                    hypothesis.proposal_residual_point_count_before_suppression
                ),
                proposal_residual_point_count_after_suppression=(
                    hypothesis.proposal_residual_point_count_after_suppression
                ),
                native_center_uv=hypothesis.native_center_uv,
                native_orientation_radians=hypothesis.native_orientation_radians,
            )
            for hypothesis in hypotheses
        ),
        common_nht_scene_units_per_metre=common_scale,
        maximum_relative_scale_deviation=maximum_deviation,
        selection=selection,
        evaluation=AlignmentEvaluationDiagnostics(
            policy=(
                AlignmentEvaluationPolicy.FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1
            ),
            evaluation_index=0,
            fit_camera_ids=tuple(camera.camera_id for camera in fit_cameras),
            holdout_camera_ids=tuple(camera.camera_id for camera in holdout_cameras),
            candidate_ids=tuple(item.candidate_id for item in hypotheses),
            fit_evaluation_count=1,
            holdout_evaluation_count=1,
            outcome=AlignmentEvaluationOutcome.FULL_VALIDATION_PASS,
        ),
        determinism=determinism,
        proposal_search=proposal_search,
        excluded_cameras=excluded_cameras,
    )
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=tuple(camera.camera_id for camera in fit_cameras),
            holdout_camera_ids=tuple(camera.camera_id for camera in holdout_cameras),
        ),
        candidates=candidates,
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera.camera_id,
                points_nht_scene=projected_by_camera[camera.camera_id].points_nht_scene,
            )
            for camera in camera_order
        ),
        complex_points_scene=metric_adapter.metric_from_nht_points(
            np.asarray(scene.points_scene[:, :3], dtype=np.float64)
        ),
        primary_candidate_id=hypotheses[0].candidate_id,
        metric_adapter=metric_adapter,
        diagnostics=diagnostics,
        whole_court_settings=settings.candidate_fit.whole_court_evidence(
            minimum_matches_per_offset_level=(
                settings.correspondences.minimum_correspondences_per_camera
            )
        ),
    )


def _resolve_common_scale(
    candidate_scales: NDArray[np.float64],
    *,
    maximum_relative_deviation: float,
) -> tuple[float, float]:
    """Resolve one robust physical scale without privileging the first candidate."""
    scales = np.asarray(candidate_scales, dtype=np.float64)
    if scales.ndim != 1 or len(scales) == 0 or not np.isfinite(scales).all():
        raise ValueError("Court candidate scales must be a non-empty finite vector.")
    if np.any(scales <= 0.0):
        raise ValueError("Court candidate scales must be positive.")
    if (
        not math.isfinite(maximum_relative_deviation)
        or maximum_relative_deviation <= 0.0
    ):
        raise ValueError("maximum_relative_deviation must be positive and finite.")
    common_scale = float(np.median(scales))
    deviation = float(np.max(np.abs(scales / common_scale - 1.0)))
    if deviation > maximum_relative_deviation:
        raise ValueError(
            "Court candidates have scale-inconsistent native hypotheses: "
            f"scales={scales.tolist()}, common={common_scale:.9g}, "
            f"maximum_relative_deviation={deviation:.6f} exceeds "
            f"{maximum_relative_deviation:.6f}."
        )
    return common_scale, deviation


class ProductionAlignmentEvidenceSource(MeasuredAlignmentEvidenceSource):
    """Public constructor that always uses the configured trained line detector."""

    def __init__(
        self,
        settings: AlignmentEvidenceSettings,
        resolver: PathResolver,
        policy: AlignmentAcceptancePolicy,
    ) -> None:
        super().__init__(
            settings,
            ProductionCourtLineDetector(
                settings.line_model, resolver, seed=settings.seed
            ),
            policy,
        )
        self._cached_scene_key: tuple[str, Path] | None = None
        self._cached_evaluation: EvaluatedAlignment | None = None

    def preflight(self, scene: StandardSceneExport) -> None:
        """Measure complete evidence so invalidation cannot conceal an evidence failure."""
        key = (scene.scene_id, scene.scene_path)
        if self._cached_scene_key == key and self._cached_evaluation is not None:
            return
        super().preflight(scene)
        evaluation = self._collect_after_preflight(scene)
        self._cached_scene_key = key
        self._cached_evaluation = evaluation

    def collect_evaluated(self, scene: StandardSceneExport) -> EvaluatedAlignment:
        """Return the one evaluation proven during preflight."""
        key = (scene.scene_id, scene.scene_path)
        if self._cached_scene_key != key or self._cached_evaluation is None:
            self.preflight(scene)
        evaluation = self._cached_evaluation
        if evaluation is None:
            raise RuntimeError(
                "Production alignment preflight did not retain an evaluation."
            )
        return evaluation

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        """Return evidence proven during preflight, measuring if called standalone."""
        return self.collect_evaluated(scene).evidence


def create_production_alignment_handler(
    *,
    settings: AlignmentEvidenceSettings,
    policy: AlignmentAcceptancePolicy,
    resolver: PathResolver,
) -> AlignmentStageHandler:
    """Bind the executable measured evidence source into the canonical stage."""
    return AlignmentStageHandler(
        evidence_source=ProductionAlignmentEvidenceSource(settings, resolver, policy),
        policy=policy,
    )


def _select_cameras(
    cameras: tuple[SceneCamera, ...],
    *,
    maximum: int,
) -> tuple[SceneCamera, ...]:
    ordered = tuple(
        sorted(cameras, key=lambda item: (item.source_frame_index, item.camera_id))
    )
    if not ordered:
        raise ValueError("NHT scene has no public cameras for alignment evidence.")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise TypeError("maximum must be a positive integer.")
    selection_order = _farthest_gap_indices(len(ordered))
    return tuple(ordered[index] for index in selection_order[:maximum])


def _fixed_camera_selection(
    cameras: tuple[SceneCamera, ...],
    *,
    settings: AlignmentEvidenceSettings,
) -> _FixedCameraSelection:
    """Build exactly one evidence-independent nested-uniform prefix."""
    settings.require_available_cameras(available_camera_count=len(cameras))
    return _FixedCameraSelection(
        ordered_cameras=_select_cameras(
            cameras, maximum=settings.camera_prefix_count
        ),
    )


def _farthest_gap_indices(total: int) -> tuple[int, ...]:
    """Return a nested uniform order by deterministic temporal gap bisection."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise TypeError("total must be a positive integer.")
    if total == 1:
        return (0,)
    result = [0, total - 1]
    intervals: list[tuple[int, int, int]] = [(-(total - 1), 0, total - 1)]
    while intervals:
        _negative_width, lower, upper = heapq.heappop(intervals)
        midpoint = (lower + upper) // 2
        if midpoint == lower:
            continue
        result.append(midpoint)
        if midpoint - lower > 1:
            heapq.heappush(
                intervals,
                (-(midpoint - lower), lower, midpoint),
            )
        if upper - midpoint > 1:
            heapq.heappush(
                intervals,
                (-(upper - midpoint), midpoint, upper),
            )
    if len(result) != total or len(set(result)) != total:
        raise RuntimeError("Farthest-gap camera ordering lost an index.")
    return tuple(result)


def _partition_cameras(
    cameras: tuple[SceneCamera, ...],
    *,
    settings: AlignmentEvidenceSettings,
) -> tuple[tuple[SceneCamera, ...], tuple[SceneCamera, ...]]:
    unit = settings.minimum_fit_cameras + settings.minimum_holdout_cameras
    if len(cameras) != settings.camera_prefix_count:
        raise ValueError(
            "Fixed alignment partition requires the exact selected camera count: "
            f"{len(cameras)} != {settings.camera_prefix_count}."
        )
    holdout_slots = set(_even_indices(unit, settings.minimum_holdout_cameras))
    holdout_indices = {
        index for index in range(len(cameras)) if index % unit in holdout_slots
    }
    fit = tuple(
        camera for index, camera in enumerate(cameras) if index not in holdout_indices
    )
    holdout = tuple(
        camera for index, camera in enumerate(cameras) if index in holdout_indices
    )
    unit_count = len(cameras) // unit
    if (len(fit), len(holdout)) != (
        unit_count * settings.minimum_fit_cameras,
        unit_count * settings.minimum_holdout_cameras,
    ):
        raise ValueError(
            "Deterministic camera partitioning violated fixed-unit ownership."
        )
    return fit, holdout


def _retain_observable_cameras(
    fit_cameras: tuple[SceneCamera, ...],
    holdout_cameras: tuple[SceneCamera, ...],
    *,
    projected_by_camera: Mapping[str, _ProjectedLineEvidence],
    settings: AlignmentEvidenceSettings,
) -> tuple[
    tuple[SceneCamera, ...],
    tuple[SceneCamera, ...],
    tuple[ExcludedCameraDiagnostics, ...],
]:
    """Exclude unobservable cameras without repartitioning or replacement."""
    observable = _classify_observable_cameras(
        fit_cameras,
        holdout_cameras,
        projected_by_camera=projected_by_camera,
        settings=settings,
    )
    _require_observable_camera_minima(observable, settings=settings)
    return observable.fit, observable.holdout, observable.exclusions


def _classify_observable_cameras(
    fit_cameras: tuple[SceneCamera, ...],
    holdout_cameras: tuple[SceneCamera, ...],
    *,
    projected_by_camera: Mapping[str, _ProjectedLineEvidence],
    settings: AlignmentEvidenceSettings,
) -> _ObservableCameraSelection:
    """Classify every immutable assignment without enforcing aggregate minima."""
    minimum_points = settings.projection.minimum_projected_points_per_camera
    retained: dict[CameraEvidencePartition, list[SceneCamera]] = {
        CameraEvidencePartition.FIT: [],
        CameraEvidencePartition.HOLDOUT: [],
    }
    exclusions: list[ExcludedCameraDiagnostics] = []
    for partition, cameras in (
        (CameraEvidencePartition.FIT, fit_cameras),
        (CameraEvidencePartition.HOLDOUT, holdout_cameras),
    ):
        for camera in cameras:
            if camera.camera_id not in projected_by_camera:
                raise ValueError(
                    "Selected camera is missing measured projection evidence: "
                    f"{camera.camera_id!r}."
                )
            projected = projected_by_camera[camera.camera_id]
            projected_count = len(projected.points_nht_scene)
            if projected_count >= minimum_points:
                retained[partition].append(camera)
                continue
            exclusions.append(
                ExcludedCameraDiagnostics(
                    camera_id=camera.camera_id,
                    original_partition=partition,
                    selected_line_pixel_count=projected.selected_line_pixel_count,
                    projected_line_point_count=projected_count,
                    reason=(
                        CameraExclusionReason.NO_DETECTED_LINE_PIXELS
                        if projected.selected_line_pixel_count == 0
                        else CameraExclusionReason.INSUFFICIENT_PROJECTED_POINTS
                    ),
                )
            )
    return _ObservableCameraSelection(
        fit=tuple(retained[CameraEvidencePartition.FIT]),
        holdout=tuple(retained[CameraEvidencePartition.HOLDOUT]),
        exclusions=tuple(exclusions),
    )


def _require_observable_camera_minima(
    observable: _ObservableCameraSelection,
    *,
    settings: AlignmentEvidenceSettings,
) -> None:
    """Fail explicitly when exclusions make either immutable aggregate too small."""
    if (
        len(observable.fit) < settings.minimum_fit_cameras
        or len(observable.holdout) < settings.minimum_holdout_cameras
    ):
        rendered = ";".join(
            (
                f"{item.camera_id}({item.original_partition.value},"
                f"selected={item.selected_line_pixel_count},"
                f"projected={item.projected_line_point_count},"
                f"reason={item.reason.value})"
            )
            for item in observable.exclusions
        )
        raise ValueError(
            "Observable alignment camera aggregate is below configured minima: "
            f"fit={len(observable.fit)}/{settings.minimum_fit_cameras},"
            f"holdout={len(observable.holdout)}/{settings.minimum_holdout_cameras};"
            f"exclusions=[{rendered}]."
        )


def _even_indices(total: int, count: int) -> tuple[int, ...]:
    if count < 1 or count > total:
        raise ValueError("Even-index selection requires 1 <= count <= total.")
    return tuple((2 * index + 1) * total // (2 * count) for index in range(count))


def _load_rgb_image(camera: SceneCamera) -> NDArray[np.uint8]:
    path = Path(camera.image_path)
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"Unable to decode exported image for {camera.camera_id!r}."
        ) from error
    if rgb.shape != (camera.height, camera.width, 3):
        raise ValueError(f"Exported image shape disagrees for {camera.camera_id!r}.")
    return rgb


def _estimate_ground_plane(
    points: NDArray[np.float64],
    cameras: tuple[SceneCamera, ...],
    *,
    seed: int,
    settings: GroundPlaneSettings,
) -> _GroundPlane:
    if len(points) < settings.minimum_candidate_points:
        raise ValueError(
            "Public NHT point cloud is too small for ground-plane fitting."
        )
    if len(cameras) < 3:
        raise ValueError("Ground-plane fitting requires at least three fit cameras.")
    poses = np.stack([camera.camera_to_scene.matrix() for camera in cameras])
    camera_centers = poses[:, :3, 3]
    camera_ups = poses[:, :3, :3] @ np.asarray([0.0, -1.0, 0.0])
    up = np.mean(camera_ups, axis=0)
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1.0e-10:
        raise ValueError("Fit cameras do not establish a stable scene-up direction.")
    up /= up_norm
    if float(np.min(camera_ups @ up)) < 0.90:
        raise ValueError("Fit-camera up vectors are not mutually consistent.")

    footprint_u = _project_axis_to_plane(np.asarray([1.0, 0.0, 0.0]), normal=up)
    footprint_v = np.cross(up, footprint_u)
    footprint_basis = np.stack((footprint_u, footprint_v), axis=1)
    point_uv = points @ footprint_basis
    camera_uv = camera_centers @ footprint_basis
    low = (
        np.quantile(camera_uv, settings.footprint_quantile, axis=0)
        - settings.footprint_margin
    )
    high = (
        np.quantile(camera_uv, 1.0 - settings.footprint_quantile, axis=0)
        + settings.footprint_margin
    )
    footprint_mask = np.all((point_uv >= low) & (point_uv <= high), axis=1)
    point_heights = points @ up
    camera_heights = camera_centers @ up
    camera_median = float(np.median(camera_heights))
    eligible = (
        footprint_mask
        & (point_heights >= camera_median - settings.maximum_camera_height)
        & (point_heights <= camera_median - settings.minimum_camera_height)
    )
    eligible_heights = point_heights[eligible]
    if len(eligible_heights) < settings.minimum_candidate_points:
        raise ValueError(
            "Too few public points lie in the configured ground-height band."
        )
    edges = np.arange(
        camera_median - settings.maximum_camera_height,
        camera_median - settings.minimum_camera_height + settings.histogram_bin_width,
        settings.histogram_bin_width,
    )
    if len(edges) < 2:
        raise ValueError("Ground-height histogram settings produce no bins.")
    histogram, edges = np.histogram(eligible_heights, bins=edges)
    peak_index = int(np.argmax(histogram))
    peak_height = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
    candidates = points[
        footprint_mask
        & (np.abs(point_heights - peak_height) <= settings.candidate_half_width)
    ]
    if len(candidates) < settings.minimum_candidate_points:
        raise ValueError("Ground-height peak has insufficient public point support.")
    normal, offset = _ransac_plane(
        candidates,
        up=up,
        seed=seed,
        settings=settings,
    )
    for _ in range(settings.refine_iterations):
        residuals = np.abs(candidates @ normal + offset)
        inliers = candidates[residuals <= settings.refine_threshold]
        if len(inliers) < settings.minimum_support_points:
            raise ValueError("Ground-plane refinement has insufficient support points.")
        normal, offset = _fit_plane_svd(inliers, up=up)
    support = points[
        footprint_mask & (np.abs(points @ normal + offset) <= settings.refine_threshold)
    ]
    if len(support) < settings.minimum_support_points:
        raise ValueError("Accepted ground plane has insufficient public point support.")
    if float(normal @ up) < settings.minimum_normal_up_cosine:
        raise ValueError(
            "Ground-plane normal disagrees with the camera-derived up direction."
        )
    signed_camera_heights = camera_centers @ normal + offset
    if (
        float(np.mean(signed_camera_heights > 0.0))
        < settings.minimum_positive_camera_fraction
    ):
        raise ValueError(
            "Too many fit cameras lie on or below the measured ground plane."
        )
    basis_u = _project_axis_to_plane(np.asarray([1.0, 0.0, 0.0]), normal=normal)
    basis_v = np.cross(normal, basis_u)
    origin = -offset * normal
    support_uv = (support - origin) @ np.stack((basis_u, basis_v), axis=1)
    quantile = settings.support_bounds_quantile
    uv_low = np.quantile(support_uv, quantile, axis=0)
    uv_high = np.quantile(support_uv, 1.0 - quantile, axis=0)
    if np.any(uv_low >= uv_high):
        raise ValueError("Measured ground-plane support bounds have no area.")
    return _GroundPlane(
        normal=normal,
        offset=offset,
        origin=origin,
        basis_u=basis_u,
        basis_v=np.asarray(basis_v, dtype=np.float64),
        support_uv_bounds=(
            float(uv_low[0]),
            float(uv_high[0]),
            float(uv_low[1]),
            float(uv_high[1]),
        ),
    )


def _ransac_plane(
    candidates: NDArray[np.float64],
    *,
    up: NDArray[np.float64],
    seed: int,
    settings: GroundPlaneSettings,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(seed)
    sample_count = min(settings.ransac_sample_limit, len(candidates))
    sample = candidates[rng.choice(len(candidates), size=sample_count, replace=False)]
    best_score = -1
    best_median = math.inf
    best_normal: NDArray[np.float64] | None = None
    best_offset = 0.0
    for _ in range(settings.ransac_iterations):
        triplet = sample[rng.choice(len(sample), size=3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-10:
            continue
        normal /= norm
        if float(normal @ up) < 0.0:
            normal = -normal
        if float(normal @ up) < settings.minimum_normal_up_cosine:
            continue
        offset = -float(normal @ triplet[0])
        residuals = np.abs(sample @ normal + offset)
        inlier_residuals = residuals[residuals <= settings.ransac_threshold]
        score = len(inlier_residuals)
        median = float(np.median(inlier_residuals)) if score else math.inf
        if score > best_score or (score == best_score and median < best_median):
            best_score = score
            best_median = median
            best_normal = normal.copy()
            best_offset = offset
    if best_normal is None:
        raise ValueError("Ground-plane RANSAC produced no valid hypothesis.")
    return best_normal, best_offset


def _fit_plane_svd(
    points: NDArray[np.float64],
    *,
    up: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    centroid = np.mean(points, axis=0)
    _, _, right = np.linalg.svd(points - centroid, full_matrices=False)
    normal = right[-1]
    if float(normal @ up) < 0.0:
        normal = -normal
    return np.asarray(normal, dtype=np.float64), -float(normal @ centroid)


def _project_axis_to_plane(
    axis: NDArray[np.float64],
    *,
    normal: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = axis - normal * float(axis @ normal)
    if float(np.linalg.norm(projected)) < 0.1:
        fallback = np.asarray([0.0, 1.0, 0.0])
        projected = fallback - normal * float(fallback @ normal)
    projected /= np.linalg.norm(projected)
    return np.asarray(projected, dtype=np.float64)


def _project_probability_to_ground(
    probability: NDArray[np.float32],
    *,
    camera: SceneCamera,
    plane: _GroundPlane,
    model_settings: CourtLineModelSettings,
    projection_settings: LineProjectionSettings,
) -> _ProjectedLineEvidence:
    selected_y, selected_x = np.nonzero(
        probability >= model_settings.probability_threshold
    )
    if len(selected_y) == 0:
        return _ProjectedLineEvidence(
            points_nht_scene=np.empty((0, 3), dtype=np.float64),
            points_uv=np.empty((0, 2), dtype=np.float64),
            selected_line_pixel_count=0,
        )
    selected_probability = probability[selected_y, selected_x]
    maximum = model_settings.maximum_selected_pixels_per_camera
    if len(selected_y) > maximum:
        order = np.argsort(-selected_probability, kind="stable")[:maximum]
        selected_y = selected_y[order]
        selected_x = selected_x[order]
    output_height, output_width = probability.shape
    pixels = np.column_stack(
        (
            selected_x.astype(np.float64) * (camera.width - 1) / (output_width - 1),
            selected_y.astype(np.float64) * (camera.height - 1) / (output_height - 1),
        )
    )
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    pose = camera.camera_to_scene.matrix()
    directions_camera = (
        np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsics).T
    )
    directions_nht = directions_camera @ pose[:3, :3].T
    directions_nht /= np.linalg.norm(directions_nht, axis=1, keepdims=True)
    camera_center = pose[:3, 3]
    denominator = directions_nht @ plane.normal
    numerator = -(float(camera_center @ plane.normal) + plane.offset)
    valid = np.abs(denominator) >= projection_settings.minimum_ray_plane_cosine
    distances = np.divide(
        numerator,
        denominator,
        out=np.full(len(pixels), np.nan, dtype=np.float64),
        where=valid,
    )
    valid &= np.isfinite(distances)
    valid &= distances > 0.0
    valid &= distances <= projection_settings.maximum_ray_distance
    points_nht = camera_center + distances[valid, None] * directions_nht[valid]
    points_uv = plane.to_uv(points_nht)
    u_min, u_max, v_min, v_max = plane.support_uv_bounds
    margin = projection_settings.bounds_margin
    in_bounds = (
        (points_uv[:, 0] >= u_min - margin)
        & (points_uv[:, 0] <= u_max + margin)
        & (points_uv[:, 1] >= v_min - margin)
        & (points_uv[:, 1] <= v_max + margin)
    )
    return _ProjectedLineEvidence(
        points_nht_scene=np.asarray(points_nht[in_bounds], dtype=np.float64),
        points_uv=np.asarray(points_uv[in_bounds], dtype=np.float64),
        selected_line_pixel_count=len(pixels),
    )


def _fit_court_hypotheses(
    fit_points_uv: NDArray[np.float64],
    *,
    bounds: tuple[float, float, float, float],
    seed: int,
    settings: CourtCandidateFitSettings,
) -> tuple[tuple[_CourtHypothesis, ...], float, float, ProposalSearchDiagnostics]:
    """Exhaust bounded topology-aware residual states, then refit at one scale."""
    points = np.asarray(fit_points_uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Measured fit line points must be a finite (N, 2) array.")
    if len(points) < 3:
        raise ValueError(
            "Measured fit line evidence is insufficient for court fitting."
        )
    original_points = points
    if len(points) > settings.maximum_fit_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(len(points), size=settings.maximum_fit_points, replace=False)
        )
        points = points[indices]
    template = sample_court_line_template(settings.samples_per_metre)
    u_min, u_max, v_min, v_max = bounds
    orientation_bands = _orientation_search_bands(settings)
    maximum_tile_width = _maximum_center_tile_width_scene_units(settings)
    center_tiles = _center_space_tiles(bounds, maximum_width=maximum_tile_width)
    resources = _proposal_search_resource_bounds(
        candidate_count=settings.candidate_count,
        orientation_band_count=len(orientation_bands),
        center_tile_count=len(center_tiles),
    )
    measured_maximum_tile_width = max(
        max(
            tile.logical_u_upper - tile.u_bounds[0],
            tile.logical_v_upper - tile.v_bounds[0],
        )
        for tile in center_tiles
    )
    if measured_maximum_tile_width > maximum_tile_width + 1.0e-12:
        raise RuntimeError("Derived center tile exceeded its physical width bound.")
    maximum_bound_magnitude = max(abs(float(item)) for item in bounds)
    center_tolerance = max(
        maximum_tile_width * 1.0e-9,
        float(np.finfo(np.float64).eps)
        * max(1.0, maximum_bound_magnitude)
        * 64.0,
    )
    rejection_reasons: list[str] = []
    states: tuple[_ProposalSearchState, ...] = (
        _ProposalSearchState(
            selected=(),
            residual=points.copy(),
            orientation_band_indices=(),
            center_tile_indices=(),
        ),
    )
    explored_tile_state_count = 0
    impossible_tile_state_count = 0
    feasible_proposal_count = 0
    duplicate_proposal_count = 0
    retained_proposal_count = 0
    expanded_state_count = 0
    residual_state_count = 0
    residual_tree_build_count = 0
    canonical_band_indices = {
        band: index for index, band in enumerate(sorted(orientation_bands))
    }
    template_bounds_by_band = {
        band: _template_point_relative_bounds(
            template,
            orientation_band=band,
            minimum_scale=settings.minimum_nht_scene_units_per_metre,
            maximum_scale=settings.maximum_nht_scene_units_per_metre,
        )
        for band in orientation_bands
    }
    for _depth in range(settings.candidate_count):
        expanded: list[_ProposalSearchState] = []
        for state in states:
            if len(state.residual) < 3:
                rejection_reasons.append(
                    f"residual_exhausted(points={len(state.residual)})"
                )
                continue
            tiled_proposals: list[_TiledProposal] = []
            residual_evidence = _prepare_residual_evidence(state.residual)
            residual_state_count += 1
            residual_tree_build_count += 1
            for orientation_band in orientation_bands:
                band_index = canonical_band_indices[orientation_band]
                possible_tiles: list[_CenterTile] = []
                for center_tile in center_tiles:
                    if _tile_is_geometrically_impossible(
                        tree=residual_evidence.nearest_tree,
                        tile=center_tile,
                        template_relative_bounds=(
                            template_bounds_by_band[orientation_band]
                        ),
                        settings=settings,
                        selected=state.selected,
                    ):
                        impossible_tile_state_count += 1
                        continue
                    possible_tiles.append(center_tile)
                explored_tile_state_count += len(possible_tiles)
                optimized_tiles = _optimize_center_tiles(
                    residual_evidence,
                    template=template,
                    center_tiles=possible_tiles,
                    orientation_band=orientation_band,
                    seed=seed,
                    settings=settings,
                    selected=state.selected,
                )
                for center_tile, parameters, measured_score in optimized_tiles:
                    proposal = _NativeProposal(
                        parameters=parameters,
                        measured_score=measured_score,
                        orientation_band_radians=orientation_band,
                        residual_point_count=len(state.residual),
                    )
                    if proposal.measured_score < settings.minimum_template_score:
                        rejection_reasons.append(
                            "score_below_minimum("
                            f"{proposal.measured_score:.6f}<"
                            f"{settings.minimum_template_score:.6f})"
                        )
                        continue
                    parameters = proposal.parameters
                    saturation_reason = _scale_bound_saturation_reason(
                        float(parameters[3]),
                        settings=settings,
                    )
                    if saturation_reason is not None:
                        rejection_reasons.append(saturation_reason)
                        continue
                    provisional = _CourtHypothesis(
                        candidate_id="proposal",
                        center_uv=(float(parameters[0]), float(parameters[1])),
                        orientation_radians=float(parameters[2]),
                        nht_scene_units_per_metre=float(parameters[3]),
                        template_score=proposal.measured_score,
                        native_nht_scene_units_per_metre=float(parameters[3]),
                        native_template_score=proposal.measured_score,
                        native_center_uv=(float(parameters[0]), float(parameters[1])),
                        native_orientation_radians=float(parameters[2]),
                        common_scale_refit_center_displacement_metres=0.0,
                        maximum_common_scale_refit_center_displacement_metres=(
                            settings.maximum_center_refit_displacement_metres()
                        ),
                        proposal_orientation_band_radians=(
                            proposal.orientation_band_radians
                        ),
                        proposal_residual_point_count_before_suppression=(
                            proposal.residual_point_count
                        ),
                        proposal_residual_point_count_after_suppression=0,
                    )
                    if not _proposal_topology_compatible(
                        provisional,
                        selected=state.selected,
                        settings=settings,
                    ):
                        rejection_reasons.append(
                            "topology_incompatible("
                            f"center={provisional.center_uv},"
                            f"angle={provisional.orientation_radians:.9g})"
                        )
                        continue
                    tiled_proposals.append(
                        _TiledProposal(
                            proposal=proposal,
                            orientation_band_index=band_index,
                            center_tile_index=center_tile.flat_index,
                        )
                    )
            feasible_proposal_count += len(tiled_proposals)
            unique_proposals, duplicate_count = _deduplicate_tiled_proposals(
                tiled_proposals,
                center_tolerance=center_tolerance,
            )
            duplicate_proposal_count += duplicate_count
            retained_proposal_count += len(unique_proposals)
            for tiled in unique_proposals:
                proposal = tiled.proposal
                parameters = proposal.parameters
                suppressed = _suppress_assigned_points(
                    state.residual,
                    parameters=parameters,
                    assignment_distance_metres=(
                        settings.evidence_assignment_distance_metres
                    ),
                )
                if len(suppressed) >= len(state.residual):
                    rejection_reasons.append(
                        "proposal_suppressed_no_evidence("
                        f"center={(float(parameters[0]), float(parameters[1]))})"
                    )
                    continue
                selected = _CourtHypothesis(
                    candidate_id="proposal",
                    center_uv=(float(parameters[0]), float(parameters[1])),
                    orientation_radians=float(parameters[2]),
                    nht_scene_units_per_metre=float(parameters[3]),
                    template_score=proposal.measured_score,
                    native_nht_scene_units_per_metre=(
                        float(parameters[3])
                    ),
                    native_template_score=proposal.measured_score,
                    native_center_uv=(float(parameters[0]), float(parameters[1])),
                    native_orientation_radians=float(parameters[2]),
                    common_scale_refit_center_displacement_metres=0.0,
                    maximum_common_scale_refit_center_displacement_metres=(
                        settings.maximum_center_refit_displacement_metres()
                    ),
                    proposal_orientation_band_radians=(
                        proposal.orientation_band_radians
                    ),
                    proposal_residual_point_count_before_suppression=len(state.residual),
                    proposal_residual_point_count_after_suppression=len(suppressed),
                )
                expanded.append(
                    _ProposalSearchState(
                        selected=(*state.selected, selected),
                        residual=suppressed,
                        orientation_band_indices=(
                            *state.orientation_band_indices,
                            tiled.orientation_band_index,
                        ),
                        center_tile_indices=(
                            *state.center_tile_indices,
                            tiled.center_tile_index,
                        ),
                    )
                )
                expanded_state_count += 1
        states = tuple(sorted(expanded, key=_proposal_state_sort_key))
        if not states:
            break

    complete_states: list[
        tuple[_ProposalSearchState, float, float, int, float]
    ] = []
    for state in states:
        if len(state.selected) != settings.candidate_count:
            continue
        try:
            common_scale, maximum_deviation = _resolve_common_scale(
                np.asarray(
                    [
                        item.native_nht_scene_units_per_metre
                        for item in state.selected
                    ],
                    dtype=np.float64,
                ),
                maximum_relative_deviation=(
                    settings.common_scale_relative_tolerance
                ),
            )
        except ValueError as error:
            rejection_reasons.append(f"scale_incompatible({error})")
            continue
        original_residual = original_points
        for item in state.selected:
            original_residual = _suppress_assigned_points(
                original_residual,
                parameters=np.asarray(
                    (
                        item.native_center_uv[0],
                        item.native_center_uv[1],
                        item.native_orientation_radians,
                        item.native_nht_scene_units_per_metre,
                    ),
                    dtype=np.float64,
                ),
                assignment_distance_metres=(
                    settings.evidence_assignment_distance_metres
                ),
            )
        explained_count = len(original_points) - len(original_residual)
        native_score_sum = float(
            sum(item.native_template_score for item in state.selected)
        )
        complete_states.append(
            (
                state,
                common_scale,
                maximum_deviation,
                explained_count,
                native_score_sum,
            )
        )

    if not complete_states:
        rendered = ";".join(rejection_reasons)
        raise ValueError(
            "Deterministic multicourt proposal search exhausted before the "
            f"required feasible complete set: required={settings.candidate_count}, "
            f"remaining_states={len(states)}, "
            f"rejections=[{rendered}]."
        )
    complete_states.sort(key=_complete_proposal_state_sort_key)
    (
        selected_state,
        common_scale,
        maximum_deviation,
        explained_count,
        native_score_sum,
    ) = complete_states[0]
    native_hypotheses = list(
        _refine_selected_native_hypotheses(
            selected_state,
            points=points,
            template=template,
            orientation_bands=orientation_bands,
            center_tiles=center_tiles,
            seed=seed,
            settings=settings,
        )
    )
    common_scale, maximum_deviation = _resolve_common_scale(
        np.asarray(
            [
                item.native_nht_scene_units_per_metre
                for item in native_hypotheses
            ],
            dtype=np.float64,
        ),
        maximum_relative_deviation=settings.common_scale_relative_tolerance,
    )
    original_residual = original_points
    for item in native_hypotheses:
        original_residual = _suppress_assigned_points(
            original_residual,
            parameters=np.asarray(
                (
                    item.native_center_uv[0],
                    item.native_center_uv[1],
                    item.native_orientation_radians,
                    item.native_nht_scene_units_per_metre,
                ),
                dtype=np.float64,
            ),
            assignment_distance_metres=settings.evidence_assignment_distance_metres,
        )
    explained_count = len(original_points) - len(original_residual)
    native_score_sum = float(
        sum(item.native_template_score for item in native_hypotheses)
    )
    native_hypotheses = sorted(native_hypotheses, key=_hypothesis_sort_key)
    native_hypotheses = [
        _CourtHypothesis(
            candidate_id=f"candidate-{index:03d}",
            center_uv=item.center_uv,
            orientation_radians=item.orientation_radians,
            nht_scene_units_per_metre=item.nht_scene_units_per_metre,
            template_score=item.template_score,
            native_nht_scene_units_per_metre=(item.native_nht_scene_units_per_metre),
            native_template_score=item.native_template_score,
            native_center_uv=item.native_center_uv,
            native_orientation_radians=item.native_orientation_radians,
            common_scale_refit_center_displacement_metres=0.0,
            maximum_common_scale_refit_center_displacement_metres=(
                item.maximum_common_scale_refit_center_displacement_metres
            ),
            proposal_orientation_band_radians=(item.proposal_orientation_band_radians),
            proposal_residual_point_count_before_suppression=(
                item.proposal_residual_point_count_before_suppression
            ),
            proposal_residual_point_count_after_suppression=(
                item.proposal_residual_point_count_after_suppression
            ),
        )
        for index, item in enumerate(native_hypotheses)
    ]

    refitted: list[_CourtHypothesis] = []
    maximum_displacement_metres = settings.maximum_center_refit_displacement_metres()
    maximum_displacement_scene = common_scale * maximum_displacement_metres
    for candidate_index, native in enumerate(native_hypotheses):
        angle_tolerance = settings.family_orientation_tolerance_radians
        pose_bounds = [
            (
                max(u_min, native.center_uv[0] - maximum_displacement_scene),
                min(u_max, native.center_uv[0] + maximum_displacement_scene),
            ),
            (
                max(v_min, native.center_uv[1] - maximum_displacement_scene),
                min(v_max, native.center_uv[1] + maximum_displacement_scene),
            ),
            (
                max(
                    settings.orientation_minimum_radians,
                    native.orientation_radians - angle_tolerance,
                ),
                min(
                    settings.orientation_maximum_radians,
                    native.orientation_radians + angle_tolerance,
                ),
            ),
        ]
        pose, refit_score = _optimize_court(
            points,
            template=template,
            bounds=pose_bounds,
            seed=seed + settings.candidate_count + candidate_index,
            settings=settings,
            fixed_scale=common_scale,
            center_limit=(
                np.asarray(native.center_uv, dtype=np.float64),
                maximum_displacement_scene,
            ),
        )
        if refit_score < settings.minimum_template_score:
            raise ValueError(
                f"Common-scale refit candidate {candidate_index} score "
                f"{refit_score:.6f} is below {settings.minimum_template_score:.6f}."
            )
        center_displacement_metres = float(
            np.linalg.norm(np.asarray(pose[:2]) - np.asarray(native.center_uv))
            / common_scale
        )
        if center_displacement_metres > maximum_displacement_metres + 1.0e-10:
            raise ValueError(
                f"Common-scale refit candidate {candidate_index} moved "
                f"{center_displacement_metres:.9g} m from its native center, above "
                f"the derived bound {maximum_displacement_metres:.9g} m."
            )
        refitted.append(
            _CourtHypothesis(
                candidate_id=native.candidate_id,
                center_uv=(float(pose[0]), float(pose[1])),
                orientation_radians=float(pose[2]),
                nht_scene_units_per_metre=common_scale,
                template_score=refit_score,
                native_nht_scene_units_per_metre=(
                    native.native_nht_scene_units_per_metre
                ),
                native_template_score=native.native_template_score,
                native_center_uv=native.center_uv,
                native_orientation_radians=native.native_orientation_radians,
                common_scale_refit_center_displacement_metres=(
                    center_displacement_metres
                ),
                maximum_common_scale_refit_center_displacement_metres=(
                    maximum_displacement_metres
                ),
                proposal_orientation_band_radians=(
                    native.proposal_orientation_band_radians
                ),
                proposal_residual_point_count_before_suppression=(
                    native.proposal_residual_point_count_before_suppression
                ),
                proposal_residual_point_count_after_suppression=(
                    native.proposal_residual_point_count_after_suppression
                ),
            )
        )
    proposal_search = ProposalSearchDiagnostics(
        orientation_band_count=len(orientation_bands),
        center_tile_count=len(center_tiles),
        maximum_center_tile_width_scene_units=maximum_tile_width,
        maximum_complete_branch_count=resources.maximum_complete_branch_count,
        maximum_tile_state_count=resources.maximum_tile_state_count,
        maximum_residual_state_count=resources.maximum_residual_state_count,
        residual_state_count=residual_state_count,
        residual_tree_build_count=residual_tree_build_count,
        explored_tile_state_count=explored_tile_state_count,
        geometrically_impossible_tile_state_count=impossible_tile_state_count,
        feasible_proposal_count_before_deduplication=feasible_proposal_count,
        duplicate_proposal_count=duplicate_proposal_count,
        retained_proposal_count=retained_proposal_count,
        expanded_state_count=expanded_state_count,
        feasible_complete_state_count=len(complete_states),
        selected_orientation_band_indices=(
            selected_state.orientation_band_indices
        ),
        selected_center_tile_indices=selected_state.center_tile_indices,
        original_point_count=len(original_points),
        selected_residual_point_count=len(original_points) - explained_count,
        selected_explained_point_count=explained_count,
        selected_native_score_sum=native_score_sum,
    )
    return tuple(refitted), common_scale, maximum_deviation, proposal_search


def _refine_selected_native_hypotheses(
    selected_state: _ProposalSearchState,
    *,
    points: NDArray[np.float64],
    template: NDArray[np.float64],
    orientation_bands: Sequence[tuple[float, float]],
    center_tiles: Sequence[_CenterTile],
    seed: int,
    settings: CourtCandidateFitSettings,
) -> tuple[_CourtHypothesis, ...]:
    """Fully refine only the ranked branch, preserving its spatial basins."""

    if not (
        len(selected_state.selected)
        == len(selected_state.orientation_band_indices)
        == len(selected_state.center_tile_indices)
    ):
        raise RuntimeError("Selected proposal branch metadata is incomplete.")
    residual = points
    refined: list[_CourtHypothesis] = []
    for depth, (band_index, tile_index) in enumerate(
        zip(
            selected_state.orientation_band_indices,
            selected_state.center_tile_indices,
            strict=True,
        )
    ):
        orientation_band = orientation_bands[band_index]
        center_tile = center_tiles[tile_index]
        parameters, measured_score = _optimize_court(
            residual,
            template=template,
            bounds=[
                center_tile.u_bounds,
                center_tile.v_bounds,
                orientation_band,
                (
                    settings.minimum_nht_scene_units_per_metre,
                    settings.maximum_nht_scene_units_per_metre,
                ),
            ],
            seed=_proposal_branch_seed(
                seed,
                selected=refined,
                orientation_band=orientation_band,
                center_tile=center_tile,
            ),
            settings=settings,
            selected=refined,
        )
        if measured_score < settings.minimum_template_score:
            raise ValueError(
                f"Selected basin refinement at depth {depth} scored "
                f"{measured_score:.6f}, below "
                f"{settings.minimum_template_score:.6f}."
            )
        saturation_reason = _scale_bound_saturation_reason(
            float(parameters[3]),
            settings=settings,
        )
        if saturation_reason is not None:
            raise ValueError(
                f"Selected basin refinement at depth {depth} is invalid: "
                f"{saturation_reason}."
            )
        hypothesis = _CourtHypothesis(
            candidate_id="proposal",
            center_uv=(float(parameters[0]), float(parameters[1])),
            orientation_radians=float(parameters[2]),
            nht_scene_units_per_metre=float(parameters[3]),
            template_score=measured_score,
            native_nht_scene_units_per_metre=float(parameters[3]),
            native_template_score=measured_score,
            native_center_uv=(float(parameters[0]), float(parameters[1])),
            native_orientation_radians=float(parameters[2]),
            common_scale_refit_center_displacement_metres=0.0,
            maximum_common_scale_refit_center_displacement_metres=(
                settings.maximum_center_refit_displacement_metres()
            ),
            proposal_orientation_band_radians=orientation_band,
            proposal_residual_point_count_before_suppression=len(residual),
            proposal_residual_point_count_after_suppression=0,
        )
        if not _proposal_topology_compatible(
            hypothesis,
            selected=refined,
            settings=settings,
        ):
            raise ValueError(
                f"Selected basin refinement at depth {depth} violates topology."
            )
        suppressed = _suppress_assigned_points(
            residual,
            parameters=parameters,
            assignment_distance_metres=settings.evidence_assignment_distance_metres,
        )
        if len(suppressed) >= len(residual):
            raise ValueError(
                f"Selected basin refinement at depth {depth} suppressed no evidence."
            )
        refined.append(
            _CourtHypothesis(
                candidate_id=hypothesis.candidate_id,
                center_uv=hypothesis.center_uv,
                orientation_radians=hypothesis.orientation_radians,
                nht_scene_units_per_metre=hypothesis.nht_scene_units_per_metre,
                template_score=hypothesis.template_score,
                native_nht_scene_units_per_metre=(
                    hypothesis.native_nht_scene_units_per_metre
                ),
                native_template_score=hypothesis.native_template_score,
                native_center_uv=hypothesis.native_center_uv,
                native_orientation_radians=hypothesis.native_orientation_radians,
                common_scale_refit_center_displacement_metres=0.0,
                maximum_common_scale_refit_center_displacement_metres=(
                    hypothesis.maximum_common_scale_refit_center_displacement_metres
                ),
                proposal_orientation_band_radians=orientation_band,
                proposal_residual_point_count_before_suppression=len(residual),
                proposal_residual_point_count_after_suppression=len(suppressed),
            )
        )
        residual = suppressed
    return tuple(refined)


def _optimize_center_tiles(
    evidence: _ResidualEvidenceContext,
    *,
    template: NDArray[np.float64],
    center_tiles: Sequence[_CenterTile],
    orientation_band: tuple[float, float],
    seed: int,
    settings: CourtCandidateFitSettings,
    selected: Sequence[_CourtHypothesis],
) -> tuple[tuple[_CenterTile, NDArray[np.float64], float], ...]:
    """Optimize independent tiles concurrently while retaining canonical order."""

    tiles = tuple(center_tiles)
    if not tiles:
        return ()
    per_tile_iterations = max(
        settings.optimizer_population_size,
        math.ceil(settings.optimizer_maximum_iterations / len(tiles)),
    )

    def optimize_with_budget(
        center_tile: _CenterTile,
    ) -> tuple[_CenterTile, NDArray[np.float64], float]:
        search_bounds = [
            center_tile.u_bounds,
            center_tile.v_bounds,
            orientation_band,
            (
                settings.minimum_nht_scene_units_per_metre,
                settings.maximum_nht_scene_units_per_metre,
            ),
        ]
        parameters, measured_score = _optimize_court(
            evidence.points,
            template=template,
            bounds=search_bounds,
            seed=_proposal_branch_seed(
                seed,
                selected=selected,
                orientation_band=orientation_band,
                center_tile=center_tile,
            ),
            settings=settings,
            selected=selected,
            maximum_iterations=per_tile_iterations,
            polish=False,
            evidence_context=evidence,
        )
        return center_tile, parameters, measured_score

    worker_count = min(_MAXIMUM_TILE_OPTIMIZER_WORKERS, len(tiles))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(optimize_with_budget, tiles))


def _optimize_court(
    points: NDArray[np.float64],
    *,
    template: NDArray[np.float64],
    bounds: Sequence[tuple[float, float]],
    seed: int,
    settings: CourtCandidateFitSettings,
    fixed_scale: float | None = None,
    center_limit: tuple[NDArray[np.float64], float] | None = None,
    selected: Sequence[_CourtHypothesis] = (),
    maximum_iterations: int | None = None,
    polish: bool = True,
    evidence_context: _ResidualEvidenceContext | None = None,
) -> tuple[NDArray[np.float64], float]:
    if evidence_context is None:
        tree = cKDTree(points)
    else:
        if points is not evidence_context.points:
            raise ValueError(
                "Prepared residual evidence must be passed with its exact point view."
            )
        tree = evidence_context.nearest_tree

    def scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
        population = np.asarray(values, dtype=np.float64)
        if population.ndim != 2:
            raise ValueError("Optimizer population must be a two-dimensional array.")
        if fixed_scale is None:
            parameters = population
        else:
            parameters = np.column_stack(
                (population, np.full(len(population), fixed_scale, dtype=np.float64))
            )
        valid: NDArray[np.bool_] = np.ones(len(parameters), dtype=np.bool_)
        if center_limit is not None:
            center, maximum_distance = center_limit
            valid &= np.linalg.norm(parameters[:, :2] - center, axis=1) <= maximum_distance
        if fixed_scale is None and selected:
            valid &= np.asarray(
                [
                    _native_parameters_topology_compatible(
                        item,
                        selected=selected,
                        settings=settings,
                    )
                    for item in parameters
                ],
                dtype=np.bool_,
            )
        result: NDArray[np.float64] = np.zeros(
            len(parameters), dtype=np.float64
        )
        if not np.any(valid):
            return result
        active = parameters[valid]
        cosine = np.cos(active[:, 2])[:, None]
        sine = np.sin(active[:, 2])[:, None]
        template_x = template[:, 0][None, :]
        template_y = template[:, 1][None, :]
        transformed_x = (
            (template_x * cosine - template_y * sine) * active[:, 3, None]
            + active[:, 0, None]
        )
        transformed_y = (
            (template_x * sine + template_y * cosine) * active[:, 3, None]
            + active[:, 1, None]
        )
        transformed = np.stack((transformed_x, transformed_y), axis=2)
        distances, _indices = tree.query(
            transformed.reshape(-1, 2),
            k=1,
            workers=1,
        )
        distance_matrix = np.asarray(distances, dtype=np.float64).reshape(
            len(active), len(template)
        )
        widths = active[:, 3] * settings.score_distance_metres
        result[valid] = np.mean(
            np.exp(-0.5 * np.square(distance_matrix / widths[:, None])),
            axis=1,
        )
        return result

    def score(values: NDArray[np.float64]) -> float:
        return float(scores(np.asarray(values, dtype=np.float64)[None, :])[0])

    def objective(values: NDArray[np.float64]) -> float | NDArray[np.float64]:
        population = np.asarray(values, dtype=np.float64)
        if population.ndim == 1:
            return -score(population)
        if population.shape[0] == len(bounds):
            population = population.T
        return -scores(population)

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=(
            settings.optimizer_maximum_iterations
            if maximum_iterations is None
            else maximum_iterations
        ),
        popsize=settings.optimizer_population_size,
        tol=settings.optimizer_tolerance,
        polish=polish,
        workers=1,
        updating="deferred",
        vectorized=True,
    )
    optimized = np.asarray(result.x, dtype=np.float64)
    return optimized, score(optimized)


def _orientation_search_bands(
    settings: CourtCandidateFitSettings,
) -> tuple[tuple[float, float], ...]:
    """Partition the configured orientation domain into <=pi/2 basins."""
    bands: list[tuple[float, float]] = []
    lower = settings.orientation_minimum_radians
    maximum = settings.orientation_maximum_radians
    while lower < maximum - 1.0e-12:
        upper = min(maximum, lower + math.pi / 2.0)
        bands.append((lower, upper))
        lower = upper
    if not bands:
        raise ValueError("Orientation search produced no non-empty bands.")
    return tuple(bands)


def _proposal_search_resource_bounds(
    *,
    candidate_count: int,
    orientation_band_count: int,
    center_tile_count: int,
) -> _ProposalResourceBounds:
    """Derive and enforce the exact configured tiled-search state bounds."""
    for value, name in (
        (candidate_count, "candidate_count"),
        (orientation_band_count, "orientation_band_count"),
        (center_tile_count, "center_tile_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TypeError(f"{name} must be a positive integer.")
    branch_factor = orientation_band_count * center_tile_count
    maximum_complete_branches = branch_factor**candidate_count
    maximum_tile_states = sum(
        branch_factor**depth for depth in range(1, candidate_count + 1)
    )
    maximum_residual_states = sum(
        branch_factor**depth for depth in range(candidate_count)
    )
    configured = {
        "candidate_count": (candidate_count, _MAXIMUM_PROPOSAL_CANDIDATE_COUNT),
        "orientation_band_count": (
            orientation_band_count,
            _MAXIMUM_ORIENTATION_BAND_COUNT,
        ),
        "center_tile_count": (center_tile_count, _MAXIMUM_CENTER_TILE_COUNT),
        "branch_factor": (branch_factor, _MAXIMUM_BRANCH_FACTOR),
        "maximum_complete_branch_count": (
            maximum_complete_branches,
            _MAXIMUM_COMPLETE_BRANCH_COUNT,
        ),
        "maximum_tile_state_count": (
            maximum_tile_states,
            _MAXIMUM_TILE_STATE_COUNT,
        ),
        "maximum_residual_state_count": (
            maximum_residual_states,
            _MAXIMUM_RESIDUAL_STATE_COUNT,
        ),
    }
    exceeded = {
        name: (actual, maximum)
        for name, (actual, maximum) in configured.items()
        if actual > maximum
    }
    if exceeded:
        raise ValueError(
            "Configured spatial multicourt search exceeds its exact resource cap: "
            f"{exceeded}."
        )
    return _ProposalResourceBounds(
        branch_factor=branch_factor,
        maximum_complete_branch_count=maximum_complete_branches,
        maximum_tile_state_count=maximum_tile_states,
        maximum_residual_state_count=maximum_residual_states,
    )


def _maximum_center_tile_width_scene_units(
    settings: CourtCandidateFitSettings,
) -> float:
    """Derive a basin-isolating width from physical compatibility geometry."""
    return float(
        settings.minimum_center_separation_metres
        * settings.minimum_nht_scene_units_per_metre
        / 2.0
    )


def _prepare_residual_evidence(
    points: NDArray[np.float64],
) -> _ResidualEvidenceContext:
    """Build the sole nearest-neighbour tree for one residual proposal state."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Residual proposal evidence must be a finite (N, 2) array.")
    read_only = values.view()
    read_only.setflags(write=False)
    return _ResidualEvidenceContext(
        points=read_only,
        nearest_tree=cKDTree(read_only),
    )


def _center_space_tiles(
    bounds: tuple[float, float, float, float],
    *,
    maximum_width: float,
) -> tuple[_CenterTile, ...]:
    """Tile exact support bounds with deterministic half-open ownership."""
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Center support bounds must contain four finite values.")
    u_min, u_max, v_min, v_max = (float(item) for item in values)
    if u_min >= u_max or v_min >= v_max:
        raise ValueError("Center support bounds must have positive two-dimensional area.")
    if not math.isfinite(maximum_width) or maximum_width <= 0.0:
        raise ValueError("Maximum center tile width must be positive and finite.")
    u_count = int(math.ceil((u_max - u_min) / maximum_width))
    v_count = int(math.ceil((v_max - v_min) / maximum_width))
    tile_count = u_count * v_count
    if tile_count > _MAXIMUM_CENTER_TILE_COUNT:
        raise ValueError(
            "Derived center-space tiling exceeds its resource bound: "
            f"{tile_count} > {_MAXIMUM_CENTER_TILE_COUNT}."
        )
    u_edges_list = [
        u_min + (u_max - u_min) * index / u_count
        for index in range(u_count + 1)
    ]
    v_edges_list = [
        v_min + (v_max - v_min) * index / v_count
        for index in range(v_count + 1)
    ]
    u_edges_list[-1] = u_max
    v_edges_list[-1] = v_max
    u_edges = tuple(u_edges_list)
    v_edges = tuple(v_edges_list)
    tiles: list[_CenterTile] = []
    for u_index in range(u_count):
        logical_u_upper = u_edges[u_index + 1]
        optimizer_u_upper = (
            logical_u_upper
            if u_index == u_count - 1
            else float(np.nextafter(logical_u_upper, u_edges[u_index]))
        )
        for v_index in range(v_count):
            logical_v_upper = v_edges[v_index + 1]
            optimizer_v_upper = (
                logical_v_upper
                if v_index == v_count - 1
                else float(np.nextafter(logical_v_upper, v_edges[v_index]))
            )
            tiles.append(
                _CenterTile(
                    flat_index=len(tiles),
                    u_index=u_index,
                    v_index=v_index,
                    u_bounds=(u_edges[u_index], optimizer_u_upper),
                    v_bounds=(v_edges[v_index], optimizer_v_upper),
                    logical_u_upper=logical_u_upper,
                    logical_v_upper=logical_v_upper,
                )
            )
    return tuple(tiles)


def _tile_is_geometrically_impossible(
    *,
    tree: cKDTree,
    tile: _CenterTile,
    template_relative_bounds: NDArray[np.float64],
    settings: CourtCandidateFitSettings,
    selected: Sequence[_CourtHypothesis],
) -> bool:
    """Prove a tile cannot reach the score or center-separation acceptance gates."""
    minimum_separation_scene = (
        settings.minimum_center_separation_metres
        * settings.minimum_nht_scene_units_per_metre
    )
    corners = np.asarray(
        (
            (tile.u_bounds[0], tile.v_bounds[0]),
            (tile.u_bounds[0], tile.logical_v_upper),
            (tile.logical_u_upper, tile.v_bounds[0]),
            (tile.logical_u_upper, tile.logical_v_upper),
        ),
        dtype=np.float64,
    )
    for existing in selected:
        maximum_separation = float(
            np.max(
                np.linalg.norm(
                    corners - np.asarray(existing.native_center_uv, dtype=np.float64),
                    axis=1,
                )
            )
        )
        if maximum_separation < minimum_separation_scene:
            return True

    box_minimum = template_relative_bounds[:, :2] + np.asarray(
        (tile.u_bounds[0], tile.v_bounds[0]), dtype=np.float64
    )
    box_maximum = template_relative_bounds[:, 2:] + np.asarray(
        (tile.logical_u_upper, tile.logical_v_upper), dtype=np.float64
    )
    box_centers = (box_minimum + box_maximum) / 2.0
    box_half_diagonals = np.linalg.norm((box_maximum - box_minimum) / 2.0, axis=1)
    nearest_to_centers, _indices = tree.query(box_centers, k=1, workers=1)
    distance_lower_bounds = np.maximum(
        np.asarray(nearest_to_centers, dtype=np.float64) - box_half_diagonals,
        0.0,
    )
    maximum_score_width = (
        settings.maximum_nht_scene_units_per_metre * settings.score_distance_metres
    )
    score_upper_bound = float(
        np.mean(
            np.exp(
                -0.5 * np.square(distance_lower_bounds / maximum_score_width)
            )
        )
    )
    return bool(score_upper_bound < settings.minimum_template_score)


def _template_point_relative_bounds(
    template: NDArray[np.float64],
    *,
    orientation_band: tuple[float, float],
    minimum_scale: float,
    maximum_scale: float,
) -> NDArray[np.float64]:
    """Enclose every rotated/scaled template sample exactly over one band."""
    lower, upper = orientation_band
    result: NDArray[np.float64] = np.empty((len(template), 4), dtype=np.float64)
    for index, point in enumerate(template):
        x, y = (float(item) for item in point)
        phase = math.atan2(y, x)
        angles = [lower, upper]
        for base in (-phase, math.pi / 2.0 - phase):
            minimum_k = math.ceil((lower - base) / math.pi)
            maximum_k = math.floor((upper - base) / math.pi)
            angles.extend(
                base + k * math.pi for k in range(minimum_k, maximum_k + 1)
            )
        coordinates = np.asarray(
            [
                (x * math.cos(angle) - y * math.sin(angle),
                 x * math.sin(angle) + y * math.cos(angle))
                for angle in angles
            ],
            dtype=np.float64,
        )
        scaled = np.concatenate(
            (coordinates * minimum_scale, coordinates * maximum_scale),
            axis=0,
        )
        result[index] = (
            float(np.min(scaled[:, 0])),
            float(np.min(scaled[:, 1])),
            float(np.max(scaled[:, 0])),
            float(np.max(scaled[:, 1])),
        )
    return result


def _deduplicate_tiled_proposals(
    proposals: Sequence[_TiledProposal],
    *,
    center_tolerance: float,
) -> tuple[tuple[_TiledProposal, ...], int]:
    """Remove only numerically equivalent adjacent-boundary optima."""
    ordered = sorted(
        proposals,
        key=lambda item: (
            _proposal_sort_key(item.proposal),
            item.orientation_band_index,
            item.center_tile_index,
        ),
    )
    retained: list[_TiledProposal] = []
    duplicate_count = 0
    for item in ordered:
        values = item.proposal.parameters
        duplicate = False
        for existing in retained:
            existing_values = existing.proposal.parameters
            center_distance = float(np.linalg.norm(values[:2] - existing_values[:2]))
            orientation_difference = abs(
                (float(values[2] - existing_values[2]) + math.pi / 2.0)
                % math.pi
                - math.pi / 2.0
            )
            scale_difference = abs(float(values[3] / existing_values[3] - 1.0))
            if (
                center_distance <= center_tolerance
                and orientation_difference <= 1.0e-9
                and scale_difference <= 1.0e-9
            ):
                duplicate = True
                break
        if duplicate:
            duplicate_count += 1
        else:
            retained.append(item)
    return tuple(retained), duplicate_count


def _proposal_branch_seed(
    seed: int,
    *,
    selected: Sequence[_CourtHypothesis],
    orientation_band: tuple[float, float],
    center_tile: _CenterTile,
) -> int:
    """Tie an optimizer seed to geometry, independent of branch iteration order."""
    value = seed & 0xFFFFFFFF
    tokens = [
        coordinate
        for item in selected
        for coordinate in (
            *item.native_center_uv,
            item.native_orientation_radians,
            item.native_nht_scene_units_per_metre,
        )
    ]
    tokens.extend(orientation_band)
    tokens.extend(
        tile_endpoint
        for tile_bounds in (center_tile.u_bounds, center_tile.v_bounds)
        for tile_endpoint in tile_bounds
    )
    for coordinate in tokens:
        token = int(round((coordinate + 4.0 * math.pi) * 1.0e12))
        value = (value * 1_664_525 + token + 1_013_904_223) & 0xFFFFFFFF
    return value


def _proposal_state_sort_key(
    state: _ProposalSearchState,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[float, float, float, float, float], ...],
]:
    """Canonicalize retained state ordering between exhaustive depths."""
    return (
        state.orientation_band_indices,
        state.center_tile_indices,
        tuple(_hypothesis_sort_key(item) for item in state.selected),
    )


def _complete_proposal_state_sort_key(
    value: tuple[_ProposalSearchState, float, float, int, float],
) -> tuple[
    int,
    float,
    tuple[tuple[float, float, float, float, float], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Rank feasible complete sets by explained evidence, then native score."""
    state, _common_scale, _maximum_deviation, explained_count, native_score_sum = (
        value
    )
    return (
        -explained_count,
        -native_score_sum,
        tuple(sorted(_hypothesis_sort_key(item) for item in state.selected)),
        state.orientation_band_indices,
        state.center_tile_indices,
    )


def _proposal_sort_key(
    proposal: _NativeProposal,
) -> tuple[float, float, float, float, float]:
    parameters = proposal.parameters
    return (
        -proposal.measured_score,
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        float(parameters[3]),
    )


def _hypothesis_sort_key(
    hypothesis: _CourtHypothesis,
) -> tuple[float, float, float, float, float]:
    return (
        -hypothesis.native_template_score,
        hypothesis.native_center_uv[0],
        hypothesis.native_center_uv[1],
        hypothesis.orientation_radians,
        hypothesis.native_nht_scene_units_per_metre,
    )


def _proposal_topology_compatible(
    proposal: _CourtHypothesis,
    *,
    selected: Sequence[_CourtHypothesis],
    settings: CourtCandidateFitSettings,
) -> bool:
    """Reject native proposals whose regulation footprints overlap an identity."""
    return _native_parameters_topology_compatible(
        np.asarray(
            (
                proposal.native_center_uv[0],
                proposal.native_center_uv[1],
                proposal.native_orientation_radians,
                proposal.native_nht_scene_units_per_metre,
            ),
            dtype=np.float64,
        ),
        selected=selected,
        settings=settings,
    )


def _native_parameters_topology_compatible(
    parameters: NDArray[np.float64],
    *,
    selected: Sequence[_CourtHypothesis],
    settings: CourtCandidateFitSettings,
) -> bool:
    """Return whether a native optimizer placement can represent a new court."""
    center_uv = (float(parameters[0]), float(parameters[1]))
    orientation_radians = float(parameters[2])
    scale = float(parameters[3])
    for existing in selected:
        orientation_difference = abs(
            (orientation_radians - existing.native_orientation_radians + math.pi / 2.0)
            % math.pi
            - math.pi / 2.0
        )
        if orientation_difference > settings.family_orientation_tolerance_radians:
            return False
        scale_difference = abs(scale / existing.native_nht_scene_units_per_metre - 1.0)
        if scale_difference > settings.family_scale_relative_tolerance:
            return False
        common_scale = float(
            np.median(
                np.asarray(
                    (
                        existing.native_nht_scene_units_per_metre,
                        scale,
                    ),
                    dtype=np.float64,
                )
            )
        )
        center_delta_scene = np.asarray(center_uv, dtype=np.float64) - np.asarray(
            existing.native_center_uv,
            dtype=np.float64,
        )
        if (
            float(np.linalg.norm(center_delta_scene)) / common_scale
            < settings.minimum_center_separation_metres - 1.0e-10
        ):
            return False
        if _native_rectangles_have_no_positive_overlap(
            center_delta_scene=center_delta_scene,
            existing_orientation=existing.native_orientation_radians,
            proposal_orientation=orientation_radians,
            common_scale=common_scale,
        ):
            continue
        topology = evaluate_court_topology(
            (
                ("existing", _native_metric_transform(existing, common_scale)),
                (
                    "proposal",
                    _native_parameter_metric_transform(
                        center_uv=center_uv,
                        orientation_radians=orientation_radians,
                        common_scale=common_scale,
                    ),
                ),
            )
        )
        if len(topology) != 1:
            raise RuntimeError("Native proposal topology did not produce one pair.")
        if (
            topology[0].footprint_overlap_fraction
            > settings.maximum_court_footprint_overlap_fraction
        ):
            return False
    return True


def _native_rectangles_have_no_positive_overlap(
    *,
    center_delta_scene: NDArray[np.float64],
    existing_orientation: float,
    proposal_orientation: float,
    common_scale: float,
) -> bool:
    """Use the exact rectangle separating axes to bypass polygon clipping."""

    def axes(orientation: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cosine = math.cos(orientation)
        sine = math.sin(orientation)
        return (
            np.asarray((cosine, sine), dtype=np.float64),
            np.asarray((-sine, cosine), dtype=np.float64),
        )

    existing_axes = axes(existing_orientation)
    proposal_axes = axes(proposal_orientation)
    half_width_scene = common_scale * HALF_DOUBLES_WIDTH
    half_length_scene = common_scale * HALF_LENGTH
    for axis in (*existing_axes, *proposal_axes):
        existing_radius = (
            half_width_scene * abs(float(axis @ existing_axes[0]))
            + half_length_scene * abs(float(axis @ existing_axes[1]))
        )
        proposal_radius = (
            half_width_scene * abs(float(axis @ proposal_axes[0]))
            + half_length_scene * abs(float(axis @ proposal_axes[1]))
        )
        separation = abs(float(axis @ center_delta_scene))
        if separation >= existing_radius + proposal_radius - 1.0e-12:
            return True
    return False


def _native_metric_transform(
    hypothesis: _CourtHypothesis,
    common_scale: float,
) -> RigidTransform:
    return _native_parameter_metric_transform(
        center_uv=hypothesis.center_uv,
        orientation_radians=hypothesis.orientation_radians,
        common_scale=common_scale,
    )


def _native_parameter_metric_transform(
    *,
    center_uv: tuple[float, float],
    orientation_radians: float,
    common_scale: float,
) -> RigidTransform:
    cosine = math.cos(orientation_radians)
    sine = math.sin(orientation_radians)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = ((cosine, -sine), (sine, cosine))
    matrix[:2, 3] = np.asarray(center_uv) / common_scale
    return RigidTransform.from_matrix(matrix)


def _scale_bound_saturation_reason(
    scale: float,
    *,
    settings: CourtCandidateFitSettings,
) -> str | None:
    lower = settings.minimum_nht_scene_units_per_metre
    upper = settings.maximum_nht_scene_units_per_metre
    margin = (upper - lower) * settings.scale_bound_margin_relative
    if scale - lower <= margin:
        return (
            "scale_bound_saturated(lower; "
            f"scale={scale:.9g}, lower={lower:.9g}, margin={margin:.9g})"
        )
    if upper - scale <= margin:
        return (
            "scale_bound_saturated(upper; "
            f"scale={scale:.9g}, upper={upper:.9g}, margin={margin:.9g})"
        )
    return None


def _suppress_assigned_points(
    points: NDArray[np.float64],
    *,
    parameters: NDArray[np.float64],
    assignment_distance_metres: float,
) -> NDArray[np.float64]:
    """Suppress proposal evidence by exact distance to regulation segments."""
    values = np.asarray(parameters, dtype=np.float64)
    center = values[:2]
    orientation = float(values[2])
    scale = float(values[3])
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    points_court = (np.asarray(points, dtype=np.float64) - center) @ rotation / scale
    minimum_distances: NDArray[np.float64] = np.full(
        len(points_court), np.inf, dtype=np.float64
    )
    for segment in COURT_LINE_SEGMENTS:
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        direction = end - start
        fraction = np.clip(
            ((points_court - start) @ direction) / float(direction @ direction),
            0.0,
            1.0,
        )
        closest = start + fraction[:, None] * direction
        distances = np.linalg.norm(points_court - closest, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    return np.asarray(
        points[minimum_distances > assignment_distance_metres],
        dtype=np.float64,
    )


def _assign_candidate_evidence(
    hypotheses: tuple[_CourtHypothesis, ...],
    *,
    plane: _GroundPlane,
    projected_by_camera: Mapping[str, _ProjectedLineEvidence],
    settings: CourtCandidateFitSettings,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Assign each measured point to at most one nearest fixed-scale court."""
    template = sample_court_line_template(settings.samples_per_metre)
    predicted = tuple(
        plane.from_uv(
            transform_template_2d(
                template,
                np.asarray(
                    (
                        hypothesis.center_uv[0],
                        hypothesis.center_uv[1],
                        hypothesis.orientation_radians,
                        hypothesis.nht_scene_units_per_metre,
                    ),
                    dtype=np.float64,
                ),
            )
        )
        for hypothesis in hypotheses
    )
    trees = tuple(cKDTree(points) for points in predicted)
    maximum_distance = (
        hypotheses[0].nht_scene_units_per_metre
        * settings.evidence_assignment_distance_metres
    )
    assigned: dict[str, dict[str, NDArray[np.float64]]] = {
        hypothesis.candidate_id: {} for hypothesis in hypotheses
    }
    for camera_id, evidence in projected_by_camera.items():
        observed = evidence.points_nht_scene
        distances = np.column_stack(
            [tree.query(observed, k=1, workers=1)[0] for tree in trees]
        )
        winner = np.argmin(distances, axis=1)
        winner_distance = distances[np.arange(len(observed)), winner]
        for index, hypothesis in enumerate(hypotheses):
            mask = (winner == index) & (winner_distance <= maximum_distance)
            assigned[hypothesis.candidate_id][camera_id] = np.asarray(
                observed[mask],
                dtype=np.float64,
            )
    return assigned


def _candidate_evidence(
    hypothesis: _CourtHypothesis,
    *,
    plane: _GroundPlane,
    metric_adapter: MetricSceneAdapter,
    fit_cameras: tuple[SceneCamera, ...],
    holdout_cameras: tuple[SceneCamera, ...],
    assigned_by_camera: Mapping[str, NDArray[np.float64]],
    settings: AlignmentEvidenceSettings,
) -> CandidateEvidence:
    template = sample_court_line_template(settings.candidate_fit.samples_per_metre)
    points_court = np.column_stack((template, np.zeros(len(template))))
    parameters = np.asarray(
        (
            hypothesis.center_uv[0],
            hypothesis.center_uv[1],
            hypothesis.orientation_radians,
            hypothesis.nht_scene_units_per_metre,
        ),
        dtype=np.float64,
    )
    predicted_nht = plane.from_uv(transform_template_2d(template, parameters))

    def correspondences(cameras: tuple[SceneCamera, ...]) -> CorrespondenceSet:
        court_parts: list[NDArray[np.float64]] = []
        metric_parts: list[NDArray[np.float64]] = []
        camera_ids: list[str] = []
        maximum_distance_nht = (
            settings.correspondences.maximum_match_distance_metres
            * metric_adapter.nht_scene_units_per_metre
        )
        predicted_tree = cKDTree(predicted_nht)
        for camera in cameras:
            observed_nht = assigned_by_camera[camera.camera_id]
            if len(observed_nht) == 0:
                continue
            distances, template_indices = predicted_tree.query(
                observed_nht,
                k=1,
                workers=1,
            )
            matched = np.flatnonzero(distances <= maximum_distance_nht)
            if (
                len(matched)
                > settings.correspondences.maximum_correspondences_per_camera
            ):
                positions = _even_indices(
                    len(matched),
                    settings.correspondences.maximum_correspondences_per_camera,
                )
                matched = matched[np.asarray(positions, dtype=np.int64)]
            if (
                len(matched)
                < settings.correspondences.minimum_correspondences_per_camera
            ):
                continue
            observed = observed_nht[matched]
            matched_template = np.asarray(template_indices[matched], dtype=np.int64)
            court_parts.append(points_court[matched_template])
            metric_parts.append(metric_adapter.metric_from_nht_points(observed))
            camera_ids.extend([camera.camera_id] * len(matched))
        if not court_parts:
            raise ValueError(
                f"Candidate {hypothesis.candidate_id!r} has no camera with the required "
                "uniquely assigned measured line correspondences in this partition."
            )
        return CorrespondenceSet(
            points_court=np.concatenate(court_parts),
            points_scene=np.concatenate(metric_parts),
            camera_ids=tuple(camera_ids),
        )

    return CandidateEvidence(
        court_instance_id=hypothesis.candidate_id.replace("candidate", "court"),
        candidate_id=hypothesis.candidate_id,
        fit=correspondences(fit_cameras),
        holdout=correspondences(holdout_cameras),
    )


def _plain_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_value(item) for item in value]
    return value


def _line_bundle_model_state(
    raw_state: Mapping[object, object],
) -> dict[str, torch.Tensor]:
    """Translate the trained single-line head namespace to its bundle head."""
    model_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key.startswith("model."):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Court-line tensor {key!r} is not a Tensor.")
        model_state[key.removeprefix("model.")] = value
    if not model_state:
        raise ValueError("Court-line checkpoint contains no model tensors.")

    legacy_head = {"final_conv.weight", "final_conv.bias"}
    bundle_head = {"heads.line.weight", "heads.line.bias"}
    legacy_present = legacy_head.intersection(model_state)
    bundle_present = bundle_head.intersection(model_state)
    if legacy_present and bundle_present:
        raise ValueError("Court-line checkpoint mixes legacy and bundle head tensors.")
    if legacy_present != legacy_head:
        raise ValueError("Court-line checkpoint has an incomplete legacy line head.")
    if bundle_present and bundle_present != bundle_head:
        raise ValueError("Court-line checkpoint has an incomplete bundle line head.")
    if legacy_present:
        model_state["heads.line.weight"] = model_state.pop("final_conv.weight")
        model_state["heads.line.bias"] = model_state.pop("final_conv.bias")
    return model_state


def _validate_embedded_architecture(
    settings: CourtLineModelSettings,
    model: Mapping[str, Any],
) -> None:
    """Cross-check explicit inference architecture against trained metadata."""
    architecture = settings.architecture
    expected_model_values: dict[str, object] = {
        "in_channels": 3,
        "num_classes": 1,
    }
    for key, expected in expected_model_values.items():
        if key not in model or model[key] != expected:
            raise ValueError(
                f"Court-line checkpoint model.{key} disagrees with {expected!r}."
            )
    encoder = _required_mapping(model, "encoder")
    expected_encoder_values: dict[str, object] = {
        "name": "dinov3",
        "backbone_name": architecture.backbone_name,
        "strict": architecture.backbone_strict,
        "train_mode": architecture.backbone_train_mode,
        "last_n_blocks": architecture.backbone_last_n_blocks,
        "out_indices": list(architecture.backbone_out_indices),
    }
    for key, expected in expected_encoder_values.items():
        if key not in encoder or encoder[key] != expected:
            raise ValueError(
                f"Court-line checkpoint encoder.{key} disagrees with explicit "
                f"architecture {expected!r}."
            )
    lora = _required_mapping(encoder, "lora")
    expected_lora_values: dict[str, object] = {
        "enabled": architecture.lora_enabled,
        "rank": architecture.lora_rank,
        "alpha": architecture.lora_alpha,
        "dropout": architecture.lora_dropout,
        "target_modules": list(architecture.lora_target_modules),
    }
    for key, expected in expected_lora_values.items():
        if key not in lora or lora[key] != expected:
            raise ValueError(
                f"Court-line checkpoint LoRA {key} disagrees with explicit "
                f"architecture {expected!r}."
            )
    decoder = _required_mapping(model, "decoder")
    expected_decoder_values: dict[str, object] = {
        "name": "dpt",
        "channels": architecture.decoder_channels,
        "reassemble_factors": list(architecture.decoder_reassemble_factors),
    }
    for key, expected in expected_decoder_values.items():
        if key not in decoder or decoder[key] != expected:
            raise ValueError(
                f"Court-line checkpoint decoder.{key} disagrees with explicit "
                f"architecture {expected!r}."
            )


def _required_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in value:
        raise ValueError(f"Court-line checkpoint has no {key} mapping.")
    nested = value[key]
    if not isinstance(nested, dict):
        raise ValueError(f"Court-line checkpoint has no {key} mapping.")
    return nested


__all__ = [
    "LineProbabilityDetector",
    "MeasuredAlignmentEvidenceSource",
    "ProductionAlignmentEvidenceSource",
    "ProductionCourtLineDetector",
    "create_production_alignment_handler",
]
