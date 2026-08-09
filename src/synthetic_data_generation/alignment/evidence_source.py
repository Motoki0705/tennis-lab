"""Measured production alignment evidence from public NHT scene files only."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    CameraLineDiagnostics,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    MeasuredCameraLines,
    MetricSceneAdapter,
)
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.settings import (
    AlignmentEvidenceSettings,
    CourtCandidateFitSettings,
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import (
    CourtDecoderConfig,
    CourtEncoderConfig,
    CourtLoRAConfig,
    CourtModelConfig,
)
from src.tasks.court_detection.inference import CourtLinePredictor
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtLineModelIO,
)
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.utils.configuration import PathResolver, PathRole
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)


class LineProbabilityDetector(Protocol):
    """Explicit trained detector boundary used by measured evidence collection."""

    def preflight(self) -> None:
        """Validate and load the configured detector without writing outputs."""

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Return finite court-line probabilities for one real image."""


class ProductionCourtLineDetector:
    """Court-line predictor loaded from explicit repository/checkpoint authority."""

    def __init__(
        self, settings: CourtLineModelSettings, resolver: PathResolver
    ) -> None:
        self._settings = settings
        self._resolver = resolver
        self._predictor: CourtLinePredictor | None = None

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
            num_classes=1,
            encoder=encoder_config,
            decoder=CourtDecoderConfig(
                name="dpt",
                channels=architecture.decoder_channels,
                reassemble_factors=architecture.decoder_reassemble_factors,
            ),
        )
        model = CourtHierarchicalModel.from_config(model_config)
        raw_state = raw.get("state_dict")
        if not isinstance(raw_state, Mapping):
            raise ValueError("Court-line checkpoint has no state_dict mapping.")
        model_state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if isinstance(key, str) and key.startswith("model."):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Court-line tensor {key!r} is not a Tensor.")
                model_state[key.removeprefix("model.")] = value
        if not model_state:
            raise ValueError("Court-line checkpoint contains no model tensors.")
        model.load_state_dict(model_state, strict=True)
        spec = CourtModelSpec(
            task="line",
            in_channels=3,
            output_channels=1,
            short_side=settings.expected_short_side,
            encoder_kind="dinov3",
        )
        adapter = CourtLineModelIO(
            spec,
            bce_weight=architecture.line_bce_weight,
            dice_weight=architecture.line_dice_weight,
            pos_weight=architecture.line_positive_weight,
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

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Run the trained model; no heuristic or identity fallback is available."""
        self.preflight()
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("Court-line predictor was not loaded by preflight.")
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


class MeasuredAlignmentEvidenceSource:
    """Deterministic fit/holdout evidence over public images, cameras, and points."""

    def __init__(
        self,
        settings: AlignmentEvidenceSettings,
        detector: LineProbabilityDetector,
    ) -> None:
        self._settings = settings
        self._detector = detector

    def preflight(self, scene: StandardSceneExport) -> None:
        """Validate all real evidence and model requirements before mutation."""
        selected = _select_cameras(
            scene.cameras, maximum=self._settings.maximum_cameras
        )
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
        self.preflight(scene)
        return self._collect_after_preflight(scene)

    def _collect_after_preflight(
        self,
        scene: StandardSceneExport,
    ) -> AlignmentEvidence:
        """Collect after input/model checks; production preflight caches this result."""
        selected = _select_cameras(
            scene.cameras, maximum=self._settings.maximum_cameras
        )
        fit_cameras, holdout_cameras = _partition_cameras(
            selected,
            settings=self._settings,
        )
        plane = _estimate_ground_plane(
            np.asarray(scene.points_scene[:, :3], dtype=np.float64),
            fit_cameras,
            seed=self._settings.seed,
            settings=self._settings.ground_plane,
        )
        projected_by_camera: dict[str, _ProjectedLineEvidence] = {}
        for camera in fit_cameras + holdout_cameras:
            image = _load_rgb_image(camera)
            probability = self._detector.predict_probability(image)
            projected = _project_probability_to_ground(
                probability,
                camera=camera,
                plane=plane,
                model_settings=self._settings.line_model,
                projection_settings=self._settings.projection,
            )
            if (
                len(projected.points_nht_scene)
                < self._settings.projection.minimum_projected_points_per_camera
            ):
                raise ValueError(
                    f"Camera {camera.camera_id!r} has insufficient measured projected "
                    f"court-line evidence: {len(projected.points_nht_scene)} < "
                    f"{self._settings.projection.minimum_projected_points_per_camera}."
                )
            projected_by_camera[camera.camera_id] = projected

        fit_points_uv = np.concatenate(
            [projected_by_camera[camera.camera_id].points_uv for camera in fit_cameras]
        )
        hypotheses = _fit_court_hypotheses(
            fit_points_uv,
            bounds=plane.support_uv_bounds,
            seed=self._settings.seed,
            settings=self._settings.candidate_fit,
        )
        candidate_scales = np.asarray(
            [item.nht_scene_units_per_metre for item in hypotheses],
            dtype=np.float64,
        )
        common_scale, maximum_deviation = _resolve_primary_scale(
            candidate_scales,
            maximum_relative_deviation=(
                self._settings.candidate_fit.common_scale_relative_tolerance
            ),
        )
        nht_from_metric = np.eye(4, dtype=np.float64)
        nht_from_metric[:3, :3] *= common_scale
        metric_adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(
            nht_from_metric
        )
        candidates = tuple(
            _candidate_evidence(
                hypothesis,
                plane=plane,
                metric_adapter=metric_adapter,
                fit_cameras=fit_cameras,
                holdout_cameras=holdout_cameras,
                projected_by_camera=projected_by_camera,
                settings=self._settings,
            )
            for hypothesis in hypotheses
        )
        camera_order = fit_cameras + holdout_cameras
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
                    nht_scene_units_per_metre=hypothesis.nht_scene_units_per_metre,
                    template_score=hypothesis.template_score,
                )
                for hypothesis in hypotheses
            ),
            common_nht_scene_units_per_metre=common_scale,
            maximum_relative_scale_deviation=maximum_deviation,
        )
        return AlignmentEvidence(
            partitions=AlignmentPartitions(
                fit_camera_ids=tuple(camera.camera_id for camera in fit_cameras),
                holdout_camera_ids=tuple(
                    camera.camera_id for camera in holdout_cameras
                ),
            ),
            candidates=candidates,
            measured_camera_lines=tuple(
                MeasuredCameraLines(
                    camera_id=camera.camera_id,
                    points_nht_scene=projected_by_camera[
                        camera.camera_id
                    ].points_nht_scene,
                )
                for camera in camera_order
            ),
            complex_points_scene=metric_adapter.metric_from_nht_points(
                np.asarray(scene.points_scene[:, :3], dtype=np.float64)
            ),
            primary_candidate_id=hypotheses[0].candidate_id,
            metric_adapter=metric_adapter,
            diagnostics=diagnostics,
        )


def _resolve_primary_scale(
    candidate_scales: NDArray[np.float64],
    *,
    maximum_relative_deviation: float,
) -> tuple[float, float]:
    """Use the best-scoring hypothesis as the sole metric-scene scale authority."""
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
    primary_scale = float(scales[0])
    deviation = float(np.max(np.abs(scales / primary_scale - 1.0)))
    if deviation > maximum_relative_deviation:
        raise ValueError(
            "Court candidates do not establish one metric scene scale relative to "
            f"the primary hypothesis: {deviation:.6f} exceeds "
            f"{maximum_relative_deviation:.6f}."
        )
    return primary_scale, deviation


class ProductionAlignmentEvidenceSource(MeasuredAlignmentEvidenceSource):
    """Public constructor that always uses the configured trained line detector."""

    def __init__(
        self,
        settings: AlignmentEvidenceSettings,
        resolver: PathResolver,
    ) -> None:
        super().__init__(
            settings,
            ProductionCourtLineDetector(settings.line_model, resolver),
        )
        self._cached_scene_key: tuple[str, Path] | None = None
        self._cached_evidence: AlignmentEvidence | None = None

    def preflight(self, scene: StandardSceneExport) -> None:
        """Measure complete evidence so invalidation cannot conceal an evidence failure."""
        super().preflight(scene)
        evidence = self._collect_after_preflight(scene)
        self._cached_scene_key = (scene.scene_id, scene.scene_path)
        self._cached_evidence = evidence

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        """Return evidence proven during preflight, measuring if called standalone."""
        key = (scene.scene_id, scene.scene_path)
        if self._cached_scene_key != key or self._cached_evidence is None:
            self.preflight(scene)
        evidence = self._cached_evidence
        if evidence is None:
            raise RuntimeError(
                "Production alignment preflight did not retain evidence."
            )
        return evidence


def create_production_alignment_handler(
    *,
    settings: AlignmentEvidenceSettings,
    policy: AlignmentAcceptancePolicy,
    resolver: PathResolver,
) -> AlignmentStageHandler:
    """Bind the executable measured evidence source into the canonical stage."""
    return AlignmentStageHandler(
        evidence_source=ProductionAlignmentEvidenceSource(settings, resolver),
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
    if len(ordered) <= maximum:
        return ordered
    indices = _even_indices(len(ordered), maximum)
    return tuple(ordered[index] for index in indices)


def _partition_cameras(
    cameras: tuple[SceneCamera, ...],
    *,
    settings: AlignmentEvidenceSettings,
) -> tuple[tuple[SceneCamera, ...], tuple[SceneCamera, ...]]:
    minimum_total = settings.minimum_fit_cameras + settings.minimum_holdout_cameras
    if len(cameras) < minimum_total:
        raise ValueError(
            "NHT scene cannot satisfy independent alignment camera partitions: "
            f"{len(cameras)} < {minimum_total}."
        )
    holdout_count = max(
        settings.minimum_holdout_cameras,
        int(round(len(cameras) * settings.holdout_fraction)),
    )
    holdout_count = min(holdout_count, len(cameras) - settings.minimum_fit_cameras)
    holdout_indices = set(_even_indices(len(cameras), holdout_count))
    fit = tuple(
        camera for index, camera in enumerate(cameras) if index not in holdout_indices
    )
    holdout = tuple(
        camera for index, camera in enumerate(cameras) if index in holdout_indices
    )
    if (
        len(fit) < settings.minimum_fit_cameras
        or len(holdout) < settings.minimum_holdout_cameras
    ):
        raise ValueError(
            "Deterministic camera partitioning violated configured minima."
        )
    return fit, holdout


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
        raise ValueError(
            f"Camera {camera.camera_id!r} has no detected court-line pixels."
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
) -> tuple[_CourtHypothesis, ...]:
    points = np.asarray(fit_points_uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Measured fit line points must be a finite (N, 2) array.")
    if len(points) < 3:
        raise ValueError(
            "Measured fit line evidence is insufficient for court fitting."
        )
    if len(points) > settings.maximum_fit_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(len(points), size=settings.maximum_fit_points, replace=False)
        )
        points = points[indices]
    tree = cKDTree(points)
    template = _sample_court_line_template(settings.samples_per_metre)
    u_min, u_max, v_min, v_max = bounds
    search_bounds = [
        (u_min, u_max),
        (v_min, v_max),
        (settings.orientation_minimum_radians, settings.orientation_maximum_radians),
        (
            settings.minimum_nht_scene_units_per_metre,
            settings.maximum_nht_scene_units_per_metre,
        ),
    ]

    def score(parameters: NDArray[np.float64]) -> float:
        transformed = _transform_template(template, parameters)
        distances, _ = tree.query(transformed, k=1, workers=1)
        width = float(parameters[3]) * settings.score_distance_metres
        return float(np.mean(np.exp(-0.5 * np.square(distances / width))))

    parameters_and_scores: list[tuple[NDArray[np.float64], float]] = []
    for candidate_index in range(settings.candidate_count):
        bounds_for_candidate = list(search_bounds)
        if parameters_and_scores:
            reference = parameters_and_scores[0][0]
            angle = float(reference[2])
            scale = float(reference[3])
            angle_tolerance = settings.family_orientation_tolerance_radians
            scale_tolerance = settings.family_scale_relative_tolerance
            bounds_for_candidate[2] = (
                max(settings.orientation_minimum_radians, angle - angle_tolerance),
                min(settings.orientation_maximum_radians, angle + angle_tolerance),
            )
            bounds_for_candidate[3] = (
                max(
                    settings.minimum_nht_scene_units_per_metre,
                    scale * (1.0 - scale_tolerance),
                ),
                min(
                    settings.maximum_nht_scene_units_per_metre,
                    scale * (1.0 + scale_tolerance),
                ),
            )

        def objective(parameters: Sequence[float]) -> float:
            array = np.asarray(parameters, dtype=np.float64)
            penalty = 0.0
            for existing, _existing_score in parameters_and_scores:
                required = (
                    settings.minimum_center_separation_metres
                    * 0.5
                    * (float(array[3]) + float(existing[3]))
                )
                distance = float(np.linalg.norm(array[:2] - existing[:2]))
                penalty += settings.separation_penalty * max(0.0, required - distance)
            return -score(array) + penalty

        result = differential_evolution(
            objective,
            bounds_for_candidate,
            seed=seed + candidate_index,
            maxiter=settings.optimizer_maximum_iterations,
            popsize=settings.optimizer_population_size,
            tol=settings.optimizer_tolerance,
            polish=True,
            workers=1,
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        measured_score = score(parameters)
        if measured_score < settings.minimum_template_score:
            raise ValueError(
                f"Measured court candidate {candidate_index} score {measured_score:.6f} "
                f"is below {settings.minimum_template_score:.6f}."
            )
        parameters_and_scores.append((parameters, measured_score))
    return tuple(
        _CourtHypothesis(
            candidate_id=f"candidate-{index:03d}",
            center_uv=(float(parameters[0]), float(parameters[1])),
            orientation_radians=float(parameters[2]),
            nht_scene_units_per_metre=float(parameters[3]),
            template_score=score_value,
        )
        for index, (parameters, score_value) in enumerate(parameters_and_scores)
    )


def _candidate_evidence(
    hypothesis: _CourtHypothesis,
    *,
    plane: _GroundPlane,
    metric_adapter: MetricSceneAdapter,
    fit_cameras: tuple[SceneCamera, ...],
    holdout_cameras: tuple[SceneCamera, ...],
    projected_by_camera: Mapping[str, _ProjectedLineEvidence],
    settings: AlignmentEvidenceSettings,
) -> CandidateEvidence:
    template = _sample_court_line_template(settings.candidate_fit.samples_per_metre)
    points_court = np.column_stack((template, np.zeros(len(template))))
    parameters = np.asarray(
        (
            hypothesis.center_uv[0],
            hypothesis.center_uv[1],
            hypothesis.orientation_radians,
            metric_adapter.nht_scene_units_per_metre,
        ),
        dtype=np.float64,
    )
    predicted_nht = plane.from_uv(_transform_template(template, parameters))

    def correspondences(cameras: tuple[SceneCamera, ...]) -> CorrespondenceSet:
        court_parts: list[NDArray[np.float64]] = []
        metric_parts: list[NDArray[np.float64]] = []
        camera_ids: list[str] = []
        maximum_distance_nht = (
            settings.correspondences.maximum_match_distance_metres
            * metric_adapter.nht_scene_units_per_metre
        )
        for camera in cameras:
            observed_nht = projected_by_camera[camera.camera_id].points_nht_scene
            tree = cKDTree(observed_nht)
            distances, observed_indices = tree.query(predicted_nht, k=1, workers=1)
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
            observed = observed_nht[
                np.asarray(observed_indices[matched], dtype=np.int64)
            ]
            court_parts.append(points_court[matched])
            metric_parts.append(metric_adapter.metric_from_nht_points(observed))
            camera_ids.extend([camera.camera_id] * len(matched))
        if not court_parts:
            raise ValueError(
                f"Candidate {hypothesis.candidate_id!r} has no camera with the required "
                "measured line correspondences in this partition."
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


def _court_line_segments() -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    ...,
]:
    xd = HALF_DOUBLES_WIDTH
    xs = HALF_SINGLES_WIDTH
    yb = HALF_LENGTH
    ys = SERVICE_LINE_DISTANCE
    return (
        ((-xd, -yb), (-xd, yb)),
        ((xd, -yb), (xd, yb)),
        ((-xs, -yb), (-xs, yb)),
        ((xs, -yb), (xs, yb)),
        ((-xd, -yb), (xd, -yb)),
        ((-xd, yb), (xd, yb)),
        ((-xs, -ys), (xs, -ys)),
        ((-xs, ys), (xs, ys)),
        ((0.0, -ys), (0.0, ys)),
    )


def _sample_court_line_template(samples_per_metre: float) -> NDArray[np.float64]:
    parts: list[NDArray[np.float64]] = []
    for start, end in _court_line_segments():
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        count = max(
            16, int(np.linalg.norm(end_array - start_array) * samples_per_metre)
        )
        fraction = np.linspace(0.0, 1.0, count)[:, None]
        parts.append(start_array * (1.0 - fraction) + end_array * fraction)
    return np.asarray(np.concatenate(parts), dtype=np.float64)


def _transform_template(
    template: NDArray[np.float64],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    center_u, center_v, orientation, scale = parameters
    cosine = math.cos(float(orientation))
    sine = math.sin(float(orientation))
    rotation_transpose = np.asarray(
        ((cosine, sine), (-sine, cosine)),
        dtype=np.float64,
    )
    return np.asarray(
        template @ rotation_transpose * float(scale) + np.asarray((center_u, center_v)),
        dtype=np.float64,
    )


def _plain_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_value(item) for item in value]
    return value


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
