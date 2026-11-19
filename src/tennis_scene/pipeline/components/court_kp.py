"""Court keypoint detection on the first frame of one camera."""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyConfig as CourtKPPostprocessConfig,
)
from src.tasks.court_detection.inference.contracts import (
    validate_hybrid_inference_config,
)
from src.tasks.court_detection.inference.regions import (
    CourtRegionPrediction,
    CourtRegionSearchConfig,
    Region,
    assess_region_prediction,
    predict_court_region,
    region_proposals,
    select_court_region,
)
from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchAddress,
    BatchSource,
    BatchUnit,
    SourceMemory,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, SourceVideo
from src.tennis_scene.pipeline.court_assets import CourtAssetCatalog
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.utils.configuration import PathResolver
from src.utils.geometry.keypoints import normalize_grid_keypoints
from src.utils.video import read_video_frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tasks.court_detection.inference.predictor import CourtPredictor
    from src.tasks.court_detection.model_io.images import PreparedCourtImage

LOGGER = logging.getLogger(__name__)

NUM_COURT_KEYPOINTS = 14
KEYPOINT_CONTRACT = "camera_view_v2"
"""Camera-local KP14 order: the only court keypoint contract of the pipeline."""


@dataclass(frozen=True)
class CourtDetectionInput:
    video: SourceVideo


@dataclass(frozen=True)
class CourtImagePrediction:
    points_px: NDArray[np.float32]
    valid: NDArray[np.bool_]
    diagnostics: dict[str, Any]
    region_selection: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class CourtKPConfig:
    """Court detector checkpoint and its hybrid KP+LINE postprocess."""

    checkpoint: Path
    device: str
    subpixel_refine: bool
    postprocess: CourtKPPostprocessConfig
    region_search: CourtRegionSearchConfig
    resolver: PathResolver
    asset_catalog: CourtAssetCatalog = field(default_factory=CourtAssetCatalog, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        validate_hybrid_inference_config(self.postprocess)


@dataclass
class CourtKPResult:
    """Court keypoints ``(N, T, 14, 2)`` normalised by ``(W-1, H-1)``."""

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_indices: NDArray[np.int32]
    diagnostics: dict[str, Any] | None = None

    def validate(self) -> tuple[bool, list[str]]:
        """Return ``(is_valid, errors)`` for the shape and value contract."""
        errors: list[str] = []
        if self.keypoints.ndim != 4 or self.keypoints.shape[2:] != (NUM_COURT_KEYPOINTS, 2):
            errors.append(f"keypoints shape must be (N, T, {NUM_COURT_KEYPOINTS}, 2), got {self.keypoints.shape}")
        if self.visibility.shape != self.keypoints.shape[:3]:
            errors.append(f"visibility shape must match keypoints[:3], got {self.visibility.shape} for {self.keypoints.shape}")
        if self.keypoints.ndim >= 2 and self.frame_indices.shape != (self.keypoints.shape[1],):
            errors.append(f"frame_indices shape must match (T,), got {self.frame_indices.shape}")
        if not np.isfinite(self.keypoints).all():
            errors.append("keypoints contain non-finite values")
        if not np.isfinite(self.visibility).all():
            errors.append("visibility contains non-finite values")
        tol = 1e-6
        if self.visibility.shape == self.keypoints.shape[:3]:
            visible_points = self.keypoints[self.visibility > 0]
            if np.any(visible_points < -tol) or np.any(visible_points > 1.0 + tol):
                errors.append("visible keypoints must be normalized to [0, 1]")
        if np.any(self.visibility < -tol) or np.any(self.visibility > 1.0 + tol):
            errors.append("visibility must be normalized to [0, 1]")
        return len(errors) == 0, errors


@dataclass(frozen=True)
class _CourtBatchImage:
    rgb: NDArray[np.uint8]
    prepared: PreparedCourtImage
    region: Region
    selection: dict[str, Any] | None


class CourtKPModule(InferenceComponent[CourtDetectionInput, CourtKPResult]):
    """Detect camera-local KP14 on frame 0 of one static camera.

    Only frame 0 is inferred; ``court_calibration`` broadcasts the static
    geometry explicitly and records that later frames were not inferred.
    """

    io = ComponentIO("court_detection", CourtDetectionInput, CourtKPResult, {}, "court_observations", 2)
    batch_model = "court"
    batch_alias = "court"

    def __init__(self, config: CourtKPConfig) -> None:
        self.config = config
        self._checkpoint_identity: dict[str, Any] | None = None

    def model_specs(self) -> dict[str, ModelSpec[CourtPredictor]]:
        self._checkpoint_identity = self.config.asset_catalog.checkpoint_identity(self.config.checkpoint, self.config.resolver)
        return {"court": self._model_spec(self.config.asset_catalog.paths(self.config.checkpoint, self.config.resolver))}

    def _model_spec(self, weights: dict[str, Path]) -> ModelSpec[CourtPredictor]:
        return ModelSpec("court", weights, "checkpoint_target_bundle",
            self.config.device, "float32", {"subpixel_refine": self.config.subpixel_refine,
                "postprocess": self.config.postprocess, "region_search": self.config.region_search}, self._load_predictor)

    def asset_identities(self) -> dict[str, Any]:
        identities: dict[str, Any] = self.config.asset_catalog.identities(self.config.checkpoint, self.config.resolver)
        return identities

    def model_identity(self) -> dict[str, Any]:
        spec = self._model_spec({})
        return {"architecture": spec.architecture, "device": spec.device, "dtype": spec.dtype, "settings": dict(spec.settings)}

    def _load_predictor(self) -> CourtPredictor:
        LOGGER.info("Loading Court KP model from %s", self.config.checkpoint)
        from src.tasks.court_detection.inference.predictor import CourtPredictor

        predictor = CourtPredictor.load_from_checkpoint(
            self.config.checkpoint,
            resolver=self.config.resolver,
            device=self.config.device,
            subpixel_refine=self.config.subpixel_refine,
            hybrid_config=self.config.postprocess,
        )
        _require_camera_view_schema(predictor.adapter.spec.target_bundle.targets["kp"].schema)
        return predictor

    def process_with_models(self, inputs: CourtDetectionInput, models: ModelBorrower) -> CourtKPResult:
        video = inputs.video
        predictor: CourtPredictor = models.borrow("court")
        return self._result(video, self.predict_image(self._read_rgb(video), predictor))

    def batch_adapter(self) -> BatchAdapter:
        return self

    def open_batch(self, clip_id: str, inputs: CourtDetectionInput, models: ModelBorrower,
                   memory: SourceMemory) -> BatchSource:
        video = inputs.video
        # BGR/RGB, full-image quality assessment and candidate diagnostics.
        memory.reserve(video.width * video.height * 24)
        output: CourtKPResult | None = None

        def units() -> Generator[BatchUnit, None, None]:
            predictor: CourtPredictor = models.borrow(self.batch_alias)
            rgb = self._read_rgb(video)
            full = (0, 0, video.width, video.height)
            channels = int(sum(head.output_channels for head in predictor.adapter.spec.target_bundle.targets.values()))

            def estimate(region: Region) -> int:
                x0, y0, x1, y1 = region
                height, width = y1 - y0, x1 - x0
                h, w = predictor.batch_image_size(height, width)
                return int(h * w * (3 * 4 * 3 + channels * 4 * 3) + height * width * 16)

            selection = None
            region = full
            if self.config.region_search.enabled:
                candidates = region_proposals(rgb)
                # Search uses the existing ordered, single-image candidate policy.
                with memory.workspace(2 * max(estimate(item) for item in candidates)):
                    chosen, selection = self._select_region(rgb, predictor)
                if chosen is None:
                    raise RuntimeError("Enabled Court region search must select a region")
                region = chosen
            x0, y0, x1, y1 = region
            item_bytes = estimate(region)
            with memory.workspace(item_bytes):
                prepared = predictor.prepare_batch_image(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            yield BatchUnit(BatchAddress(clip_id, video.camera_id, 0), tuple(prepared.images.shape),
                            _CourtBatchImage(rgb, prepared, region, selection), item_bytes)

        def write(unit: BatchUnit, result: CourtImagePrediction) -> None:
            nonlocal output
            if unit.address.source_frame != 0 or output is not None:
                raise ValueError("Court must write frame 0 exactly once")
            output = self._result(video, result)

        def finish() -> CourtKPResult:
            if output is None:
                raise ValueError("Court did not predict source frame 0")
            return output

        return BatchSource(units(), write, finish)

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]:
        predictor: CourtPredictor = models.borrow(self.batch_alias)
        items: list[_CourtBatchImage] = [unit.payload for unit in units]
        decoded = predictor.predict_prepared([item.prepared for item in items], postprocess="hybrid", heads=("kp", "line"))
        results: list[CourtImagePrediction | ValueError] = []
        for item, prediction in zip(items, decoded, strict=True):
            if isinstance(prediction, ValueError):
                results.append(prediction)
                continue
            try:
                assessed = assess_region_prediction(prediction, item.rgb, item.region, self.config.region_search)
                points, valid, diagnostic = self._geometry(assessed)
                results.append(CourtImagePrediction(points, valid, diagnostic, item.selection))
            except ValueError as error:
                results.append(error)
        return results

    @staticmethod
    def _read_rgb(video: SourceVideo) -> NDArray[np.uint8]:
        packet = read_video_frame(video.path, 0)
        if tuple(packet.original_size) != (video.width, video.height):
            raise ValueError(f"Court frame size {packet.original_size} disagrees with source {(video.width, video.height)}")
        rgb = cast("NDArray[np.uint8]", cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 1:
            raise ValueError("Court image must be uint8 RGB (H,W,3)")
        return rgb

    def _result(self, video: SourceVideo, prediction: CourtImagePrediction) -> CourtKPResult:
        if self._checkpoint_identity is None:
            raise RuntimeError("Court assets must be bound before inference")
        checkpoint = self._checkpoint_identity
        keypoints = normalize_grid_keypoints(prediction.points_px, video.width, video.height)
        result = CourtKPResult(
            keypoints[None, None].astype(np.float32), prediction.valid[None, None].astype(np.float32), np.array([0], np.int32),
            {"schema": "court_kp_quality_v2", "checkpoint": checkpoint,
             "postprocess": asdict(self.config.postprocess), "region_search": asdict(self.config.region_search),
             "output_keypoint_contract": KEYPOINT_CONTRACT,
             "cameras": [{"camera_id": video.camera_id, "video_path": str(video.path), "region_selection": prediction.region_selection,
                          "frames": [{"frame_index": 0, **prediction.diagnostics}]}],
             "temporal_policy": "static_first_frame", "observed_frame_indices": [0], "source_frame_count": video.num_frames},
        )
        valid, errors = result.validate()
        if not valid:
            raise ValueError(f"Invalid CourtKP result: {errors}")
        return result

    def predict_image(self, rgb: NDArray[np.uint8], predictor: CourtPredictor) -> CourtImagePrediction:
        """Run the same camera-local geometry on an explicit RGB image.

        The caller owns the predictor. No video/source identity is invented for
        JPEG stores. Region/model errors propagate unchanged.
        """
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 1:
            raise ValueError("Court image must be uint8 RGB (H,W,3)")
        region, selection = self._select_region(rgb, predictor)
        points, valid, diagnostic = self._predict_frame_geometry(rgb, predictor, region=region)
        return CourtImagePrediction(points, valid, diagnostic, selection)

    def _select_region(self, rgb: NDArray[np.uint8], predictor: CourtPredictor) -> tuple[Region | None, dict[str, Any] | None]:
        if not self.config.region_search.enabled:
            return None, None
        selection = select_court_region(predictor, rgb, self.config.region_search)
        return selection.region, {"frame_index": 0, "region_xyxy": list(selection.region),
            "candidates": list(selection.candidates), "original_size_hw": list(rgb.shape[:2])}

    def _predict_frame_geometry(
        self,
        frame_rgb: NDArray[np.uint8],
        predictor: CourtPredictor,
        *,
        region: Region | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
        if region is None:
            prediction = predictor.predict(frame_rgb, postprocess="hybrid", heads=("kp", "line"))
            assessed = assess_region_prediction(prediction, frame_rgb, (0, 0, frame_rgb.shape[1], frame_rgb.shape[0]), self.config.region_search)
        else:
            assessed = predict_court_region(predictor, frame_rgb, region, quality_config=self.config.region_search)
        return self._geometry(assessed)

    @staticmethod
    def _geometry(prediction: CourtRegionPrediction) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
        points, visible = prediction.downstream_keypoints()
        diagnostic: dict[str, Any] = prediction.geometry_diagnostics()
        if points.shape != (NUM_COURT_KEYPOINTS, 2) or visible.shape != (NUM_COURT_KEYPOINTS,):
            raise ValueError("Hybrid court geometry must contain ordered KP14")
        _require_camera_view_schema(prediction.cropped.keypoint_schema)
        diagnostic["output_keypoint_contract"] = KEYPOINT_CONTRACT
        return points, visible, diagnostic


def _require_camera_view_schema(schema: str | None) -> None:
    camera_view = schema is not None and (
        schema.startswith("synthetic_camera_view_kp14_") or schema == "tennis_court_detector_kp14:gaussian_max_v1"
    )
    if not camera_view:
        raise ValueError(f"{KEYPOINT_CONTRACT} requires a camera-view KP14 checkpoint schema, got {schema!r}")
