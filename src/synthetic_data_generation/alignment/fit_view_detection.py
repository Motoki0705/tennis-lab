"""Fit-side court inference and immutable detection-artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.provider.bundle import LoadedSceneProviderBundle
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.court_detection.evaluation.contracts import (
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.evaluation.homography_quality import (
    evaluate_homography_quality,
)
from src.tasks.court_detection.evaluation.image_evidence import line_edge_support

FIT_VIEW_COURT_DETECTIONS_SCHEMA = "fit_view_court_detections_v1"
_SHA256_LENGTH = 64
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CourtKeypointPrediction:
    """One detector result in the provider image's original pixel coordinates."""

    keypoints_xy: NDArray[np.float32]
    peak_scores: NDArray[np.float32]

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_xy, dtype=np.float32)
        scores = np.asarray(self.peak_scores, dtype=np.float32)
        if keypoints.shape != (14, 2) or not np.isfinite(keypoints).all():
            raise ValueError(
                "keypoints_xy must be finite with shape (14, 2), "
                f"got {keypoints.shape}."
            )
        if scores.shape != (14,) or not np.isfinite(scores).all():
            raise ValueError(
                f"peak_scores must be finite with shape (14,), got {scores.shape}."
            )
        if bool(np.any(scores < 0.0)) or bool(np.any(scores > 1.0)):
            raise ValueError("peak_scores must lie in [0, 1].")
        object.__setattr__(self, "keypoints_xy", keypoints)
        object.__setattr__(self, "peak_scores", scores)


class CourtKeypointPredictorPort(Protocol):
    """Narrow inference boundary used by the scene-alignment pipeline."""

    def predict_rgb(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> CourtKeypointPrediction:
        """Predict 14 court keypoints from one uint8 sRGB image."""
        ...


@dataclass(frozen=True)
class FitViewDetectionSettings:
    """Frozen selection and acceptance policy for fit-side inference."""

    artifact_id: str
    holdout_group_ids: tuple[int, ...]
    min_peak_score: float
    min_confident_keypoints: int
    homography: HomographyEvaluationCriteria

    def __post_init__(self) -> None:
        if _ARTIFACT_ID_PATTERN.fullmatch(self.artifact_id) is None:
            raise ValueError(
                "artifact_id must be a non-empty path-safe identifier, "
                f"got {self.artifact_id!r}."
            )
        groups = tuple(self.holdout_group_ids)
        if not groups or len(groups) != len(set(groups)):
            raise ValueError("holdout_group_ids must be non-empty and unique.")
        if any(isinstance(group, bool) or group < 0 for group in groups):
            raise ValueError("holdout_group_ids must contain non-negative integers.")
        if not 0.0 <= self.min_peak_score <= 1.0:
            raise ValueError("min_peak_score must lie in [0, 1].")
        if not 1 <= self.min_confident_keypoints <= 14:
            raise ValueError("min_confident_keypoints must lie in [1, 14].")
        object.__setattr__(self, "holdout_group_ids", groups)


def infer_fit_view_court_detections(
    bundle: LoadedSceneProviderBundle,
    predictor: CourtKeypointPredictorPort,
    *,
    settings: FitViewDetectionSettings,
    detector: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_at_utc: str,
    image_loader: Callable[[Path], NDArray[np.uint8]] | None = None,
) -> dict[str, Any]:
    """Infer only fit groups and return a fingerprinted v1 artifact payload.

    Holdout cameras are partitioned before any image path is resolved or image
    bytes are decoded. Their inventory is recorded, but their pixels never
    cross the predictor boundary.
    """
    cameras = tuple(bundle.manifest.cameras)
    available_groups = {camera.group_id for camera in cameras}
    missing_holdout = sorted(set(settings.holdout_group_ids) - available_groups)
    if missing_holdout:
        raise ValueError(
            f"Configured holdout groups are absent from provider: {missing_holdout}."
        )
    fit_cameras, holdout_cameras = partition_fit_and_holdout_cameras(
        cameras,
        holdout_group_ids=settings.holdout_group_ids,
    )
    if not fit_cameras or not holdout_cameras:
        raise ValueError("Both fit and holdout camera partitions must be non-empty.")

    load_image = image_loader or load_provider_rgb_image
    image_files = {image.camera_id: image for image in bundle.manifest.images}
    records: list[dict[str, Any]] = []
    for camera in fit_cameras:
        image = load_image(bundle.image_path(camera.camera_id))
        _validate_provider_image(image, camera=camera)
        prediction = predictor.predict_rgb(image)
        quality = evaluate_homography_quality(
            prediction.keypoints_xy,
            image_width=camera.width,
            image_height=camera.height,
            criteria=settings.homography,
        )
        confident_mask = prediction.peak_scores >= settings.min_peak_score
        confident_count = int(confident_mask.sum())
        reasons: list[str] = []
        if confident_count < settings.min_confident_keypoints:
            reasons.append("insufficient_detector_confidence")
        reasons.extend(quality.rejection_reasons)

        line_support: float | None = None
        if quality.projected_keypoints_normalized is not None:
            gray: NDArray[np.uint8] = np.asarray(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                dtype=np.uint8,
            )
            line_support = line_edge_support(
                gray,
                quality.projected_keypoints_normalized,
                distance_tolerance_px=(settings.homography.line_distance_tolerance_px),
                max_side=settings.homography.line_evidence_max_side,
            )
            if line_support < settings.homography.min_line_edge_support:
                reasons.append("weak_line_evidence")

        records.append(
            {
                "camera_id": camera.camera_id,
                "source_camera_id": camera.source_camera_id,
                "source_frame_index": camera.source_frame_index,
                "group_id": camera.group_id,
                "image": {
                    "relative_path": image_files[camera.camera_id].file.relative_path,
                    "sha256": image_files[camera.camera_id].file.sha256,
                    "width": camera.width,
                    "height": camera.height,
                },
                "keypoints_xy": prediction.keypoints_xy.astype(float).tolist(),
                "peak_scores": prediction.peak_scores.astype(float).tolist(),
                "confident_keypoint_mask": confident_mask.astype(int).tolist(),
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "homography": _array_list(quality.homography),
                "projected_keypoints_normalized": _array_list(
                    quality.projected_keypoints_normalized
                ),
                "inlier_mask": quality.inlier_mask.astype(int).tolist(),
                "residuals_normalized": quality.residuals_normalized.astype(
                    float
                ).tolist(),
                "metrics": {
                    "confident_keypoint_count": confident_count,
                    "peak_score_min": float(np.min(prediction.peak_scores)),
                    "peak_score_mean": float(np.mean(prediction.peak_scores)),
                    "peak_score_max": float(np.max(prediction.peak_scores)),
                    **quality.metrics,
                    "line_edge_support": line_support,
                },
            }
        )

    payload: dict[str, Any] = {
        "schema": FIT_VIEW_COURT_DETECTIONS_SCHEMA,
        "artifact_id": settings.artifact_id,
        "created_at_utc": created_at_utc,
        "provider": {
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_fingerprint": bundle.manifest.bundle_fingerprint,
            "scene_fingerprint": bundle.manifest.scene_fingerprint,
            "camera_array_sha256": bundle.manifest.camera_array_sha256,
            "shared_intrinsics_sha256": bundle.manifest.shared_intrinsics_sha256,
            "image_set_sha256": bundle.manifest.image_set_sha256,
        },
        "coordinate_contract": {
            "camera_axes": bundle.manifest.camera_axes,
            "pixel_coordinates": bundle.manifest.pixel_coordinates,
            "provider_image_color_space": bundle.manifest.image_color_space,
            "detector_input_color_space": "srgb8-rgb",
            "detector_output_coordinates": (
                "provider_original_zero_based_pixel_centres"
            ),
            "camera_image_dimensions": sorted(
                {
                    f"{camera.width}x{camera.height}"
                    for camera in bundle.manifest.cameras
                }
            ),
        },
        "split": {
            "group_definition": "provider SceneCamera.group_id",
            "fit_group_ids": sorted({camera.group_id for camera in fit_cameras}),
            "holdout_group_ids": list(settings.holdout_group_ids),
            "fit_camera_ids": [camera.camera_id for camera in fit_cameras],
            "holdout_camera_ids": [camera.camera_id for camera in holdout_cameras],
            "holdout_inference_status": "not_run",
        },
        "detector": dict(detector),
        "criteria": {
            "min_peak_score": settings.min_peak_score,
            "min_confident_keypoints": settings.min_confident_keypoints,
            "homography": asdict(settings.homography),
        },
        "records": records,
        "summary": summarize_fit_view_detections(records),
        "provenance": dict(provenance),
    }
    payload["artifact_fingerprint"] = compute_detection_artifact_fingerprint(payload)
    validate_fit_view_court_detections(payload, bundle=bundle)
    return payload


def partition_fit_and_holdout_cameras(
    cameras: Sequence[SceneCamera],
    *,
    holdout_group_ids: Sequence[int],
) -> tuple[tuple[SceneCamera, ...], tuple[SceneCamera, ...]]:
    """Partition cameras by immutable provider group without opening images."""
    holdout_groups = set(holdout_group_ids)
    fit = tuple(camera for camera in cameras if camera.group_id not in holdout_groups)
    holdout = tuple(camera for camera in cameras if camera.group_id in holdout_groups)
    return fit, holdout


def load_provider_rgb_image(path: Path) -> NDArray[np.uint8]:
    """Decode one provider image explicitly as uint8 RGB."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb


def summarize_fit_view_detections(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize acceptance, rejection, confidence, and group coverage."""
    accepted = [record for record in records if bool(record["accepted"])]
    reasons = Counter(
        str(reason)
        for record in records
        for reason in cast(Sequence[object], record["rejection_reasons"])
    )
    all_scores = np.asarray(
        [
            score
            for record in records
            for score in cast(Sequence[float], record["peak_scores"])
        ],
        dtype=np.float64,
    )
    accepted_by_group = Counter(int(record["group_id"]) for record in accepted)
    input_by_group = Counter(int(record["group_id"]) for record in records)
    return {
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(records) - len(accepted),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "rejection_reasons": dict(reasons.most_common()),
        "input_count_by_group": {
            str(key): value for key, value in sorted(input_by_group.items())
        },
        "accepted_count_by_group": {
            str(key): value for key, value in sorted(accepted_by_group.items())
        },
        "peak_score_quantiles": (
            {
                "q00": float(np.quantile(all_scores, 0.0)),
                "q25": float(np.quantile(all_scores, 0.25)),
                "q50": float(np.quantile(all_scores, 0.5)),
                "q75": float(np.quantile(all_scores, 0.75)),
                "q100": float(np.quantile(all_scores, 1.0)),
            }
            if all_scores.size
            else {}
        ),
    }


def compute_detection_artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash canonical artifact content while excluding its declared hash."""
    unhashed = {
        key: value for key, value in payload.items() if key != "artifact_fingerprint"
    }
    encoded = json.dumps(
        unhashed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_fit_view_court_detections(
    payload: Mapping[str, Any],
    *,
    bundle: LoadedSceneProviderBundle | None = None,
) -> None:
    """Validate schema, fingerprint, quarantine, and optional provider binding."""
    required = {
        "schema",
        "artifact_id",
        "artifact_fingerprint",
        "created_at_utc",
        "provider",
        "coordinate_contract",
        "split",
        "detector",
        "criteria",
        "records",
        "summary",
        "provenance",
    }
    if set(payload) != required:
        raise ValueError(
            "Detection artifact keys do not match v1 schema: "
            f"expected {sorted(required)}, got {sorted(payload)}."
        )
    if payload["schema"] != FIT_VIEW_COURT_DETECTIONS_SCHEMA:
        raise ValueError(
            f"Unsupported detection artifact schema: {payload['schema']!r}."
        )
    fingerprint = payload["artifact_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != _SHA256_LENGTH
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        raise ValueError("artifact_fingerprint must be a SHA-256 hex digest.")
    expected = compute_detection_artifact_fingerprint(payload)
    if fingerprint != expected:
        raise ValueError(
            "Detection artifact fingerprint mismatch: "
            f"declared {fingerprint}, computed {expected}."
        )

    split = _mapping(payload["split"], name="split")
    if split.get("holdout_inference_status") != "not_run":
        raise ValueError("Holdout inference status must remain 'not_run'.")
    fit_ids = _string_sequence(split.get("fit_camera_ids"), name="fit_camera_ids")
    holdout_ids = _string_sequence(
        split.get("holdout_camera_ids"),
        name="holdout_camera_ids",
    )
    if not fit_ids or not holdout_ids or set(fit_ids).intersection(holdout_ids):
        raise ValueError("Fit and holdout camera ids must be non-empty and disjoint.")
    records = _sequence(payload["records"], name="records")
    record_ids = [
        str(_mapping(record, name="record").get("camera_id")) for record in records
    ]
    if record_ids != fit_ids:
        raise ValueError("Detection records must exactly match ordered fit_camera_ids.")
    for record_value in records:
        record = _mapping(record_value, name="record")
        accepted = record.get("accepted")
        reasons = _sequence(record.get("rejection_reasons"), name="rejection_reasons")
        if not isinstance(accepted, bool) or accepted != (not reasons):
            raise ValueError("Record accepted flag must equal absence of rejections.")
        if len(_sequence(record.get("keypoints_xy"), name="keypoints_xy")) != 14:
            raise ValueError("Every detection record must contain 14 keypoints.")
        if len(_sequence(record.get("peak_scores"), name="peak_scores")) != 14:
            raise ValueError("Every detection record must contain 14 peak scores.")

    if bundle is not None:
        provider = _mapping(payload["provider"], name="provider")
        if provider.get("bundle_fingerprint") != bundle.manifest.bundle_fingerprint:
            raise ValueError("Detection artifact provider fingerprint mismatch.")
        provider_ids = [camera.camera_id for camera in bundle.manifest.cameras]
        if set(fit_ids).union(holdout_ids) != set(provider_ids):
            raise ValueError("Detection split does not cover the provider cameras.")


def publish_fit_view_court_detections(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Atomically publish one fingerprint-named JSON file without replacement."""
    validate_fit_view_court_detections(payload)
    artifact_id = str(payload["artifact_id"])
    fingerprint = str(payload["artifact_fingerprint"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{artifact_id}-{fingerprint[:16]}.json"
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{artifact_id}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite court detection artifact: {destination}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_fit_view_court_detections(
    path: Path,
    *,
    bundle: LoadedSceneProviderBundle | None = None,
) -> dict[str, Any]:
    """Load and strictly validate one fit-view detection artifact."""
    with path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    payload = dict(_mapping(raw, name="detection artifact"))
    validate_fit_view_court_detections(payload, bundle=bundle)
    return payload


def _validate_provider_image(
    image: NDArray[np.uint8],
    *,
    camera: SceneCamera,
) -> None:
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape != (camera.height, camera.width, 3)
    ):
        raise ValueError(
            f"Provider image for {camera.camera_id!r} must have shape "
            f"{(camera.height, camera.width, 3)} and dtype uint8, "
            f"got shape={image.shape}, dtype={image.dtype}."
        )


def _array_list(array: np.ndarray | None) -> list[Any] | None:
    return array.astype(float).tolist() if array is not None else None


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence.")
    return value


def _string_sequence(value: object, *, name: str) -> list[str]:
    sequence = _sequence(value, name=name)
    if not all(isinstance(item, str) and item for item in sequence):
        raise ValueError(f"{name} must contain non-empty strings.")
    result = cast(list[str], list(sequence))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates.")
    return result
