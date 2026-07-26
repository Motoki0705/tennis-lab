"""Camera-ray projection and proximity-weighted ground-line aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.synthetic_data_generation.provider.bundle import sha256_file
from src.synthetic_data_generation.scene_contract import SceneCamera

GROUND_LINE_MAP_SCHEMA = "ground_line_map_v1"
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GroundLineMapSettings:
    """Frozen probability, projection, weighting, and raster settings."""

    probability_threshold: float = 0.5
    proximity_scale: float = 0.35
    proximity_power: float = 2.0
    min_ray_plane_cosine: float = 0.05
    max_ray_distance: float = 3.0
    bounds_margin: float = 0.05
    grid_spacing: float = 0.0025
    min_projected_pixels: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1].")
        for name, value in (
            ("proximity_scale", self.proximity_scale),
            ("proximity_power", self.proximity_power),
            ("min_ray_plane_cosine", self.min_ray_plane_cosine),
            ("max_ray_distance", self.max_ray_distance),
            ("bounds_margin", self.bounds_margin),
            ("grid_spacing", self.grid_spacing),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.min_ray_plane_cosine >= 1.0:
            raise ValueError("min_ray_plane_cosine must be smaller than one.")
        if (
            isinstance(self.min_projected_pixels, bool)
            or self.min_projected_pixels <= 0
        ):
            raise ValueError("min_projected_pixels must be a positive integer.")


@dataclass(frozen=True)
class ProjectedLinePixels:
    """Valid line pixels intersected with the bounded ground plane."""

    points_scene: NDArray[np.float64]
    points_uv: NDArray[np.float64]
    probabilities: NDArray[np.float32]
    camera_ranges: NDArray[np.float64]
    proximity_weights: NDArray[np.float64]
    input_count: int
    invalid_parallel_count: int
    invalid_behind_count: int
    invalid_range_count: int
    invalid_bounds_count: int


class GroundLineAccumulator:
    """Accumulate at most one proximity-weighted contribution per view/cell."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        grid_spacing: float,
    ) -> None:
        u_min, u_max, v_min, v_max = bounds
        if u_min >= u_max or v_min >= v_max:
            raise ValueError("Ground-line raster bounds must have positive area.")
        if grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive.")
        self.bounds = tuple(float(value) for value in bounds)
        self.grid_spacing = float(grid_spacing)
        self.width = int(np.ceil((u_max - u_min) / grid_spacing)) + 1
        self.height = int(np.ceil((v_max - v_min) / grid_spacing)) + 1
        self.evidence_sum: NDArray[np.float32] = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        self.weight_sum: NDArray[np.float32] = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        self.view_count: NDArray[np.uint16] = np.zeros(
            (self.height, self.width),
            dtype=np.uint16,
        )

    def add_view(self, projection: ProjectedLinePixels) -> int:
        """Rasterize one view with a max reducer, then add it globally."""
        if len(projection.points_uv) == 0:
            return 0
        u_min, _, v_min, _ = self.bounds
        columns = np.rint(
            (projection.points_uv[:, 0] - u_min) / self.grid_spacing
        ).astype(np.int64)
        rows = np.rint((projection.points_uv[:, 1] - v_min) / self.grid_spacing).astype(
            np.int64
        )
        valid = (
            (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)
        )
        flat_indices = rows[valid] * self.width + columns[valid]
        if len(flat_indices) == 0:
            return 0
        evidence = (
            projection.probabilities[valid].astype(np.float64)
            * projection.proximity_weights[valid]
        )
        weights = projection.proximity_weights[valid]
        view_evidence: NDArray[np.float32] = np.zeros(
            self.height * self.width,
            dtype=np.float32,
        )
        view_weight: NDArray[np.float32] = np.zeros(
            self.height * self.width,
            dtype=np.float32,
        )
        np.maximum.at(view_evidence, flat_indices, evidence.astype(np.float32))
        np.maximum.at(view_weight, flat_indices, weights.astype(np.float32))
        view_mask = view_weight > 0.0
        self.evidence_sum.ravel()[view_mask] += view_evidence[view_mask]
        self.weight_sum.ravel()[view_mask] += view_weight[view_mask]
        self.view_count.ravel()[view_mask] += 1
        return int(view_mask.sum())

    def arrays(self) -> dict[str, NDArray[Any]]:
        """Return aggregate evidence, weight, support, and normalized score."""
        mean_probability = np.divide(
            self.evidence_sum,
            self.weight_sum,
            out=np.zeros_like(self.evidence_sum),
            where=self.weight_sum > 0.0,
        )
        return {
            "evidence_sum": self.evidence_sum,
            "weight_sum": self.weight_sum,
            "view_count": self.view_count,
            "mean_probability": mean_probability,
        }


def expanded_plane_bounds(
    plane: GroundPlaneEstimate,
    *,
    margin: float,
) -> tuple[float, float, float, float]:
    """Expand point-supported plane bounds by a fixed scene-unit margin."""
    if margin <= 0.0:
        raise ValueError("margin must be positive.")
    u_min, u_max, v_min, v_max = plane.support_uv_bounds
    return (
        u_min - margin,
        u_max + margin,
        v_min - margin,
        v_max + margin,
    )


def project_line_pixels_to_ground(
    camera: SceneCamera,
    pixels_xy: NDArray[np.floating[Any]],
    probabilities: NDArray[np.floating[Any]],
    *,
    plane: GroundPlaneEstimate,
    bounds: tuple[float, float, float, float],
    settings: GroundLineMapSettings,
) -> ProjectedLinePixels:
    """Back-project original-image pixels and apply proximity weighting."""
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float32)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape (N, 2), got {pixels.shape}.")
    if probability.shape != (len(pixels),):
        raise ValueError("probabilities must have shape (N,).")
    if not np.isfinite(pixels).all() or not np.isfinite(probability).all():
        raise ValueError("Projection inputs must contain only finite values.")
    if bool(np.any(probability < 0.0)) or bool(np.any(probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")

    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(camera.camera_to_scene, dtype=np.float64).reshape(4, 4)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    directions_camera = homogeneous @ np.linalg.inv(intrinsics).T
    directions_scene = directions_camera @ pose[:3, :3].T
    directions_scene /= np.linalg.norm(directions_scene, axis=1, keepdims=True)
    camera_center = pose[:3, 3]
    normal = np.asarray(plane.normal, dtype=np.float64)
    denominator = directions_scene @ normal
    numerator = -(float(camera_center @ normal) + plane.offset)
    parallel = np.abs(denominator) < settings.min_ray_plane_cosine
    distances = np.divide(
        numerator,
        denominator,
        out=np.full(len(pixels), np.nan, dtype=np.float64),
        where=~parallel,
    )
    behind = distances <= 0.0
    excessive_range = distances > settings.max_ray_distance
    finite_forward = ~(parallel | behind | excessive_range | ~np.isfinite(distances))
    selected_indices = np.flatnonzero(finite_forward)
    selected_ranges = distances[selected_indices]
    points_scene = (
        camera_center + selected_ranges[:, None] * directions_scene[selected_indices]
    )
    points_uv = plane.to_uv(points_scene)
    u_min, u_max, v_min, v_max = bounds
    outside = (
        (points_uv[:, 0] < u_min)
        | (points_uv[:, 0] > u_max)
        | (points_uv[:, 1] < v_min)
        | (points_uv[:, 1] > v_max)
    )
    in_bounds = ~outside
    selected_ranges = selected_ranges[in_bounds]
    weights = 1.0 / (
        1.0
        + np.power(
            selected_ranges / settings.proximity_scale,
            settings.proximity_power,
        )
    )
    return ProjectedLinePixels(
        points_scene=points_scene[in_bounds],
        points_uv=points_uv[in_bounds],
        probabilities=probability[selected_indices[in_bounds]],
        camera_ranges=selected_ranges,
        proximity_weights=weights,
        input_count=len(pixels),
        invalid_parallel_count=int(parallel.sum()),
        invalid_behind_count=int((~parallel & behind).sum()),
        invalid_range_count=int((~parallel & ~behind & excessive_range).sum()),
        invalid_bounds_count=int(outside.sum()),
    )


def publish_ground_line_map_artifact(
    payload: dict[str, Any],
    *,
    arrays: dict[str, NDArray[Any]],
    output_dir: Path,
) -> Path:
    """Atomically publish a fingerprinted manifest, NPZ, and preview directory."""
    _validate_payload_core(payload)
    required_arrays = {
        "evidence_sum",
        "weight_sum",
        "view_count",
        "mean_probability",
    }
    if set(arrays) != required_arrays:
        raise ValueError(
            f"Ground-line arrays must be exactly {sorted(required_arrays)}."
        )
    shapes = {np.asarray(value).shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("All ground-line arrays must share one raster shape.")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{payload['artifact_id']}.",
            suffix=".tmp",
            dir=output_dir,
        )
    )
    try:
        arrays_path = temporary_dir / "ground_line_map.npz"
        np.savez_compressed(arrays_path, **arrays)
        preview_path = temporary_dir / "aggregate_evidence.png"
        preview = render_ground_line_preview(
            np.asarray(arrays["evidence_sum"], dtype=np.float32),
            np.asarray(arrays["view_count"], dtype=np.uint16),
        )
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Failed to write ground-line preview: {preview_path}")
        manifest = dict(payload)
        manifest["files"] = {
            "arrays": _file_record(arrays_path),
            "preview": _file_record(preview_path),
        }
        manifest["artifact_fingerprint"] = _canonical_fingerprint(manifest)
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        destination = output_dir / (
            f"{payload['artifact_id']}-{manifest['artifact_fingerprint'][:16]}"
        )
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite ground-line artifact: {destination}"
            )
        os.rename(temporary_dir, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def load_ground_line_map_artifact(
    path: Path,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Strict-load and hash-verify a published ground-line map."""
    root = path.resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Ground-line manifest must be a JSON object.")
    manifest = dict(raw)
    _validate_payload_core(manifest)
    fingerprint = manifest.get("artifact_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("Invalid ground-line artifact fingerprint.")
    expected = _canonical_fingerprint(manifest)
    if fingerprint != expected:
        raise ValueError(
            "Ground-line artifact fingerprint mismatch: "
            f"declared {fingerprint}, computed {expected}."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"arrays", "preview"}:
        raise ValueError("Ground-line manifest files are invalid.")
    for record_value in files.values():
        if not isinstance(record_value, dict):
            raise ValueError("Ground-line file record must be an object.")
        relative_path = record_value.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).name != relative_path
        ):
            raise ValueError("Ground-line artifact paths must be plain file names.")
        file_path = root / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing ground-line artifact file: {file_path}")
        if file_path.stat().st_size != record_value.get("size_bytes"):
            raise ValueError(f"Ground-line artifact size mismatch: {file_path}")
        if sha256_file(file_path) != record_value.get("sha256"):
            raise ValueError(f"Ground-line artifact hash mismatch: {file_path}")
    arrays_record = files["arrays"]
    arrays_path = root / str(arrays_record["relative_path"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return manifest, arrays


def render_ground_line_preview(
    evidence_sum: NDArray[np.float32],
    view_count: NDArray[np.uint16],
) -> NDArray[np.uint8]:
    """Render log-scaled aggregate evidence with +plane-v pointing upward."""
    evidence = np.asarray(evidence_sum, dtype=np.float32)
    support = np.asarray(view_count, dtype=np.uint16)
    if evidence.shape != support.shape or evidence.ndim != 2:
        raise ValueError("Preview inputs must be same-shape 2D arrays.")
    positive = evidence[evidence > 0.0]
    scale = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
    normalized = np.clip(
        np.log1p(evidence) / np.log1p(max(scale, 1.0e-6)),
        0.0,
        1.0,
    )
    intensity = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    colored[support == 0] = 0
    return np.flipud(colored)


def _validate_payload_core(payload: dict[str, Any]) -> None:
    if payload.get("schema") != GROUND_LINE_MAP_SCHEMA:
        raise ValueError(f"Unsupported ground-line schema: {payload.get('schema')!r}.")
    artifact_id = payload.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or _ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None
    ):
        raise ValueError("Ground-line artifact_id must be path-safe.")
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "provider",
        "split",
        "detector",
        "ground_plane",
        "projection",
        "records",
        "summary",
        "provenance",
    }
    optional = {"files", "artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Ground-line manifest keys do not match v1 schema.")
    split = payload.get("split")
    if (
        not isinstance(split, dict)
        or split.get("holdout_inference_status") != "not_run"
    ):
        raise ValueError("Ground-line holdout inference status must remain 'not_run'.")
    fit_ids = split.get("fit_camera_ids")
    holdout_ids = split.get("holdout_camera_ids")
    if (
        not isinstance(fit_ids, list)
        or not isinstance(holdout_ids, list)
        or not fit_ids
        or not holdout_ids
        or set(fit_ids).intersection(holdout_ids)
    ):
        raise ValueError("Ground-line fit/holdout camera ids must be disjoint.")
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or [record.get("camera_id") for record in records if isinstance(record, dict)]
        != fit_ids
    ):
        raise ValueError("Ground-line records must exactly match fit_camera_ids.")


def _canonical_fingerprint(manifest: dict[str, Any]) -> str:
    unhashed = {
        key: value for key, value in manifest.items() if key != "artifact_fingerprint"
    }
    encoded = json.dumps(
        unhashed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
