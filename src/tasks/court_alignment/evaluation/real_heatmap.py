"""Evaluate KP14 court alignment on measured ground-UV line heatmaps.

The numeric ``mean_probability`` array in an ``alignment_line_heatmaps_v2``
archive is the evidence authority.  PNG files are deliberately not accepted as
model inputs.  This module keeps archive validation, coordinate transforms,
preprocessing, accepted-alignment projection, checkpoint loading, inference,
metrics, and artifact persistence as separately testable boundaries.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.base.training.repro import resolve_queue_repro_dir
from src.tasks.court_alignment.geometry.court import canonical_court_keypoints
from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    SimilarityFit2D,
    fit_similarity_2d,
)
from src.tasks.court_alignment.inference.predictor import CourtAlignmentPredictor
from src.tasks.court_alignment.models.checkpoint import (
    load_court_alignment_model_checkpoint,
)
from src.tasks.court_alignment.models.cnn import (
    validate_court_alignment_input,
    validate_court_alignment_output,
)
from src.tasks.court_alignment.training.metrics import compute_alignment_metrics
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX

LINE_HEATMAP_ARCHIVE_SCHEMA = "alignment_line_heatmaps_v2"
LINE_HEATMAP_MANIFEST_SCHEMA = "alignment_line_heatmap_manifest_v2"
ALIGNMENT_SCHEMA = "semantic_multi_court_alignment_v2"
GROUND_PLANE_CONVENTION = (
    "right_handed_metric_scene_ground_plane_uv;units=metres;"
    "u=dot(metric_point-origin,basis_u);"
    "v=dot(metric_point-origin,basis_v);normal=cross(basis_u,basis_v)"
)
REFERENCE_TYPE = "accepted_alignment"
REFERENCE_LIMITATION = (
    "System-relative comparison against the accepted alignment; this is not "
    "independent ground truth."
)
NUM_KEYPOINTS = 14
POSE_DIAGNOSTIC_MINIMUM_KEYPOINTS = 4

_ARCHIVE_KEYS = frozenset(
    {
        "schema",
        "coordinate_convention",
        "coordinate_units",
        "camera_ids",
        "included_in_aggregate",
        "bounds_uv",
        "grid_spacing",
        "proximity_scale",
        "proximity_power",
        "probability_shapes",
        "probability_offsets",
        "probability_values",
        "projected_offsets",
        "projected_points_uv",
        "projected_probabilities",
        "proximity_weights",
        "evidence_sum",
        "weight_sum",
        "view_count",
        "mean_probability",
    }
)

FloatArray: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
PreprocessFunction = Callable[[FloatArray, tuple[int, int]], FloatArray]


def _ordinary_file(path: Path, *, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} must be an ordinary file: {path}")


def _json_object(path: Path, *, name: str) -> dict[str, object]:
    _ordinary_file(path, name=name)
    try:
        value: object = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {name} {path}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must contain a JSON object: {path}")
    return dict(value)


def _scalar_text(value: NDArray[Any], *, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"Archive field {name} must be a Unicode scalar.")
    return str(value.item())


def _finite_vector(
    value: object,
    *,
    size: int,
    name: str,
) -> Float64Array:
    array = np.asarray(value)
    if (
        array.dtype.kind not in {"f", "i", "u"}
        or array.shape != (size,)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite numeric vector of length {size}.")
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RealHeatmapArchive:
    """Validated numeric aggregate and its ground-UV raster metadata."""

    mean_probability: FloatArray
    bounds_uv: tuple[float, float, float, float]
    grid_spacing_m: float
    coordinate_convention: str
    aggregate_view_count: int
    archive_path: Path
    manifest_path: Path

    def __post_init__(self) -> None:
        raster = np.asarray(self.mean_probability)
        if raster.dtype != np.float32 or raster.ndim != 2 or min(raster.shape) < 2:
            raise ValueError("mean_probability must be a non-trivial float32 raster.")
        if not np.isfinite(raster).all() or np.any((raster < 0.0) | (raster > 1.0)):
            raise ValueError("mean_probability must be finite and lie in [0,1].")
        bounds = _finite_vector(self.bounds_uv, size=4, name="bounds_uv")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("bounds_uv must have positive U and V extents.")
        spacing = float(self.grid_spacing_m)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("grid_spacing_m must be finite and positive.")
        expected_width = int(math.ceil((bounds[1] - bounds[0]) / spacing)) + 1
        expected_height = int(math.ceil((bounds[3] - bounds[2]) / spacing)) + 1
        if raster.shape != (expected_height, expected_width):
            raise ValueError(
                "mean_probability shape disagrees with bounds_uv/grid_spacing: "
                f"{raster.shape} != {(expected_height, expected_width)}."
            )
        if self.coordinate_convention != GROUND_PLANE_CONVENTION:
            raise ValueError("Unsupported ground-UV coordinate convention.")
        if type(self.aggregate_view_count) is not int or self.aggregate_view_count <= 0:
            raise ValueError("aggregate_view_count must be a positive integer.")
        owned = np.array(raster, dtype=np.float32, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "mean_probability", owned)
        object.__setattr__(self, "bounds_uv", tuple(float(item) for item in bounds))
        object.__setattr__(self, "grid_spacing_m", spacing)

    @property
    def raster_shape(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.mean_probability.shape)

    @classmethod
    def load(cls, archive_path: Path, manifest_path: Path) -> RealHeatmapArchive:
        """Strict-load the NPZ authority and cross-check its JSON manifest."""
        _ordinary_file(archive_path, name="Line-heatmap archive")
        _ordinary_file(manifest_path, name="Line-heatmap manifest")
        with np.load(archive_path, allow_pickle=False) as loaded:
            if set(loaded.files) != _ARCHIVE_KEYS:
                missing = sorted(_ARCHIVE_KEYS - set(loaded.files))
                extra = sorted(set(loaded.files) - _ARCHIVE_KEYS)
                raise ValueError(
                    "Line-heatmap archive keys do not match the v2 schema; "
                    f"missing={missing}, extra={extra}."
                )
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        if _scalar_text(arrays["schema"], name="schema") != LINE_HEATMAP_ARCHIVE_SCHEMA:
            raise ValueError("Unsupported line-heatmap archive schema.")
        convention = _scalar_text(
            arrays["coordinate_convention"], name="coordinate_convention"
        )
        if convention != GROUND_PLANE_CONVENTION:
            raise ValueError("Unsupported archive coordinate convention.")
        if (
            _scalar_text(arrays["coordinate_units"], name="coordinate_units")
            != "metres"
        ):
            raise ValueError("Line-heatmap archive coordinate units must be metres.")
        bounds_array = arrays["bounds_uv"]
        if bounds_array.dtype != np.float64:
            raise TypeError("Archive bounds_uv must have float64 dtype.")
        bounds = _finite_vector(bounds_array, size=4, name="archive bounds_uv")
        spacing_array = arrays["grid_spacing"]
        if spacing_array.ndim != 0 or spacing_array.dtype != np.float64:
            raise TypeError("Archive grid_spacing must be a float64 scalar.")
        spacing = float(spacing_array.item())
        mean_probability = arrays["mean_probability"]
        if mean_probability.dtype != np.float32:
            raise TypeError("Archive mean_probability must have float32 dtype.")
        included = arrays["included_in_aggregate"]
        camera_ids = arrays["camera_ids"]
        if (
            included.dtype != np.bool_
            or included.ndim != 1
            or camera_ids.ndim != 1
            or camera_ids.dtype.kind != "U"
            or len(included) != len(camera_ids)
        ):
            raise ValueError("Archive camera aggregate inventory is malformed.")

        manifest = _json_object(manifest_path, name="Line-heatmap manifest")
        expected_manifest_values: tuple[tuple[str, object], ...] = (
            ("schema", LINE_HEATMAP_MANIFEST_SCHEMA),
            ("coordinate_convention", convention),
            ("coordinate_units", "metres"),
            ("archive", archive_path.name),
            ("ground_png_orientation", "top row is maximum v (vertical flip)"),
        )
        for key, expected in expected_manifest_values:
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Manifest field {key!r} disagrees with the numeric archive: "
                    f"{manifest.get(key)!r} != {expected!r}."
                )
        manifest_bounds = _finite_vector(
            manifest.get("bounds_uv"), size=4, name="manifest bounds_uv"
        )
        if not np.array_equal(manifest_bounds, bounds):
            raise ValueError("Manifest bounds_uv disagrees with the numeric archive.")
        manifest_spacing = manifest.get("grid_spacing")
        if type(manifest_spacing) not in {float, int}:
            raise ValueError("Manifest grid_spacing must be numeric.")
        if float(cast(float | int, manifest_spacing)) != spacing:
            raise ValueError(
                "Manifest grid_spacing disagrees with the numeric archive."
            )
        raster_shape = manifest.get("raster_shape")
        if not isinstance(raster_shape, list) or raster_shape != list(
            mean_probability.shape
        ):
            raise ValueError("Manifest raster_shape disagrees with mean_probability.")
        aggregate_count = int(np.count_nonzero(included))
        if manifest.get("aggregate_view_count") != aggregate_count:
            raise ValueError(
                "Manifest aggregate_view_count disagrees with the archive inventory."
            )
        return cls(
            mean_probability=cast(FloatArray, mean_probability),
            bounds_uv=cast(
                tuple[float, float, float, float], tuple(float(item) for item in bounds)
            ),
            grid_spacing_m=spacing,
            coordinate_convention=convention,
            aggregate_view_count=aggregate_count,
            archive_path=archive_path,
            manifest_path=manifest_path,
        )


@dataclass(frozen=True, slots=True)
class PreprocessOptions:
    """Explicit measured-raster preprocessing configuration."""

    method: str
    output_size: tuple[int, int]
    padding_value: float = 0.0
    content_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in REAL_HEATMAP_PREPROCESSORS:
            raise ValueError(
                f"Unknown real-heatmap preprocess method {self.method!r}; "
                f"available={sorted(REAL_HEATMAP_PREPROCESSORS)}."
            )
        if len(self.output_size) != 2 or any(
            type(item) is not int or item <= 0 for item in self.output_size
        ):
            raise ValueError("output_size must contain two positive integers.")
        padding = float(self.padding_value)
        if not math.isfinite(padding) or not 0.0 <= padding <= 1.0:
            raise ValueError("padding_value must be finite and in [0,1].")
        fraction = float(self.content_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("content_fraction must be finite and in (0,1].")
        object.__setattr__(self, "padding_value", padding)
        object.__setattr__(self, "content_fraction", fraction)


@dataclass(frozen=True, slots=True)
class PixelUVTransform:
    """Invertible UV/native-raster/letterboxed-model coordinate transform."""

    bounds_uv: tuple[float, float, float, float]
    grid_spacing_m: float
    source_shape: tuple[int, int]
    content_shape: tuple[int, int]
    output_shape: tuple[int, int]
    padding_xy: tuple[int, int]
    vertical_flip: bool = True

    def __post_init__(self) -> None:
        bounds = _finite_vector(self.bounds_uv, size=4, name="transform bounds_uv")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("transform bounds_uv must have positive area.")
        if not math.isfinite(self.grid_spacing_m) or self.grid_spacing_m <= 0.0:
            raise ValueError("transform grid_spacing_m must be finite and positive.")
        for name, shape in (
            ("source_shape", self.source_shape),
            ("content_shape", self.content_shape),
            ("output_shape", self.output_shape),
        ):
            if len(shape) != 2 or any(
                type(item) is not int or item <= 0 for item in shape
            ):
                raise ValueError(f"{name} must contain two positive integers.")
        pad_x, pad_y = self.padding_xy
        if any(type(item) is not int or item < 0 for item in (pad_x, pad_y)):
            raise ValueError("padding_xy must contain non-negative integers.")
        content_h, content_w = self.content_shape
        output_h, output_w = self.output_shape
        if pad_x + content_w > output_w or pad_y + content_h > output_h:
            raise ValueError("Letterbox content exceeds the output canvas.")
        if self.vertical_flip is not True:
            raise ValueError(
                "The alignment archive requires the explicit vertical flip."
            )

    @property
    def resize_scale_xy(self) -> tuple[float, float]:
        source_h, source_w = self.source_shape
        content_h, content_w = self.content_shape
        return content_w / source_w, content_h / source_h

    @property
    def pixels_per_metre(self) -> float:
        scale_x, scale_y = self.resize_scale_xy
        return math.sqrt(scale_x * scale_y) / self.grid_spacing_m

    @property
    def metres_per_pixel_xy(self) -> tuple[float, float]:
        scale_x, scale_y = self.resize_scale_xy
        return self.grid_spacing_m / scale_x, self.grid_spacing_m / scale_y

    def pixels_to_metres(self, distance_px: float) -> float:
        distance = float(distance_px)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("distance_px must be finite and non-negative.")
        return distance / self.pixels_per_metre

    @staticmethod
    def _points(
        value: NDArray[Any] | Sequence[Sequence[float]], *, name: str
    ) -> Float64Array:
        points = np.asarray(value)
        if (
            points.dtype.kind not in {"f", "i", "u"}
            or points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"{name} must be a finite numeric (N,2) array.")
        return np.asarray(points, dtype=np.float64)

    def uv_to_model_px(
        self, value: NDArray[Any] | Sequence[Sequence[float]]
    ) -> Float64Array:
        """Map metric UV points to model ``(x,y-down)`` pixel coordinates."""
        points = self._points(value, name="UV points")
        u_min, _, v_min, _ = self.bounds_uv
        source_h, _source_w = self.source_shape
        source_x = (points[:, 0] - u_min) / self.grid_spacing_m
        source_y = (source_h - 1.0) - ((points[:, 1] - v_min) / self.grid_spacing_m)
        scale_x, scale_y = self.resize_scale_xy
        pad_x, pad_y = self.padding_xy
        model_x = (source_x + 0.5) * scale_x - 0.5 + pad_x
        model_y = (source_y + 0.5) * scale_y - 0.5 + pad_y
        return np.column_stack((model_x, model_y)).astype(np.float64, copy=False)

    def model_px_to_uv(
        self, value: NDArray[Any] | Sequence[Sequence[float]]
    ) -> Float64Array:
        """Invert :meth:`uv_to_model_px` without rounding to raster cells."""
        points = self._points(value, name="model pixel points")
        scale_x, scale_y = self.resize_scale_xy
        pad_x, pad_y = self.padding_xy
        source_x = (points[:, 0] - pad_x + 0.5) / scale_x - 0.5
        source_y = (points[:, 1] - pad_y + 0.5) / scale_y - 0.5
        u_min, _, v_min, _ = self.bounds_uv
        source_h, _source_w = self.source_shape
        u = u_min + source_x * self.grid_spacing_m
        v = v_min + (source_h - 1.0 - source_y) * self.grid_spacing_m
        return np.column_stack((u, v)).astype(np.float64, copy=False)

    def to_metadata(self) -> dict[str, object]:
        scale_x, scale_y = self.resize_scale_xy
        metres_x, metres_y = self.metres_per_pixel_xy
        return {
            "bounds_uv_m": list(self.bounds_uv),
            "grid_spacing_m": self.grid_spacing_m,
            "source_shape_hw": list(self.source_shape),
            "content_shape_hw": list(self.content_shape),
            "output_shape_hw": list(self.output_shape),
            "padding_xy": list(self.padding_xy),
            "resize_scale_xy": [scale_x, scale_y],
            "metres_per_pixel_xy": [metres_x, metres_y],
            "pixels_per_metre": self.pixels_per_metre,
            "vertical_flip": self.vertical_flip,
            "pixel_convention": "x_right_y_down; interpolate_half_pixel_centres",
        }


def _content_shape(
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    content_fraction: float,
) -> tuple[int, int]:
    source_h, source_w = source_shape
    output_h, output_w = output_shape
    scale = min(output_h / source_h, output_w / source_w) * content_fraction
    return (
        min(output_h, max(1, int(round(source_h * scale)))),
        min(output_w, max(1, int(round(source_w * scale)))),
    )


def _resize_max(source: FloatArray, size: tuple[int, int]) -> FloatArray:
    """Reduce source pixels into target cells with a peak-preserving maximum."""
    target_h, target_w = size
    source_h, source_w = source.shape
    if target_h > source_h or target_w > source_w:
        raise ValueError(
            "max preprocessing is a reducer and requires target cells no larger "
            "than the source raster."
        )
    row_indices = np.floor(
        (np.arange(source_h, dtype=np.float64) + 0.5) * target_h / source_h
    ).astype(np.int64)
    column_indices = np.floor(
        (np.arange(source_w, dtype=np.float64) + 0.5) * target_w / source_w
    ).astype(np.int64)
    np.clip(row_indices, 0, target_h - 1, out=row_indices)
    np.clip(column_indices, 0, target_w - 1, out=column_indices)
    flat_targets = (row_indices[:, None] * target_w + column_indices[None, :]).reshape(
        -1
    )
    output: FloatArray = np.zeros((target_h, target_w), dtype=np.float32)
    np.maximum.at(output.reshape(-1), flat_targets, source.reshape(-1))
    return output


def _torch_resize(mode: str) -> PreprocessFunction:
    def resize(source: FloatArray, size: tuple[int, int]) -> FloatArray:
        value = torch.from_numpy(np.ascontiguousarray(source)).view(1, 1, *source.shape)
        if mode == "bilinear":
            resized = F.interpolate(value, size=size, mode=mode, align_corners=False)
        else:
            resized = F.interpolate(value, size=size, mode=mode)
        return cast(
            FloatArray, resized[0, 0].cpu().numpy().astype(np.float32, copy=False)
        )

    return resize


REAL_HEATMAP_PREPROCESSORS: Mapping[str, PreprocessFunction] = {
    "max": _resize_max,
    "bilinear": _torch_resize("bilinear"),
    "area": _torch_resize("area"),
    "nearest": _torch_resize("nearest"),
}


def letterbox_heatmap(
    source: FloatArray,
    *,
    bounds_uv: tuple[float, float, float, float],
    grid_spacing_m: float,
    options: PreprocessOptions,
) -> tuple[FloatArray, PixelUVTransform]:
    """Vertically flip, resize, and letterbox one authoritative UV raster."""
    value = np.asarray(source)
    if value.dtype != np.float32 or value.ndim != 2:
        raise TypeError("source heatmap must be a float32 2-D array.")
    if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("source heatmap must be finite and lie in [0,1].")
    source_shape = cast(tuple[int, int], value.shape)
    content_shape = _content_shape(
        source_shape,
        options.output_size,
        content_fraction=options.content_fraction,
    )
    # Numeric archive rows increase with V.  Model pixels increase downward,
    # so this flip is part of the coordinate contract, not an augmentation.
    image_oriented = np.flip(value, axis=0).copy(order="C")
    resized = REAL_HEATMAP_PREPROCESSORS[options.method](image_oriented, content_shape)
    output_h, output_w = options.output_size
    content_h, content_w = content_shape
    pad_x = (output_w - content_w) // 2
    pad_y = (output_h - content_h) // 2
    canvas: FloatArray = np.full(
        (output_h, output_w), options.padding_value, dtype=np.float32
    )
    canvas[pad_y : pad_y + content_h, pad_x : pad_x + content_w] = resized
    transform = PixelUVTransform(
        bounds_uv=bounds_uv,
        grid_spacing_m=grid_spacing_m,
        source_shape=source_shape,
        content_shape=content_shape,
        output_shape=options.output_size,
        padding_xy=(pad_x, pad_y),
    )
    return np.ascontiguousarray(canvas, dtype=np.float32), transform


@dataclass(frozen=True, slots=True)
class AcceptedCourtAlignment:
    court_instance_id: str
    candidate_id: str
    scene_from_court: Float64Array

    def __post_init__(self) -> None:
        if not self.court_instance_id or not self.candidate_id:
            raise ValueError("Accepted alignment IDs must be non-empty.")
        matrix = np.asarray(self.scene_from_court)
        if matrix.dtype != np.float64 or matrix.shape != (4, 4):
            raise TypeError("scene_from_court must be a float64 4x4 matrix.")
        if not np.isfinite(matrix).all():
            raise ValueError("scene_from_court must be finite.")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
            raise ValueError("scene_from_court must be affine homogeneous.")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
            raise ValueError("scene_from_court rotation must be orthonormal.")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
            raise ValueError("scene_from_court rotation must be right-handed.")
        owned = np.array(matrix, dtype=np.float64, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "scene_from_court", owned)


def load_accepted_alignments(path: Path) -> tuple[AcceptedCourtAlignment, ...]:
    """Load all and only explicitly accepted alignment candidates."""
    payload = _json_object(path, name="Alignment result")
    if payload.get("schema") != ALIGNMENT_SCHEMA:
        raise ValueError(f"Unsupported alignment schema: {payload.get('schema')!r}.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Alignment candidates must be a non-empty list.")
    accepted: list[AcceptedCourtAlignment] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise TypeError(f"Alignment candidate {index} must be an object.")
        if type(item.get("accepted")) is not bool:
            raise TypeError(f"Alignment candidate {index}.accepted must be boolean.")
        if not item["accepted"]:
            continue
        court_id = item.get("court_instance_id")
        candidate_id = item.get("candidate_id")
        if not isinstance(court_id, str) or not isinstance(candidate_id, str):
            raise TypeError("Accepted alignment IDs must be strings.")
        flat = _finite_vector(
            item.get("scene_from_court"),
            size=16,
            name=f"candidate[{index}].scene_from_court",
        )
        accepted.append(
            AcceptedCourtAlignment(
                court_instance_id=court_id,
                candidate_id=candidate_id,
                scene_from_court=flat.reshape(4, 4),
            )
        )
    if not accepted:
        raise ValueError("Alignment contains no explicitly accepted candidate.")
    court_ids = [item.court_instance_id for item in accepted]
    candidate_ids = [item.candidate_id for item in accepted]
    if len(court_ids) != len(set(court_ids)) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError("Accepted alignment IDs must be unique.")
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class AlignmentGroundPlane:
    """Ground-plane frame reconstructed from accepted rigid court poses."""

    origin_scene_m: Float64Array
    basis_u_scene: Float64Array
    basis_v_scene: Float64Array
    normal_scene: Float64Array

    def __post_init__(self) -> None:
        values = tuple(
            _finite_vector(value, size=3, name=name)
            for value, name in (
                (self.origin_scene_m, "origin_scene_m"),
                (self.basis_u_scene, "basis_u_scene"),
                (self.basis_v_scene, "basis_v_scene"),
                (self.normal_scene, "normal_scene"),
            )
        )
        basis = np.column_stack(values[1:])
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-8, rtol=0.0):
            raise ValueError("Ground-plane basis must be orthonormal.")
        if not np.allclose(np.cross(values[1], values[2]), values[3], atol=1.0e-8):
            raise ValueError("Ground-plane basis must be right-handed.")
        for field_name, value in zip(
            ("origin_scene_m", "basis_u_scene", "basis_v_scene", "normal_scene"),
            values,
            strict=True,
        ):
            owned = np.array(value, dtype=np.float64, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, field_name, owned)

    @classmethod
    def from_alignments(
        cls, alignments: Sequence[AcceptedCourtAlignment]
    ) -> AlignmentGroundPlane:
        if not alignments:
            raise ValueError("At least one accepted alignment is required.")
        first = alignments[0].scene_from_court
        normal = first[:3, 2]
        offset = float(normal @ first[:3, 3])
        for item in alignments[1:]:
            candidate_normal = item.scene_from_court[:3, 2]
            candidate_offset = float(candidate_normal @ item.scene_from_court[:3, 3])
            if not np.allclose(candidate_normal, normal, atol=1.0e-8, rtol=0.0):
                raise ValueError(
                    "Accepted courts do not share one ground-plane normal."
                )
            if not math.isclose(candidate_offset, offset, abs_tol=1.0e-8):
                raise ValueError(
                    "Accepted courts do not share one ground-plane offset."
                )
        origin = normal * offset
        scene_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        basis_u = scene_x - normal * float(scene_x @ normal)
        norm = float(np.linalg.norm(basis_u))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError(
                "Ground plane is degenerate against the metric scene X axis."
            )
        basis_u /= norm
        basis_v = np.asarray(np.cross(normal, basis_u), dtype=np.float64)
        return cls(
            origin_scene_m=origin,
            basis_u_scene=basis_u,
            basis_v_scene=basis_v,
            normal_scene=normal,
        )

    def scene_to_uv(self, points_scene_m: NDArray[Any]) -> Float64Array:
        points = np.asarray(points_scene_m)
        if (
            points.dtype.kind not in {"f", "i", "u"}
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("Scene points must be a finite numeric (N,3) array.")
        basis = np.column_stack((self.basis_u_scene, self.basis_v_scene))
        return np.asarray(
            (np.asarray(points, dtype=np.float64) - self.origin_scene_m) @ basis,
            dtype=np.float64,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "origin_scene_m": self.origin_scene_m.tolist(),
            "basis_u_scene": self.basis_u_scene.tolist(),
            "basis_v_scene": self.basis_v_scene.tolist(),
            "normal_scene": self.normal_scene.tolist(),
            "coordinate_convention": GROUND_PLANE_CONVENTION,
        }


@dataclass(frozen=True, slots=True)
class ProjectedReference:
    keypoints_px: FloatArray
    valid: NDArray[np.bool_]
    centers_px: FloatArray
    court_instance_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    poses: tuple[dict[str, object], ...]


def _similarity_payload(fit: SimilarityFit2D) -> dict[str, object]:
    return {
        "translation_px": [float(item) for item in fit.translation_px.cpu()],
        "rotation_deg": math.degrees(float(fit.rotation_rad)),
        "scale_px_per_metre": float(fit.scale_px_per_metre),
        "residual_px": float(fit.residual_px),
    }


def project_accepted_reference(
    alignments: Sequence[AcceptedCourtAlignment],
    *,
    ground_plane: AlignmentGroundPlane,
    transform: PixelUVTransform,
) -> ProjectedReference:
    """Project accepted court poses into the model's image-view KP14 convention.

    Archive rows are flipped into ``y-down`` model pixels.  To retain the
    proper Sim(2) KP convention learned from synthetic images, model-canonical
    ``+y`` maps to physical court ``-y``.  This changes semantic ordering only;
    the unlabelled physical court-line set is unchanged.
    """
    canonical = canonical_court_keypoints(dtype=torch.float64).cpu().numpy()
    physical = canonical.copy()
    physical[:, 1] *= -1.0
    homogeneous = np.column_stack(
        (physical, np.zeros(NUM_KEYPOINTS), np.ones(NUM_KEYPOINTS))
    )
    all_keypoints: list[FloatArray] = []
    all_centers: list[FloatArray] = []
    all_valid: list[NDArray[np.bool_]] = []
    all_poses: list[dict[str, object]] = []
    output_h, output_w = transform.output_shape
    for item in alignments:
        scene = (item.scene_from_court @ homogeneous.T).T[:, :3]
        distances = (scene - ground_plane.origin_scene_m) @ ground_plane.normal_scene
        if not np.allclose(distances, 0.0, atol=1.0e-6, rtol=0.0):
            raise ValueError("Accepted court keypoints do not lie on the ground plane.")
        uv = ground_plane.scene_to_uv(scene)
        keypoints = transform.uv_to_model_px(uv).astype(np.float32)
        center_scene = item.scene_from_court[:3, 3][None]
        center_uv = ground_plane.scene_to_uv(center_scene)
        center_px = transform.uv_to_model_px(center_uv).astype(np.float32)
        valid = (
            (keypoints[:, 0] >= 0.0)
            & (keypoints[:, 0] <= output_w - 1.0)
            & (keypoints[:, 1] >= 0.0)
            & (keypoints[:, 1] <= output_h - 1.0)
        )
        if int(np.count_nonzero(valid)) < 2:
            raise ValueError(
                f"Accepted court {item.court_instance_id!r} has fewer than two "
                "visible model-canvas keypoints."
            )
        fit = fit_similarity_2d(
            torch.from_numpy(canonical[valid]).to(torch.float32),
            torch.from_numpy(keypoints[valid]),
        )
        pose = {
            "court_instance_id": item.court_instance_id,
            "candidate_id": item.candidate_id,
            "center_uv_m": center_uv[0].tolist(),
            "center_px": center_px[0].tolist(),
            "visible_keypoint_count": int(np.count_nonzero(valid)),
            **_similarity_payload(fit),
        }
        all_keypoints.append(keypoints)
        all_centers.append(center_px[0])
        all_valid.append(valid)
        all_poses.append(pose)
    return ProjectedReference(
        keypoints_px=np.stack(all_keypoints).astype(np.float32, copy=False),
        valid=np.stack(all_valid).astype(np.bool_, copy=False),
        centers_px=np.stack(all_centers).astype(np.float32, copy=False),
        court_instance_ids=tuple(item.court_instance_id for item in alignments),
        candidate_ids=tuple(item.candidate_id for item in alignments),
        poses=tuple(all_poses),
    )


@dataclass(frozen=True, slots=True)
class DecoderOptions:
    threshold: float
    nms_kernel: int
    max_peaks: int
    subpixel_refine: bool
    cluster_distance_px: float
    max_instances: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("decoder.threshold must be finite and in [0,1].")
        if (
            type(self.nms_kernel) is not int
            or self.nms_kernel <= 0
            or self.nms_kernel % 2 == 0
        ):
            raise ValueError("decoder.nms_kernel must be a positive odd integer.")
        if type(self.max_peaks) is not int or self.max_peaks <= 0:
            raise ValueError("decoder.max_peaks must be a positive integer.")
        if type(self.subpixel_refine) is not bool:
            raise TypeError("decoder.subpixel_refine must be boolean.")
        if (
            not math.isfinite(self.cluster_distance_px)
            or self.cluster_distance_px <= 0.0
        ):
            raise ValueError("decoder.cluster_distance_px must be finite and positive.")
        if type(self.max_instances) is not int or self.max_instances <= 0:
            raise ValueError("decoder.max_instances must be a positive integer.")


@dataclass(frozen=True, slots=True)
class MetricOptions:
    match_max_error_px: float
    minimum_common_keypoints: int
    minimum_visible_keypoints: int
    minimum_visible_fraction: float
    minimum_sim2_keypoints: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.match_max_error_px) or self.match_max_error_px <= 0.0:
            raise ValueError("metrics.match_max_error_px must be finite and positive.")
        for name, value, minimum in (
            ("minimum_common_keypoints", self.minimum_common_keypoints, 1),
            ("minimum_visible_keypoints", self.minimum_visible_keypoints, 1),
            ("minimum_sim2_keypoints", self.minimum_sim2_keypoints, 2),
        ):
            if type(value) is not int or not minimum <= value <= NUM_KEYPOINTS:
                raise ValueError(
                    f"metrics.{name} must be an integer in [{minimum},{NUM_KEYPOINTS}]."
                )
        if (
            not math.isfinite(self.minimum_visible_fraction)
            or not 0.0 < self.minimum_visible_fraction <= 1.0
        ):
            raise ValueError("metrics.minimum_visible_fraction must be in (0,1].")


@dataclass(frozen=True, slots=True)
class RealHeatmapEvaluationRequest:
    archive_path: Path
    manifest_path: Path
    alignment_path: Path
    checkpoint_path: Path
    output_dir: Path
    device: str
    preprocess: PreprocessOptions
    decoder: DecoderOptions
    metrics: MetricOptions
    training_scale_range_px_per_metre: tuple[float, float]

    def __post_init__(self) -> None:
        for path_name in (
            "archive_path",
            "manifest_path",
            "alignment_path",
            "checkpoint_path",
            "output_dir",
        ):
            path = getattr(self, path_name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{path_name} must be an absolute Path.")
        if self.preprocess.output_size != (256, 256):
            raise ValueError("Measured inference requires output_size=(256,256).")
        try:
            torch.device(self.device)
        except RuntimeError as error:
            raise ValueError(f"Invalid evaluation device {self.device!r}.") from error
        low, high = self.training_scale_range_px_per_metre
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low <= 0.0
            or high <= 0.0
            or low > high
        ):
            raise ValueError(
                "training_scale_range_px_per_metre must be finite, positive, and ordered."
            )


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, object]:
    """Strict-load a Lightning ``model.`` state dict into a bare CNN."""
    return load_court_alignment_model_checkpoint(model, checkpoint_path)


def _prediction_poses(
    sample: CourtInstanceBatch, transform: PixelUVTransform
) -> list[dict[str, object]]:
    canonical = canonical_court_keypoints(
        dtype=sample.keypoints_px.dtype, device=sample.keypoints_px.device
    )
    result: list[dict[str, object]] = []
    for index in range(sample.num_instances):
        valid = sample.valid[index]
        payload: dict[str, object] = {
            "prediction_index": index,
            "center_px": [float(item) for item in sample.centers_px[index].cpu()],
            "center_uv_m": transform.model_px_to_uv(
                sample.centers_px[index].detach().cpu().numpy()[None]
            )[0].tolist(),
            "semantic_count": int(valid.sum()),
            "aggregate_score": float(sample.scores[index, valid].sum()),
        }
        if int(valid.sum()) >= 2:
            try:
                fit = fit_similarity_2d(
                    canonical[valid], sample.keypoints_px[index, valid]
                )
            except ValueError as error:
                payload["similarity_fit"] = {
                    "status": "unavailable",
                    "reason": str(error),
                }
            else:
                payload["similarity_fit"] = {
                    "status": "available",
                    **_similarity_payload(fit),
                }
        else:
            payload["similarity_fit"] = {
                "status": "unavailable",
                "reason": "fewer than two decoded semantic keypoints",
            }
        result.append(payload)
    return result


@dataclass(frozen=True, slots=True)
class _PoseFitCandidate:
    correspondence: str
    fit: SimilarityFit2D
    prediction_indices: Tensor
    reference_indices: Tensor
    reconstructed_keypoints_px: Tensor


def _apply_similarity_fit(fit: SimilarityFit2D, points: Tensor) -> Tensor:
    cosine = torch.cos(fit.rotation_rad)
    sine = torch.sin(fit.rotation_rad)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        )
    )
    return points @ (fit.scale_px_per_metre * rotation).T + fit.translation_px


def _error_statistics(
    errors_px: Tensor,
    *,
    transform: PixelUVTransform,
) -> dict[str, object]:
    values = errors_px.detach().to(device="cpu", dtype=torch.float64).numpy()
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Pose diagnostic errors must be a non-empty finite vector.")
    pixel_values = {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "q95_px": float(np.quantile(values, 0.95)),
        "max_px": float(np.max(values)),
    }
    return {
        "count": int(values.size),
        **pixel_values,
        **{
            key.replace("_px", "_m"): transform.pixels_to_metres(value)
            for key, value in pixel_values.items()
        },
    }


def _add_flat_error_statistics(
    summary: dict[str, object],
    *,
    prefix: str,
    statistics: Mapping[str, object] | None,
) -> None:
    for statistic in ("mean", "median", "q95", "max"):
        for unit in ("px", "m"):
            key = f"{prefix}_error_{statistic}_{unit}"
            summary[key] = (
                None if statistics is None else statistics[f"{statistic}_{unit}"]
            )


def compute_pose_alignment_diagnostics(
    sample: CourtInstanceBatch,
    reference: ProjectedReference,
    transform: PixelUVTransform,
    *,
    match_max_error_px: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Match independently fitted prediction poses to full-KP14 references.

    This is deliberately separate from the established instance metric.  Each
    prediction proposes direct and half-turn Sim(2) fits from its available
    semantic keypoints.  Full reconstructed KP14 error supplies a pair cost,
    and one Hungarian assignment prevents two predictions from claiming the
    same accepted court.
    """

    if not isinstance(sample, CourtInstanceBatch):
        raise TypeError("Pose diagnostics require a CourtInstanceBatch.")
    if not isinstance(reference, ProjectedReference):
        raise TypeError("Pose diagnostics require a ProjectedReference.")
    if not isinstance(transform, PixelUVTransform):
        raise TypeError("Pose diagnostics require a PixelUVTransform.")
    if not math.isfinite(match_max_error_px) or match_max_error_px <= 0.0:
        raise ValueError(
            "Pose diagnostic match_max_error_px must be finite and positive."
        )
    reference_keypoints = np.asarray(reference.keypoints_px)
    reference_count = int(reference_keypoints.shape[0])
    if reference_keypoints.shape != (reference_count, NUM_KEYPOINTS, 2):
        raise ValueError(
            "Pose diagnostic reference keypoints must have shape (N,14,2)."
        )
    if not np.isfinite(reference_keypoints).all():
        raise ValueError("Pose diagnostic reference keypoints must be finite.")
    if not (
        len(reference.court_instance_ids)
        == len(reference.candidate_ids)
        == len(reference.poses)
        == reference_count
    ):
        raise ValueError(
            "Pose diagnostic reference metadata count must match keypoints."
        )

    device = sample.keypoints_px.device
    dtype = sample.keypoints_px.dtype
    canonical = canonical_court_keypoints(dtype=dtype, device=device)
    half_turn = torch.as_tensor(
        CAMERA_VIEW_HALF_TURN_INDEX,
        dtype=torch.long,
        device=device,
    )
    targets = torch.as_tensor(reference_keypoints, dtype=dtype, device=device)
    candidates_by_prediction: dict[int, tuple[_PoseFitCandidate, ...]] = {}
    prediction_fits: list[dict[str, object]] = []
    for prediction_index in range(sample.num_instances):
        prediction_indices = torch.nonzero(
            sample.valid[prediction_index], as_tuple=False
        ).flatten()
        valid_count = int(prediction_indices.numel())
        prediction_payload: dict[str, object] = {
            "prediction_index": prediction_index,
            "valid_keypoint_count": valid_count,
        }
        if valid_count < POSE_DIAGNOSTIC_MINIMUM_KEYPOINTS:
            prediction_payload.update(
                {
                    "status": "unavailable",
                    "reason": (
                        f"requires at least {POSE_DIAGNOSTIC_MINIMUM_KEYPOINTS} "
                        f"valid semantic keypoints; got {valid_count}"
                    ),
                    "available_correspondences": [],
                }
            )
            prediction_fits.append(prediction_payload)
            continue

        candidates: list[_PoseFitCandidate] = []
        failure_reasons: list[str] = []
        for correspondence, reference_indices in (
            ("direct", prediction_indices),
            ("half_turn", half_turn[prediction_indices]),
        ):
            try:
                fit = fit_similarity_2d(
                    canonical[reference_indices],
                    sample.keypoints_px[prediction_index, prediction_indices],
                )
            except ValueError as error:
                failure_reasons.append(f"{correspondence}: {error}")
                continue
            candidates.append(
                _PoseFitCandidate(
                    correspondence=correspondence,
                    fit=fit,
                    prediction_indices=prediction_indices,
                    reference_indices=reference_indices,
                    reconstructed_keypoints_px=_apply_similarity_fit(fit, canonical),
                )
            )
        if not candidates:
            prediction_payload.update(
                {
                    "status": "unavailable",
                    "reason": "; ".join(failure_reasons),
                    "available_correspondences": [],
                }
            )
            prediction_fits.append(prediction_payload)
            continue
        candidates_by_prediction[prediction_index] = tuple(candidates)
        prediction_payload.update(
            {
                "status": "available",
                "available_correspondences": [
                    candidate.correspondence for candidate in candidates
                ],
            }
        )
        if failure_reasons:
            prediction_payload["failed_correspondences"] = failure_reasons
        prediction_fits.append(prediction_payload)

    fit_prediction_indices = tuple(sorted(candidates_by_prediction))
    pair_choices: dict[tuple[int, int], tuple[_PoseFitCandidate, Tensor]] = {}
    costs: NDArray[np.float64] = np.full(
        (reference_count, len(fit_prediction_indices)),
        np.inf,
        dtype=np.float64,
    )
    for reference_index in range(reference_count):
        for column, prediction_index in enumerate(fit_prediction_indices):
            evaluated: list[tuple[int, _PoseFitCandidate, Tensor]] = []
            for candidate_index, candidate in enumerate(
                candidates_by_prediction[prediction_index]
            ):
                reconstructed_errors = torch.linalg.vector_norm(
                    candidate.reconstructed_keypoints_px - targets[reference_index],
                    dim=-1,
                )
                evaluated.append((candidate_index, candidate, reconstructed_errors))
            _candidate_index, candidate, reconstructed_errors = min(
                evaluated,
                key=lambda item: (float(item[2].mean()), item[0]),
            )
            pair_choices[reference_index, prediction_index] = (
                candidate,
                reconstructed_errors,
            )
            costs[reference_index, column] = float(reconstructed_errors.mean())

    assigned: list[tuple[int, int]] = []
    rejected_over_gate: list[dict[str, object]] = []
    if costs.shape[0] > 0 and costs.shape[1] > 0:
        # Preserve the ordinary forced assignment only as evidence of pairs
        # rejected by the gate.  Accepted matches use the same dummy-column
        # gated Hungarian policy as the established instance metric, so a
        # distant forced pair cannot block a lower-cost valid pair.
        reference_indices, prediction_columns = linear_sum_assignment(costs)
        for reference_index, column in zip(
            reference_indices.tolist(),
            prediction_columns.tolist(),
            strict=True,
        ):
            prediction_index = fit_prediction_indices[int(column)]
            pair_cost_px = float(costs[int(reference_index), int(column)])
            if pair_cost_px <= match_max_error_px:
                continue
            candidate, _errors = pair_choices[int(reference_index), prediction_index]
            rejected_over_gate.append(
                {
                    "prediction_index": prediction_index,
                    "reference_index": int(reference_index),
                    "court_instance_id": reference.court_instance_ids[
                        int(reference_index)
                    ],
                    "candidate_id": reference.candidate_ids[int(reference_index)],
                    "correspondence": candidate.correspondence,
                    "pair_cost_px": pair_cost_px,
                    "pair_cost_m": transform.pixels_to_metres(pair_cost_px),
                    "status": "unavailable",
                    "reason": (
                        f"pair cost exceeds match_max_error_px={match_max_error_px}"
                    ),
                }
            )
        accepted = np.isfinite(costs) & (costs <= match_max_error_px)
        unmatched_penalty = (reference_count + 1) * match_max_error_px
        gated_costs: NDArray[np.float64] = np.full(
            (reference_count, len(fit_prediction_indices) + reference_count),
            np.inf,
            dtype=np.float64,
        )
        gated_costs[:, : len(fit_prediction_indices)][accepted] = costs[accepted]
        for reference_index in range(reference_count):
            gated_costs[
                reference_index,
                len(fit_prediction_indices) + reference_index,
            ] = unmatched_penalty
        gated_references, gated_columns = linear_sum_assignment(gated_costs)
        assigned = [
            (int(reference_index), fit_prediction_indices[int(column)])
            for reference_index, column in zip(
                gated_references.tolist(),
                gated_columns.tolist(),
                strict=True,
            )
            if column < len(fit_prediction_indices)
        ]

    raw_error_vectors: list[Tensor] = []
    reconstructed_error_vectors: list[Tensor] = []
    matches: list[dict[str, object]] = []
    matched_prediction_indices: set[int] = set()
    matched_reference_indices: set[int] = set()
    half_turn_match_count = 0
    for reference_index, prediction_index in assigned:
        candidate, reconstructed_errors = pair_choices[
            reference_index, prediction_index
        ]
        raw_errors = torch.linalg.vector_norm(
            sample.keypoints_px[prediction_index, candidate.prediction_indices]
            - targets[reference_index, candidate.reference_indices],
            dim=-1,
        )
        raw_statistics = _error_statistics(raw_errors, transform=transform)
        reconstructed_statistics = _error_statistics(
            reconstructed_errors,
            transform=transform,
        )
        raw_error_vectors.append(raw_errors)
        reconstructed_error_vectors.append(reconstructed_errors)
        matched_prediction_indices.add(prediction_index)
        matched_reference_indices.add(reference_index)
        half_turn_match_count += int(candidate.correspondence == "half_turn")
        matches.append(
            {
                "prediction_index": prediction_index,
                "reference_index": reference_index,
                "court_instance_id": reference.court_instance_ids[reference_index],
                "candidate_id": reference.candidate_ids[reference_index],
                "correspondence": candidate.correspondence,
                "half_turn_selected": candidate.correspondence == "half_turn",
                "pair_cost_px": float(reconstructed_errors.mean()),
                "pair_cost_m": transform.pixels_to_metres(
                    float(reconstructed_errors.mean())
                ),
                "raw_detected_keypoints": {
                    **raw_statistics,
                    "coverage": int(raw_errors.numel()) / NUM_KEYPOINTS,
                },
                "reconstructed_full_keypoints": reconstructed_statistics,
                "similarity_fit": _similarity_payload(candidate.fit),
            }
        )

    matched_count = len(matches)
    unmatched_prediction_indices = sorted(
        set(range(sample.num_instances)) - matched_prediction_indices
    )
    unmatched_reference_indices = sorted(
        set(range(reference_count)) - matched_reference_indices
    )
    fit_unavailable_count = sample.num_instances - len(fit_prediction_indices)
    if matched_count == 0:
        status = "unavailable"
    elif unmatched_prediction_indices or unmatched_reference_indices:
        status = "partial"
    else:
        status = "available"
    aggregate_raw_statistics = (
        _error_statistics(torch.cat(raw_error_vectors), transform=transform)
        if raw_error_vectors
        else None
    )
    aggregate_reconstructed_statistics = (
        _error_statistics(
            torch.cat(reconstructed_error_vectors),
            transform=transform,
        )
        if reconstructed_error_vectors
        else None
    )
    raw_count = sum(int(errors.numel()) for errors in raw_error_vectors)
    reconstructed_count = sum(
        int(errors.numel()) for errors in reconstructed_error_vectors
    )
    summary: dict[str, object] = {
        "pose_diagnostic_status": status,
        "pose_minimum_keypoints": POSE_DIAGNOSTIC_MINIMUM_KEYPOINTS,
        "pose_prediction_instance_count": sample.num_instances,
        "pose_reference_instance_count": reference_count,
        "pose_fit_available_prediction_count": len(fit_prediction_indices),
        "pose_fit_unavailable_prediction_count": fit_unavailable_count,
        "pose_matched_instance_count": matched_count,
        "pose_rejected_over_gate_pair_count": len(rejected_over_gate),
        "pose_unmatched_prediction_count": len(unmatched_prediction_indices),
        "pose_unmatched_reference_count": len(unmatched_reference_indices),
        "pose_half_turn_match_count": half_turn_match_count,
        "pose_raw_kp_count": raw_count,
        "pose_raw_kp_coverage": (
            raw_count / (matched_count * NUM_KEYPOINTS) if matched_count > 0 else 0.0
        ),
        "pose_reconstructed_kp_count": reconstructed_count,
    }
    _add_flat_error_statistics(
        summary,
        prefix="pose_raw_kp",
        statistics=aggregate_raw_statistics,
    )
    _add_flat_error_statistics(
        summary,
        prefix="pose_reconstructed_kp",
        statistics=aggregate_reconstructed_statistics,
    )
    diagnostic: dict[str, object] = {
        "schema": "court_alignment_pose_diagnostics_v1",
        "status": status,
        "matching": ("full14_reconstructed_sim2_pair_cost_gated_hungarian_one_to_one"),
        "minimum_valid_semantic_keypoints": POSE_DIAGNOSTIC_MINIMUM_KEYPOINTS,
        "match_max_error_px": match_max_error_px,
        "summary": summary,
        "prediction_fits": prediction_fits,
        "matches": matches,
        "rejected_over_gate_pairs": rejected_over_gate,
        "unmatched_prediction_indices": unmatched_prediction_indices,
        "unmatched_references": [
            {
                "reference_index": index,
                "court_instance_id": reference.court_instance_ids[index],
                "candidate_id": reference.candidate_ids[index],
            }
            for index in unmatched_reference_indices
        ],
        "unavailable_policy": (
            "Unavailable or unmatched instances are counted explicitly and are "
            "never silently omitted into an all-instance pose average."
        ),
    }
    return summary, diagnostic


def _input_tensor(image: FloatArray) -> Tensor:
    if image.shape != (256, 256) or image.dtype != np.float32:
        raise ValueError(
            "Preprocessed image must have shape (256,256) and float32 dtype."
        )
    tensor = torch.from_numpy(np.ascontiguousarray(image)).view(1, 1, 256, 256)
    validate_court_alignment_input(tensor, expected_dtype=torch.float32)
    if tensor.shape != (1, 1, 256, 256) or tensor.dtype != torch.float32:
        raise AssertionError("Measured model input contract was not preserved.")
    return tensor


def _atomic_write(path: Path, payload: bytes) -> None:
    """Atomically replace one artifact with already-materialized bytes."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_evaluation_artifacts(
    output_dir: Path,
    *,
    metrics: Mapping[str, object],
    diagnostic: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Atomically persist normal artifacts and an optional queue repro mirror."""
    queue_repro_dir = resolve_queue_repro_dir()
    destinations = [output_dir]
    if queue_repro_dir is not None:
        queue_predictions = queue_repro_dir / "predictions"
        if queue_predictions == output_dir:
            raise ValueError(
                "Configured output_dir must differ from TENNIS_REPRO_DIR/predictions."
            )
        destinations.append(queue_predictions)
    for destination in destinations:
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(
                f"Evaluation output path is not a directory: {destination}"
            )

    serialized_json = {
        name: (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for name, payload in (
            ("metrics.json", metrics),
            ("diagnostic_metrics.json", diagnostic),
        )
    }
    npz_buffer = io.BytesIO()
    cast(Any, np.savez_compressed)(npz_buffer, **arrays)
    serialized_npz = npz_buffer.getvalue()

    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in serialized_json.items():
            _atomic_write(destination / name, payload)
        _atomic_write(destination / "pred_test.npz", serialized_npz)


def evaluate_real_heatmap(
    request: RealHeatmapEvaluationRequest,
    model: nn.Module,
) -> dict[str, object]:
    """Run measured B00-style inference and write the reproducible artifact bundle."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    archive = RealHeatmapArchive.load(request.archive_path, request.manifest_path)
    image, transform = letterbox_heatmap(
        archive.mean_probability,
        bounds_uv=archive.bounds_uv,
        grid_spacing_m=archive.grid_spacing_m,
        options=request.preprocess,
    )
    tensor = _input_tensor(image)
    alignments = load_accepted_alignments(request.alignment_path)
    ground_plane = AlignmentGroundPlane.from_alignments(alignments)
    reference = project_accepted_reference(
        alignments, ground_plane=ground_plane, transform=transform
    )
    checkpoint_metadata = load_model_checkpoint(model, request.checkpoint_path)
    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device}")
    model.to(device)
    predictor = CourtAlignmentPredictor(
        model,
        threshold=request.decoder.threshold,
        nms_kernel=request.decoder.nms_kernel,
        max_peaks=request.decoder.max_peaks,
        subpixel_refine=request.decoder.subpixel_refine,
        cluster_distance_px=request.decoder.cluster_distance_px,
        max_instances=request.decoder.max_instances,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        output = validate_court_alignment_output(model(tensor.to(device)))
    predictions = predictor.decode(output.heatmap_logits, output.center_votes)
    sample = predictions[0]
    reference_keypoints = torch.from_numpy(reference.keypoints_px).to(device)[None]
    reference_valid = torch.from_numpy(reference.valid).to(device)[None]
    reference_centers = torch.from_numpy(reference.centers_px).to(device)[None]
    raw_metrics = compute_alignment_metrics(
        predictions,
        reference_keypoints,
        reference_valid,
        centers=reference_centers,
        num_courts=torch.tensor([len(alignments)], dtype=torch.long, device=device),
        image_size=(256, 256),
        target_normalized=False,
        match_max_error_px=request.metrics.match_max_error_px,
        minimum_common_keypoints=request.metrics.minimum_common_keypoints,
        minimum_visible_keypoints=request.metrics.minimum_visible_keypoints,
        minimum_visible_fraction=request.metrics.minimum_visible_fraction,
        minimum_sim2_keypoints=request.metrics.minimum_sim2_keypoints,
    )
    pose_metrics, pose_diagnostic = compute_pose_alignment_diagnostics(
        sample,
        reference,
        transform,
        match_max_error_px=request.metrics.match_max_error_px,
    )
    metric_values: dict[str, object] = dict(raw_metrics)
    metric_values.update(pose_metrics)
    for pixel_key, metre_key in (
        ("instance_kp_mean_error_px", "instance_kp_mean_error_m"),
        ("matched_center_mean_error_px", "matched_center_mean_error_m"),
        ("sim2_translation_error_px", "sim2_translation_error_m"),
    ):
        metric_values[metre_key] = transform.pixels_to_metres(raw_metrics[pixel_key])
    metric_values.update(
        {
            "reference_type": REFERENCE_TYPE,
            "reference_is_independent_ground_truth": False,
            "reference_limitation": REFERENCE_LIMITATION,
            "scene_id": "B00",
            "checkpoint_path": str(request.checkpoint_path),
            "preprocess_method": request.preprocess.method,
            "decoder_threshold": request.decoder.threshold,
            "reference_instance_count": len(alignments),
            "predicted_instance_count_integer": sample.num_instances,
            "kp_mean_error_px": raw_metrics["instance_kp_mean_error_px"],
            "kp_mean_error_m": metric_values["instance_kp_mean_error_m"],
            "center_mean_error_px": raw_metrics["matched_center_mean_error_px"],
            "center_mean_error_m": metric_values["matched_center_mean_error_m"],
        }
    )
    reference_scales = [
        float(cast(float, pose["scale_px_per_metre"])) for pose in reference.poses
    ]
    scale_low, scale_high = request.training_scale_range_px_per_metre
    ood_flags = [scale < scale_low or scale > scale_high for scale in reference_scales]
    diagnostic: dict[str, object] = {
        "schema": "court_alignment_real_heatmap_diagnostics_v2",
        "reference": {
            "type": REFERENCE_TYPE,
            "is_independent_ground_truth": False,
            "limitation": REFERENCE_LIMITATION,
            "semantic_projection": (
                "model y-down KP14; physical scene_from_court local y equals "
                "negative model-canonical y"
            ),
            "court_instance_ids": list(reference.court_instance_ids),
            "candidate_ids": list(reference.candidate_ids),
            "poses": list(reference.poses),
        },
        "archive": {
            "path": str(archive.archive_path),
            "manifest_path": str(archive.manifest_path),
            "authority_array": "mean_probability",
            "bounds_uv_m": list(archive.bounds_uv),
            "grid_spacing_m": archive.grid_spacing_m,
            "raster_shape_hw": list(archive.raster_shape),
            "aggregate_view_count": archive.aggregate_view_count,
        },
        "ground_plane": ground_plane.to_metadata(),
        "transform": transform.to_metadata(),
        "preprocess": asdict(request.preprocess),
        "input_stats": {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
            "mean": float(tensor.mean()),
            "nonzero_fraction": float(torch.count_nonzero(tensor) / tensor.numel()),
        },
        "decoder": asdict(request.decoder),
        "metrics_options": asdict(request.metrics),
        "predicted_instance_count": sample.num_instances,
        "predicted_poses": _prediction_poses(sample, transform),
        "pose_alignment": pose_diagnostic,
        "raw_existing_metrics": raw_metrics,
        "checkpoint": checkpoint_metadata,
        "scale_domain": {
            "training_range_px_per_metre": [scale_low, scale_high],
            "reference_scale_px_per_metre": reference_scales,
            "reference_instance_ood_flags": ood_flags,
            "any_reference_instance_ood": any(ood_flags),
            "raster_effective_pixels_per_metre": transform.pixels_per_metre,
        },
    }
    transform_json = json.dumps(
        transform.to_metadata(), sort_keys=True, separators=(",", ":")
    )
    arrays: dict[str, NDArray[Any]] = {
        "input": tensor.cpu().numpy(),
        "pred_keypoints_px": sample.keypoints_px.detach().cpu().numpy(),
        "pred_scores": sample.scores.detach().cpu().numpy(),
        "pred_valid": sample.valid.detach().cpu().numpy(),
        "pred_centers_px": sample.centers_px.detach().cpu().numpy(),
        "pred_heatmap_probability": torch.sigmoid(output.heatmap_logits)
        .detach()
        .cpu()
        .numpy(),
        "pred_center_votes": output.center_votes.detach().cpu().numpy(),
        "reference_keypoints_px": reference.keypoints_px,
        "reference_valid": reference.valid,
        "reference_centers_px": reference.centers_px,
        "reference_court_instance_ids": np.asarray(reference.court_instance_ids),
        "reference_candidate_ids": np.asarray(reference.candidate_ids),
        "transform_json": np.asarray(transform_json),
        "reference_type": np.asarray(REFERENCE_TYPE),
        "reference_is_independent_ground_truth": np.asarray(False),
    }
    write_evaluation_artifacts(
        request.output_dir,
        metrics=metric_values,
        diagnostic=diagnostic,
        arrays=arrays,
    )
    return metric_values


__all__ = [
    "AlignmentGroundPlane",
    "DecoderOptions",
    "MetricOptions",
    "PixelUVTransform",
    "PreprocessOptions",
    "REAL_HEATMAP_PREPROCESSORS",
    "RealHeatmapArchive",
    "RealHeatmapEvaluationRequest",
    "compute_pose_alignment_diagnostics",
    "evaluate_real_heatmap",
    "letterbox_heatmap",
    "load_accepted_alignments",
    "load_model_checkpoint",
    "project_accepted_reference",
    "write_evaluation_artifacts",
]
