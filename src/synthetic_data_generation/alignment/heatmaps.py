"""Strict line-heatmap diagnostics for measured court alignment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

LINE_HEATMAP_DIRECTORY = "line-heatmaps"
LINE_HEATMAP_ARCHIVE_FILE = "heatmaps.npz"
LINE_HEATMAP_MANIFEST_FILE = "manifest.json"
WEIGHTED_PROJECTION_HEATMAP_FILE = "weighted-projection.png"
LINE_HEATMAP_VIEWS_DIRECTORY = "views"

GROUND_PLANE_UV_COORDINATE_CONVENTION = (
    "right_handed_metric_scene_ground_plane_uv;units=metres;"
    "u=dot(metric_point-origin,basis_u);"
    "v=dot(metric_point-origin,basis_v);normal=cross(basis_u,basis_v)"
)
_ARCHIVE_SCHEMA = "alignment_line_heatmaps_v2"
_MANIFEST_SCHEMA = "alignment_line_heatmap_manifest_v2"
_WEIGHT_MODEL = "1/(1+(camera_range/proximity_scale)^power)"
_RASTER_REDUCER = "per-view cell max then weighted global sum"
_ARCHIVE_KEYS = {
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


class _SavezCompressed(Protocol):
    def __call__(self, file: Path, **arrays: NDArray[Any]) -> None: ...


_savez_compressed = cast(_SavezCompressed, np.savez_compressed)


@dataclass(frozen=True, slots=True)
class AlignmentLineHeatmapView:
    """Raw detector heatmap and its valid weighted ground projection for one view."""

    camera_id: str
    probability: NDArray[np.float32]
    points_uv: NDArray[np.float64]
    projected_probabilities: NDArray[np.float32]
    proximity_weights: NDArray[np.float64]
    included_in_aggregate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise TypeError("camera_id must be a non-empty string.")
        if type(self.included_in_aggregate) is not bool:
            raise TypeError("included_in_aggregate must be a boolean.")
        probability = _readonly_array(
            self.probability,
            dtype=np.dtype(np.float32),
            name="probability",
        )
        if probability.ndim != 2 or min(probability.shape) < 2:
            raise ValueError("probability must be a non-trivial 2-D heatmap.")
        _require_unit_interval(probability, name="probability")

        points_uv = _readonly_array(
            self.points_uv,
            dtype=np.dtype(np.float64),
            name="points_uv",
        )
        projected_probabilities = _readonly_array(
            self.projected_probabilities,
            dtype=np.dtype(np.float32),
            name="projected_probabilities",
        )
        proximity_weights = _readonly_array(
            self.proximity_weights,
            dtype=np.dtype(np.float64),
            name="proximity_weights",
        )
        if points_uv.ndim != 2 or points_uv.shape[1:] != (2,):
            raise ValueError("points_uv must have shape (N, 2).")
        if projected_probabilities.shape != (len(points_uv),):
            raise ValueError(
                "projected_probabilities must have one value per projected point."
            )
        if proximity_weights.shape != (len(points_uv),):
            raise ValueError("proximity_weights must have one value per projected point.")
        _require_unit_interval(
            projected_probabilities,
            name="projected_probabilities",
        )
        if len(proximity_weights) and (
            np.any(proximity_weights <= 0.0) or np.any(proximity_weights > 1.0)
        ):
            raise ValueError("proximity_weights must lie in (0, 1].")
        if self.included_in_aggregate and len(points_uv) == 0:
            raise ValueError("An aggregate view must contain projected line evidence.")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "points_uv", points_uv)
        object.__setattr__(
            self,
            "projected_probabilities",
            projected_probabilities,
        )
        object.__setattr__(self, "proximity_weights", proximity_weights)


@dataclass(frozen=True, slots=True)
class AlignmentLineHeatmaps:
    """Complete raw/per-view/aggregate line heatmap evidence for one evaluation."""

    bounds_uv: tuple[float, float, float, float]
    grid_spacing: float
    proximity_scale: float
    proximity_power: float
    views: tuple[AlignmentLineHeatmapView, ...]
    coordinate_convention: str = GROUND_PLANE_UV_COORDINATE_CONVENTION
    coordinate_units: str = "metres"

    def __post_init__(self) -> None:
        if self.coordinate_convention != GROUND_PLANE_UV_COORDINATE_CONVENTION:
            raise ValueError("Unsupported line-heatmap coordinate convention.")
        if self.coordinate_units != "metres":
            raise ValueError("Line-heatmap UV coordinates must use metres.")
        bounds = tuple(float(value) for value in self.bounds_uv)
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError("bounds_uv must contain four finite values.")
        u_min, u_max, v_min, v_max = bounds
        if u_min >= u_max or v_min >= v_max:
            raise ValueError("bounds_uv must define positive ground-plane area.")
        for name in ("grid_spacing", "proximity_scale", "proximity_power"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        views = tuple(self.views)
        if not views or any(
            not isinstance(view, AlignmentLineHeatmapView) for view in views
        ):
            raise TypeError("views must contain AlignmentLineHeatmapView values.")
        camera_ids = tuple(view.camera_id for view in views)
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Line-heatmap camera IDs must be unique.")
        if len(views) > np.iinfo(np.uint16).max:
            raise ValueError("Line-heatmap view count exceeds uint16 raster capacity.")
        for view in views:
            if len(view.points_uv) == 0:
                continue
            if (
                np.any(view.points_uv[:, 0] < u_min)
                or np.any(view.points_uv[:, 0] > u_max)
                or np.any(view.points_uv[:, 1] < v_min)
                or np.any(view.points_uv[:, 1] > v_max)
            ):
                raise ValueError(
                    f"Projected heatmap points exceed bounds for {view.camera_id!r}."
                )
        object.__setattr__(self, "bounds_uv", bounds)
        object.__setattr__(self, "grid_spacing", float(self.grid_spacing))
        object.__setattr__(self, "proximity_scale", float(self.proximity_scale))
        object.__setattr__(self, "proximity_power", float(self.proximity_power))
        object.__setattr__(self, "views", views)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return the fixed selected-camera order."""
        return tuple(view.camera_id for view in self.views)

    @property
    def aggregate_camera_ids(self) -> tuple[str, ...]:
        """Return selected cameras retained by the alignment evidence gate."""
        return tuple(
            view.camera_id for view in self.views if view.included_in_aggregate
        )

    @property
    def raster_shape(self) -> tuple[int, int]:
        """Return deterministic ``(height, width)`` for the common UV grid."""
        u_min, u_max, v_min, v_max = self.bounds_uv
        width = int(np.ceil((u_max - u_min) / self.grid_spacing)) + 1
        height = int(np.ceil((v_max - v_min) / self.grid_spacing)) + 1
        return height, width


@dataclass(frozen=True, slots=True)
class LineHeatmapRasters:
    """Aggregate weighted evidence and support arrays on the common UV grid."""

    evidence_sum: NDArray[np.float32]
    weight_sum: NDArray[np.float32]
    view_count: NDArray[np.uint16]
    mean_probability: NDArray[np.float32]


def rasterize_weighted_view(
    heatmaps: AlignmentLineHeatmaps,
    view: AlignmentLineHeatmapView,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Rasterize one view with independent max reducers for evidence and weight."""
    if view.camera_id not in heatmaps.camera_ids:
        raise ValueError("Line-heatmap view does not belong to the supplied bundle.")
    height, width = heatmaps.raster_shape
    view_evidence: NDArray[np.float32] = np.zeros(
        height * width,
        dtype=np.float32,
    )
    view_weight: NDArray[np.float32] = np.zeros(
        height * width,
        dtype=np.float32,
    )
    if len(view.points_uv) == 0:
        return view_evidence.reshape(height, width), view_weight.reshape(height, width)
    u_min, _, v_min, _ = heatmaps.bounds_uv
    columns = np.rint(
        (view.points_uv[:, 0] - u_min) / heatmaps.grid_spacing
    ).astype(np.int64)
    rows = np.rint(
        (view.points_uv[:, 1] - v_min) / heatmaps.grid_spacing
    ).astype(np.int64)
    valid = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    flat_indices = rows[valid] * width + columns[valid]
    weighted = (
        view.projected_probabilities[valid].astype(np.float64)
        * view.proximity_weights[valid]
    )
    np.maximum.at(view_evidence, flat_indices, weighted.astype(np.float32))
    np.maximum.at(
        view_weight,
        flat_indices,
        view.proximity_weights[valid].astype(np.float32),
    )
    return view_evidence.reshape(height, width), view_weight.reshape(height, width)


def aggregate_line_heatmaps(heatmaps: AlignmentLineHeatmaps) -> LineHeatmapRasters:
    """Apply per-view cell max followed by weighted global summation."""
    height, width = heatmaps.raster_shape
    evidence_sum: NDArray[np.float32] = np.zeros(
        (height, width),
        dtype=np.float32,
    )
    weight_sum: NDArray[np.float32] = np.zeros(
        (height, width),
        dtype=np.float32,
    )
    view_count: NDArray[np.uint16] = np.zeros(
        (height, width),
        dtype=np.uint16,
    )
    for view in heatmaps.views:
        if not view.included_in_aggregate:
            continue
        view_evidence, view_weight = rasterize_weighted_view(heatmaps, view)
        mask = view_weight > 0.0
        evidence_sum[mask] += view_evidence[mask]
        weight_sum[mask] += view_weight[mask]
        view_count[mask] += 1
    mean_probability = np.divide(
        evidence_sum,
        weight_sum,
        out=np.zeros_like(evidence_sum),
        where=weight_sum > 0.0,
    )
    return LineHeatmapRasters(
        evidence_sum=evidence_sum,
        weight_sum=weight_sum,
        view_count=view_count,
        mean_probability=mean_probability,
    )


def write_line_heatmaps(
    output_path: Path,
    *,
    heatmaps: AlignmentLineHeatmaps,
) -> None:
    """Write the exact numeric and PNG heatmap inventory beneath a fresh path."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Line-heatmap output already exists: {output_path}")
    output_path.mkdir(parents=False, exist_ok=False)
    views_path = output_path / LINE_HEATMAP_VIEWS_DIRECTORY
    views_path.mkdir(parents=False, exist_ok=False)
    rasters = aggregate_line_heatmaps(heatmaps)
    _savez_compressed(
        output_path / LINE_HEATMAP_ARCHIVE_FILE,
        **_archive_arrays(heatmaps, rasters=rasters),
    )
    _write_json(
        output_path / LINE_HEATMAP_MANIFEST_FILE,
        _manifest_payload(heatmaps),
    )
    _write_png(
        output_path / WEIGHTED_PROJECTION_HEATMAP_FILE,
        _render_aggregate(rasters.evidence_sum, rasters.view_count),
    )
    for index, view in enumerate(heatmaps.views):
        weighted, _weight = rasterize_weighted_view(heatmaps, view)
        _write_png(
            views_path / _view_heatmap_name(index),
            _render_probability(view.probability, flip_vertical=False),
        )
        _write_png(
            views_path / _view_weighted_heatmap_name(index),
            _render_probability(weighted, flip_vertical=True),
        )


def validate_line_heatmaps(output_path: Path) -> AlignmentLineHeatmaps:
    """Strict-load and cross-check numeric, manifest, and rendered heatmaps."""
    if not output_path.is_dir() or output_path.is_symlink():
        raise ValueError(
            f"Line-heatmap output must be an ordinary directory: {output_path}"
        )
    expected = {
        LINE_HEATMAP_ARCHIVE_FILE,
        LINE_HEATMAP_MANIFEST_FILE,
        WEIGHTED_PROJECTION_HEATMAP_FILE,
        LINE_HEATMAP_VIEWS_DIRECTORY,
    }
    actual = {path.name for path in output_path.iterdir()}
    if actual != expected:
        raise ValueError("Line-heatmap output inventory does not match the fixed schema.")
    views_path = output_path / LINE_HEATMAP_VIEWS_DIRECTORY
    if not views_path.is_dir() or views_path.is_symlink():
        raise ValueError("Line-heatmap views must be an ordinary directory.")
    for name in expected - {LINE_HEATMAP_VIEWS_DIRECTORY}:
        _require_ordinary_file(output_path / name)

    heatmaps, stored_rasters = _load_archive(
        output_path / LINE_HEATMAP_ARCHIVE_FILE
    )
    recomputed = aggregate_line_heatmaps(heatmaps)
    _require_rasters_equal(stored_rasters, recomputed)
    manifest = _load_json_object(output_path / LINE_HEATMAP_MANIFEST_FILE)
    if manifest != _manifest_payload(heatmaps):
        raise ValueError("Line-heatmap manifest disagrees with the numeric archive.")

    expected_views = {
        name
        for index in range(len(heatmaps.views))
        for name in (_view_heatmap_name(index), _view_weighted_heatmap_name(index))
    }
    actual_views = {path.name for path in views_path.iterdir()}
    if actual_views != expected_views:
        raise ValueError("Per-view heatmap inventory does not match the manifest.")
    _validate_png(
        output_path / WEIGHTED_PROJECTION_HEATMAP_FILE,
        _render_aggregate(recomputed.evidence_sum, recomputed.view_count),
    )
    for index, view in enumerate(heatmaps.views):
        weighted, _weight = rasterize_weighted_view(heatmaps, view)
        _validate_png(
            views_path / _view_heatmap_name(index),
            _render_probability(view.probability, flip_vertical=False),
        )
        _validate_png(
            views_path / _view_weighted_heatmap_name(index),
            _render_probability(weighted, flip_vertical=True),
        )
    return heatmaps


def _archive_arrays(
    heatmaps: AlignmentLineHeatmaps,
    *,
    rasters: LineHeatmapRasters,
) -> dict[str, NDArray[Any]]:
    probability_shapes = np.asarray(
        [view.probability.shape for view in heatmaps.views],
        dtype=np.int64,
    )
    probability_sizes = np.prod(probability_shapes, axis=1, dtype=np.int64)
    probability_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(probability_sizes, dtype=np.int64))
    )
    projected_sizes = np.asarray(
        [len(view.points_uv) for view in heatmaps.views],
        dtype=np.int64,
    )
    projected_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(projected_sizes, dtype=np.int64))
    )
    projected_points = (
        np.concatenate([view.points_uv for view in heatmaps.views])
        if int(projected_offsets[-1])
        else np.empty((0, 2), dtype=np.float64)
    )
    projected_probabilities = (
        np.concatenate([view.projected_probabilities for view in heatmaps.views])
        if int(projected_offsets[-1])
        else np.empty(0, dtype=np.float32)
    )
    proximity_weights = (
        np.concatenate([view.proximity_weights for view in heatmaps.views])
        if int(projected_offsets[-1])
        else np.empty(0, dtype=np.float64)
    )
    return {
        "schema": np.asarray(_ARCHIVE_SCHEMA),
        "coordinate_convention": np.asarray(heatmaps.coordinate_convention),
        "coordinate_units": np.asarray(heatmaps.coordinate_units),
        "camera_ids": np.asarray(heatmaps.camera_ids),
        "included_in_aggregate": np.asarray(
            [view.included_in_aggregate for view in heatmaps.views],
            dtype=np.bool_,
        ),
        "bounds_uv": np.asarray(heatmaps.bounds_uv, dtype=np.float64),
        "grid_spacing": np.asarray(heatmaps.grid_spacing, dtype=np.float64),
        "proximity_scale": np.asarray(heatmaps.proximity_scale, dtype=np.float64),
        "proximity_power": np.asarray(heatmaps.proximity_power, dtype=np.float64),
        "probability_shapes": probability_shapes,
        "probability_offsets": probability_offsets,
        "probability_values": np.concatenate(
            [view.probability.ravel() for view in heatmaps.views]
        ).astype(np.float32, copy=False),
        "projected_offsets": projected_offsets,
        "projected_points_uv": projected_points.astype(np.float64, copy=False),
        "projected_probabilities": projected_probabilities.astype(
            np.float32, copy=False
        ),
        "proximity_weights": proximity_weights.astype(np.float64, copy=False),
        "evidence_sum": rasters.evidence_sum,
        "weight_sum": rasters.weight_sum,
        "view_count": rasters.view_count,
        "mean_probability": rasters.mean_probability,
    }


def _load_archive(
    path: Path,
) -> tuple[AlignmentLineHeatmaps, LineHeatmapRasters]:
    _require_ordinary_file(path)
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != _ARCHIVE_KEYS:
            raise ValueError("Line-heatmap archive keys do not match the strict schema.")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    schema = arrays["schema"]
    if (
        schema.ndim != 0
        or schema.dtype.kind != "U"
        or str(schema.item()) != _ARCHIVE_SCHEMA
    ):
        raise ValueError("Unsupported line-heatmap archive schema.")
    for name, expected in (
        ("coordinate_convention", GROUND_PLANE_UV_COORDINATE_CONVENTION),
        ("coordinate_units", "metres"),
    ):
        value = arrays[name]
        if value.ndim != 0 or value.dtype.kind != "U" or str(value.item()) != expected:
            raise ValueError(f"Unsupported line-heatmap {name}.")
    camera_ids = arrays["camera_ids"]
    included = arrays["included_in_aggregate"]
    shapes = arrays["probability_shapes"]
    probability_offsets = arrays["probability_offsets"]
    probability_values = arrays["probability_values"]
    projected_offsets = arrays["projected_offsets"]
    projected_points = arrays["projected_points_uv"]
    projected_probabilities = arrays["projected_probabilities"]
    proximity_weights = arrays["proximity_weights"]
    view_count_value = len(camera_ids)
    if (
        camera_ids.ndim != 1
        or camera_ids.dtype.kind != "U"
        or view_count_value == 0
        or len(set(camera_ids.tolist())) != view_count_value
    ):
        raise ValueError("Line-heatmap camera_ids must be a unique Unicode vector.")
    if included.dtype != np.bool_ or included.shape != (view_count_value,):
        raise ValueError("included_in_aggregate must be a boolean camera vector.")
    if (
        shapes.dtype != np.int64
        or shapes.shape != (view_count_value, 2)
        or np.any(shapes < 2)
    ):
        raise ValueError("probability_shapes must be positive int64 (N, 2) values.")
    if probability_values.dtype != np.float32 or probability_values.ndim != 1:
        raise ValueError("probability_values must be a float32 vector.")
    _validate_offsets(
        probability_offsets,
        item_count=view_count_value,
        final_size=len(probability_values),
        name="probability_offsets",
    )
    expected_probability_sizes = np.prod(shapes, axis=1, dtype=np.int64)
    if not np.array_equal(np.diff(probability_offsets), expected_probability_sizes):
        raise ValueError("probability_offsets disagree with probability_shapes.")
    if projected_points.dtype != np.float64 or projected_points.shape[1:] != (2,):
        raise ValueError("projected_points_uv must be a float64 (N, 2) array.")
    if (
        projected_probabilities.dtype != np.float32
        or projected_probabilities.shape != (len(projected_points),)
        or proximity_weights.dtype != np.float64
        or proximity_weights.shape != (len(projected_points),)
    ):
        raise ValueError("Projected line-heatmap vectors have invalid shape or dtype.")
    _validate_offsets(
        projected_offsets,
        item_count=view_count_value,
        final_size=len(projected_points),
        name="projected_offsets",
    )
    bounds = arrays["bounds_uv"]
    if bounds.dtype != np.float64 or bounds.shape != (4,):
        raise ValueError("bounds_uv must be a float64 four-vector.")
    scalar_names = ("grid_spacing", "proximity_scale", "proximity_power")
    for name in scalar_names:
        if arrays[name].dtype != np.float64 or arrays[name].ndim != 0:
            raise ValueError(f"{name} must be a float64 scalar.")

    views: list[AlignmentLineHeatmapView] = []
    for index, camera_id in enumerate(cast(list[str], camera_ids.tolist())):
        probability_start = int(probability_offsets[index])
        probability_stop = int(probability_offsets[index + 1])
        height, width = (int(value) for value in shapes[index])
        projected_start = int(projected_offsets[index])
        projected_stop = int(projected_offsets[index + 1])
        views.append(
            AlignmentLineHeatmapView(
                camera_id=camera_id,
                probability=probability_values[
                    probability_start:probability_stop
                ].reshape(height, width),
                points_uv=projected_points[projected_start:projected_stop],
                projected_probabilities=projected_probabilities[
                    projected_start:projected_stop
                ],
                proximity_weights=proximity_weights[
                    projected_start:projected_stop
                ],
                included_in_aggregate=bool(included[index]),
            )
        )
    heatmaps = AlignmentLineHeatmaps(
        bounds_uv=(
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        ),
        grid_spacing=float(arrays["grid_spacing"].item()),
        proximity_scale=float(arrays["proximity_scale"].item()),
        proximity_power=float(arrays["proximity_power"].item()),
        views=tuple(views),
        coordinate_convention=str(arrays["coordinate_convention"].item()),
        coordinate_units=str(arrays["coordinate_units"].item()),
    )
    raster_shape = heatmaps.raster_shape
    evidence_sum = _raster_array(
        arrays["evidence_sum"],
        dtype=np.dtype(np.float32),
        shape=raster_shape,
        name="evidence_sum",
    )
    weight_sum = _raster_array(
        arrays["weight_sum"],
        dtype=np.dtype(np.float32),
        shape=raster_shape,
        name="weight_sum",
    )
    view_count = _raster_array(
        arrays["view_count"],
        dtype=np.dtype(np.uint16),
        shape=raster_shape,
        name="view_count",
    )
    mean_probability = _raster_array(
        arrays["mean_probability"],
        dtype=np.dtype(np.float32),
        shape=raster_shape,
        name="mean_probability",
    )
    return heatmaps, LineHeatmapRasters(
        evidence_sum=cast(NDArray[np.float32], evidence_sum),
        weight_sum=cast(NDArray[np.float32], weight_sum),
        view_count=cast(NDArray[np.uint16], view_count),
        mean_probability=cast(NDArray[np.float32], mean_probability),
    )


def _manifest_payload(heatmaps: AlignmentLineHeatmaps) -> dict[str, object]:
    height, width = heatmaps.raster_shape
    return {
        "schema": _MANIFEST_SCHEMA,
        "coordinate_convention": heatmaps.coordinate_convention,
        "coordinate_units": heatmaps.coordinate_units,
        "archive": LINE_HEATMAP_ARCHIVE_FILE,
        "weighted_projection_heatmap": WEIGHTED_PROJECTION_HEATMAP_FILE,
        "views_directory": LINE_HEATMAP_VIEWS_DIRECTORY,
        "bounds_uv": list(heatmaps.bounds_uv),
        "grid_spacing": heatmaps.grid_spacing,
        "raster_shape": [height, width],
        "weight_model": _WEIGHT_MODEL,
        "proximity_scale": heatmaps.proximity_scale,
        "proximity_power": heatmaps.proximity_power,
        "raster_reducer": _RASTER_REDUCER,
        "ground_png_orientation": "top row is maximum v (vertical flip)",
        "raw_heatmap_encoding": "turbo-u8-linear-[0,1]",
        "weighted_heatmap_encoding": "turbo-u8-linear-[0,1]",
        "aggregate_heatmap_encoding": "turbo-u8-log1p-q99.5-positive",
        "view_count": len(heatmaps.views),
        "aggregate_view_count": len(heatmaps.aggregate_camera_ids),
        "views": [
            {
                "index": index,
                "camera_id": view.camera_id,
                "included_in_aggregate": view.included_in_aggregate,
                "probability_shape": list(view.probability.shape),
                "projected_point_count": len(view.points_uv),
                "heatmap": f"{LINE_HEATMAP_VIEWS_DIRECTORY}/{_view_heatmap_name(index)}",
                "weighted_heatmap": (
                    f"{LINE_HEATMAP_VIEWS_DIRECTORY}/"
                    f"{_view_weighted_heatmap_name(index)}"
                ),
            }
            for index, view in enumerate(heatmaps.views)
        ],
    }


def _require_rasters_equal(
    stored: LineHeatmapRasters,
    expected: LineHeatmapRasters,
) -> None:
    for name in ("evidence_sum", "weight_sum", "view_count", "mean_probability"):
        if not np.array_equal(getattr(stored, name), getattr(expected, name)):
            raise ValueError(
                f"Stored line-heatmap {name} disagrees with projected view evidence."
            )


def _render_probability(
    values: NDArray[np.floating[Any]],
    *,
    flip_vertical: bool,
) -> NDArray[np.uint8]:
    probability = np.asarray(values)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Rendered probability heatmap must be finite and 2-D.")
    if np.any(probability < 0.0) or np.any(probability > 1.0 + 1.0e-7):
        raise ValueError("Rendered probability heatmap must lie in [0, 1].")
    if flip_vertical:
        probability = np.flipud(probability)
    intensity = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = _turbo(intensity)
    colored[probability <= 0.0] = 0
    return colored


def _render_aggregate(
    evidence_sum: NDArray[np.float32],
    view_count: NDArray[np.uint16],
) -> NDArray[np.uint8]:
    evidence = np.asarray(evidence_sum, dtype=np.float32)
    support = np.asarray(view_count, dtype=np.uint16)
    if evidence.shape != support.shape or evidence.ndim != 2:
        raise ValueError("Aggregate heatmap inputs must be same-shape 2-D arrays.")
    positive = evidence[evidence > 0.0]
    scale = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
    normalized = np.clip(
        np.log1p(evidence) / np.log1p(max(scale, 1.0e-6)),
        0.0,
        1.0,
    )
    intensity = np.rint(normalized * 255.0).astype(np.uint8)
    colored = _turbo(intensity)
    colored[support == 0] = 0
    return np.flipud(colored)


def _turbo(intensity: NDArray[np.uint8]) -> NDArray[np.uint8]:
    colored_bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    return cast(
        NDArray[np.uint8],
        np.asarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8),
    )


def _write_png(path: Path, image_rgb: NDArray[np.uint8]) -> None:
    Image.fromarray(image_rgb, mode="RGB").save(
        path,
        format="PNG",
        compress_level=6,
    )


def _validate_png(path: Path, expected_rgb: NDArray[np.uint8]) -> None:
    _require_ordinary_file(path)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Line heatmap must be an RGB PNG: {path}")
        actual = np.asarray(image, dtype=np.uint8)
    if not np.array_equal(actual, expected_rgb):
        raise ValueError(f"Rendered line heatmap disagrees with numeric evidence: {path}")


def _view_heatmap_name(index: int) -> str:
    return f"view-{index:03d}-heatmap.png"


def _view_weighted_heatmap_name(index: int) -> str:
    return f"view-{index:03d}-weighted-heatmap.png"


def _readonly_array(
    value: NDArray[Any],
    *,
    dtype: np.dtype[Any],
    name: str,
) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _require_unit_interval(value: NDArray[Any], *, name: str) -> None:
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(f"{name} must lie in [0, 1].")


def _validate_offsets(
    offsets: NDArray[Any],
    *,
    item_count: int,
    final_size: int,
    name: str,
) -> None:
    if (
        offsets.dtype != np.int64
        or offsets.shape != (item_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != final_size
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"{name} is not a valid int64 offset vector.")


def _raster_array(
    value: NDArray[Any],
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, int],
    name: str,
) -> NDArray[Any]:
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}.")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path) -> dict[str, object]:
    _require_ordinary_file(path)
    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise TypeError(f"JSON document must be an object with string keys: {path}")
    return dict(raw)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _require_ordinary_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected an ordinary line-heatmap file: {path}")


__all__ = [
    "AlignmentLineHeatmapView",
    "AlignmentLineHeatmaps",
    "GROUND_PLANE_UV_COORDINATE_CONVENTION",
    "LINE_HEATMAP_ARCHIVE_FILE",
    "LINE_HEATMAP_DIRECTORY",
    "LINE_HEATMAP_MANIFEST_FILE",
    "LINE_HEATMAP_VIEWS_DIRECTORY",
    "LineHeatmapRasters",
    "WEIGHTED_PROJECTION_HEATMAP_FILE",
    "aggregate_line_heatmaps",
    "rasterize_weighted_view",
    "validate_line_heatmaps",
    "write_line_heatmaps",
]
