"""Collect projected line evidence with one shared fit/holdout implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineAccumulator,
    GroundLineMapSettings,
)
from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)
from src.synthetic_data_generation.alignment.components.inference.line_detector import (
    LineDetector,
    infer_line_projection,
)
from src.synthetic_data_generation.alignment.components.inputs.view_inputs import (
    load_provider_rgb_image,
)
from src.synthetic_data_generation.alignment.scene_provider.bundle import (
    LoadedSceneProviderBundle,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True)
class CollectedLineEvidence:
    """Projected points, per-group rasters, and machine-readable view records."""

    points_scene: NDArray[np.float64]
    weights: NDArray[np.float64]
    points_by_group: dict[int, NDArray[np.float64]]
    weights_by_group: dict[int, NDArray[np.float64]]
    points_by_camera: dict[str, NDArray[np.float64]]
    weights_by_camera: dict[str, NDArray[np.float64]]
    evidence_by_group: dict[int, NDArray[np.float32]]
    records: tuple[dict[str, Any], ...]


def collect_projected_line_evidence(
    cameras: Sequence[SceneCamera],
    *,
    bundle: LoadedSceneProviderBundle,
    detector: LineDetector,
    plane: GroundPlaneEstimate,
    bounds: tuple[float, float, float, float],
    settings: GroundLineMapSettings,
) -> CollectedLineEvidence:
    """Infer and collect one declared camera partition without silent fallback."""
    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("At least one camera is required for line collection.")
    group_ids = sorted({camera.group_id for camera in camera_tuple})
    accumulators = {
        group_id: GroundLineAccumulator(
            bounds=bounds,
            grid_spacing=settings.grid_spacing,
        )
        for group_id in group_ids
    }
    points_lists: dict[int, list[NDArray[np.float64]]] = {
        group_id: [] for group_id in group_ids
    }
    weight_lists: dict[int, list[NDArray[np.float64]]] = {
        group_id: [] for group_id in group_ids
    }
    image_files = {image.camera_id: image for image in bundle.manifest.images}
    records: list[dict[str, Any]] = []
    points_by_camera: dict[str, NDArray[np.float64]] = {}
    weights_by_camera: dict[str, NDArray[np.float64]] = {}
    for camera in camera_tuple:
        image_rgb = load_provider_rgb_image(bundle.image_path(camera.camera_id))
        observation = infer_line_projection(
            image_rgb,
            camera,
            detector=detector,
            plane=plane,
            bounds=bounds,
            settings=settings,
        )
        projection = observation.projection
        accepted = len(projection.points_scene) >= settings.min_projected_pixels
        raster_cell_count = (
            accumulators[camera.group_id].add_view(projection) if accepted else 0
        )
        if accepted:
            camera_points = np.asarray(
                projection.points_scene,
                dtype=np.float64,
            )
            camera_weights = np.asarray(
                projection.probabilities * projection.proximity_weights,
                dtype=np.float64,
            )
            points_lists[camera.group_id].append(camera_points)
            weight_lists[camera.group_id].append(camera_weights)
            points_by_camera[camera.camera_id] = camera_points
            weights_by_camera[camera.camera_id] = camera_weights
        else:
            points_by_camera[camera.camera_id] = np.empty(
                (0, 3),
                dtype=np.float64,
            )
            weights_by_camera[camera.camera_id] = np.empty(0, dtype=np.float64)
        records.append(
            {
                "camera_id": camera.camera_id,
                "source_frame_index": camera.source_frame_index,
                "group_id": camera.group_id,
                "image_sha256": image_files[camera.camera_id].file.sha256,
                "line_output_width": observation.output_width,
                "line_output_height": observation.output_height,
                "selected_line_pixel_count": (observation.selected_line_pixel_count),
                "projected_line_pixel_count": len(projection.points_scene),
                "raster_cell_count": raster_cell_count,
                "accepted": accepted,
                "rejection_reasons": (
                    [] if accepted else ["insufficient_projected_line_pixels"]
                ),
                "projection_rejections": {
                    "parallel": projection.invalid_parallel_count,
                    "behind_camera": projection.invalid_behind_count,
                    "beyond_max_range": projection.invalid_range_count,
                    "outside_bounds": projection.invalid_bounds_count,
                },
            }
        )
    points_by_group = {
        group_id: _concatenate_points(values)
        for group_id, values in points_lists.items()
    }
    weights_by_group = {
        group_id: _concatenate_weights(values)
        for group_id, values in weight_lists.items()
    }
    return CollectedLineEvidence(
        points_scene=_concatenate_points(list(points_by_group.values())),
        weights=_concatenate_weights(list(weights_by_group.values())),
        points_by_group=points_by_group,
        weights_by_group=weights_by_group,
        points_by_camera=points_by_camera,
        weights_by_camera=weights_by_camera,
        evidence_by_group={
            group_id: accumulator.arrays()["evidence_sum"]
            for group_id, accumulator in accumulators.items()
        },
        records=tuple(records),
    )


def _concatenate_points(
    values: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    nonempty = [value for value in values if len(value)]
    return np.concatenate(nonempty) if nonempty else np.empty((0, 3), dtype=np.float64)


def _concatenate_weights(
    values: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    nonempty = [value for value in values if len(value)]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)
