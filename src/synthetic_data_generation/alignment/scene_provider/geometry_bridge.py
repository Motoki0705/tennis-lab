"""Provider-side COLMAP geometry bridge for an isolated Python subprocess.

This adapter deliberately imports pycolmap only inside the configured provider
environment. Its only interface is a strict JSON request and a temporary NPZ
response consumed by
:mod:`src.synthetic_data_generation.alignment.scene_provider.export`.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
)

_REQUEST_KEYS = {"cameras_bin", "images_bin", "points3d_bin"}


PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.geometry_bridge",
    fields=(
        BoundaryPathField(
            "request",
            PathRole.CACHE,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "output",
            PathRole.CACHE,
            PathDirection.OUTPUT,
            PathKind.FILE,
        ),
    ),
)


def _runtime_versions() -> dict[str, str]:
    import pycolmap  # type: ignore[import-not-found]

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pycolmap": str(pycolmap.__version__),
    }


def _load_request(
    path: Path,
    *,
    resolver: PathResolver,
) -> tuple[Path, Path, Path]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("Geometry bridge request must be a JSON object.")
    if any(type(key) is not str for key in value):
        raise TypeError("Geometry bridge request keys must be strings.")
    raw = {str(key): item for key, item in value.items()}
    if set(raw) != _REQUEST_KEYS:
        raise ValueError(
            "Geometry bridge request must contain exactly three role-tagged paths."
        )
    resolved: dict[str, Path] = {}
    for key in sorted(_REQUEST_KEYS):
        declaration = raw[key]
        if not isinstance(declaration, Mapping) or any(
            type(item) is not str for item in declaration
        ):
            raise TypeError(f"Geometry bridge request {key} must be an object.")
        if set(declaration) != {"role", "path"}:
            raise ValueError(
                f"Geometry bridge request {key} requires exactly role and path."
            )
        if declaration["role"] != PathRole.EXTERNAL_ASSET.value:
            raise ValueError(
                f"Geometry bridge request {key}.role must be 'external_asset'."
            )
        raw_path = declaration["path"]
        if type(raw_path) is not str or not raw_path:
            raise TypeError(f"Geometry bridge request {key}.path must be a string.")
        resolved[key] = resolver.validate(PathRole.EXTERNAL_ASSET, Path(raw_path))
    cameras_bin = resolved["cameras_bin"]
    images_bin = resolved["images_bin"]
    points3d_bin = resolved["points3d_bin"]
    expected_names = {
        cameras_bin: "cameras.bin",
        images_bin: "images.bin",
        points3d_bin: "points3D.bin",
    }
    for candidate, expected_name in expected_names.items():
        if candidate.name != expected_name or not candidate.is_file():
            raise ValueError(f"Invalid COLMAP input path: {candidate}.")
    if len({path.parent for path in expected_names}) != 1:
        raise ValueError("COLMAP binary inputs must share one sparse directory.")
    return cameras_bin, images_bin, points3d_bin


def _image_world_to_camera(image: Any) -> NDArray[np.float64]:
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    matrix: NDArray[np.float64] = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
    return matrix


def _similarity_from_cameras(
    camera_to_world: NDArray[np.float64],
) -> NDArray[np.float64]:
    translations = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3]
    ups = np.sum(rotations * np.asarray([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camera = np.asarray([0.0, -1.0, 0.0])
    cosine = float((up_camera * world_up).sum())
    cross = np.cross(world_up, up_camera)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if cosine > -1.0:
        rotation_align = np.eye(3) + skew + (skew @ skew) * (1.0 / (1.0 + cosine))
    else:
        rotation_align = np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    rotations = rotation_align @ rotations
    forwards = np.sum(rotations * np.asarray([0.0, 0.0, 1.0]), axis=-1)
    translations = (rotation_align @ translations[..., None])[..., 0]
    nearest = translations + (forwards * -translations).sum(-1)[:, None] * forwards
    translate = -np.median(nearest, axis=0)

    transform: NDArray[np.float64] = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = rotation_align
    scale = 1.0 / np.median(np.linalg.norm(translations + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    centroid = np.median(points, axis=0)
    translated = points - centroid
    covariance = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation = eigenvectors.T
    transform: NDArray[np.float64] = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def _transform_points(
    matrix: NDArray[np.float64],
    points: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    transformed: NDArray[np.float64] = points @ matrix[:3, :3].T + matrix[:3, 3]
    return transformed


def _transform_cameras(
    matrix: NDArray[np.float64],
    cameras: NDArray[np.float64],
) -> NDArray[np.float64]:
    transformed: NDArray[np.float64] = np.einsum(
        "nij,ki->nkj",
        cameras,
        matrix,
    )
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    return transformed


def _normalize_scene(
    camera_to_world: NDArray[np.float64],
    points: NDArray[np.floating[Any]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    transform_1 = _similarity_from_cameras(camera_to_world)
    cameras = _transform_cameras(transform_1, camera_to_world)
    transformed_points = _transform_points(transform_1, points)
    transform_2 = _align_principal_axes(transformed_points)
    cameras = _transform_cameras(transform_2, cameras)
    transformed_points = _transform_points(transform_2, transformed_points)
    transform = transform_2 @ transform_1
    if np.median(transformed_points[:, 2]) > np.mean(transformed_points[:, 2]):
        transform_3 = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cameras = _transform_cameras(transform_3, cameras)
        transformed_points = _transform_points(transform_3, transformed_points)
        transform = transform_3 @ transform
    return cameras, transformed_points, transform


def _export_geometry(
    request_path: Path,
    output_path: Path,
    *,
    resolver: PathResolver,
) -> None:
    import pycolmap

    cameras_bin, _, _ = _load_request(request_path, resolver=resolver)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite geometry output: {output_path}")
    reconstruction = pycolmap.Reconstruction(str(cameras_bin.parent))
    registered_images = sorted(
        (
            reconstruction.images[image_id]
            for image_id in reconstruction.reg_image_ids()
        ),
        key=lambda image: image.name,
    )
    if not registered_images:
        raise ValueError("COLMAP reconstruction contains no registered images.")
    world_to_camera = np.stack(
        [_image_world_to_camera(image) for image in registered_images],
        axis=0,
    )
    camera_to_world = np.asarray(np.linalg.inv(world_to_camera), dtype=np.float64)
    points_by_id = reconstruction.points3D
    point_ids = sorted(points_by_id)
    points = np.asarray(
        [points_by_id[point_id].xyz for point_id in point_ids],
        dtype=np.float32,
    ).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("COLMAP reconstruction contains no 3D points.")
    cameras, normalized_points, transform = _normalize_scene(
        camera_to_world,
        points,
    )
    np.savez(
        output_path,
        camera_to_scene=np.asarray(cameras, dtype=np.float64),
        points_scene=np.asarray(normalized_points, dtype=np.float64),
        normalization=np.asarray(transform, dtype=np.float64),
        image_names=np.asarray(
            [image.name for image in registered_images],
            dtype=np.str_,
        ),
        camera_ids=np.asarray(
            [int(image.camera_id) for image in registered_images],
            dtype=np.int64,
        ),
        runtime_json=np.asarray(json.dumps(_runtime_versions(), sort_keys=True)),
    )


def run_geometry_bridge(
    request: Path,
    output: Path,
    *,
    resolver: PathResolver,
) -> None:
    """Validate the shared boundary and export provider geometry."""
    paths = PATH_BOUNDARY.validate(
        {"request": request, "output": output},
        resolver=resolver,
    )
    _export_geometry(
        paths.declared("request").path,
        paths.declared("output").path,
        resolver=resolver,
    )


__all__ = ["PATH_BOUNDARY", "_runtime_versions", "run_geometry_bridge"]
