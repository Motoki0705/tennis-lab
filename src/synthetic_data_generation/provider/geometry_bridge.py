"""Provider-side COLMAP geometry bridge for an isolated Python subprocess.

This adapter deliberately imports pycolmap only inside the configured provider
environment. Its only interface is a strict JSON request and a temporary NPZ
response consumed by :mod:`src.synthetic_data_generation.provider.export`.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap  # type: ignore[import-not-found]
from numpy.typing import NDArray

_REQUEST_KEYS = {"cameras_bin", "images_bin", "points3d_bin"}


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pycolmap": str(pycolmap.__version__),
    }


def _load_request(path: Path) -> tuple[Path, Path, Path]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("Geometry bridge request must be a JSON object.")
    raw = {str(key): item for key, item in value.items()}
    if set(raw) != _REQUEST_KEYS or not all(
        isinstance(raw[key], str) for key in _REQUEST_KEYS
    ):
        raise ValueError(
            "Geometry bridge request must contain exactly three string paths."
        )
    cameras_bin = Path(raw["cameras_bin"]).resolve()
    images_bin = Path(raw["images_bin"]).resolve()
    points3d_bin = Path(raw["points3d_bin"]).resolve()
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


def _export_geometry(request_path: Path, output_path: Path) -> None:
    cameras_bin, _, _ = _load_request(request_path)
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
    camera_to_world = np.linalg.inv(world_to_camera)
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


def main() -> int:
    """Run the provider-side geometry bridge."""
    if sys.argv[1:] == ["--runtime-json"]:
        print(json.dumps(_runtime_versions(), sort_keys=True))
        return 0
    if len(sys.argv) != 3:
        raise ValueError("Usage: provider_geometry_bridge.py REQUEST.json OUTPUT.npz")
    _export_geometry(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
