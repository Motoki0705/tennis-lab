"""Read-only COLMAP/3DGS artifact exporter for scene-provider bundles.

This module verifies explicit files, delegates pycolmap normalization through
a temporary subprocess/file bridge, and reproduces gsplat undistortion. It
does not import gsplat, pycolmap, or any external application module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.scene_provider.bundle import (
    BundleFile,
    ExporterProvenance,
    ProviderImage,
    ProviderNormalization,
    SceneProviderBundle,
    compute_camera_array_sha256,
    compute_shared_intrinsics_sha256,
    load_scene_provider_bundle,
    sha256_file,
    write_scene_provider_bundle_manifest,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef, SceneCamera

_OPENCV_CAMERA_MODEL_ID = 4
_OPENCV_NUM_PARAMS = 8
_FRAME_INDEX_PATTERN = re.compile(r"^frame_(\d+)$")


@dataclass(frozen=True)
class SourceArtifactInput:
    """Explicit source artifact path and required digest."""

    artifact_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ProviderExportExpectations:
    """Frozen geometry values which the export must reproduce."""

    camera_count: int
    image_width: int
    image_height: int
    camera_array_sha256: str
    shared_intrinsics_sha256: str
    normalization_sha256: str


@dataclass(frozen=True)
class ProviderExportSettings:
    """All explicit inputs required for one provider export."""

    bundle_id: str
    provider_backend: str
    output_dir: Path
    cameras_bin: Path
    images_bin: Path
    points3d_bin: Path
    original_image_dir: Path
    factor_image_dir: Path
    geometry_python: Path
    geometry_bridge: Path
    factor: int
    group_size: int
    source_artifacts: tuple[SourceArtifactInput, ...]
    expectations: ProviderExportExpectations

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.provider_backend:
            raise ValueError("bundle_id and provider_backend must not be empty.")
        if self.factor <= 0:
            raise ValueError("factor must be positive.")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive.")
        if not self.source_artifacts:
            raise ValueError("source_artifacts must not be empty.")


@dataclass(frozen=True)
class _ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class _ColmapImage:
    image_id: int
    camera_id: int
    name: str
    world_to_camera: NDArray[np.float64]


@dataclass(frozen=True)
class _PreparedCameraModel:
    camera_id: int
    intrinsics: NDArray[np.float64]
    width: int
    height: int
    map_x: NDArray[np.float32]
    map_y: NDArray[np.float32]
    roi_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class _ProviderGeometry:
    camera_to_scene: NDArray[np.float64]
    points_scene: NDArray[np.float64]
    normalization: NDArray[np.float64]


def export_scene_provider_bundle(
    settings: ProviderExportSettings,
    *,
    exporter: ExporterProvenance,
) -> Path:
    """Export and atomically publish a verified pre-alignment provider bundle."""
    output_dir = settings.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite provider bundle: {output_dir}")
    source_artifacts = _verify_source_artifacts(settings.source_artifacts)
    _validate_input_files(settings)

    colmap_cameras = _read_cameras_binary(settings.cameras_bin)
    colmap_images = _read_images_binary(settings.images_bin)
    sorted_images = tuple(sorted(colmap_images, key=lambda item: item.name))
    geometry = _load_provider_geometry(
        settings,
        sorted_images=sorted_images,
        exporter=exporter,
    )
    factor_paths = _map_factor_images(
        settings.original_image_dir,
        settings.factor_image_dir,
    )
    prepared_models = _prepare_camera_models(
        colmap_cameras,
        factor=settings.factor,
        sample_factor_paths=factor_paths,
    )
    cameras = _build_scene_cameras(
        sorted_images,
        geometry.camera_to_scene,
        prepared_models,
        group_size=settings.group_size,
    )
    _validate_expectations(
        settings.expectations,
        cameras=cameras,
        normalization_matrix=geometry.normalization,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        images = _export_images(
            temporary_dir=temporary_dir,
            images=sorted_images,
            cameras=cameras,
            factor_paths=factor_paths,
            prepared_models=prepared_models,
        )
        point_cloud_path = temporary_dir / "points_scene.npy"
        np.save(point_cloud_path, geometry.points_scene)
        point_cloud_file = _bundle_file(
            point_cloud_path,
            root=temporary_dir,
        )
        normalization = ProviderNormalization(
            scene_from_source_world=tuple(
                float(value) for value in geometry.normalization.ravel()
            ),
            sha256=hashlib.sha256(geometry.normalization.tobytes()).hexdigest(),
        )
        manifest = SceneProviderBundle.create(
            bundle_id=settings.bundle_id,
            provider_backend=settings.provider_backend,
            source_artifacts=source_artifacts,
            cameras=cameras,
            images=images,
            point_cloud=point_cloud_file,
            point_cloud_shape=(
                int(geometry.points_scene.shape[0]),
                int(geometry.points_scene.shape[1]),
            ),
            normalization=normalization,
            exporter=exporter,
        )
        write_scene_provider_bundle_manifest(
            temporary_dir / "provider.json",
            manifest,
        )
        load_scene_provider_bundle(
            temporary_dir,
            verify_files=True,
            verify_source_artifacts=False,
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite provider bundle: {output_dir}"
            )
        os.rename(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return output_dir


def collect_exporter_provenance(
    *,
    repo_root: Path,
    code_paths: tuple[Path, ...],
    command: str,
    geometry_python: Path,
    geometry_bridge: Path,
) -> ExporterProvenance:
    """Collect git, code-content, and runtime identity for an export."""
    git_revision = _run_git(repo_root, "rev-parse", "HEAD")
    git_status = _run_git(repo_root, "status", "--porcelain")
    code_digest = hashlib.sha256()
    for path in sorted(code_paths, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            display_path = resolved.as_posix()
        code_digest.update(display_path.encode("utf-8"))
        code_digest.update(b"\0")
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                code_digest.update(chunk)
        code_digest.update(b"\0")
    geometry_runtime = _probe_geometry_runtime(
        geometry_python,
        geometry_bridge,
    )
    return ExporterProvenance(
        git_revision=git_revision,
        git_dirty=bool(git_status),
        code_sha256=code_digest.hexdigest(),
        command=command,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        opencv_version=cv2.__version__,
        geometry_python_version=geometry_runtime["python"],
        geometry_numpy_version=geometry_runtime["numpy"],
        geometry_pycolmap_version=geometry_runtime["pycolmap"],
    )


def _probe_geometry_runtime(
    geometry_python: Path,
    geometry_bridge: Path,
) -> dict[str, str]:
    if not geometry_python.is_file():
        raise FileNotFoundError(f"Missing provider geometry Python: {geometry_python}")
    if not geometry_bridge.is_file():
        raise FileNotFoundError(f"Missing provider geometry bridge: {geometry_bridge}")
    result = _run_geometry_subprocess(
        geometry_python,
        geometry_bridge,
        "--runtime-json",
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Provider geometry bridge returned invalid runtime JSON."
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "python",
        "numpy",
        "pycolmap",
    }:
        raise RuntimeError(
            "Provider geometry bridge runtime must contain python, numpy, and pycolmap."
        )
    runtime = {str(key): item for key, item in value.items()}
    if not all(isinstance(item, str) and item for item in runtime.values()):
        raise RuntimeError(
            "Provider geometry bridge runtime versions must be non-empty strings."
        )
    return runtime


def _load_provider_geometry(
    settings: ProviderExportSettings,
    *,
    sorted_images: tuple[_ColmapImage, ...],
    exporter: ExporterProvenance,
) -> _ProviderGeometry:
    with tempfile.TemporaryDirectory(prefix="tennis-scene-geometry-") as raw_dir:
        temporary_dir = Path(raw_dir)
        request_path = temporary_dir / "request.json"
        output_path = temporary_dir / "geometry.npz"
        request_path.write_text(
            json.dumps(
                {
                    "cameras_bin": str(settings.cameras_bin.resolve()),
                    "images_bin": str(settings.images_bin.resolve()),
                    "points3d_bin": str(settings.points3d_bin.resolve()),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _run_geometry_subprocess(
            settings.geometry_python,
            settings.geometry_bridge,
            str(request_path),
            str(output_path),
        )
        if not output_path.is_file():
            raise RuntimeError(
                "Provider geometry bridge did not create its declared output."
            )
        with np.load(output_path, allow_pickle=False) as payload:
            expected_keys = {
                "camera_to_scene",
                "points_scene",
                "normalization",
                "image_names",
                "camera_ids",
                "runtime_json",
            }
            if set(payload.files) != expected_keys:
                raise ValueError(
                    "Provider geometry response fields mismatch: "
                    f"expected {sorted(expected_keys)}, got {sorted(payload.files)}."
                )
            cameras = np.asarray(payload["camera_to_scene"], dtype=np.float64)
            points = np.asarray(payload["points_scene"], dtype=np.float64)
            normalization = np.asarray(
                payload["normalization"],
                dtype=np.float64,
            )
            image_names = tuple(str(value) for value in payload["image_names"].tolist())
            camera_ids = tuple(int(value) for value in payload["camera_ids"].tolist())
            runtime_raw = str(payload["runtime_json"].item())

    if cameras.shape != (len(sorted_images), 4, 4):
        raise ValueError(
            f"Provider geometry camera array shape mismatch: {cameras.shape}."
        )
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(
            f"Provider geometry point cloud must have shape [N,3], got {points.shape}."
        )
    if normalization.shape != (4, 4):
        raise ValueError(
            "Provider geometry normalization must have shape [4,4], "
            f"got {normalization.shape}."
        )
    if not all(np.isfinite(array).all() for array in (cameras, points, normalization)):
        raise ValueError("Provider geometry response contains non-finite values.")
    expected_names = tuple(image.name for image in sorted_images)
    expected_camera_ids = tuple(image.camera_id for image in sorted_images)
    if image_names != expected_names or camera_ids != expected_camera_ids:
        raise ValueError(
            "Provider geometry camera association does not match the independent "
            "COLMAP binary inventory."
        )
    try:
        runtime_value = json.loads(runtime_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Provider geometry response contains invalid runtime JSON."
        ) from exc
    expected_runtime = {
        "python": exporter.geometry_python_version,
        "numpy": exporter.geometry_numpy_version,
        "pycolmap": exporter.geometry_pycolmap_version,
    }
    if runtime_value != expected_runtime:
        raise ValueError(
            "Provider geometry runtime changed during export: "
            f"expected {expected_runtime}, got {runtime_value}."
        )
    return _ProviderGeometry(
        camera_to_scene=cameras,
        points_scene=points,
        normalization=normalization,
    )


def _run_geometry_subprocess(
    geometry_python: Path,
    geometry_bridge: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(geometry_python), str(geometry_bridge), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Provider geometry bridge failed "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
    return result


def _read_cameras_binary(path: Path) -> dict[int, _ColmapCamera]:
    with path.open("rb") as handle:
        count = _read_struct(handle, "<Q", context="camera count")[0]
        cameras: dict[int, _ColmapCamera] = {}
        for _ in range(int(count)):
            camera_id, model_id, width, height = _read_struct(
                handle,
                "<iiQQ",
                context="camera header",
            )
            if model_id != _OPENCV_CAMERA_MODEL_ID:
                raise ValueError(
                    "Provider export v1 requires COLMAP OPENCV cameras; "
                    f"camera {camera_id} has model id {model_id}."
                )
            params = _read_struct(
                handle,
                "<" + "d" * _OPENCV_NUM_PARAMS,
                context="camera parameters",
            )
            cameras[int(camera_id)] = _ColmapCamera(
                camera_id=int(camera_id),
                model_id=int(model_id),
                width=int(width),
                height=int(height),
                params=tuple(float(value) for value in params),
            )
        if handle.read(1):
            raise ValueError(f"Unexpected trailing bytes in cameras binary: {path}")
    if not cameras:
        raise ValueError("COLMAP cameras binary is empty.")
    return cameras


def _read_images_binary(path: Path) -> tuple[_ColmapImage, ...]:
    with path.open("rb") as handle:
        count = int(_read_struct(handle, "<Q", context="image count")[0])
        images: list[_ColmapImage] = []
        for _ in range(count):
            properties = _read_struct(
                handle,
                "<i" + "d" * 7 + "i",
                context="image properties",
            )
            image_id = int(properties[0])
            quaternion = np.asarray(properties[1:5], dtype=np.float64)
            translation = np.asarray(properties[5:8], dtype=np.float64)
            camera_id = int(properties[8])
            name = _read_c_string(handle, context=f"image {image_id} name")
            points2d_count = int(
                _read_struct(
                    handle,
                    "<Q",
                    context=f"image {image_id} point count",
                )[0]
            )
            handle.seek(points2d_count * 24, os.SEEK_CUR)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = _quaternion_to_rotation(quaternion)
            world_to_camera[:3, 3] = translation
            images.append(
                _ColmapImage(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    world_to_camera=world_to_camera,
                )
            )
        if handle.read(1):
            raise ValueError(f"Unexpected trailing bytes in images binary: {path}")
    if not images:
        raise ValueError("COLMAP images binary contains no registered images.")
    if len({image.name for image in images}) != len(images):
        raise ValueError("COLMAP registered image names are not unique.")
    return tuple(images)


def _read_points3d_binary(path: Path) -> NDArray[np.float32]:
    with path.open("rb") as handle:
        count = int(_read_struct(handle, "<Q", context="point count")[0])
        points_by_id: list[tuple[int, tuple[float, float, float]]] = []
        for _ in range(count):
            values = _read_struct(
                handle,
                "<QdddBBBd",
                context="point properties",
            )
            point_id = int(values[0])
            xyz = (float(values[1]), float(values[2]), float(values[3]))
            track_length = int(
                _read_struct(handle, "<Q", context="point track length")[0]
            )
            handle.seek(track_length * 8, os.SEEK_CUR)
            points_by_id.append((point_id, xyz))
        if handle.read(1):
            raise ValueError(f"Unexpected trailing bytes in points binary: {path}")
    if not points_by_id:
        raise ValueError("COLMAP point cloud is empty.")
    points_by_id.sort(key=lambda item: item[0])
    return np.asarray([item[1] for item in points_by_id], dtype=np.float32)


def _normalize_scene(
    camera_to_world: NDArray[np.float64],
    points: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
            ],
            dtype=np.float64,
        )
        cameras = _transform_cameras(transform_3, cameras)
        transformed_points = _transform_points(transform_3, transformed_points)
        transform = transform_3 @ transform
    return cameras, transformed_points, transform


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

    transform = np.eye(4)
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
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def _transform_points(
    matrix: NDArray[np.float64],
    points: NDArray[np.floating],
) -> NDArray[np.float64]:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(
    matrix: NDArray[np.float64],
    cameras: NDArray[np.float64],
) -> NDArray[np.float64]:
    transformed = np.einsum("nij,ki->nkj", cameras, matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    return transformed


def _prepare_camera_models(
    cameras: dict[int, _ColmapCamera],
    *,
    factor: int,
    sample_factor_paths: dict[str, Path],
) -> dict[int, _PreparedCameraModel]:
    if len(cameras) != 1:
        raise ValueError(
            f"Provider bundle v1 expects one shared camera, got {len(cameras)}."
        )
    sample_path = next(iter(sample_factor_paths.values()))
    sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if sample is None:
        raise ValueError(f"Failed to decode factor image: {sample_path}")
    actual_height, actual_width = sample.shape[:2]
    prepared: dict[int, _PreparedCameraModel] = {}
    for camera_id, camera in cameras.items():
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        intrinsics = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        intrinsics[:2, :] /= factor
        width = camera.width // factor
        height = camera.height // factor
        scale_width = actual_width / width
        scale_height = actual_height / height
        intrinsics[0, :] *= scale_width
        intrinsics[1, :] *= scale_height
        width = int(width * scale_width)
        height = int(height * scale_height)
        # gsplat's OPENCV adapter intentionally materializes distortion as
        # float32 before asking OpenCV for the undistorted camera matrix.
        distortion = np.asarray([k1, k2, p1, p2], dtype=np.float32)
        undistorted_intrinsics, raw_roi = cv2.getOptimalNewCameraMatrix(
            intrinsics,
            distortion,
            (width, height),
            0,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics,
            distortion,
            None,
            undistorted_intrinsics,
            (width, height),
            cv2.CV_32FC1,
        )
        roi = tuple(int(value) for value in raw_roi)
        prepared[camera_id] = _PreparedCameraModel(
            camera_id=camera_id,
            intrinsics=np.asarray(undistorted_intrinsics, dtype=np.float64),
            width=roi[2],
            height=roi[3],
            map_x=map_x,
            map_y=map_y,
            roi_xywh=(roi[0], roi[1], roi[2], roi[3]),
        )
    return prepared


def _build_scene_cameras(
    images: tuple[_ColmapImage, ...],
    camera_to_scene: NDArray[np.float64],
    models: dict[int, _PreparedCameraModel],
    *,
    group_size: int,
) -> tuple[SceneCamera, ...]:
    cameras: list[SceneCamera] = []
    for image, pose in zip(images, camera_to_scene, strict=True):
        model = models.get(image.camera_id)
        if model is None:
            raise ValueError(
                f"Image {image.name!r} references unknown camera {image.camera_id}."
            )
        stem = Path(image.name).stem
        match = _FRAME_INDEX_PATTERN.fullmatch(stem)
        if match is None:
            raise ValueError(
                f"Provider image name does not encode source frame index: {image.name!r}."
            )
        frame_index = int(match.group(1))
        cameras.append(
            SceneCamera(
                camera_id=stem,
                source_camera_id=f"colmap-{image.camera_id}",
                image_uri=f"images/{stem}.png",
                source_frame_index=frame_index,
                group_id=frame_index // group_size,
                width=model.width,
                height=model.height,
                intrinsics=tuple(float(value) for value in model.intrinsics.ravel()),
                camera_to_scene=tuple(float(value) for value in pose.ravel()),
            )
        )
    return tuple(cameras)


def _export_images(
    *,
    temporary_dir: Path,
    images: tuple[_ColmapImage, ...],
    cameras: tuple[SceneCamera, ...],
    factor_paths: dict[str, Path],
    prepared_models: dict[int, _PreparedCameraModel],
) -> tuple[ProviderImage, ...]:
    image_dir = temporary_dir / "images"
    image_dir.mkdir()
    records: list[ProviderImage] = []
    for image, camera in zip(images, cameras, strict=True):
        source_path = factor_paths.get(image.name)
        if source_path is None:
            raise FileNotFoundError(
                f"No factor image maps to registered image {image.name!r}."
            )
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"Failed to decode factor image: {source_path}")
        model = prepared_models[image.camera_id]
        remapped = cv2.remap(
            source,
            model.map_x,
            model.map_y,
            cv2.INTER_LINEAR,
        )
        x, y, width, height = model.roi_xywh
        cropped = remapped[y : y + height, x : x + width]
        if cropped.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"Processed image size mismatch for {image.name}: "
                f"{cropped.shape[:2]} vs {(camera.height, camera.width)}."
            )
        output_path = temporary_dir / camera.image_uri
        if not cv2.imwrite(
            str(output_path),
            cropped,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"Failed to write provider image: {output_path}")
        records.append(
            ProviderImage(
                camera_id=camera.camera_id,
                source_image_name=image.name,
                file=_bundle_file(output_path, root=temporary_dir),
            )
        )
    return tuple(records)


def _map_factor_images(
    original_image_dir: Path,
    factor_image_dir: Path,
) -> dict[str, Path]:
    original_rel = sorted(
        path.relative_to(original_image_dir).as_posix()
        for path in original_image_dir.rglob("*")
        if path.is_file()
    )
    factor_rel = sorted(
        path.relative_to(factor_image_dir).as_posix()
        for path in factor_image_dir.rglob("*")
        if path.is_file()
    )
    if not original_rel or len(original_rel) != len(factor_rel):
        raise ValueError(
            "Original and factor image inventories must be non-empty and equal: "
            f"{len(original_rel)} vs {len(factor_rel)}."
        )
    mapping: dict[str, Path] = {}
    for original_name, factor_name in zip(original_rel, factor_rel, strict=True):
        if Path(original_name).stem != Path(factor_name).stem:
            raise ValueError(
                f"Image inventory stem mismatch: {original_name!r} vs {factor_name!r}."
            )
        if Path(factor_name).suffix.lower() != ".png":
            raise ValueError(f"Factor image must be lossless PNG, got {factor_name!r}.")
        mapping[original_name] = factor_image_dir / factor_name
    return mapping


def _verify_source_artifacts(
    inputs: tuple[SourceArtifactInput, ...],
) -> tuple[ArtifactRef, ...]:
    artifacts: list[ArtifactRef] = []
    ids: set[str] = set()
    for item in inputs:
        if item.artifact_id in ids:
            raise ValueError(f"Duplicate source artifact id: {item.artifact_id!r}.")
        ids.add(item.artifact_id)
        path = item.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing source artifact: {path}")
        digest = sha256_file(path)
        if digest != item.sha256.lower():
            raise ValueError(
                f"Source artifact hash mismatch for {path}: "
                f"expected {item.sha256}, computed {digest}."
            )
        artifacts.append(
            ArtifactRef(
                artifact_id=item.artifact_id,
                uri=path.as_uri(),
                sha256=digest,
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(artifacts)


def _validate_input_files(settings: ProviderExportSettings) -> None:
    for path in (
        settings.cameras_bin,
        settings.images_bin,
        settings.points3d_bin,
        settings.geometry_python,
        settings.geometry_bridge,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing provider input file: {path}")
    for path in (settings.original_image_dir, settings.factor_image_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing provider image directory: {path}")


def _validate_expectations(
    expectations: ProviderExportExpectations,
    *,
    cameras: tuple[SceneCamera, ...],
    normalization_matrix: NDArray[np.float64],
) -> None:
    if len(cameras) != expectations.camera_count:
        raise ValueError(
            f"Provider camera count mismatch: expected {expectations.camera_count}, "
            f"got {len(cameras)}."
        )
    sizes = {(camera.width, camera.height) for camera in cameras}
    expected_size = (expectations.image_width, expectations.image_height)
    if sizes != {expected_size}:
        raise ValueError(
            f"Provider image size mismatch: expected {expected_size}, got {sizes}."
        )
    actual_hashes = {
        "camera array": compute_camera_array_sha256(cameras),
        "shared intrinsics": compute_shared_intrinsics_sha256(cameras),
        "normalization": hashlib.sha256(
            np.asarray(normalization_matrix, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    expected_hashes = {
        "camera array": expectations.camera_array_sha256.lower(),
        "shared intrinsics": expectations.shared_intrinsics_sha256.lower(),
        "normalization": expectations.normalization_sha256.lower(),
    }
    for name, actual in actual_hashes.items():
        expected = expected_hashes[name]
        if actual != expected:
            raise ValueError(
                f"Provider {name} hash mismatch: expected {expected}, got {actual}."
            )


def _bundle_file(path: Path, *, root: Path) -> BundleFile:
    return BundleFile(
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _quaternion_to_rotation(
    quaternion: NDArray[np.float64],
) -> NDArray[np.float64]:
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("COLMAP quaternion must be finite with shape [4].")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("COLMAP quaternion norm must be positive.")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * y * y - 2.0 * z * z,
                2.0 * x * y - 2.0 * w * z,
                2.0 * x * z + 2.0 * w * y,
            ],
            [
                2.0 * x * y + 2.0 * w * z,
                1.0 - 2.0 * x * x - 2.0 * z * z,
                2.0 * y * z - 2.0 * w * x,
            ],
            [
                2.0 * x * z - 2.0 * w * y,
                2.0 * y * z + 2.0 * w * x,
                1.0 - 2.0 * x * x - 2.0 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def _read_struct(
    handle: BinaryIO,
    format_string: str,
    *,
    context: str,
) -> tuple[int | float, ...]:
    size = struct.calcsize(format_string)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated COLMAP binary while reading {context}.")
    return cast(tuple[int | float, ...], struct.unpack(format_string, data))


def _read_c_string(handle: BinaryIO, *, context: str) -> str:
    value = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError(f"Truncated COLMAP binary while reading {context}.")
        if byte == b"\0":
            break
        value.extend(byte)
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 while reading {context}.") from exc


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


__all__ = [
    "ProviderExportExpectations",
    "ProviderExportSettings",
    "SourceArtifactInput",
    "collect_exporter_provenance",
    "export_scene_provider_bundle",
]
