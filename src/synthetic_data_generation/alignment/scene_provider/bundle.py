"""Versioned scene-provider bundle contract owned by the alignment boundary.

Provider bundles are the file boundary between an external reconstruction and
tennis-lab alignment. They contain calibrated cameras, exact processed images,
normalized point-cloud support, and content-addressed source artifacts, but no
court alignment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self
from urllib.parse import unquote, urlparse

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import (
    CAMERA_AXES_OPENCV,
    PIXEL_COORDINATES,
    ArtifactRef,
    SceneCamera,
    SimilarityTransform,
    compute_scene_fingerprint,
)

SCENE_PROVIDER_BUNDLE_SCHEMA = "tennis_scene_provider_bundle_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MATRIX_ATOL = 1.0e-6


@dataclass(frozen=True)
class BundleFile:
    """One immutable file stored relative to the bundle root."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        digest = _digest(self.sha256, name="bundle file sha256")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("Bundle file size_bytes must be non-negative.")
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict bundle-file record."""
        raw = _strict_mapping(
            value,
            name="bundle file",
            keys={"relative_path", "sha256", "size_bytes"},
        )
        return cls(
            relative_path=_string(raw["relative_path"], name="relative_path"),
            sha256=_string(raw["sha256"], name="sha256"),
            size_bytes=_integer(raw["size_bytes"], name="size_bytes"),
        )


@dataclass(frozen=True)
class ProviderImage:
    """Processed RGB image associated one-to-one with a scene camera."""

    camera_id: str
    source_image_name: str
    file: BundleFile

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("Provider image camera_id must not be empty.")
        if not self.source_image_name or Path(self.source_image_name).name != (
            self.source_image_name
        ):
            raise ValueError("source_image_name must be a plain file name.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "camera_id": self.camera_id,
            "source_image_name": self.source_image_name,
            "file": self.file.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict provider-image record."""
        raw = _strict_mapping(
            value,
            name="provider image",
            keys={"camera_id", "source_image_name", "file"},
        )
        return cls(
            camera_id=_string(raw["camera_id"], name="camera_id"),
            source_image_name=_string(
                raw["source_image_name"],
                name="source_image_name",
            ),
            file=BundleFile.from_dict(raw["file"]),
        )


@dataclass(frozen=True)
class ProviderNormalization:
    """Exact homogeneous Sim(3) used to normalize COLMAP world coordinates."""

    scene_from_source_world: tuple[float, ...]
    sha256: str

    def __post_init__(self) -> None:
        matrix_values = _number_sequence(
            self.scene_from_source_world,
            length=16,
            name="scene_from_source_world",
        )
        matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
        if not np.allclose(
            matrix[3],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            atol=_MATRIX_ATOL,
            rtol=0.0,
        ):
            raise ValueError("Provider normalization must be homogeneous.")
        scale = float(np.linalg.norm(matrix[0, :3]))
        if scale <= 0.0:
            raise ValueError("Provider normalization scale must be positive.")
        rotation = matrix[:3, :3] / scale
        SimilarityTransform(
            scale=scale,
            rotation=tuple(float(item) for item in rotation.ravel()),
            translation=tuple(float(item) for item in matrix[:3, 3]),
        )
        expected = hashlib.sha256(matrix.tobytes()).hexdigest()
        digest = _digest(self.sha256, name="normalization sha256")
        if digest != expected:
            raise ValueError(
                "Provider normalization hash mismatch: "
                f"declared {digest}, computed {expected}."
            )
        object.__setattr__(self, "scene_from_source_world", matrix_values)
        object.__setattr__(self, "sha256", digest)

    def matrix(self) -> NDArray[np.float64]:
        """Return the exact float64 homogeneous transform."""
        matrix: NDArray[np.float64] = np.asarray(
            self.scene_from_source_world,
            dtype=np.float64,
        )
        return matrix.reshape(4, 4)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "scene_from_source_world": list(self.scene_from_source_world),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict normalization record."""
        raw = _strict_mapping(
            value,
            name="provider normalization",
            keys={"scene_from_source_world", "sha256"},
        )
        return cls(
            scene_from_source_world=_number_sequence(
                raw["scene_from_source_world"],
                length=16,
                name="scene_from_source_world",
            ),
            sha256=_string(raw["sha256"], name="sha256"),
        )


@dataclass(frozen=True)
class ExporterProvenance:
    """Code and runtime identity of the one-way provider export."""

    git_revision: str
    git_dirty: bool
    code_sha256: str
    command: str
    python_version: str
    numpy_version: str
    opencv_version: str
    geometry_python_version: str
    geometry_numpy_version: str
    geometry_pycolmap_version: str

    def __post_init__(self) -> None:
        if not self.git_revision:
            raise ValueError("Exporter git_revision must not be empty.")
        _digest(self.code_sha256, name="exporter code_sha256")
        for name, value in (
            ("command", self.command),
            ("python_version", self.python_version),
            ("numpy_version", self.numpy_version),
            ("opencv_version", self.opencv_version),
            ("geometry_python_version", self.geometry_python_version),
            ("geometry_numpy_version", self.geometry_numpy_version),
            ("geometry_pycolmap_version", self.geometry_pycolmap_version),
        ):
            if not value:
                raise ValueError(f"Exporter {name} must not be empty.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "git_revision": self.git_revision,
            "git_dirty": self.git_dirty,
            "code_sha256": self.code_sha256,
            "command": self.command,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "geometry_python_version": self.geometry_python_version,
            "geometry_numpy_version": self.geometry_numpy_version,
            "geometry_pycolmap_version": self.geometry_pycolmap_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict exporter-provenance record."""
        raw = _strict_mapping(
            value,
            name="exporter provenance",
            keys={
                "git_revision",
                "git_dirty",
                "code_sha256",
                "command",
                "python_version",
                "numpy_version",
                "opencv_version",
                "geometry_python_version",
                "geometry_numpy_version",
                "geometry_pycolmap_version",
            },
        )
        return cls(
            git_revision=_string(raw["git_revision"], name="git_revision"),
            git_dirty=_boolean(raw["git_dirty"], name="git_dirty"),
            code_sha256=_string(raw["code_sha256"], name="code_sha256"),
            command=_string(raw["command"], name="command"),
            python_version=_string(raw["python_version"], name="python_version"),
            numpy_version=_string(raw["numpy_version"], name="numpy_version"),
            opencv_version=_string(raw["opencv_version"], name="opencv_version"),
            geometry_python_version=_string(
                raw["geometry_python_version"],
                name="geometry_python_version",
            ),
            geometry_numpy_version=_string(
                raw["geometry_numpy_version"],
                name="geometry_numpy_version",
            ),
            geometry_pycolmap_version=_string(
                raw["geometry_pycolmap_version"],
                name="geometry_pycolmap_version",
            ),
        )


@dataclass(frozen=True)
class SceneProviderBundle:
    """Strict pre-alignment provider manifest consumed by alignment code."""

    schema: str
    bundle_id: str
    bundle_fingerprint: str
    scene_fingerprint: str
    provider_backend: str
    camera_axes: str
    pixel_coordinates: str
    image_color_space: str
    source_artifacts: tuple[ArtifactRef, ...]
    cameras: tuple[SceneCamera, ...]
    images: tuple[ProviderImage, ...]
    point_cloud: BundleFile
    point_cloud_shape: tuple[int, int]
    normalization: ProviderNormalization
    camera_array_sha256: str
    shared_intrinsics_sha256: str
    image_set_sha256: str
    exporter: ExporterProvenance

    def __post_init__(self) -> None:
        if self.schema != SCENE_PROVIDER_BUNDLE_SCHEMA:
            raise ValueError(
                f"Unsupported provider bundle schema {self.schema!r}; "
                f"expected {SCENE_PROVIDER_BUNDLE_SCHEMA!r}."
            )
        if not self.bundle_id or not self.provider_backend:
            raise ValueError("bundle_id and provider_backend must not be empty.")
        if self.camera_axes != CAMERA_AXES_OPENCV:
            raise ValueError(f"Unsupported camera axes: {self.camera_axes!r}.")
        if self.pixel_coordinates != PIXEL_COORDINATES:
            raise ValueError(
                f"Unsupported pixel coordinates: {self.pixel_coordinates!r}."
            )
        if self.image_color_space != "srgb8":
            raise ValueError(
                f"Unsupported provider image color space: {self.image_color_space!r}."
            )
        artifacts = tuple(self.source_artifacts)
        cameras = tuple(self.cameras)
        images = tuple(self.images)
        point_cloud_shape = tuple(self.point_cloud_shape)
        if not artifacts or not cameras or not images:
            raise ValueError(
                "Provider artifacts, cameras, and images must all be non-empty."
            )
        if len(cameras) != len(images):
            raise ValueError("Provider camera/image counts must match.")
        if len(point_cloud_shape) != 2 or point_cloud_shape[1] != 3:
            raise ValueError("point_cloud_shape must be [N, 3].")
        if point_cloud_shape[0] <= 0:
            raise ValueError("Provider point cloud must be non-empty.")
        _require_unique(
            [artifact.artifact_id for artifact in artifacts],
            name="source artifact ids",
        )
        camera_ids = [camera.camera_id for camera in cameras]
        image_camera_ids = [image.camera_id for image in images]
        _require_unique(camera_ids, name="camera ids")
        _require_unique(image_camera_ids, name="image camera ids")
        if set(camera_ids) != set(image_camera_ids):
            raise ValueError("Provider images must map one-to-one to cameras.")
        images_by_camera = {image.camera_id: image for image in images}
        for camera in cameras:
            image = images_by_camera[camera.camera_id]
            if camera.image_uri != image.file.relative_path:
                raise ValueError(
                    f"Camera {camera.camera_id!r} image_uri does not match its file."
                )

        camera_hash = compute_camera_array_sha256(cameras)
        intrinsics_hash = compute_shared_intrinsics_sha256(cameras)
        image_hash = compute_image_set_sha256(images)
        scene_hash = compute_scene_fingerprint(
            provider_backend=self.provider_backend,
            artifacts=artifacts,
            cameras=cameras,
        )
        declared_scene_hash = _digest(
            self.scene_fingerprint,
            name="scene_fingerprint",
        )
        declared_camera_hash = _digest(
            self.camera_array_sha256,
            name="camera_array_sha256",
        )
        declared_intrinsics_hash = _digest(
            self.shared_intrinsics_sha256,
            name="shared_intrinsics_sha256",
        )
        declared_image_hash = _digest(
            self.image_set_sha256,
            name="image_set_sha256",
        )
        for name, declared, computed in (
            ("scene", declared_scene_hash, scene_hash),
            ("camera array", declared_camera_hash, camera_hash),
            ("shared intrinsics", declared_intrinsics_hash, intrinsics_hash),
            ("image set", declared_image_hash, image_hash),
        ):
            if declared != computed:
                raise ValueError(
                    f"Provider {name} hash mismatch: "
                    f"declared {declared}, computed {computed}."
                )
        expected_bundle_hash = compute_bundle_fingerprint(
            bundle_id=self.bundle_id,
            scene_fingerprint=scene_hash,
            image_set_sha256=image_hash,
            point_cloud=self.point_cloud,
            normalization=self.normalization,
            exporter=self.exporter,
        )
        declared_bundle_hash = _digest(
            self.bundle_fingerprint,
            name="bundle_fingerprint",
        )
        if declared_bundle_hash != expected_bundle_hash:
            raise ValueError(
                "Provider bundle fingerprint mismatch: "
                f"declared {declared_bundle_hash}, computed {expected_bundle_hash}."
            )
        object.__setattr__(self, "source_artifacts", artifacts)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "point_cloud_shape", point_cloud_shape)
        object.__setattr__(self, "scene_fingerprint", declared_scene_hash)
        object.__setattr__(self, "camera_array_sha256", declared_camera_hash)
        object.__setattr__(
            self,
            "shared_intrinsics_sha256",
            declared_intrinsics_hash,
        )
        object.__setattr__(self, "image_set_sha256", declared_image_hash)
        object.__setattr__(self, "bundle_fingerprint", declared_bundle_hash)

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        provider_backend: str,
        source_artifacts: Sequence[ArtifactRef],
        cameras: Sequence[SceneCamera],
        images: Sequence[ProviderImage],
        point_cloud: BundleFile,
        point_cloud_shape: tuple[int, int],
        normalization: ProviderNormalization,
        exporter: ExporterProvenance,
    ) -> Self:
        """Create a provider bundle with all canonical fingerprints."""
        artifacts_tuple = tuple(source_artifacts)
        camera_tuple = tuple(cameras)
        image_tuple = tuple(images)
        scene_hash = compute_scene_fingerprint(
            provider_backend=provider_backend,
            artifacts=artifacts_tuple,
            cameras=camera_tuple,
        )
        image_hash = compute_image_set_sha256(image_tuple)
        bundle_hash = compute_bundle_fingerprint(
            bundle_id=bundle_id,
            scene_fingerprint=scene_hash,
            image_set_sha256=image_hash,
            point_cloud=point_cloud,
            normalization=normalization,
            exporter=exporter,
        )
        return cls(
            schema=SCENE_PROVIDER_BUNDLE_SCHEMA,
            bundle_id=bundle_id,
            bundle_fingerprint=bundle_hash,
            scene_fingerprint=scene_hash,
            provider_backend=provider_backend,
            camera_axes=CAMERA_AXES_OPENCV,
            pixel_coordinates=PIXEL_COORDINATES,
            image_color_space="srgb8",
            source_artifacts=artifacts_tuple,
            cameras=camera_tuple,
            images=image_tuple,
            point_cloud=point_cloud,
            point_cloud_shape=point_cloud_shape,
            normalization=normalization,
            camera_array_sha256=compute_camera_array_sha256(camera_tuple),
            shared_intrinsics_sha256=compute_shared_intrinsics_sha256(camera_tuple),
            image_set_sha256=image_hash,
            exporter=exporter,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "schema": self.schema,
            "bundle_id": self.bundle_id,
            "bundle_fingerprint": self.bundle_fingerprint,
            "scene_fingerprint": self.scene_fingerprint,
            "provider_backend": self.provider_backend,
            "camera_axes": self.camera_axes,
            "pixel_coordinates": self.pixel_coordinates,
            "image_color_space": self.image_color_space,
            "source_artifacts": [
                artifact.to_dict() for artifact in self.source_artifacts
            ],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "images": [image.to_dict() for image in self.images],
            "point_cloud": self.point_cloud.to_dict(),
            "point_cloud_shape": list(self.point_cloud_shape),
            "normalization": self.normalization.to_dict(),
            "camera_array_sha256": self.camera_array_sha256,
            "shared_intrinsics_sha256": self.shared_intrinsics_sha256,
            "image_set_sha256": self.image_set_sha256,
            "exporter": self.exporter.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and validate a strict v1 provider bundle."""
        raw = _strict_mapping(
            value,
            name="provider bundle",
            keys={
                "schema",
                "bundle_id",
                "bundle_fingerprint",
                "scene_fingerprint",
                "provider_backend",
                "camera_axes",
                "pixel_coordinates",
                "image_color_space",
                "source_artifacts",
                "cameras",
                "images",
                "point_cloud",
                "point_cloud_shape",
                "normalization",
                "camera_array_sha256",
                "shared_intrinsics_sha256",
                "image_set_sha256",
                "exporter",
            },
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            bundle_id=_string(raw["bundle_id"], name="bundle_id"),
            bundle_fingerprint=_string(
                raw["bundle_fingerprint"],
                name="bundle_fingerprint",
            ),
            scene_fingerprint=_string(
                raw["scene_fingerprint"],
                name="scene_fingerprint",
            ),
            provider_backend=_string(
                raw["provider_backend"],
                name="provider_backend",
            ),
            camera_axes=_string(raw["camera_axes"], name="camera_axes"),
            pixel_coordinates=_string(
                raw["pixel_coordinates"],
                name="pixel_coordinates",
            ),
            image_color_space=_string(
                raw["image_color_space"],
                name="image_color_space",
            ),
            source_artifacts=tuple(
                ArtifactRef.from_dict(item)
                for item in _sequence(
                    raw["source_artifacts"],
                    name="source_artifacts",
                )
            ),
            cameras=tuple(
                SceneCamera.from_dict(item)
                for item in _sequence(raw["cameras"], name="cameras")
            ),
            images=tuple(
                ProviderImage.from_dict(item)
                for item in _sequence(raw["images"], name="images")
            ),
            point_cloud=BundleFile.from_dict(raw["point_cloud"]),
            point_cloud_shape=_integer_pair(
                raw["point_cloud_shape"],
                name="point_cloud_shape",
            ),
            normalization=ProviderNormalization.from_dict(raw["normalization"]),
            camera_array_sha256=_string(
                raw["camera_array_sha256"],
                name="camera_array_sha256",
            ),
            shared_intrinsics_sha256=_string(
                raw["shared_intrinsics_sha256"],
                name="shared_intrinsics_sha256",
            ),
            image_set_sha256=_string(
                raw["image_set_sha256"],
                name="image_set_sha256",
            ),
            exporter=ExporterProvenance.from_dict(raw["exporter"]),
        )


@dataclass(frozen=True)
class LoadedSceneProviderBundle:
    """A validated manifest paired with its resolved bundle root."""

    root: Path
    manifest: SceneProviderBundle

    def image_path(self, camera_id: str) -> Path:
        """Resolve the processed image for one camera id."""
        for image in self.manifest.images:
            if image.camera_id == camera_id:
                return _resolve_bundle_file(self.root, image.file)
        raise KeyError(f"Unknown provider camera id: {camera_id!r}.")

    def point_cloud_path(self) -> Path:
        """Resolve the normalized scene point-cloud file."""
        return _resolve_bundle_file(self.root, self.manifest.point_cloud)


def compute_camera_array_sha256(cameras: Sequence[SceneCamera]) -> str:
    """Hash cameras in manifest order as canonical float64 ``[N,4,4]``."""
    array = np.asarray(
        [camera.camera_to_scene for camera in cameras],
        dtype=np.float64,
    ).reshape(-1, 4, 4)
    return hashlib.sha256(array.tobytes()).hexdigest()


def compute_shared_intrinsics_sha256(cameras: Sequence[SceneCamera]) -> str:
    """Hash the shared camera K after requiring identical intrinsics."""
    if not cameras:
        raise ValueError("Cannot hash an empty camera sequence.")
    first = np.asarray(cameras[0].intrinsics, dtype=np.float64).reshape(3, 3)
    for camera in cameras[1:]:
        candidate = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        if not np.array_equal(candidate, first):
            raise ValueError("Provider bundle v1 requires identical shared intrinsics.")
    return hashlib.sha256(first.tobytes()).hexdigest()


def compute_image_set_sha256(images: Sequence[ProviderImage]) -> str:
    """Hash the ordered image inventory without reading image bytes."""
    payload = [image.to_dict() for image in sorted(images, key=lambda x: x.camera_id)]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_bundle_fingerprint(
    *,
    bundle_id: str,
    scene_fingerprint: str,
    image_set_sha256: str,
    point_cloud: BundleFile,
    normalization: ProviderNormalization,
    exporter: ExporterProvenance,
) -> str:
    """Compute the full pre-alignment bundle identity."""
    payload = {
        "schema": SCENE_PROVIDER_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "scene_fingerprint": scene_fingerprint,
        "image_set_sha256": image_set_sha256,
        "point_cloud": point_cloud.to_dict(),
        "normalization": normalization.to_dict(),
        "exporter": exporter.to_dict(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_scene_provider_bundle(
    path: str | Path,
    *,
    verify_files: bool = True,
    verify_source_artifacts: bool = False,
) -> LoadedSceneProviderBundle:
    """Load a bundle directory or manifest and optionally verify its files."""
    candidate = Path(path)
    manifest_path = candidate / "provider.json" if candidate.is_dir() else candidate
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = SceneProviderBundle.from_dict(json.load(handle))
    root = manifest_path.parent.resolve()
    loaded = LoadedSceneProviderBundle(root=root, manifest=manifest)
    if verify_files:
        for image in manifest.images:
            _verify_bundle_file(root, image.file)
        _verify_bundle_file(root, manifest.point_cloud)
        points = np.load(loaded.point_cloud_path(), allow_pickle=False, mmap_mode="r")
        if points.dtype != np.float64 or points.shape != manifest.point_cloud_shape:
            raise ValueError(
                "Normalized point cloud does not match declared float64 shape: "
                f"dtype={points.dtype}, shape={points.shape}."
            )
    if verify_source_artifacts:
        for artifact in manifest.source_artifacts:
            _verify_source_artifact(artifact)
    return loaded


def write_scene_provider_bundle_manifest(
    path: str | Path,
    manifest: SceneProviderBundle,
) -> None:
    """Atomically write ``provider.json`` and refuse replacement."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, manifest_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite provider manifest: {manifest_path}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle_file(root: Path, file: BundleFile) -> None:
    path = _resolve_bundle_file(root, file)
    if not path.is_file():
        raise FileNotFoundError(f"Missing provider bundle file: {path}")
    if path.stat().st_size != file.size_bytes:
        raise ValueError(f"Provider bundle file size mismatch: {path}")
    digest = sha256_file(path)
    if digest != file.sha256:
        raise ValueError(
            f"Provider bundle file hash mismatch for {path}: "
            f"declared {file.sha256}, computed {digest}."
        )


def _verify_source_artifact(artifact: ArtifactRef) -> None:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(
            f"Cannot verify non-local source artifact URI: {artifact.uri!r}."
        )
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"Missing source artifact: {path}")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"Source artifact size mismatch: {path}")
    digest = sha256_file(path)
    if digest != artifact.sha256:
        raise ValueError(
            f"Source artifact hash mismatch for {path}: "
            f"declared {artifact.sha256}, computed {digest}."
        )


def _resolve_bundle_file(root: Path, file: BundleFile) -> Path:
    path = (root / file.relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Bundle file escapes root: {file.relative_path!r}.")
    return path


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Invalid bundle-relative path: {value!r}.")


def _digest(value: str, *, name: str) -> str:
    digest = value.lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} is not a SHA-256 digest: {value!r}.")
    return digest


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _number_sequence(
    value: object,
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    sequence = _sequence(value, name=name)
    if len(sequence) != length:
        raise ValueError(f"{name} must contain {length} numbers.")
    values: list[float] = []
    for item in sequence:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only numbers.")
        number = float(item)
        if not np.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers.")
        values.append(number)
    return tuple(values)


def _integer_pair(value: object, *, name: str) -> tuple[int, int]:
    sequence = _sequence(value, name=name)
    if len(sequence) != 2:
        raise ValueError(f"{name} must contain two integers.")
    return (
        _integer(sequence[0], name=f"{name}[0]"),
        _integer(sequence[1], name=f"{name}[1]"),
    )


def _require_unique(values: Sequence[str], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")


__all__ = [
    "BundleFile",
    "ExporterProvenance",
    "LoadedSceneProviderBundle",
    "ProviderImage",
    "ProviderNormalization",
    "SCENE_PROVIDER_BUNDLE_SCHEMA",
    "SceneProviderBundle",
    "compute_bundle_fingerprint",
    "compute_camera_array_sha256",
    "compute_image_set_sha256",
    "compute_shared_intrinsics_sha256",
    "load_scene_provider_bundle",
    "sha256_file",
    "write_scene_provider_bundle_manifest",
]
