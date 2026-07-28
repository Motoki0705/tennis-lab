"""Unit tests for the versioned scene-provider file boundary."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.scene_provider.bundle import (
    BundleFile,
    ExporterProvenance,
    ProviderImage,
    ProviderNormalization,
    SceneProviderBundle,
    load_scene_provider_bundle,
    sha256_file,
    write_scene_provider_bundle_manifest,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef, SceneCamera


def _camera(camera_id: str, frame_index: int) -> SceneCamera:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(frame_index) / 10.0
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="colmap-1",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=frame_index,
        group_id=frame_index // 32,
        width=16,
        height=12,
        intrinsics=(10.0, 0.0, 8.0, 0.0, 10.0, 6.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def _bundle_file(path: Path, root: Path) -> BundleFile:
    return BundleFile(
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _make_bundle(root: Path) -> SceneProviderBundle:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    cameras = (_camera("frame_000000", 0), _camera("frame_000032", 32))
    images: list[ProviderImage] = []
    for camera in cameras:
        path = root / camera.image_uri
        path.write_bytes(f"test-image-{camera.camera_id}".encode())
        images.append(
            ProviderImage(
                camera_id=camera.camera_id,
                source_image_name=f"{camera.camera_id}.jpg",
                file=_bundle_file(path, root),
            )
        )

    point_cloud_path = root / "points_scene.npy"
    np.save(
        point_cloud_path,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64),
    )
    normalization_matrix = np.eye(4, dtype=np.float64)
    normalization_matrix[:3, :3] *= 0.5
    normalization_matrix[:3, 3] = (1.0, 2.0, 3.0)
    normalization = ProviderNormalization(
        scene_from_source_world=tuple(
            float(value) for value in normalization_matrix.ravel()
        ),
        sha256=hashlib.sha256(normalization_matrix.tobytes()).hexdigest(),
    )
    return SceneProviderBundle.create(
        bundle_id="synthetic-provider-v1",
        provider_backend="test-provider@1",
        source_artifacts=(
            ArtifactRef(
                artifact_id="checkpoint",
                uri="artifact://test/checkpoint",
                sha256="a" * 64,
                size_bytes=1,
            ),
        ),
        cameras=cameras,
        images=tuple(images),
        point_cloud=_bundle_file(point_cloud_path, root),
        point_cloud_shape=(2, 3),
        normalization=normalization,
        exporter=ExporterProvenance(
            git_revision="deadbeef",
            git_dirty=True,
            code_sha256="b" * 64,
            command="python -m test",
            python_version="3.11",
            numpy_version="2.3",
            opencv_version="5.0",
            geometry_python_version="3.12",
            geometry_numpy_version="2.4",
            geometry_pycolmap_version="4.1",
        ),
    )


def test_bundle_round_trip_verifies_files_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    manifest_path = tmp_path / "provider.json"

    write_scene_provider_bundle_manifest(manifest_path, bundle)
    loaded = load_scene_provider_bundle(tmp_path)

    assert loaded.manifest == bundle
    assert loaded.image_path("frame_000032").name == "frame_000032.png"
    assert loaded.point_cloud_path().name == "points_scene.npy"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_scene_provider_bundle_manifest(manifest_path, bundle)


def test_bundle_detects_tampered_file(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    write_scene_provider_bundle_manifest(tmp_path / "provider.json", bundle)
    (tmp_path / bundle.images[0].file.relative_path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="file size mismatch|file hash mismatch"):
        load_scene_provider_bundle(tmp_path)


def test_bundle_rejects_camera_image_mismatch(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    bad_image = replace(bundle.images[0], camera_id="unknown-camera")

    with pytest.raises(ValueError, match="one-to-one"):
        replace(bundle, images=(bad_image, bundle.images[1]))


def test_bundle_rejects_non_shared_intrinsics(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    changed_intrinsics = list(bundle.cameras[1].intrinsics)
    changed_intrinsics[0] += 1.0
    bad_camera = replace(
        bundle.cameras[1],
        intrinsics=tuple(changed_intrinsics),
    )

    with pytest.raises(ValueError, match="identical shared intrinsics"):
        SceneProviderBundle.create(
            bundle_id=bundle.bundle_id,
            provider_backend=bundle.provider_backend,
            source_artifacts=bundle.source_artifacts,
            cameras=(bundle.cameras[0], bad_camera),
            images=bundle.images,
            point_cloud=bundle.point_cloud,
            point_cloud_shape=bundle.point_cloud_shape,
            normalization=bundle.normalization,
            exporter=bundle.exporter,
        )


def test_bundle_file_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="Invalid bundle-relative path"):
        BundleFile(relative_path="../outside.npy", sha256="a" * 64, size_bytes=1)
