"""Unit tests for the dependency-isolated COLMAP provider exporter."""

from __future__ import annotations

import ast
import hashlib
import inspect
import struct
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.alignment.scene_provider.export as provider_export_module
from src.synthetic_data_generation.alignment.scene_provider.export import (
    ProviderExportExpectations,
    ProviderExportSettings,
    SourceArtifactInput,
    _map_factor_images,
    _quaternion_to_rotation,
    _read_cameras_binary,
    _read_images_binary,
    _read_points3d_binary,
)
from src.synthetic_data_generation.configuration import VerifiedSystemExecutable
from src.utils.configuration import (
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)


def test_reads_minimal_colmap_binary_contract(tmp_path: Path) -> None:
    cameras_path = tmp_path / "cameras.bin"
    cameras_path.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<iiQQ", 1, 4, 1920, 1080)
        + struct.pack(
            "<8d",
            1000.0,
            1001.0,
            960.0,
            540.0,
            0.01,
            -0.02,
            0.001,
            -0.001,
        )
    )
    images_path = tmp_path / "images.bin"
    images_path.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack(
            "<i7di",
            7,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            1,
        )
        + b"frame_000007.jpg\0"
        + struct.pack("<Q", 0)
    )
    points_path = tmp_path / "points3D.bin"
    points_path.write_bytes(
        struct.pack("<Q", 2)
        + struct.pack("<QdddBBBdQ", 9, 1.0, 2.0, 3.0, 1, 2, 3, 0.5, 0)
        + struct.pack("<QdddBBBdQ", 2, -1.0, 4.0, 5.0, 4, 5, 6, 0.2, 0)
    )

    cameras = _read_cameras_binary(cameras_path)
    images = _read_images_binary(images_path)
    points = _read_points3d_binary(points_path)

    assert cameras[1].params[:4] == (1000.0, 1001.0, 960.0, 540.0)
    assert images[0].name == "frame_000007.jpg"
    np.testing.assert_array_equal(images[0].world_to_camera[:3, 3], (1, 2, 3))
    np.testing.assert_array_equal(
        points,
        np.asarray([[-1.0, 4.0, 5.0], [1.0, 2.0, 3.0]], dtype=np.float32),
    )


def test_rejects_unsupported_colmap_camera_model(tmp_path: Path) -> None:
    path = tmp_path / "cameras.bin"
    path.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<iiQQ", 1, 1, 1920, 1080)
        + struct.pack("<8d", *([0.0] * 8))
    )

    with pytest.raises(ValueError, match="requires COLMAP OPENCV cameras"):
        _read_cameras_binary(path)


def test_quaternion_conversion_rejects_zero_and_recovers_rotation() -> None:
    with pytest.raises(ValueError, match="norm must be positive"):
        _quaternion_to_rotation(np.zeros(4, dtype=np.float64))

    angle = np.pi / 2.0
    quaternion = np.asarray(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float64,
    )
    rotation = _quaternion_to_rotation(quaternion)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(
        rotation @ np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        atol=1.0e-12,
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_factor_mapping_requires_one_to_one_lossless_inventory(
    tmp_path: Path,
) -> None:
    originals = tmp_path / "images"
    factors = tmp_path / "images_2"
    originals.mkdir()
    factors.mkdir()
    (originals / "frame_000000.jpg").write_bytes(b"original")
    (factors / "frame_000000.png").write_bytes(b"factor")

    mapping = _map_factor_images(originals, factors)
    assert mapping == {"frame_000000.jpg": factors / "frame_000000.png"}

    (factors / "frame_000001.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventories must be non-empty and equal"):
        _map_factor_images(originals, factors)


def test_provider_export_has_no_external_application_imports() -> None:
    tree = ast.parse(inspect.getsource(provider_export_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden = ("gsplat", "pycolmap", "src.tasks")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden
    )


def _resolver(tmp_path: Path) -> PathResolver:
    roots = {
        f"{role.value}_root": str((tmp_path / f"{role.value}-root").resolve())
        for role in PathRole
    }
    return PathResolver(
        RuntimePathRoots.from_mapping(
            roots,
            repository_root=tmp_path.resolve(),
        )
    )


def _system_executable(tmp_path: Path) -> VerifiedSystemExecutable:
    root = (tmp_path / "system-bin").resolve()
    root.mkdir()
    path = root / "geometry-python"
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return VerifiedSystemExecutable(
        root=root,
        relative_path=Path(path.name),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _settings(
    tmp_path: Path,
    *,
    source_artifact: Path,
    geometry_executable: VerifiedSystemExecutable,
) -> ProviderExportSettings:
    resolver = _resolver(tmp_path)
    asset_scope = resolver.resolve(PathRole.EXTERNAL_ASSET, "provider")
    return ProviderExportSettings(
        bundle_id="bundle",
        provider_backend="provider",
        output_dir=resolver.resolve(PathRole.DATA, "provider-bundle"),
        external_asset_scope=asset_scope,
        cameras_bin=resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            "provider/scene/cameras.bin",
        ),
        images_bin=resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            "provider/scene/images.bin",
        ),
        points3d_bin=resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            "provider/scene/points3D.bin",
        ),
        original_image_dir=resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            "provider/scene/images",
        ),
        factor_image_dir=resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            "provider/scene/images_2",
        ),
        geometry_executable=geometry_executable,
        geometry_bridge=resolver.resolve(PathRole.PROJECT, "src/geometry_bridge.py"),
        resolver=resolver,
        factor=2,
        group_size=32,
        source_artifacts=(
            SourceArtifactInput(
                artifact_id="source",
                path=source_artifact,
                sha256="0" * 64,
            ),
        ),
        expectations=ProviderExportExpectations(
            camera_count=1,
            image_width=1,
            image_height=1,
            camera_array_sha256="0" * 64,
            shared_intrinsics_sha256="0" * 64,
            normalization_sha256="0" * 64,
        ),
    )


def test_export_settings_reject_out_of_root_source_before_output_creation(
    tmp_path: Path,
) -> None:
    executable = _system_executable(tmp_path)
    output = tmp_path / "data-root/provider-bundle"

    with pytest.raises(PathContractError, match="outside its root"):
        _settings(
            tmp_path,
            source_artifact=(tmp_path / "outside/source.bin").resolve(),
            geometry_executable=executable,
        )

    assert not output.exists()


def test_export_settings_reject_source_outside_narrow_asset_scope(
    tmp_path: Path,
) -> None:
    executable = _system_executable(tmp_path)
    sibling_source = (
        tmp_path / "external_asset-root/sibling-project/source.bin"
    ).resolve()
    output = tmp_path / "data-root/provider-bundle"

    with pytest.raises(PathContractError, match="outside its declared scope"):
        _settings(
            tmp_path,
            source_artifact=sibling_source,
            geometry_executable=executable,
        )

    assert not output.exists()


def test_export_settings_recheck_executable_digest_before_output_creation(
    tmp_path: Path,
) -> None:
    executable = _system_executable(tmp_path)
    executable.path.write_bytes(b"tampered\n")
    resolver = _resolver(tmp_path)
    source_artifact = resolver.resolve(
        PathRole.EXTERNAL_ASSET,
        "provider/scene/source.bin",
    )
    output = tmp_path / "data-root/provider-bundle"

    with pytest.raises(PathContractError, match="SHA-256 mismatch"):
        _settings(
            tmp_path,
            source_artifact=source_artifact,
            geometry_executable=executable,
        )

    assert not output.exists()
