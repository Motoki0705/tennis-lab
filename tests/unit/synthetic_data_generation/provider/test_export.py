"""Unit tests for the dependency-isolated COLMAP provider exporter."""

from __future__ import annotations

import ast
import inspect
import struct
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.provider.export as provider_export_module
from src.synthetic_data_generation.provider.export import (
    _map_factor_images,
    _quaternion_to_rotation,
    _read_cameras_binary,
    _read_images_binary,
    _read_points3d_binary,
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
