"""Tests for strict GLB loading and metric tennis-ball normalization."""

from __future__ import annotations

import io
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallAssetMetadata,
    BLCSBallMeshAsset,
    BLCSBallRendering,
)
from src.synthetic_data_generation.dataset.blcs.mesh_asset import (
    _primitive_colors,
    load_ball_mesh_asset,
)


def test_glb_mesh_is_recentered_and_scaled_to_physical_radius(
    tmp_path: Path,
    blcs_assets,
) -> None:
    path = tmp_path / "offset tennis ball.glb"
    _write_tetrahedron_glb(path, translation=(4.0, -3.0, 8.0))
    assets = replace(
        blcs_assets,
        rendering=BLCSBallRendering.MESH,
        mesh=BLCSBallMeshAsset(
            path=path,
            data_root_relative_path="assets/blcs/offset tennis ball.glb",
            maximum_file_bytes=4096,
            maximum_source_vertices=4,
            maximum_source_faces=4,
            maximum_faces=4,
        ),
    )

    mesh = load_ball_mesh_asset(assets)

    assert mesh.source_vertex_count == 4
    assert mesh.source_face_count == 4
    assert mesh.vertices_m.shape == (4, 3)
    assert mesh.faces.shape == (4, 3)
    np.testing.assert_allclose(
        (mesh.vertices_m.min(axis=0) + mesh.vertices_m.max(axis=0)) / 2.0,
        0.0,
        atol=1.0e-7,
    )
    assert np.linalg.norm(mesh.vertices_m, axis=1).max() == pytest.approx(
        assets.settings.radius_m,
        abs=2.0e-7,
    )
    np.testing.assert_allclose(
        mesh.colors_linear_rgb[0],
        (0.5, 0.75, 0.25),
        atol=1.0e-6,
    )


def test_mesh_metadata_records_only_the_data_root_relative_glb_source(
    tmp_path: Path,
    blcs_assets,
) -> None:
    path = tmp_path / "ball.glb"
    _write_tetrahedron_glb(path)
    assets = replace(
        blcs_assets,
        rendering=BLCSBallRendering.MESH,
        mesh=BLCSBallMeshAsset(
            path=path,
            data_root_relative_path="synthetic_data_generation/assets/blcs/ball.glb",
            maximum_file_bytes=4096,
            maximum_source_vertices=16,
            maximum_source_faces=16,
            maximum_faces=16,
        ),
    )

    metadata = assets.metadata().to_dict()

    assert metadata["rendering"] == "mesh"
    assert metadata["radius_m"] == assets.settings.radius_m
    assert metadata["source"] == {
        "format": "glb",
        "appearance_model": "glb_base_color_lambertian_v1",
        "data_root_relative_path": "synthetic_data_generation/assets/blcs/ball.glb",
        "maximum_file_bytes": 4096,
        "maximum_source_vertices": 16,
        "maximum_source_faces": 16,
        "maximum_faces": 16,
    }
    assert str(tmp_path) not in json.dumps(metadata)
    assert BLCSBallAssetMetadata.from_dict(metadata).to_dict() == metadata


def test_mesh_mode_rejects_missing_or_non_glb_assets(
    tmp_path: Path,
    blcs_assets,
) -> None:
    with pytest.raises(FileNotFoundError, match="ordinary existing"):
        BLCSBallMeshAsset(
            path=tmp_path / "missing.glb",
            data_root_relative_path="assets/blcs/missing.glb",
            maximum_file_bytes=4096,
            maximum_source_vertices=16,
            maximum_source_faces=16,
            maximum_faces=16,
        )
    wrong_format = tmp_path / "ball.obj"
    wrong_format.write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".glb"):
        BLCSBallMeshAsset(
            path=wrong_format,
            data_root_relative_path="assets/blcs/ball.obj",
            maximum_file_bytes=4096,
            maximum_source_vertices=16,
            maximum_source_faces=16,
            maximum_faces=16,
        )
    with pytest.raises(TypeError, match="requires one"):
        replace(blcs_assets, rendering=BLCSBallRendering.MESH)


def test_mesh_resource_limits_reject_file_and_instanced_geometry_before_accessors(
    tmp_path: Path,
    blcs_assets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "instanced.glb"
    _write_tetrahedron_glb(path, node_count=2)
    with pytest.raises(ValueError, match="maximum_file_bytes"):
        BLCSBallMeshAsset(
            path=path,
            data_root_relative_path="assets/blcs/instanced.glb",
            maximum_file_bytes=1,
            maximum_source_vertices=8,
            maximum_source_faces=8,
            maximum_faces=4,
        )
    assets = replace(
        blcs_assets,
        rendering=BLCSBallRendering.MESH,
        mesh=BLCSBallMeshAsset(
            path=path,
            data_root_relative_path="assets/blcs/instanced.glb",
            maximum_file_bytes=4096,
            maximum_source_vertices=4,
            maximum_source_faces=4,
            maximum_faces=4,
        ),
    )

    def forbidden_accessor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("accessor arrays must not be materialized")

    monkeypatch.setattr(
        "src.synthetic_data_generation.dataset.blcs.mesh_asset._accessor",
        forbidden_accessor,
    )
    with pytest.raises(ValueError, match="maximum_source_vertices"):
        load_ball_mesh_asset(assets)
    assert assets.mesh is not None
    face_limited = replace(
        assets,
        mesh=replace(
            assets.mesh,
            maximum_source_vertices=8,
            maximum_source_faces=4,
        ),
    )
    with pytest.raises(ValueError, match="maximum_source_faces"):
        load_ball_mesh_asset(face_limited)


def test_texture_is_decoded_before_linear_base_color_factor_multiplication() -> None:
    document, binary = _textured_color_document(
        texture_srgb=(128, 128, 128),
        factor=(0.5, 0.25, 1.0, 1.0),
    )

    colors = _primitive_colors(
        document,
        binary,
        {"material": 0},
        {"TEXCOORD_0": 0},
        vertex_count=1,
    )

    sampled_linear = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4
    np.testing.assert_allclose(
        colors[0],
        sampled_linear * np.asarray((0.5, 0.25, 1.0)),
        atol=1.0e-8,
    )


def test_linear_color_zero_multiplies_material_and_is_not_ignored() -> None:
    values = np.asarray(((0.2, 0.4, 0.5),), dtype=np.float32)
    document, binary = _vertex_color_document(
        values,
        normalized=False,
        factor=(0.5, 0.25, 1.0, 1.0),
    )

    colors = _primitive_colors(
        document,
        binary,
        {"material": 0},
        {"COLOR_0": 0},
        vertex_count=1,
    )

    np.testing.assert_allclose(colors[0], (0.1, 0.1, 0.5), atol=1.0e-7)


def test_normalized_integer_color_zero_is_supported_without_a_material() -> None:
    values = np.asarray(((128, 64, 255, 255),), dtype=np.uint8)
    document, binary = _vertex_color_document(values, normalized=True, factor=None)

    colors = _primitive_colors(
        document,
        binary,
        {},
        {"COLOR_0": 0},
        vertex_count=1,
    )

    np.testing.assert_allclose(colors[0], np.asarray((128, 64, 255)) / 255.0)


@pytest.mark.parametrize(
    ("values", "normalized", "message"),
    [
        (
            np.asarray(((128, 64, 255),), dtype=np.uint8),
            False,
            "must be normalized",
        ),
        (
            np.asarray(((128, 64, 255, 254),), dtype=np.uint8),
            True,
            "alpha must equal 1",
        ),
        (
            np.asarray(((1.1, 0.5, 0.5),), dtype=np.float32),
            False,
            "finite and in \\[0,1\\]",
        ),
    ],
)
def test_color_zero_rejects_ambiguous_or_non_opaque_values(
    values: np.ndarray,
    normalized: bool,
    message: str,
) -> None:
    document, binary = _vertex_color_document(
        values, normalized=normalized, factor=None
    )

    with pytest.raises(ValueError, match=message):
        _primitive_colors(
            document,
            binary,
            {},
            {"COLOR_0": 0},
            vertex_count=1,
        )


def _write_tetrahedron_glb(
    path: Path,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    node_count: int = 1,
) -> None:
    positions = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ),
        dtype=np.float32,
    )
    indices = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.uint16,
    ).reshape(-1)
    position_bytes = positions.tobytes()
    index_offset = len(position_bytes)
    binary = position_bytes + indices.tobytes()
    while len(binary) % 4:
        binary += b"\x00"
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(node_count))}],
        "nodes": [
            {"mesh": 0, "translation": list(translation)}
            for _index in range(node_count)
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.75, 0.25, 1.0]}}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 12,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": indices.nbytes,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    encoded_json = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded_json += b" " * ((-len(encoded_json)) % 4)
    total_length = 12 + 8 + len(encoded_json) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(encoded_json), 0x4E4F534A)
        + encoded_json
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _textured_color_document(
    *,
    texture_srgb: tuple[int, int, int],
    factor: tuple[float, float, float, float],
) -> tuple[dict[str, object], bytes]:
    encoded = io.BytesIO()
    Image.new("RGB", (1, 1), texture_srgb).save(encoded, format="PNG")
    uv = np.asarray(((0.0, 0.0),), dtype=np.float32).tobytes()
    image = encoded.getvalue()
    binary = uv + image
    return (
        {
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorFactor": list(factor),
                        "baseColorTexture": {"index": 0},
                    }
                }
            ],
            "textures": [{"source": 0}],
            "images": [{"bufferView": 1, "mimeType": "image/png"}],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 1,
                    "type": "VEC2",
                }
            ],
            "bufferViews": [
                {"buffer": 0, "byteLength": len(uv)},
                {"buffer": 0, "byteOffset": len(uv), "byteLength": len(image)},
            ],
        },
        binary,
    )


def _vertex_color_document(
    values: np.ndarray,
    *,
    normalized: bool,
    factor: tuple[float, float, float, float] | None,
) -> tuple[dict[str, object], bytes]:
    component_type = {
        np.dtype(np.float32): 5126,
        np.dtype(np.uint8): 5121,
        np.dtype(np.uint16): 5123,
    }[values.dtype]
    accessor: dict[str, object] = {
        "bufferView": 0,
        "componentType": component_type,
        "count": len(values),
        "type": f"VEC{values.shape[1]}",
    }
    if normalized:
        accessor["normalized"] = True
    document: dict[str, object] = {
        "accessors": [accessor],
        "bufferViews": [
            {"buffer": 0, "byteLength": values.nbytes},
        ],
    }
    if factor is not None:
        document["materials"] = [
            {"pbrMetallicRoughness": {"baseColorFactor": list(factor)}}
        ]
    return document, values.tobytes()
