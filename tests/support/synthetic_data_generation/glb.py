"""Small GLB fixtures shared by synthetic-data-generation tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_tetrahedron_glb(
    path: Path,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    node_count: int = 1,
) -> None:
    """Write a minimal embedded GLB containing a colored tetrahedron."""
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
