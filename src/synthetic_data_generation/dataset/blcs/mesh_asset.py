"""Strict GLB loading and metric normalization for BLCS tennis-ball meshes."""

from __future__ import annotations

import io
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.dataset.blcs.contracts import BLCSCompositionAssets

_GLB_MAGIC = b"glTF"
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BINARY_CHUNK = 0x004E4942
_TRIANGLES_MODE = 4
_COMPONENT_DTYPES: dict[int, np.dtype[np.generic]] = {
    5121: np.dtype(np.uint8),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}
_ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass(frozen=True, slots=True)
class BLCSBallMesh:
    """One centered, radius-normalized and bounded-complexity triangle mesh."""

    vertices_m: NDArray[np.float32]
    faces: NDArray[np.int64]
    normals: NDArray[np.float32]
    colors_linear_rgb: NDArray[np.float32]
    source_vertex_count: int
    source_face_count: int
    source_path: Path

    def __post_init__(self) -> None:
        vertices = _array(self.vertices_m, np.float32, 2, "vertices_m")
        faces = _array(self.faces, np.int64, 2, "faces")
        normals = _array(self.normals, np.float32, 2, "normals")
        colors = _array(
            self.colors_linear_rgb,
            np.float32,
            2,
            "colors_linear_rgb",
        )
        if vertices.shape[1:] != (3,) or len(vertices) < 4:
            raise ValueError("BLCS mesh vertices must have shape [V,3] with V >= 4.")
        if faces.shape[1:] != (3,) or len(faces) < 4:
            raise ValueError("BLCS mesh faces must have shape [F,3] with F >= 4.")
        if normals.shape != vertices.shape or colors.shape != vertices.shape:
            raise ValueError("BLCS mesh normals/colors must match its vertices.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("BLCS mesh faces contain out-of-range vertex indices.")
        if any(not np.isfinite(value).all() for value in (vertices, normals, colors)):
            raise ValueError("BLCS mesh arrays must contain only finite values.")
        if np.any(colors < 0.0) or np.any(colors > 1.0):
            raise ValueError("BLCS mesh colors must be linear RGB in [0,1].")
        normal_lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(normal_lengths, 1.0, atol=2.0e-4, rtol=0.0):
            raise ValueError("BLCS mesh normals must be unit length.")
        if (
            isinstance(self.source_vertex_count, bool)
            or self.source_vertex_count < len(vertices)
            or isinstance(self.source_face_count, bool)
            or self.source_face_count < len(faces)
        ):
            raise ValueError("BLCS mesh source counts are inconsistent.")
        if not self.source_path.is_absolute() or not self.source_path.is_file():
            raise FileNotFoundError(
                "BLCS mesh source_path must identify an existing file."
            )
        for value in (vertices, faces, normals, colors):
            value.setflags(write=False)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "colors_linear_rgb", colors)


def load_ball_mesh_asset(assets: BLCSCompositionAssets) -> BLCSBallMesh:
    """Load the configured GLB, then recenter and scale it to ``radius_m``."""
    if not isinstance(assets, BLCSCompositionAssets):
        raise TypeError("assets must be BLCSCompositionAssets.")
    if assets.mesh is None:
        raise ValueError("BLCS mesh loading requires assets.rendering=mesh.")
    document, binary = _read_glb(assets.mesh.path)
    _validate_scene_resource_limits(
        document,
        maximum_source_vertices=assets.mesh.maximum_source_vertices,
        maximum_source_faces=assets.mesh.maximum_source_faces,
    )
    vertices, faces, colors = _collect_scene_mesh(document, binary)
    source_vertex_count = len(vertices)
    source_face_count = len(faces)
    normalized = _center_and_scale(vertices, radius_m=assets.settings.radius_m)
    simplified_vertices, simplified_faces, simplified_colors = (
        _simplify_vertex_clusters(
            normalized,
            faces,
            colors,
            maximum_faces=assets.mesh.maximum_faces,
        )
    )
    simplified_vertices = _center_and_scale(
        simplified_vertices,
        radius_m=assets.settings.radius_m,
    )
    normals = _vertex_normals(simplified_vertices, simplified_faces)
    result = BLCSBallMesh(
        vertices_m=simplified_vertices.astype(np.float32, copy=False),
        faces=simplified_faces.astype(np.int64, copy=False),
        normals=normals,
        colors_linear_rgb=simplified_colors.astype(np.float32, copy=False),
        source_vertex_count=source_vertex_count,
        source_face_count=source_face_count,
        source_path=assets.mesh.path,
    )
    measured_radius = float(np.linalg.norm(result.vertices_m, axis=1).max())
    if not math.isclose(
        measured_radius,
        assets.settings.radius_m,
        abs_tol=2.0e-6,
        rel_tol=1.0e-5,
    ):
        raise ValueError("Normalized BLCS mesh radius disagrees with radius_m.")
    return result


def _read_glb(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Unable to read BLCS GLB asset: {path}") from error
    if len(payload) < 20:
        raise ValueError("BLCS GLB asset is truncated.")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != _GLB_MAGIC or version != 2 or declared_length != len(payload):
        raise ValueError("BLCS mesh must be a complete glTF 2.0 binary (.glb) file.")
    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("BLCS GLB contains a truncated chunk header.")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        stop = offset + chunk_length
        if stop > len(payload) or chunk_type in chunks:
            raise ValueError("BLCS GLB contains invalid or duplicate chunks.")
        chunks[chunk_type] = payload[offset:stop]
        offset = stop
    if set(chunks) != {_GLB_JSON_CHUNK, _GLB_BINARY_CHUNK}:
        raise ValueError("BLCS GLB requires exactly one JSON and one binary chunk.")
    try:
        decoded = json.loads(chunks[_GLB_JSON_CHUNK].decode("utf-8").rstrip("\x00 "))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("BLCS GLB JSON chunk is invalid.") from error
    if not isinstance(decoded, Mapping) or decoded.get("asset") is None:
        raise ValueError("BLCS GLB JSON document is invalid.")
    asset = _mapping(decoded["asset"], name="asset")
    if asset.get("version") != "2.0":
        raise ValueError("BLCS GLB asset.version must be exactly '2.0'.")
    required = decoded.get("extensionsRequired", [])
    if required:
        raise ValueError(
            f"BLCS GLB requires unsupported extensions: {_string_sequence(required)}."
        )
    buffers = _sequence(decoded.get("buffers"), name="buffers")
    if len(buffers) != 1:
        raise ValueError("BLCS GLB must contain exactly one embedded binary buffer.")
    buffer_record = _mapping(buffers[0], name="buffers[0]")
    if "uri" in buffer_record or _integer(
        buffer_record.get("byteLength", -1), name="buffers[0].byteLength"
    ) > len(chunks[_GLB_BINARY_CHUNK]):
        raise ValueError("BLCS GLB buffer must be fully embedded in the binary chunk.")
    return cast(Mapping[str, object], decoded), chunks[_GLB_BINARY_CHUNK]


def _validate_scene_resource_limits(
    document: Mapping[str, object],
    *,
    maximum_source_vertices: int,
    maximum_source_faces: int,
) -> None:
    """Bound scene-instanced geometry before accessor arrays are materialized."""
    scenes = _sequence(document.get("scenes"), name="scenes")
    scene_index = _integer(document.get("scene"), name="scene")
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError("BLCS GLB default scene index is out of range.")
    nodes = _sequence(document.get("nodes"), name="nodes")
    meshes = _sequence(document.get("meshes"), name="meshes")
    accessors = _sequence(document.get("accessors"), name="accessors")
    roots = _integer_sequence(
        _mapping(scenes[scene_index], name="default scene").get("nodes"),
        name="scene.nodes",
    )
    if not roots:
        raise ValueError("BLCS GLB default scene has no root nodes.")
    vertex_count = 0
    face_count = 0
    active: set[int] = set()

    def accessor_count(value: object, *, name: str) -> int:
        accessor_index = _integer(value, name=name)
        if accessor_index < 0 or accessor_index >= len(accessors):
            raise ValueError(f"BLCS GLB {name} is out of range.")
        accessor = _mapping(
            accessors[accessor_index],
            name=f"accessors[{accessor_index}]",
        )
        count = _integer(accessor.get("count"), name="accessor.count")
        if count <= 0:
            raise ValueError("BLCS GLB accessor count must be positive.")
        return count

    def visit(node_index: int) -> None:
        nonlocal vertex_count, face_count
        if node_index < 0 or node_index >= len(nodes):
            raise ValueError("BLCS GLB node index is out of range.")
        if node_index in active:
            raise ValueError("BLCS GLB node graph contains a cycle.")
        active.add(node_index)
        node = _mapping(nodes[node_index], name=f"nodes[{node_index}]")
        if "mesh" in node:
            mesh_index = _integer(node["mesh"], name=f"nodes[{node_index}].mesh")
            if mesh_index < 0 or mesh_index >= len(meshes):
                raise ValueError("BLCS GLB mesh index is out of range.")
            mesh = _mapping(meshes[mesh_index], name=f"meshes[{mesh_index}]")
            primitives = _sequence(mesh.get("primitives"), name="mesh.primitives")
            for primitive_index, value in enumerate(primitives):
                primitive = _mapping(
                    value,
                    name=f"mesh.primitives[{primitive_index}]",
                )
                if primitive.get("mode", _TRIANGLES_MODE) != _TRIANGLES_MODE:
                    raise ValueError(
                        "BLCS GLB supports only indexed triangle primitives."
                    )
                if "indices" not in primitive:
                    raise ValueError("BLCS GLB triangle primitives must be indexed.")
                attributes = _mapping(
                    primitive.get("attributes"),
                    name="primitive.attributes",
                )
                if "POSITION" not in attributes:
                    raise ValueError("BLCS GLB primitive is missing POSITION data.")
                primitive_vertices = accessor_count(
                    attributes["POSITION"],
                    name="POSITION accessor",
                )
                index_count = accessor_count(
                    primitive["indices"],
                    name="indices accessor",
                )
                if index_count % 3 != 0:
                    raise ValueError(
                        "BLCS GLB triangle index count must be divisible by three."
                    )
                vertex_count += primitive_vertices
                face_count += index_count // 3
                if vertex_count > maximum_source_vertices:
                    raise ValueError(
                        "BLCS GLB exceeds maximum_source_vertices after scene "
                        "instancing."
                    )
                if face_count > maximum_source_faces:
                    raise ValueError(
                        "BLCS GLB exceeds maximum_source_faces after scene instancing."
                    )
        for child in _integer_sequence(node.get("children", []), name="node.children"):
            visit(child)
        active.remove(node_index)

    for root in roots:
        visit(root)


def _collect_scene_mesh(
    document: Mapping[str, object],
    binary: bytes,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    scenes = _sequence(document.get("scenes"), name="scenes")
    scene_index = _integer(document.get("scene"), name="scene")
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError("BLCS GLB default scene index is out of range.")
    nodes = _sequence(document.get("nodes"), name="nodes")
    meshes = _sequence(document.get("meshes"), name="meshes")
    roots = _integer_sequence(
        _mapping(scenes[scene_index], name="default scene").get("nodes"),
        name="scene.nodes",
    )
    if not roots:
        raise ValueError("BLCS GLB default scene has no root nodes.")
    vertices: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    colors: list[NDArray[np.float64]] = []
    active: set[int] = set()

    def visit(node_index: int, parent_from_node_parent: NDArray[np.float64]) -> None:
        if node_index < 0 or node_index >= len(nodes):
            raise ValueError("BLCS GLB node index is out of range.")
        if node_index in active:
            raise ValueError("BLCS GLB node graph contains a cycle.")
        active.add(node_index)
        node = _mapping(nodes[node_index], name=f"nodes[{node_index}]")
        scene_from_node = parent_from_node_parent @ _node_transform(node)
        if "mesh" in node:
            mesh_index = _integer(node["mesh"], name=f"nodes[{node_index}].mesh")
            if mesh_index < 0 or mesh_index >= len(meshes):
                raise ValueError("BLCS GLB mesh index is out of range.")
            mesh_vertices, mesh_faces, mesh_colors = _mesh_data(
                document,
                binary,
                _mapping(meshes[mesh_index], name=f"meshes[{mesh_index}]"),
                scene_from_node,
            )
            vertex_offset = sum(len(value) for value in vertices)
            vertices.append(mesh_vertices)
            faces.append(mesh_faces + vertex_offset)
            colors.append(mesh_colors)
        for child in _integer_sequence(node.get("children", []), name="node.children"):
            visit(child, scene_from_node)
        active.remove(node_index)

    for root in roots:
        visit(root, np.eye(4, dtype=np.float64))
    if not vertices:
        raise ValueError("BLCS GLB default scene contains no triangle mesh primitives.")
    combined_vertices = np.concatenate(vertices, axis=0)
    combined_faces = np.concatenate(faces, axis=0)
    combined_colors = np.concatenate(colors, axis=0)
    if len(combined_vertices) < 4 or len(combined_faces) < 4:
        raise ValueError("BLCS GLB needs at least four vertices and four triangles.")
    return combined_vertices, combined_faces, combined_colors


def _mesh_data(
    document: Mapping[str, object],
    binary: bytes,
    mesh: Mapping[str, object],
    scene_from_node: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    primitives = _sequence(mesh.get("primitives"), name="mesh.primitives")
    vertex_parts: list[NDArray[np.float64]] = []
    face_parts: list[NDArray[np.int64]] = []
    color_parts: list[NDArray[np.float64]] = []
    for primitive_index, value in enumerate(primitives):
        primitive = _mapping(value, name=f"mesh.primitives[{primitive_index}]")
        if primitive.get("mode", _TRIANGLES_MODE) != _TRIANGLES_MODE:
            raise ValueError("BLCS GLB supports only indexed triangle primitives.")
        if primitive.get("extensions"):
            raise ValueError("BLCS GLB primitive extensions are unsupported.")
        if "indices" not in primitive:
            raise ValueError("BLCS GLB triangle primitives must be indexed.")
        attributes = _mapping(primitive.get("attributes"), name="primitive.attributes")
        if "POSITION" not in attributes:
            raise ValueError("BLCS GLB primitive is missing POSITION data.")
        positions = _accessor(
            document,
            binary,
            _integer(attributes["POSITION"], name="POSITION accessor"),
        )
        if (
            positions.dtype != np.float32
            or positions.ndim != 2
            or positions.shape[1] != 3
        ):
            raise TypeError("BLCS GLB POSITION accessor must be float32 VEC3.")
        indices = _accessor(
            document,
            binary,
            _integer(primitive["indices"], name="indices accessor"),
        )
        if indices.ndim != 1 or indices.dtype not in {
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.uint32),
        }:
            raise TypeError("BLCS GLB indices must be unsigned SCALAR values.")
        if len(indices) % 3 != 0:
            raise ValueError(
                "BLCS GLB triangle index count must be divisible by three."
            )
        triangles = indices.astype(np.int64).reshape(-1, 3)
        if np.any(triangles >= len(positions)):
            raise ValueError("BLCS GLB primitive contains out-of-range indices.")
        homogeneous = np.concatenate(
            (positions.astype(np.float64), np.ones((len(positions), 1))), axis=1
        )
        transformed = (homogeneous @ scene_from_node.T)[:, :3]
        primitive_colors = _primitive_colors(
            document,
            binary,
            primitive,
            attributes,
            vertex_count=len(positions),
        )
        offset = sum(len(item) for item in vertex_parts)
        vertex_parts.append(transformed)
        face_parts.append(triangles + offset)
        color_parts.append(primitive_colors)
    if not vertex_parts:
        raise ValueError("BLCS GLB mesh has no primitives.")
    return (
        np.concatenate(vertex_parts),
        np.concatenate(face_parts),
        np.concatenate(color_parts),
    )


def _primitive_colors(
    document: Mapping[str, object],
    binary: bytes,
    primitive: Mapping[str, object],
    attributes: Mapping[str, object],
    *,
    vertex_count: int,
) -> NDArray[np.float64]:
    materials = _sequence(document.get("materials", []), name="materials")
    factor: NDArray[np.float64] = np.ones(4, dtype=np.float64)
    texture_record: object | None = None
    if "material" not in primitive:
        colors = np.repeat(factor[None, :3], vertex_count, axis=0)
    else:
        material_index = _integer(primitive["material"], name="material")
        if material_index < 0 or material_index >= len(materials):
            raise ValueError("BLCS GLB material index is out of range.")
        material = _mapping(materials[material_index], name="material")
        if material.get("alphaMode", "OPAQUE") != "OPAQUE":
            raise ValueError("BLCS mesh rendering supports only opaque GLB materials.")
        pbr = _mapping(
            material.get("pbrMetallicRoughness", {}),
            name="material.pbr",
        )
        factor_values = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))
        factor = np.asarray(
            _number_sequence(factor_values, name="baseColorFactor"),
            dtype=np.float64,
        )
        if factor.shape != (4,) or np.any(factor < 0.0) or np.any(factor > 1.0):
            raise ValueError(
                "BLCS GLB baseColorFactor must contain four values in [0,1]."
            )
        if factor[3] != 1.0:
            raise ValueError(
                "BLCS GLB opaque material baseColorFactor alpha must equal 1."
            )
        texture_record = pbr.get("baseColorTexture")
        colors = np.repeat(factor[None, :3], vertex_count, axis=0)
    if texture_record is not None:
        texture_info = _mapping(texture_record, name="baseColorTexture")
        if texture_info.get("extensions"):
            raise ValueError("BLCS GLB base-color texture extensions are unsupported.")
        texcoord_set = texture_info.get("texCoord", 0)
        if texcoord_set != 0 or "TEXCOORD_0" not in attributes:
            raise ValueError("BLCS GLB textured material requires TEXCOORD_0.")
        uv = _accessor(
            document,
            binary,
            _integer(attributes["TEXCOORD_0"], name="TEXCOORD_0 accessor"),
        )
        if uv.shape != (vertex_count, 2):
            raise ValueError("BLCS GLB TEXCOORD_0 must match POSITION as VEC2.")
        uv_values = _normalized_accessor_values(
            uv,
            document,
            attributes["TEXCOORD_0"],
        )
        image, wrap_s, wrap_t = _base_color_image(
            document,
            binary,
            _integer(texture_info.get("index"), name="baseColorTexture.index"),
        )
        sampled_srgb = _sample_texture(
            image,
            uv_values,
            wrap_s=wrap_s,
            wrap_t=wrap_t,
        )
        colors = _srgb_to_linear(sampled_srgb) * factor[None, :3]
    if "COLOR_0" in attributes:
        colors *= _vertex_colors(
            document,
            binary,
            attributes["COLOR_0"],
            vertex_count=vertex_count,
        )
    return cast(NDArray[np.float64], colors)


def _vertex_colors(
    document: Mapping[str, object],
    binary: bytes,
    accessor_value: object,
    *,
    vertex_count: int,
) -> NDArray[np.float64]:
    accessor_index = _integer(accessor_value, name="COLOR_0 accessor")
    values = _accessor(document, binary, accessor_index)
    if values.shape not in {(vertex_count, 3), (vertex_count, 4)}:
        raise ValueError("BLCS GLB COLOR_0 must match POSITION as VEC3 or VEC4.")
    accessors = _sequence(document.get("accessors"), name="accessors")
    accessor = _mapping(accessors[accessor_index], name="COLOR_0 accessor")
    if values.dtype == np.float32:
        if accessor.get("normalized", False) is not False:
            raise ValueError("Float BLCS GLB COLOR_0 cannot be normalized.")
        colors = values.astype(np.float64)
    elif values.dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        if accessor.get("normalized") is not True:
            raise ValueError("Integer BLCS GLB COLOR_0 must be normalized.")
        maximum = float(np.iinfo(values.dtype).max)
        colors = values.astype(np.float64) / maximum
    else:
        raise TypeError("BLCS GLB COLOR_0 must be float32 or normalized uint8/uint16.")
    if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("BLCS GLB COLOR_0 values must be finite and in [0,1].")
    if colors.shape[1] == 4:
        if not np.all(colors[:, 3] == 1.0):
            raise ValueError("BLCS GLB opaque COLOR_0 alpha must equal 1.")
        colors = colors[:, :3]
    return cast(NDArray[np.float64], colors)


def _base_color_image(
    document: Mapping[str, object],
    binary: bytes,
    texture_index: int,
) -> tuple[NDArray[np.float64], int, int]:
    textures = _sequence(document.get("textures"), name="textures")
    if texture_index < 0 or texture_index >= len(textures):
        raise ValueError("BLCS GLB texture index is out of range.")
    texture = _mapping(textures[texture_index], name="texture")
    if texture.get("extensions"):
        raise ValueError("BLCS GLB texture extensions are unsupported.")
    source_index = _integer(texture.get("source"), name="texture.source")
    images = _sequence(document.get("images"), name="images")
    if source_index < 0 or source_index >= len(images):
        raise ValueError("BLCS GLB image index is out of range.")
    image_record = _mapping(images[source_index], name="image")
    if "uri" in image_record or "bufferView" not in image_record:
        raise ValueError("BLCS GLB texture images must be embedded buffer views.")
    mime = image_record.get("mimeType")
    if mime not in {"image/jpeg", "image/png"}:
        raise ValueError("BLCS GLB base-color texture must be embedded JPEG or PNG.")
    encoded = _buffer_view(
        document,
        binary,
        _integer(image_record["bufferView"], name="image.bufferView"),
    )
    try:
        with Image.open(io.BytesIO(encoded)) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("BLCS GLB base-color image is invalid.") from error
    sampler_index = texture.get("sampler")
    sampler: Mapping[str, object] = {}
    if sampler_index is not None:
        samplers = _sequence(document.get("samplers"), name="samplers")
        index = _integer(sampler_index, name="texture.sampler")
        if index < 0 or index >= len(samplers):
            raise ValueError("BLCS GLB sampler index is out of range.")
        sampler = _mapping(samplers[index], name="sampler")
    return (
        image,
        _integer(sampler.get("wrapS", 10497), name="sampler.wrapS"),
        _integer(sampler.get("wrapT", 10497), name="sampler.wrapT"),
    )


def _sample_texture(
    image: NDArray[np.float64],
    uv: NDArray[np.float64],
    *,
    wrap_s: int,
    wrap_t: int,
) -> NDArray[np.float64]:
    u = _wrap_texture_coordinate(uv[:, 0], wrap_s)
    v = _wrap_texture_coordinate(uv[:, 1], wrap_t)
    height, width = image.shape[:2]
    x = np.clip(np.rint(u * (width - 1)), 0, width - 1).astype(np.int64)
    y = np.clip(np.rint((1.0 - v) * (height - 1)), 0, height - 1).astype(np.int64)
    return cast(NDArray[np.float64], image[y, x])


def _wrap_texture_coordinate(
    values: NDArray[np.float64], mode: int
) -> NDArray[np.float64]:
    if mode == 10497:
        return cast(NDArray[np.float64], values - np.floor(values))
    if mode == 33071:
        return cast(NDArray[np.float64], np.clip(values, 0.0, 1.0))
    if mode == 33648:
        period = np.mod(values, 2.0)
        return cast(
            NDArray[np.float64],
            np.where(period <= 1.0, period, 2.0 - period),
        )
    raise ValueError(f"BLCS GLB uses an unsupported texture wrap mode: {mode}.")


def _accessor(
    document: Mapping[str, object],
    binary: bytes,
    accessor_index: int,
) -> NDArray[np.generic]:
    accessors = _sequence(document.get("accessors"), name="accessors")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError("BLCS GLB accessor index is out of range.")
    accessor = _mapping(accessors[accessor_index], name=f"accessors[{accessor_index}]")
    if accessor.get("sparse") is not None or "bufferView" not in accessor:
        raise ValueError("BLCS GLB sparse or buffer-less accessors are unsupported.")
    component_type = _integer(accessor.get("componentType"), name="componentType")
    try:
        dtype = _COMPONENT_DTYPES[component_type]
        width = _ACCESSOR_WIDTHS[str(accessor.get("type"))]
    except KeyError as error:
        raise ValueError(
            "BLCS GLB accessor has an unsupported component/type."
        ) from error
    count = _integer(accessor.get("count"), name="accessor.count")
    if count <= 0:
        raise ValueError("BLCS GLB accessor count must be positive.")
    buffer_views = _sequence(document.get("bufferViews"), name="bufferViews")
    view_index = _integer(accessor["bufferView"], name="accessor.bufferView")
    if view_index < 0 or view_index >= len(buffer_views):
        raise ValueError("BLCS GLB accessor bufferView is out of range.")
    view = _mapping(buffer_views[view_index], name=f"bufferViews[{view_index}]")
    if _integer(view.get("buffer"), name="bufferView.buffer") != 0:
        raise ValueError("BLCS GLB accessors must reference the embedded buffer.")
    view_offset = _integer(view.get("byteOffset", 0), name="bufferView.byteOffset")
    accessor_offset = _integer(
        accessor.get("byteOffset", 0), name="accessor.byteOffset"
    )
    offset = view_offset + accessor_offset
    packed_width = dtype.itemsize * width
    stride = _integer(
        view.get("byteStride", packed_width), name="bufferView.byteStride"
    )
    if stride < packed_width or stride % dtype.itemsize != 0:
        raise ValueError("BLCS GLB accessor byteStride is invalid.")
    required = offset + stride * (count - 1) + packed_width
    view_stop = view_offset + _integer(
        view.get("byteLength"), name="bufferView.byteLength"
    )
    if offset < view_offset or required > view_stop or view_stop > len(binary):
        raise ValueError("BLCS GLB accessor exceeds its embedded buffer view.")
    shape = (count,) if width == 1 else (count, width)
    strides = (stride,) if width == 1 else (stride, dtype.itemsize)
    return np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=strides,
    ).copy()


def _normalized_accessor_values(
    values: NDArray[np.generic],
    document: Mapping[str, object],
    accessor_value: object,
) -> NDArray[np.float64]:
    accessor_index = _integer(accessor_value, name="accessor")
    accessors = _sequence(document.get("accessors"), name="accessors")
    accessor = _mapping(accessors[accessor_index], name="accessor")
    if values.dtype == np.float32:
        return values.astype(np.float64)
    if accessor.get("normalized") is not True:
        raise ValueError("Integer BLCS GLB texture coordinates must be normalized.")
    if values.dtype.kind != "u":
        raise TypeError("Normalized BLCS GLB texture coordinates must be unsigned.")
    maximum = float(np.iinfo(cast(Any, values.dtype)).max)
    return values.astype(np.float64) / maximum


def _buffer_view(
    document: Mapping[str, object],
    binary: bytes,
    view_index: int,
) -> bytes:
    views = _sequence(document.get("bufferViews"), name="bufferViews")
    if view_index < 0 or view_index >= len(views):
        raise ValueError("BLCS GLB bufferView index is out of range.")
    view = _mapping(views[view_index], name="bufferView")
    if _integer(view.get("buffer"), name="bufferView.buffer") != 0:
        raise ValueError("BLCS GLB bufferView must reference buffer zero.")
    start = _integer(view.get("byteOffset", 0), name="bufferView.byteOffset")
    stop = start + _integer(view.get("byteLength"), name="bufferView.byteLength")
    if start < 0 or stop > len(binary):
        raise ValueError("BLCS GLB bufferView exceeds its binary chunk.")
    return binary[start:stop]


def _node_transform(node: Mapping[str, object]) -> NDArray[np.float64]:
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            raise ValueError("BLCS GLB nodes cannot mix matrix and TRS transforms.")
        values = np.asarray(_number_sequence(node["matrix"], name="node.matrix"))
        if values.shape != (16,):
            raise ValueError("BLCS GLB node.matrix must contain 16 values.")
        matrix = values.reshape(4, 4, order="F")
    else:
        translation = np.asarray(
            _number_sequence(
                node.get("translation", (0.0, 0.0, 0.0)), name="translation"
            )
        )
        rotation = np.asarray(
            _number_sequence(
                node.get("rotation", (0.0, 0.0, 0.0, 1.0)), name="rotation"
            )
        )
        scale = np.asarray(
            _number_sequence(node.get("scale", (1.0, 1.0, 1.0)), name="scale")
        )
        if translation.shape != (3,) or rotation.shape != (4,) or scale.shape != (3,):
            raise ValueError("BLCS GLB node TRS fields have invalid dimensions.")
        if np.any(scale == 0.0):
            raise ValueError("BLCS GLB node scale must be nonzero.")
        norm = float(np.linalg.norm(rotation))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("BLCS GLB node rotation quaternion is invalid.")
        x, y, z, w = rotation / norm
        rotation_matrix = np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_matrix @ np.diag(scale)
        matrix[:3, 3] = translation
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("BLCS GLB node transform must be finite and affine.")
    if np.linalg.det(matrix[:3, :3]) <= 0.0:
        raise ValueError("BLCS GLB node transform must preserve handedness.")
    return matrix


def _center_and_scale(
    vertices: NDArray[np.floating[Any]],
    *,
    radius_m: float,
) -> NDArray[np.float64]:
    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("BLCS mesh positions must be a finite [V,3] array.")
    lower = value.min(axis=0)
    upper = value.max(axis=0)
    extents = upper - lower
    maximum_extent = float(extents.max())
    if maximum_extent <= 1.0e-12 or float(extents.min()) / maximum_extent < 0.7:
        raise ValueError("BLCS GLB geometry is degenerate or not ball-like.")
    centered = value - (lower + upper) / 2.0
    source_radius = float(np.linalg.norm(centered, axis=1).max())
    if not math.isfinite(source_radius) or source_radius <= 1.0e-12:
        raise ValueError("BLCS GLB geometry has no measurable radius.")
    return cast(NDArray[np.float64], centered * (radius_m / source_radius))


def _simplify_vertex_clusters(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    colors: NDArray[np.float64],
    *,
    maximum_faces: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    if len(faces) <= maximum_faces:
        return vertices, faces, colors
    lower = vertices.min(axis=0)
    extent = np.maximum(vertices.max(axis=0) - lower, np.finfo(np.float64).eps)

    def clustered(
        resolution: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        coordinates = np.floor((vertices - lower) / extent * resolution).astype(
            np.int64
        )
        coordinates = np.clip(coordinates, 0, resolution - 1)
        keys = (
            coordinates[:, 0] * resolution * resolution
            + coordinates[:, 1] * resolution
            + coordinates[:, 2]
        )
        _unique, inverse = np.unique(keys, return_inverse=True)
        count = int(inverse.max()) + 1
        weights = np.bincount(inverse, minlength=count).astype(np.float64)
        clustered_vertices = (
            np.column_stack(
                [
                    np.bincount(inverse, weights=vertices[:, axis], minlength=count)
                    for axis in range(3)
                ]
            )
            / weights[:, None]
        )
        clustered_colors = (
            np.column_stack(
                [
                    np.bincount(inverse, weights=colors[:, axis], minlength=count)
                    for axis in range(3)
                ]
            )
            / weights[:, None]
        )
        mapped = inverse[faces]
        mapped = mapped[
            (mapped[:, 0] != mapped[:, 1])
            & (mapped[:, 1] != mapped[:, 2])
            & (mapped[:, 0] != mapped[:, 2])
        ]
        if not len(mapped):
            return clustered_vertices, mapped.astype(np.int64), clustered_colors
        canonical = np.sort(mapped, axis=1)
        _canonical, first = np.unique(canonical, axis=0, return_index=True)
        mapped = mapped[np.sort(first)].astype(np.int64)
        used, remapped = np.unique(mapped, return_inverse=True)
        return (
            clustered_vertices[used],
            remapped.reshape(-1, 3).astype(np.int64),
            clustered_colors[used],
        )

    low = 2
    high = 128
    best: tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]] | None = (
        None
    )
    while low <= high:
        middle = (low + high) // 2
        candidate = clustered(middle)
        if 4 <= len(candidate[1]) <= maximum_faces:
            best = candidate
            low = middle + 1
        elif len(candidate[1]) > maximum_faces:
            high = middle - 1
        else:
            low = middle + 1
    if best is None:
        raise ValueError(
            "BLCS GLB could not be simplified to maximum_faces without degenerating."
        )
    return best


def _vertex_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> NDArray[np.float32]:
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError(
            "Simplified BLCS GLB contains vertices without surface normals."
        )
    normals /= lengths[:, None]
    return normals.astype(np.float32)


def _srgb_to_linear(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _array(
    value: NDArray[np.generic],
    dtype: type[np.generic],
    ndim: int,
    name: str,
) -> NDArray[Any]:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.ndim != ndim:
        raise TypeError(f"{name} must use {np.dtype(dtype)} with ndim={ndim}.")
    return np.array(result, dtype=dtype, order="C", copy=True)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"BLCS GLB {name} must be a string-keyed object.")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"BLCS GLB {name} must be an array.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"BLCS GLB {name} must be an integer.")
    return value


def _integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(_integer(item, name=name) for item in _sequence(value, name=name))


def _number_sequence(value: object, *, name: str) -> tuple[float, ...]:
    result = []
    for item in _sequence(value, name=name):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"BLCS GLB {name} must contain only numbers.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"BLCS GLB {name} must contain only finite numbers.")
        result.append(number)
    return tuple(result)


def _string_sequence(value: object) -> tuple[str, ...]:
    sequence = _sequence(value, name="string array")
    if any(not isinstance(item, str) for item in sequence):
        raise TypeError("BLCS GLB extension names must be strings.")
    return tuple(cast(str, item) for item in sequence)


__all__ = ["BLCSBallMesh", "load_ball_mesh_asset"]
