"""Strict consumer-side validation for the public NHT standard scene export."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

NHT_SCENE_SCHEMA = "nht_standard_scene_v1"
NHT_CAMERAS_SCHEMA = "nht_standard_cameras_v1"
NHT_CAMERA_COORDINATE_CONVENTION = "x-right, y-down, z-forward"
NHT_SCENE_COORDINATE_CONVENTION = (
    "NHT parser normalized world coordinates; right-handed; identical to "
    "checkpoint Gaussian means"
)
NHT_PIXEL_COORDINATE_CONVENTION = (
    "origin at top-left; x-right, y-down; pixel centers"
)
NHT_IMAGE_RESOLUTION_SEMANTICS = (
    "width and height describe the undistorted, cropped training image "
    "at the configured data factor"
)
NHT_RGB_OUTPUT_SEMANTICS = "float32 HxWx3 in [0,1] plus PNG preview"
NHT_ALPHA_OUTPUT_SEMANTICS = "float32 HxWx1 in [0,1] plus PNG preview"
NHT_DEPTH_OUTPUT_SEMANTICS = "float32 HxWx1 in canonical scene units"
NHT_TRANSFORM_SEMANTICS = (
    "camera_to_scene maps homogeneous camera coordinates to scene coordinates"
)

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MATRIX_ATOL = 1.0e-6
_SCENE_KEYS = {
    "schema",
    "scene_id",
    "camera_coordinate_convention",
    "scene_coordinate_convention",
    "pixel_coordinate_convention",
    "image_resolution_semantics",
    "camera_count",
    "cameras",
    "point_cloud",
    "image_root",
    "model_root",
    "scene_from_sfm",
    "sfm_from_scene",
    "normalization",
    "renderer",
    "sfm_summary",
    "nht_training_summary",
    "capabilities",
}
_CAMERA_KEYS = {
    "camera_id",
    "source_frame_index",
    "time_seconds",
    "split",
    "image",
    "width",
    "height",
    "intrinsics",
    "camera_to_scene",
    "source_image_processing",
    "diagnostics",
    "group",
}


@dataclass(frozen=True, slots=True)
class StandardSceneExport:
    """Validated NHT export expressed only through public files and geometry."""

    scene_id: str
    export_root: Path
    scene_path: Path
    cameras: tuple[SceneCamera, ...]
    points_scene: NDArray[np.float32]
    scene_from_sfm: tuple[float, ...]
    sfm_from_scene: tuple[float, ...]
    checkpoint_path: Path
    runtime_config_path: Path

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return camera IDs in the canonical exported order."""
        return tuple(camera.camera_id for camera in self.cameras)

    @property
    def point_count(self) -> int:
        """Return the validated number of sparse scene points."""
        return int(self.points_scene.shape[0])

    def camera(self, camera_id: str) -> SceneCamera:
        """Return exactly one exported camera, without selecting a fallback."""
        matches = [camera for camera in self.cameras if camera.camera_id == camera_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown exported NHT camera ID: {camera_id!r}.")
        return matches[0]


def validate_standard_scene_export(scene_path: str | Path) -> StandardSceneExport:
    """Load and semantically validate one self-contained NHT standard export."""
    path = Path(scene_path)
    if path.name != "scene.json" or path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary exported scene.json file: {path}")
    if path.parent.is_symlink():
        raise ValueError("The NHT export directory must not be a symbolic link.")
    export_root = path.parent.resolve(strict=True)
    scene = _mapping(_load_json(path), keys=_SCENE_KEYS, name="scene.json")

    _expect(scene["schema"], NHT_SCENE_SCHEMA, name="scene.schema")
    scene_id = _portable_id(scene["scene_id"], name="scene.scene_id")
    _expect(
        scene["camera_coordinate_convention"],
        NHT_CAMERA_COORDINATE_CONVENTION,
        name="scene.camera_coordinate_convention",
    )
    _expect(
        scene["scene_coordinate_convention"],
        NHT_SCENE_COORDINATE_CONVENTION,
        name="scene.scene_coordinate_convention",
    )
    _expect(
        scene["pixel_coordinate_convention"],
        NHT_PIXEL_COORDINATE_CONVENTION,
        name="scene.pixel_coordinate_convention",
    )
    _expect(
        scene["image_resolution_semantics"],
        NHT_IMAGE_RESOLUTION_SEMANTICS,
        name="scene.image_resolution_semantics",
    )

    cameras_reference = _expect_string(scene["cameras"], name="scene.cameras")
    if cameras_reference != "cameras.json":
        raise ValueError("scene.cameras must be the canonical cameras.json reference.")
    cameras_path = _resolve_reference(
        export_root,
        cameras_reference,
        name="scene.cameras",
        kind="file",
    )
    cameras_payload = _mapping(
        _load_json(cameras_path),
        keys={
            "schema",
            "camera_coordinate_convention",
            "transform_semantics",
            "cameras",
        },
        name="cameras.json",
    )
    _expect(cameras_payload["schema"], NHT_CAMERAS_SCHEMA, name="cameras.schema")
    _expect(
        cameras_payload["camera_coordinate_convention"],
        NHT_CAMERA_COORDINATE_CONVENTION,
        name="cameras.camera_coordinate_convention",
    )
    _expect(
        cameras_payload["transform_semantics"],
        NHT_TRANSFORM_SEMANTICS,
        name="cameras.transform_semantics",
    )

    image_root_reference = _expect_string(scene["image_root"], name="scene.image_root")
    if image_root_reference != "images":
        raise ValueError("scene.image_root must be the canonical images directory.")
    image_root = _resolve_reference(
        export_root,
        image_root_reference,
        name="scene.image_root",
        kind="directory",
    )

    raw_cameras = _sequence(cameras_payload["cameras"], name="cameras.cameras")
    if not raw_cameras:
        raise ValueError("cameras.cameras must contain at least one camera.")
    cameras = tuple(
        _parse_camera(item, index=index, export_root=export_root, image_root=image_root)
        for index, item in enumerate(raw_cameras)
    )
    camera_ids = [camera.camera_id for camera in cameras]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Exported camera IDs must be unique.")
    camera_count = _integer(scene["camera_count"], name="scene.camera_count", minimum=1)
    if camera_count != len(cameras):
        raise ValueError(
            f"scene.camera_count={camera_count} does not match {len(cameras)} camera records."
        )

    point_cloud = _mapping(
        scene["point_cloud"],
        keys={"path", "shape", "dtype", "columns", "color_range"},
        name="scene.point_cloud",
    )
    points_reference = _expect_string(point_cloud["path"], name="point_cloud.path")
    if points_reference != "points_scene.npy":
        raise ValueError(
            "point_cloud.path must be the canonical points_scene.npy reference."
        )
    points_path = _resolve_reference(
        export_root,
        points_reference,
        name="point_cloud.path",
        kind="file",
    )
    declared_shape = _integer_pair(
        point_cloud["shape"], name="point_cloud.shape", minimum=0
    )
    if declared_shape[1] != 6:
        raise ValueError("point_cloud.shape must be [N, 6].")
    _expect(point_cloud["dtype"], "float32", name="point_cloud.dtype")
    _expect_sequence(
        point_cloud["columns"],
        ("x", "y", "z", "red", "green", "blue"),
        name="point_cloud.columns",
    )
    _expect_sequence(
        point_cloud["color_range"], (0.0, 1.0), name="point_cloud.color_range"
    )
    points_scene = np.load(points_path, allow_pickle=False)
    if points_scene.dtype != np.dtype(np.float32):
        raise TypeError(
            f"points_scene.npy must have dtype float32, got {points_scene.dtype}."
        )
    if points_scene.ndim != 2 or points_scene.shape != declared_shape:
        raise ValueError(
            "points_scene.npy shape disagrees with point_cloud.shape: "
            f"array={points_scene.shape}, declared={declared_shape}."
        )
    if not np.isfinite(points_scene).all():
        raise ValueError("points_scene.npy must contain only finite values.")
    if points_scene.size and (
        np.any(points_scene[:, 3:] < 0.0) or np.any(points_scene[:, 3:] > 1.0)
    ):
        raise ValueError("points_scene.npy RGB values must stay in [0, 1].")
    points_scene.setflags(write=False)

    scene_from_sfm = _similarity_matrix(
        scene["scene_from_sfm"], name="scene.scene_from_sfm"
    )
    sfm_from_scene = _similarity_matrix(
        scene["sfm_from_scene"], name="scene.sfm_from_scene"
    )
    if not np.allclose(
        scene_from_sfm @ sfm_from_scene, np.eye(4), atol=1.0e-5, rtol=0.0
    ):
        raise ValueError(
            "scene_from_sfm and sfm_from_scene must be reciprocal transforms."
        )
    if not np.allclose(
        sfm_from_scene @ scene_from_sfm, np.eye(4), atol=1.0e-5, rtol=0.0
    ):
        raise ValueError(
            "sfm_from_scene and scene_from_sfm must be reciprocal transforms."
        )

    normalization = _mapping(
        scene["normalization"],
        keys={
            "applied",
            "camera_similarity",
            "principal_axis_alignment",
            "upside_down_correction",
        },
        name="scene.normalization",
    )
    if normalization["applied"] is not True:
        raise ValueError("scene.normalization.applied must be true.")
    _similarity_matrix(
        normalization["camera_similarity"],
        name="normalization.camera_similarity",
    )
    _rigid_matrix(
        normalization["principal_axis_alignment"],
        name="normalization.principal_axis_alignment",
    )
    _rigid_matrix(
        normalization["upside_down_correction"],
        name="normalization.upside_down_correction",
    )

    model_root_reference = _expect_string(scene["model_root"], name="scene.model_root")
    if model_root_reference != "model":
        raise ValueError("scene.model_root must be the canonical model directory.")
    model_root = _resolve_reference(
        export_root,
        model_root_reference,
        name="scene.model_root",
        kind="directory",
    )
    renderer = _mapping(
        scene["renderer"],
        keys={"command", "model", "runtime_config", "checkpoint", "outputs"},
        name="scene.renderer",
    )
    _expect(renderer["command"], "nht-render", name="renderer.command")
    _expect(renderer["model"], "model", name="renderer.model")
    outputs = _mapping(
        renderer["outputs"],
        keys={"rgb", "alpha", "depth"},
        name="renderer.outputs",
    )
    _expect(outputs["rgb"], NHT_RGB_OUTPUT_SEMANTICS, name="renderer.outputs.rgb")
    _expect(
        outputs["alpha"],
        NHT_ALPHA_OUTPUT_SEMANTICS,
        name="renderer.outputs.alpha",
    )
    _expect(
        outputs["depth"],
        NHT_DEPTH_OUTPUT_SEMANTICS,
        name="renderer.outputs.depth",
    )
    checkpoint_path = _resolve_model_file(
        export_root,
        model_root,
        renderer["checkpoint"],
        name="renderer.checkpoint",
    )
    runtime_config_path = _resolve_model_file(
        export_root,
        model_root,
        renderer["runtime_config"],
        name="renderer.runtime_config",
    )
    _validate_runtime_config(runtime_config_path)

    _mapping(scene["sfm_summary"], name="scene.sfm_summary")
    _mapping(scene["nht_training_summary"], name="scene.nht_training_summary")
    capabilities = _sequence(scene["capabilities"], name="scene.capabilities")
    if any(not isinstance(item, str) or not item for item in capabilities):
        raise TypeError("scene.capabilities must contain non-empty strings.")
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("scene.capabilities must not contain duplicates.")
    if "nht_rendering_model" not in capabilities:
        raise ValueError("scene.capabilities must include nht_rendering_model.")

    return StandardSceneExport(
        scene_id=scene_id,
        export_root=export_root,
        scene_path=path.resolve(strict=True),
        cameras=cameras,
        points_scene=points_scene,
        scene_from_sfm=tuple(float(value) for value in scene_from_sfm.ravel()),
        sfm_from_scene=tuple(float(value) for value in sfm_from_scene.ravel()),
        checkpoint_path=checkpoint_path,
        runtime_config_path=runtime_config_path,
    )


def _parse_camera(
    value: object,
    *,
    index: int,
    export_root: Path,
    image_root: Path,
) -> SceneCamera:
    name = f"cameras.cameras[{index}]"
    raw = _mapping(value, keys=_CAMERA_KEYS, name=name)
    camera_id = _portable_id(raw["camera_id"], name=f"{name}.camera_id")
    source_frame_index = _integer(
        raw["source_frame_index"],
        name=f"{name}.source_frame_index",
        minimum=0,
    )
    _number(raw["time_seconds"], name=f"{name}.time_seconds", minimum=0.0)
    if raw["split"] not in {"train", "validation"}:
        raise ValueError(f"{name}.split must be train or validation.")
    width = _integer(raw["width"], name=f"{name}.width", minimum=2)
    height = _integer(raw["height"], name=f"{name}.height", minimum=2)

    intrinsics = _mapping(
        raw["intrinsics"],
        keys={"model", "distortion_model", "params", "matrix"},
        name=f"{name}.intrinsics",
    )
    _expect(intrinsics["model"], "PINHOLE", name=f"{name}.intrinsics.model")
    _expect(
        intrinsics["distortion_model"],
        "NONE",
        name=f"{name}.intrinsics.distortion_model",
    )
    intrinsic_matrix = _intrinsic_matrix(
        intrinsics["matrix"],
        width=width,
        height=height,
        name=f"{name}.intrinsics.matrix",
    )
    parameters = _number_tuple(
        intrinsics["params"], length=4, name=f"{name}.intrinsics.params"
    )
    expected_parameters = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    if not np.allclose(parameters, expected_parameters, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name}.intrinsics.params disagree with the PINHOLE matrix.")

    camera_to_scene_matrix = _rigid_matrix(
        raw["camera_to_scene"],
        name=f"{name}.camera_to_scene",
    )
    image_reference = _expect_string(raw["image"], name=f"{name}.image")
    image_path = _resolve_reference(
        export_root,
        image_reference,
        name=f"{name}.image",
        kind="file",
    )
    if not image_path.is_relative_to(image_root):
        raise ValueError(f"{name}.image must resolve inside scene.image_root.")
    try:
        with Image.open(image_path) as image:
            actual_size = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"{name}.image is not a readable image: {image_path}"
        ) from error
    if actual_size != (width, height):
        raise ValueError(
            f"{name}.image resolution {actual_size} disagrees with {(width, height)}."
        )

    processing = _mapping(
        raw["source_image_processing"],
        keys={"source_resolution", "crop_xywh", "undistorted", "data_factor"},
        name=f"{name}.source_image_processing",
    )
    source_width, source_height = _integer_pair(
        processing["source_resolution"],
        name=f"{name}.source_image_processing.source_resolution",
        minimum=1,
    )
    crop = _integer_tuple(
        processing["crop_xywh"],
        length=4,
        name=f"{name}.source_image_processing.crop_xywh",
        minimum=0,
    )
    if crop[2] < 1 or crop[3] < 1:
        raise ValueError(f"{name}.source_image_processing crop size must be positive.")
    if crop[0] + crop[2] > source_width or crop[1] + crop[3] > source_height:
        raise ValueError(
            f"{name}.source_image_processing crop escapes the source image."
        )
    if processing["undistorted"] is not True:
        raise ValueError(f"{name}.source_image_processing.undistorted must be true.")
    _integer(
        processing["data_factor"],
        name=f"{name}.source_image_processing.data_factor",
        minimum=1,
    )

    diagnostics = _mapping(
        raw["diagnostics"],
        keys={"sfm_camera_id", "sfm_camera_to_world"},
        name=f"{name}.diagnostics",
    )
    _integer(diagnostics["sfm_camera_id"], name=f"{name}.sfm_camera_id", minimum=0)
    _rigid_matrix(
        diagnostics["sfm_camera_to_world"],
        name=f"{name}.sfm_camera_to_world",
    )
    _expect_string(raw["group"], name=f"{name}.group")

    return SceneCamera(
        camera_id=camera_id,
        source_frame_index=source_frame_index,
        width=width,
        height=height,
        intrinsics=tuple(float(item) for item in intrinsic_matrix.ravel()),
        camera_to_scene=RigidTransform.from_matrix(camera_to_scene_matrix),
        image_path=str(image_path),
    )


def _validate_runtime_config(path: Path) -> None:
    raw = _mapping(
        _load_json(path),
        keys={
            "schema",
            "camera_model",
            "pose_opt",
            "primitive_type",
            "antialiased",
            "packed",
            "tile_size",
            "with_ut",
            "with_eval3d",
            "post_processing",
            "near_plane",
            "far_plane",
            "deferred_opt_feature_dim",
            "deferred_opt_enable_view_encoding",
            "deferred_opt_view_encoding_type",
            "deferred_mlp_hidden_dim",
            "deferred_mlp_num_layers",
            "deferred_opt_sh_degree",
            "deferred_opt_sh_scale",
            "deferred_opt_fourier_num_freqs",
            "deferred_opt_center_ray_encoding",
            "deferred_decode_activation",
        },
        name="renderer.runtime_config",
    )
    _expect(raw["schema"], "nht_runtime_config_v1", name="runtime.schema")
    _expect(raw["camera_model"], "pinhole", name="runtime.camera_model")
    if raw["pose_opt"] is not False:
        raise ValueError("runtime.pose_opt must be false.")
    if raw["post_processing"] is not None:
        raise ValueError("runtime.post_processing must be null.")
    _expect(raw["primitive_type"], "3dgs", name="runtime.primitive_type")
    _boolean(raw["antialiased"], name="runtime.antialiased")
    _boolean(raw["packed"], name="runtime.packed")
    _integer(raw["tile_size"], name="runtime.tile_size", minimum=1)
    with_ut = _boolean(raw["with_ut"], name="runtime.with_ut")
    with_eval3d = _boolean(raw["with_eval3d"], name="runtime.with_eval3d")
    if with_ut and not with_eval3d:
        raise ValueError("runtime.with_ut requires runtime.with_eval3d.")
    _integer(
        raw["deferred_opt_feature_dim"],
        name="runtime.deferred_opt_feature_dim",
        minimum=1,
    )
    _boolean(
        raw["deferred_opt_enable_view_encoding"],
        name="runtime.deferred_opt_enable_view_encoding",
    )
    view_encoding = _expect_string(
        raw["deferred_opt_view_encoding_type"],
        name="runtime.deferred_opt_view_encoding_type",
    )
    if view_encoding not in {"sh", "fourier"}:
        raise ValueError(
            "runtime.deferred_opt_view_encoding_type must be 'sh' or 'fourier'."
        )
    _integer(
        raw["deferred_mlp_hidden_dim"],
        name="runtime.deferred_mlp_hidden_dim",
        minimum=1,
    )
    _integer(
        raw["deferred_mlp_num_layers"],
        name="runtime.deferred_mlp_num_layers",
        minimum=1,
    )
    _integer(
        raw["deferred_opt_sh_degree"],
        name="runtime.deferred_opt_sh_degree",
        minimum=0,
    )
    _number(
        raw["deferred_opt_sh_scale"],
        name="runtime.deferred_opt_sh_scale",
        minimum=0.0,
        exclusive=True,
    )
    _integer(
        raw["deferred_opt_fourier_num_freqs"],
        name="runtime.deferred_opt_fourier_num_freqs",
        minimum=1,
    )
    _boolean(
        raw["deferred_opt_center_ray_encoding"],
        name="runtime.deferred_opt_center_ray_encoding",
    )
    decode_activation = _expect_string(
        raw["deferred_decode_activation"],
        name="runtime.deferred_decode_activation",
    )
    if decode_activation not in {"sigmoid", "relu_clamp"}:
        raise ValueError(
            "runtime.deferred_decode_activation must be 'sigmoid' or 'relu_clamp'."
        )
    near = _number(
        raw["near_plane"], name="runtime.near_plane", minimum=0.0, exclusive=True
    )
    far = _number(
        raw["far_plane"], name="runtime.far_plane", minimum=0.0, exclusive=True
    )
    if near >= far:
        raise ValueError("runtime near_plane must be smaller than far_plane.")


def _resolve_model_file(
    export_root: Path,
    model_root: Path,
    value: object,
    *,
    name: str,
) -> Path:
    reference = _expect_string(value, name=name)
    path = _resolve_reference(export_root, reference, name=name, kind="file")
    if not path.is_relative_to(model_root):
        raise ValueError(f"{name} must resolve inside scene.model_root.")
    return path


def _resolve_reference(root: Path, reference: str, *, name: str, kind: str) -> Path:
    if "\\" in reference:
        raise ValueError(f"{name} must use an export-relative POSIX path.")
    pure = PurePosixPath(reference)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(
            f"{name} must be a contained export-relative path: {reference!r}."
        )
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {candidate}") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes the export root through its resolved path.")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{name} must reference a file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{name} must reference a directory: {resolved}")
    return resolved


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON number {value!r} is not allowed in {path}.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}.")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error


def _mapping(
    value: object,
    *,
    name: str,
    keys: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    result = dict(value)
    if keys is not None and set(result) != keys:
        raise ValueError(
            f"{name} schema mismatch; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _expect(value: object, expected: object, *, name: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{name} must be {expected!r}, got {value!r}.")


def _expect_sequence(value: object, expected: tuple[object, ...], *, name: str) -> None:
    sequence = _sequence(value, name=name)
    if tuple(sequence) != expected:
        raise ValueError(f"{name} must be {list(expected)!r}, got {sequence!r}.")


def _expect_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean.")
    return value


def _portable_id(value: object, *, name: str) -> str:
    result = _expect_string(value, name=name)
    if _PORTABLE_ID.fullmatch(result) is None:
        raise ValueError(f"{name} must be a portable identifier: {result!r}.")
    return result


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        relation = ">" if exclusive else ">="
        raise ValueError(f"{name} must be {relation} {minimum}.")
    return result


def _number_tuple(value: object, *, length: int, name: str) -> tuple[float, ...]:
    raw = _sequence(value, name=name)
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers.")
    return tuple(
        _number(item, name=f"{name}[{index}]") for index, item in enumerate(raw)
    )


def _integer_tuple(
    value: object,
    *,
    length: int,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    raw = _sequence(value, name=name)
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} integers.")
    return tuple(
        _integer(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(raw)
    )


def _integer_pair(value: object, *, name: str, minimum: int) -> tuple[int, int]:
    pair = _integer_tuple(value, length=2, name=name, minimum=minimum)
    return pair[0], pair[1]


def _matrix(
    value: object, *, rows: int, columns: int, name: str
) -> NDArray[np.float64]:
    raw_rows = _sequence(value, name=name)
    if len(raw_rows) != rows:
        raise ValueError(f"{name} must have {rows} rows.")
    parsed = [
        _number_tuple(row, length=columns, name=f"{name}[{index}]")
        for index, row in enumerate(raw_rows)
    ]
    return np.asarray(parsed, dtype=np.float64)


def _rigid_matrix(value: object, *, name: str) -> NDArray[np.float64]:
    matrix = _matrix(value, rows=4, columns=4, name=name)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must contain a proper rotation with determinant +1.")
    return matrix


def _similarity_matrix(value: object, *, name: str) -> NDArray[np.float64]:
    matrix = _matrix(value, rows=4, columns=4, name=name)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1].")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError(f"{name} must contain a positive-scale proper rotation.")
    scale = determinant ** (1.0 / 3.0)
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have uniform scale and an orthonormal rotation.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must contain a proper rotation with determinant +1.")
    return matrix


def _intrinsic_matrix(
    value: object,
    *,
    width: int,
    height: int,
    name: str,
) -> NDArray[np.float64]:
    matrix = _matrix(value, rows=3, columns=3, name=name)
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive.")
    if not np.allclose(
        matrix,
        (
            (matrix[0, 0], 0.0, matrix[0, 2]),
            (0.0, matrix[1, 1], matrix[1, 2]),
            (0.0, 0.0, 1.0),
        ),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be a canonical PINHOLE intrinsic matrix.")
    if not 0.0 <= matrix[0, 2] < width or not 0.0 <= matrix[1, 2] < height:
        raise ValueError(f"{name} principal point must lie inside the image.")
    return matrix


__all__ = [
    "NHT_CAMERAS_SCHEMA",
    "NHT_CAMERA_COORDINATE_CONVENTION",
    "NHT_IMAGE_RESOLUTION_SEMANTICS",
    "NHT_PIXEL_COORDINATE_CONVENTION",
    "NHT_SCENE_COORDINATE_CONVENTION",
    "NHT_SCENE_SCHEMA",
    "NHT_TRANSFORM_SEMANTICS",
    "StandardSceneExport",
    "validate_standard_scene_export",
]
