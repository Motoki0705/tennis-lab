"""Typed client for public joint background/dynamic-Gaussian NHT rendering."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht.client import (
    NHTRenderClient,
    _load_render_result,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCommandRequest,
    NHTRenderResult,
)

NHT_COMPOSED_RENDER_RESULT_SCHEMA = "nht_composed_render_result_v1"

_RESULT_KEYS = {
    "schema",
    "scene_schema",
    "scene_id",
    "coordinate_space",
    "background",
    "composition",
    "chunks",
    "cuda_peak_bytes",
}
_COMPOSITION_KEYS = {
    "request_schema",
    "frame_count",
    "object_count",
    "asset_gaussian_count",
    "appearance_model",
    "rasterization",
    "visibility_threshold",
}
_CHUNK_KEYS = {
    "chunk_id",
    "frame_indices",
    "camera_ids",
    "sample_count",
    "pixel_count",
    "arrays",
}
_ARRAY_KEYS = {
    "frame_indices",
    "camera_indices",
    "offsets",
    "pixel_indices",
    "rgb",
    "alpha",
    "depth",
    "instance_ids",
}


@dataclass(frozen=True, slots=True)
class NHTComposedRenderCommandRequest:
    """One standard NHT camera request plus one dynamic composition request."""

    base: NHTRenderCommandRequest
    composition_request_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.base, NHTRenderCommandRequest):
            raise TypeError("base must be NHTRenderCommandRequest.")
        path = self.composition_request_path
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("composition_request_path must be an absolute pathlib.Path.")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"NHT composition request must be an ordinary file: {path}"
            )
        output = self.base.output_directory.resolve(strict=False)
        request = path.resolve(strict=True)
        if request == output or request.is_relative_to(output):
            raise ValueError("NHT composition request must remain outside render output.")
        workspace = self.base.scene_path.parent.parent.resolve(strict=True)
        if request.is_relative_to(workspace):
            raise ValueError(
                "NHT composition request must stay outside the reconstruction workspace."
            )

    @property
    def expected_camera_ids(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.base.expected_camera_ids)

    def argv(self) -> tuple[str, ...]:
        base = list(self.base.argv())
        output_index = base.index("--output")
        base[output_index:output_index] = (
            "--composition",
            str(self.composition_request_path),
        )
        return tuple(base)


@dataclass(frozen=True, slots=True)
class NHTComposedChunkArrays:
    """One validated sparse joint render chunk in canonical NHT depth units."""

    frame_indices: NDArray[np.int64]
    camera_indices: NDArray[np.int32]
    offsets: NDArray[np.int64]
    pixel_indices: NDArray[np.int32]
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    instance_ids: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class NHTComposedChunkRecord:
    """Metadata and immutable file reference for one composed output chunk."""

    chunk_id: str
    frame_indices: tuple[int, ...]
    camera_ids: tuple[str, ...]
    sample_count: int
    pixel_count: int
    arrays_path: Path
    width: int
    height: int
    object_count: int

    def load_arrays(self) -> NHTComposedChunkArrays:
        """Load and rescan one sparse public payload before dataset publication."""
        if self.arrays_path.is_symlink() or not self.arrays_path.is_file():
            raise FileNotFoundError(
                f"NHT composed chunk arrays are unavailable: {self.arrays_path}"
            )
        with np.load(self.arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise ValueError("NHT composed chunk has unknown or missing arrays.")
            arrays = NHTComposedChunkArrays(
                frame_indices=np.array(archive["frame_indices"], copy=True),
                camera_indices=np.array(archive["camera_indices"], copy=True),
                offsets=np.array(archive["offsets"], copy=True),
                pixel_indices=np.array(archive["pixel_indices"], copy=True),
                rgb=np.array(archive["rgb"], copy=True),
                alpha=np.array(archive["alpha"], copy=True),
                depth=np.array(archive["depth"], copy=True),
                instance_ids=np.array(archive["instance_ids"], copy=True),
            )
        self._validate_arrays(arrays)
        for value in (
            arrays.frame_indices,
            arrays.camera_indices,
            arrays.offsets,
            arrays.pixel_indices,
            arrays.rgb,
            arrays.alpha,
            arrays.depth,
            arrays.instance_ids,
        ):
            value.setflags(write=False)
        return arrays

    def _validate_arrays(self, arrays: NHTComposedChunkArrays) -> None:
        expected_dtypes: dict[str, np.dtype[np.generic]] = {
            "frame_indices": np.dtype(np.int64),
            "camera_indices": np.dtype(np.int32),
            "offsets": np.dtype(np.int64),
            "pixel_indices": np.dtype(np.int32),
            "rgb": np.dtype(np.float32),
            "alpha": np.dtype(np.float32),
            "depth": np.dtype(np.float32),
            "instance_ids": np.dtype(np.int32),
        }
        for name, dtype in expected_dtypes.items():
            value = cast(NDArray[np.generic], getattr(arrays, name))
            if value.dtype != dtype:
                raise TypeError(f"NHT composed {name} must use {dtype}.")
        expected_frame_indices: NDArray[np.int64] = np.repeat(
            np.asarray(self.frame_indices, dtype=np.int64), len(self.camera_ids)
        )
        expected_camera_indices: NDArray[np.int32] = np.tile(
            np.arange(len(self.camera_ids), dtype=np.int32), len(self.frame_indices)
        )
        if arrays.frame_indices.shape != (self.sample_count,) or not np.array_equal(
            arrays.frame_indices, expected_frame_indices
        ):
            raise ValueError("NHT composed frame sample order differs from its request.")
        if arrays.camera_indices.shape != (self.sample_count,) or not np.array_equal(
            arrays.camera_indices, expected_camera_indices
        ):
            raise ValueError("NHT composed camera sample order differs from its request.")
        if (
            arrays.offsets.shape != (self.sample_count + 1,)
            or arrays.offsets[0] != 0
            or np.any(np.diff(arrays.offsets) < 0)
            or int(arrays.offsets[-1]) != self.pixel_count
        ):
            raise ValueError("NHT composed sparse offsets are invalid.")
        expected_shapes = {
            "pixel_indices": (self.pixel_count,),
            "rgb": (self.pixel_count, 3),
            "alpha": (self.pixel_count,),
            "depth": (self.pixel_count,),
            "instance_ids": (self.pixel_count,),
        }
        for name, shape in expected_shapes.items():
            if getattr(arrays, name).shape != shape:
                raise ValueError(f"NHT composed {name} has the wrong shape.")
        if any(
            not np.isfinite(value).all()
            for value in (arrays.rgb, arrays.alpha, arrays.depth)
        ):
            raise ValueError("NHT composed floating arrays must be finite.")
        if (
            np.any(arrays.rgb < 0.0)
            or np.any(arrays.rgb > 1.0)
            or np.any(arrays.alpha < 0.0)
            or np.any(arrays.alpha > 1.0)
            or np.any(arrays.depth <= 0.0)
        ):
            raise ValueError("NHT composed RGB/alpha/depth values are outside contract.")
        if np.any(arrays.instance_ids <= 0) or np.any(
            arrays.instance_ids > self.object_count
        ):
            raise ValueError("NHT composed instance IDs are outside the object inventory.")
        maximum_pixel = self.width * self.height
        for sample_index in range(self.sample_count):
            start = int(arrays.offsets[sample_index])
            stop = int(arrays.offsets[sample_index + 1])
            pixels = arrays.pixel_indices[start:stop]
            if len(pixels) and (
                pixels[0] < 0
                or pixels[-1] >= maximum_pixel
                or np.any(np.diff(pixels) <= 0)
            ):
                raise ValueError(
                    "NHT composed sample pixels must be sorted, unique, and in frame."
                )


@dataclass(frozen=True, slots=True)
class NHTComposedRenderResult:
    """Complete public background plus joint-rasterized dynamic chunks."""

    scene_id: str
    output_directory: Path
    background: NHTRenderResult
    chunks: tuple[NHTComposedChunkRecord, ...]
    appearance_model: str
    rasterization: str
    cuda_peak_bytes: int


@dataclass(frozen=True, slots=True)
class _ExpectedComposition:
    """Strict values from the public request that the result must reproduce."""

    schema: str
    frame_count: int
    object_count: int
    asset_gaussian_count: int
    chunks: tuple[tuple[int, ...], ...]
    visibility_threshold: float
    appearance_model: str


class NHTComposedRenderClient(NHTRenderClient):
    """Invoke the composed mode of ``nht-render`` and validate every output."""

    def render_composed(
        self,
        request: NHTComposedRenderCommandRequest,
        *,
        environment: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> NHTComposedRenderResult:
        if not isinstance(request, NHTComposedRenderCommandRequest):
            raise TypeError("request must be NHTComposedRenderCommandRequest.")
        if timeout_seconds is not None and timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive when provided.")
        expected = _load_expected_composition(request.composition_request_path)
        scene, _cache_hit = self._validated_scene(request.base.scene_path)
        cameras = request.base.arbitrary_cameras
        camera_path = request.base.arbitrary_request_path
        if cameras is None or camera_path is None:
            raise ValueError("Composed NHT rendering requires arbitrary cameras.")
        cameras.write(camera_path)
        child_environment: dict[str, str] | None = None
        if environment is not None:
            child_environment = dict(os.environ)
            child_environment.update(environment)
        subprocess.run(
            list(request.argv()),
            check=True,
            shell=False,
            timeout=timeout_seconds,
            env=child_environment,
        )
        return _load_composed_result(request, scene, expected)


def _load_composed_result(
    request: NHTComposedRenderCommandRequest,
    scene: StandardSceneExport,
    expected: _ExpectedComposition,
) -> NHTComposedRenderResult:
    output = request.base.output_directory
    marker = output / "render.json"
    if marker.is_symlink() or not marker.is_file():
        raise FileNotFoundError(f"nht-render did not publish composed render.json: {marker}")
    payload = _mapping(_load_json(marker), _RESULT_KEYS, "result")
    _expect(payload["schema"], NHT_COMPOSED_RENDER_RESULT_SCHEMA, "result.schema")
    _expect(payload["scene_schema"], "nht_standard_scene_v1", "result.scene_schema")
    _expect(payload["scene_id"], scene.scene_id, "result.scene_id")
    _expect(
        payload["coordinate_space"],
        "canonical NHT scene space",
        "result.coordinate_space",
    )
    _expect(payload["background"], "background/render.json", "result.background")
    composition = _mapping(payload["composition"], _COMPOSITION_KEYS, "composition")
    _expect(composition["request_schema"], expected.schema, "request_schema")
    _expect(composition["frame_count"], expected.frame_count, "frame_count")
    _expect(composition["object_count"], expected.object_count, "object_count")
    _expect(
        composition["asset_gaussian_count"],
        expected.asset_gaussian_count,
        "asset_gaussian_count",
    )
    _expect(
        composition["visibility_threshold"],
        expected.visibility_threshold,
        "visibility_threshold",
    )
    _expect(
        composition["appearance_model"],
        expected.appearance_model,
        "appearance_model",
    )
    _expect(
        composition["rasterization"],
        "joint_3dgs_eval3d_transmittance_v1",
        "rasterization",
    )

    background_request = NHTRenderCommandRequest(
        scene_path=request.base.scene_path,
        output_directory=output / "background",
        arbitrary_cameras=request.base.arbitrary_cameras,
        arbitrary_request_path=request.base.arbitrary_request_path,
        executable=request.base.executable,
    )
    background = _load_render_result(background_request, scene)
    expected_camera_ids = request.expected_camera_ids
    arbitrary_cameras = request.base.arbitrary_cameras
    if arbitrary_cameras is None:
        raise ValueError("Composed NHT rendering requires arbitrary cameras.")
    first_camera = arbitrary_cameras.cameras[0]
    if any(
        camera.width != first_camera.width or camera.height != first_camera.height
        for camera in arbitrary_cameras.cameras
    ):
        raise ValueError("Composed NHT chunks require one camera resolution.")
    raw_chunks = payload["chunks"]
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(expected.chunks):
        raise ValueError("NHT composed chunk inventory differs from its request.")
    chunks = []
    for index, (raw_value, expected_frames) in enumerate(
        zip(raw_chunks, expected.chunks, strict=True)
    ):
        raw = _mapping(raw_value, _CHUNK_KEYS, f"chunks[{index}]")
        chunk_id = f"chunk-{index:06d}"
        _expect(raw["chunk_id"], chunk_id, f"chunks[{index}].chunk_id")
        frames = expected_frames
        _expect(raw["frame_indices"], list(frames), f"chunks[{index}].frame_indices")
        _expect(raw["camera_ids"], list(expected_camera_ids), f"chunks[{index}].camera_ids")
        sample_count = _integer(raw["sample_count"], f"chunks[{index}].sample_count")
        if sample_count != len(frames) * len(expected_camera_ids):
            raise ValueError("NHT composed sample_count differs from frame×camera inventory.")
        pixel_count = _integer(raw["pixel_count"], f"chunks[{index}].pixel_count", minimum=0)
        relative = raw["arrays"]
        if relative != f"chunks/{chunk_id}/composed.npz":
            raise ValueError("NHT composed chunk arrays path is not canonical.")
        arrays_path = output / relative
        if arrays_path.is_symlink() or not arrays_path.is_file():
            raise FileNotFoundError(f"NHT composed chunk arrays are missing: {arrays_path}")
        record = NHTComposedChunkRecord(
            chunk_id=chunk_id,
            frame_indices=frames,
            camera_ids=expected_camera_ids,
            sample_count=sample_count,
            pixel_count=pixel_count,
            arrays_path=arrays_path,
            width=first_camera.width,
            height=first_camera.height,
            object_count=expected.object_count,
        )
        chunks.append(record)
    cuda_peak = _integer(payload["cuda_peak_bytes"], "cuda_peak_bytes", minimum=0)
    return NHTComposedRenderResult(
        scene_id=scene.scene_id,
        output_directory=output.resolve(strict=True),
        background=background,
        chunks=tuple(chunks),
        appearance_model=expected.appearance_model,
        rasterization="joint_3dgs_eval3d_transmittance_v1",
        cuda_peak_bytes=cuda_peak,
    )


def _load_expected_composition(path: Path) -> _ExpectedComposition:
    payload = _mapping(
        _load_json(path),
        {"schema", "asset", "timeline", "visibility_threshold"},
        "composition request",
    )
    _expect(
        payload["schema"],
        "nht_composed_render_request_v1",
        "composition request schema",
    )
    asset = _mapping(
        payload["asset"],
        {
            "asset_id",
            "coordinate_space",
            "appearance_model",
            "gaussian_count",
            "tensors",
        },
        "composition asset",
    )
    _expect(
        asset["coordinate_space"],
        "right_handed_asset_local_metres",
        "asset coordinate_space",
    )
    _expect(
        asset["appearance_model"],
        "direct_linear_rgb",
        "asset appearance_model",
    )
    _ordinary_sibling(path, asset["tensors"], name="asset tensors")
    asset_count = _integer(asset["gaussian_count"], "asset gaussian_count")

    timeline = _mapping(
        payload["timeline"],
        {
            "coordinate_space",
            "frame_count",
            "object_count",
            "object_ids",
            "instance_ids",
            "tensors",
            "chunks",
        },
        "composition timeline",
    )
    _expect(
        timeline["coordinate_space"],
        "canonical NHT scene space",
        "timeline coordinate_space",
    )
    _ordinary_sibling(path, timeline["tensors"], name="timeline tensors")
    frame_count = _integer(timeline["frame_count"], "timeline frame_count")
    object_count = _integer(timeline["object_count"], "timeline object_count")
    object_ids = _list(timeline["object_ids"], "timeline object_ids")
    if (
        len(object_ids) != object_count
        or any(not isinstance(value, str) or not value for value in object_ids)
        or len(set(object_ids)) != object_count
    ):
        raise ValueError("Timeline object IDs must be non-empty, unique, and complete.")
    instance_ids = _list(timeline["instance_ids"], "timeline instance IDs")
    if instance_ids != list(range(1, object_count + 1)):
        raise ValueError("Timeline instance IDs must exactly equal 1..object_count.")
    raw_chunks = _list(timeline["chunks"], "timeline chunks")
    chunks: list[tuple[int, ...]] = []
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        chunk = _mapping(
            raw_chunk,
            {"chunk_index", "frame_indices"},
            f"timeline chunks[{chunk_index}]",
        )
        _expect(chunk["chunk_index"], chunk_index, f"timeline chunks[{chunk_index}].index")
        frames = tuple(
            _integer(value, f"timeline chunks[{chunk_index}].frame", minimum=0)
            for value in _list(
                chunk["frame_indices"],
                f"timeline chunks[{chunk_index}].frame_indices",
            )
        )
        if not frames:
            raise ValueError("Timeline chunks must not be empty.")
        chunks.append(frames)
    if tuple(frame for chunk in chunks for frame in chunk) != tuple(range(frame_count)):
        raise ValueError("Timeline chunks must cover every frame exactly once in order.")

    visibility = _finite_number(payload["visibility_threshold"], "visibility threshold")
    if not 0.0 < visibility < 1.0:
        raise ValueError("Composition visibility threshold is out of range.")
    return _ExpectedComposition(
        schema="nht_composed_render_request_v1",
        frame_count=frame_count,
        object_count=object_count,
        asset_gaussian_count=asset_count,
        chunks=tuple(chunks),
        visibility_threshold=visibility,
        appearance_model="direct_linear_rgb",
    )


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"JSON file must be an ordinary file: {path}")

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON number {value!r} is forbidden in {path}.")

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


def _mapping(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} contains unknown or missing fields.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return dict(value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _ordinary_sibling(owner: Path, value: object, *, name: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise ValueError(f"{name} must be one sibling filename.")
    path = owner.parent / value
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {path}")
    return path


def _expect(actual: object, expected: object, name: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{name} disagrees with the request: {actual!r} != {expected!r}.")


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


__all__ = [
    "NHT_COMPOSED_RENDER_RESULT_SCHEMA",
    "NHTComposedChunkArrays",
    "NHTComposedChunkRecord",
    "NHTComposedRenderClient",
    "NHTComposedRenderCommandRequest",
    "NHTComposedRenderResult",
]
