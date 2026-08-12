"""Subprocess client for the independent NHT renderer's public file boundary."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
    validate_standard_scene_export,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHT_RENDER_RESULT_SCHEMA,
    NHTRenderArrays,
    NHTRenderCommandRequest,
    NHTRenderEvidence,
    NHTRenderRecord,
    NHTRenderResult,
)

_RESULT_KEYS = {
    "schema",
    "scene_schema",
    "scene_id",
    "coordinate_space",
    "export_validation",
    "renders",
}
_RECORD_KEYS = {
    "camera_id",
    "request_source",
    "width",
    "height",
    "rgb",
    "rgb_preview",
    "alpha",
    "alpha_preview",
    "depth",
}


class NHTRenderClient:
    """Validate inputs, invoke ``nht-render``, and validate complete result files."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._scene_cache: dict[Path, _SceneCacheEntry] = {}
        self._invocation_count = 0

    def validate_scene(self, scene_path: Path) -> StandardSceneExport:
        """Validate or reuse one unchanged public scene in this client session."""
        scene, _cache_hit = self._validated_scene(scene_path)
        return scene

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> NHTRenderResult:
        """Execute one command without importing or inspecting renderer internals."""
        if timeout_seconds is not None and timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive when provided.")
        scene, scene_cache_hit = self._validated_scene(request.scene_path)
        observed = {camera.camera_id: camera for camera in scene.cameras}
        unknown_observed = set(request.observed_camera_ids) - set(observed)
        if unknown_observed:
            raise ValueError(
                f"NHT render request references unknown observed cameras: {sorted(unknown_observed)}."
            )
        if request.arbitrary_cameras is not None:
            assert request.arbitrary_request_path is not None
            request.arbitrary_cameras.write(request.arbitrary_request_path)

        child_environment = None
        if environment is not None:
            child_environment = dict(os.environ)
            child_environment.update(environment)
        with self._lock:
            self._invocation_count += 1
            invocation_index = self._invocation_count
        started = time.perf_counter()
        subprocess.run(
            list(request.argv()),
            check=True,
            shell=False,
            timeout=timeout_seconds,
            env=child_environment,
        )
        subprocess_wall_seconds = time.perf_counter() - started
        result = _load_render_result(request, scene)
        loaded_array_bytes = sum(
            record.validated_array_byte_count for record in result.records
        )
        maximum_live_array_bytes = max(
            record.validated_array_byte_count for record in result.records
        )
        result._bind_evidence(
            NHTRenderEvidence(
                invocation_index=invocation_index,
                scene_validation_count=0 if scene_cache_hit else 1,
                scene_cache_hit=scene_cache_hit,
                camera_count=len(result.records),
                complete_payload_scan_count=len(result.records),
                array_file_load_count=3 * len(result.records),
                preview_validation_count=2 * len(result.records),
                loaded_array_bytes=loaded_array_bytes,
                maximum_live_array_bytes=maximum_live_array_bytes,
                retained_array_bytes=sum(
                    record.retained_array_byte_count for record in result.records
                ),
                subprocess_wall_seconds=subprocess_wall_seconds,
            )
        )
        return result

    def _validated_scene(
        self,
        scene_path: Path,
    ) -> tuple[StandardSceneExport, bool]:
        """Validate once, then reuse only while every validated input is unchanged."""
        if not isinstance(scene_path, Path) or not scene_path.is_absolute():
            raise ValueError("NHT scene cache requires an absolute pathlib.Path.")
        resolved = scene_path.resolve(strict=True)
        with self._lock:
            cached = self._scene_cache.get(resolved)
            if cached is not None and cached.matches(scene_path):
                return cached.scene, True
            scene = validate_standard_scene_export(scene_path)
            entry = _SceneCacheEntry.capture(scene_path, scene)
            self._scene_cache[resolved] = entry
            return scene, False


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    """Non-content identity used only to invalidate one process-local cache."""

    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def capture(cls, path: Path) -> _FileIdentity:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FileNotFoundError(
                f"Validated NHT scene dependency is unavailable: {path}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Validated NHT scene dependency is not an ordinary file: {path}"
            )
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class _SceneCacheEntry:
    """One validated export and identities for every file its validation consumed."""

    requested_scene_path: Path
    scene: StandardSceneExport
    dependencies: tuple[tuple[Path, _FileIdentity], ...]

    @classmethod
    def capture(
        cls,
        requested_scene_path: Path,
        scene: StandardSceneExport,
    ) -> _SceneCacheEntry:
        paths = {
            requested_scene_path,
            scene.scene_path,
            scene.export_root / "cameras.json",
            scene.export_root / "points_scene.npy",
            scene.checkpoint_path,
            scene.runtime_config_path,
            *(Path(camera.image_path) for camera in scene.cameras),
        }
        dependencies = tuple(
            (path, _FileIdentity.capture(path))
            for path in sorted(paths, key=lambda item: str(item))
        )
        return cls(
            requested_scene_path=requested_scene_path,
            scene=scene,
            dependencies=dependencies,
        )

    def matches(self, requested_scene_path: Path) -> bool:
        if requested_scene_path != self.requested_scene_path:
            return False
        if requested_scene_path.is_symlink() or requested_scene_path.parent.is_symlink():
            return False
        try:
            return all(
                _FileIdentity.capture(path) == expected
                for path, expected in self.dependencies
            )
        except (FileNotFoundError, ValueError):
            return False


def _load_render_result(
    request: NHTRenderCommandRequest,
    scene: StandardSceneExport,
) -> NHTRenderResult:
    output = request.output_directory
    marker = output / "render.json"
    if marker.is_symlink() or not marker.is_file():
        raise FileNotFoundError(
            f"nht-render did not publish an ordinary render.json: {marker}"
        )
    payload = _mapping(_load_json(marker), keys=_RESULT_KEYS, name="render.json")
    _expect(payload["schema"], NHT_RENDER_RESULT_SCHEMA, name="render.schema")
    _expect(
        payload["scene_schema"], "nht_standard_scene_v1", name="render.scene_schema"
    )
    _expect(payload["scene_id"], scene.scene_id, name="render.scene_id")
    _expect(
        payload["coordinate_space"],
        "canonical NHT scene space",
        name="render.coordinate_space",
    )
    _mapping(payload["export_validation"], name="render.export_validation")
    raw_records = _sequence(payload["renders"], name="render.renders")
    if len(raw_records) != len(request.expected_camera_ids):
        raise ValueError(
            "NHT render result count disagrees with the request: "
            f"expected={len(request.expected_camera_ids)}, actual={len(raw_records)}."
        )

    observed = {camera.camera_id: camera for camera in scene.cameras}
    arbitrary = (
        {camera.camera_id: camera for camera in request.arbitrary_cameras.cameras}
        if request.arbitrary_cameras is not None
        else {}
    )
    records: list[NHTRenderRecord] = []
    for index, (raw_record, expected_id) in enumerate(
        zip(raw_records, request.expected_camera_ids, strict=True)
    ):
        name = f"render.renders[{index}]"
        raw = _mapping(raw_record, keys=_RECORD_KEYS, name=name)
        _expect(raw["camera_id"], expected_id, name=f"{name}.camera_id")
        expected_source = "observed" if expected_id in observed else "arbitrary"
        _expect(raw["request_source"], expected_source, name=f"{name}.request_source")
        expected_camera = observed.get(expected_id) or arbitrary[expected_id]
        width = _integer(raw["width"], name=f"{name}.width", minimum=1)
        height = _integer(raw["height"], name=f"{name}.height", minimum=1)
        if (width, height) != (expected_camera.width, expected_camera.height):
            raise ValueError(f"{name} resolution disagrees with the requested camera.")

        rgb_path = _result_file(
            output,
            raw["rgb"],
            camera_id=expected_id,
            filename="rgb.npy",
            name=f"{name}.rgb",
        )
        alpha_path = _result_file(
            output,
            raw["alpha"],
            camera_id=expected_id,
            filename="alpha.npy",
            name=f"{name}.alpha",
        )
        depth_path = _result_file(
            output,
            raw["depth"],
            camera_id=expected_id,
            filename="depth.npy",
            name=f"{name}.depth",
        )
        rgb_preview_path = _result_file(
            output,
            raw["rgb_preview"],
            camera_id=expected_id,
            filename="rgb.png",
            name=f"{name}.rgb_preview",
        )
        alpha_preview_path = _result_file(
            output,
            raw["alpha_preview"],
            camera_id=expected_id,
            filename="alpha.png",
            name=f"{name}.alpha_preview",
        )
        arrays = NHTRenderArrays(
            rgb=np.load(rgb_path, allow_pickle=False),
            alpha=np.load(alpha_path, allow_pickle=False),
            depth=np.load(depth_path, allow_pickle=False),
        )
        if (arrays.width, arrays.height) != (width, height):
            raise ValueError(f"{name} array resolution disagrees with the request.")
        _validate_preview(
            rgb_preview_path, width=width, height=height, name=f"{name}.rgb_preview"
        )
        _validate_preview(
            alpha_preview_path,
            width=width,
            height=height,
            name=f"{name}.alpha_preview",
        )
        record = NHTRenderRecord(
            camera_id=expected_id,
            request_source=expected_source,
            width=width,
            height=height,
            rgb_path=rgb_path,
            rgb_preview_path=rgb_preview_path,
            alpha_path=alpha_path,
            alpha_preview_path=alpha_preview_path,
            depth_path=depth_path,
        )
        record._bind_arrays(arrays)
        del arrays
        records.append(record)

    return NHTRenderResult(
        scene_id=scene.scene_id,
        output_directory=output.resolve(strict=True),
        records=tuple(records),
    )


def _validate_preview(path: Path, *, width: int, height: int, name: str) -> None:
    try:
        with Image.open(path) as image:
            actual_size = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"{name} is not a readable preview image.") from error
    if actual_size != (width, height):
        raise ValueError(
            f"{name} resolution {actual_size} disagrees with {(width, height)}."
        )


def _result_file(
    output: Path,
    value: object,
    *,
    camera_id: str,
    filename: str,
    name: str,
) -> Path:
    reference = _string(value, name=name)
    if "\\" in reference:
        raise ValueError(f"{name} must use a result-relative POSIX path.")
    pure = PurePosixPath(reference)
    if pure.parts != (camera_id, filename):
        raise ValueError(f"{name} must be exactly {camera_id}/{filename}.")
    root = output.resolve(strict=True)
    candidate = output.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {candidate}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"{name} must resolve to a file inside the render output.")
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


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


__all__ = ["NHTRenderClient"]
