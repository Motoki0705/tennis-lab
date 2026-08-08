"""Typed public request/result contracts for ``nht-render`` file boundaries."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

NHT_RENDER_COMMAND = "nht-render"
NHT_RENDER_REQUEST_SCHEMA = "nht_render_request_v1"
NHT_RENDER_RESULT_SCHEMA = "nht_render_result_v1"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class NHTRenderCamera:
    """One arbitrary PINHOLE camera expressed in canonical NHT scene space."""

    camera_id: str
    width: int
    height: int
    intrinsics: tuple[float, ...]
    camera_to_scene: RigidTransform

    def __post_init__(self) -> None:
        validated = SceneCamera(
            camera_id=self.camera_id,
            source_frame_index=0,
            width=self.width,
            height=self.height,
            intrinsics=self.intrinsics,
            camera_to_scene=self.camera_to_scene,
            image_path="request-only",
        )
        matrix = np.asarray(validated.intrinsics, dtype=np.float64).reshape(3, 3)
        if not np.allclose(
            matrix,
            (
                (matrix[0, 0], 0.0, matrix[0, 2]),
                (0.0, matrix[1, 1], matrix[1, 2]),
                (0.0, 0.0, 1.0),
            ),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "NHT render intrinsics must be a canonical PINHOLE matrix."
            )
        object.__setattr__(self, "intrinsics", validated.intrinsics)

    @classmethod
    def from_scene_camera(
        cls, camera: SceneCamera, *, camera_id: str | None = None
    ) -> NHTRenderCamera:
        """Construct an arbitrary render request from validated scene camera geometry."""
        return cls(
            camera_id=camera.camera_id if camera_id is None else camera_id,
            width=camera.width,
            height=camera.height,
            intrinsics=camera.intrinsics,
            camera_to_scene=camera.camera_to_scene,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict ``nht_render_request_v1`` camera representation."""
        matrix = np.asarray(self.intrinsics, dtype=np.float64).reshape(3, 3)
        return {
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "intrinsics": {
                "model": "PINHOLE",
                "distortion_model": "NONE",
                "params": [
                    float(matrix[0, 0]),
                    float(matrix[1, 1]),
                    float(matrix[0, 2]),
                    float(matrix[1, 2]),
                ],
                "matrix": matrix.tolist(),
            },
            "camera_to_scene": self.camera_to_scene.matrix().tolist(),
        }


@dataclass(frozen=True, slots=True)
class NHTRenderRequest:
    """A strict, non-empty arbitrary-camera request written to JSON."""

    cameras: tuple[NHTRenderCamera, ...]

    def __post_init__(self) -> None:
        cameras = tuple(self.cameras)
        if not cameras:
            raise ValueError("NHT render request must contain at least one camera.")
        if any(not isinstance(camera, NHTRenderCamera) for camera in cameras):
            raise TypeError(
                "NHT render request cameras must be NHTRenderCamera values."
            )
        camera_ids = [camera.camera_id for camera in cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("NHT render request camera IDs must be unique.")
        object.__setattr__(self, "cameras", cameras)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical public request payload."""
        return {
            "schema": NHT_RENDER_REQUEST_SCHEMA,
            "cameras": [camera.to_dict() for camera in self.cameras],
        }

    def write(self, path: str | Path) -> Path:
        """Atomically replace one ordinary JSON request file."""
        target = Path(path)
        if not target.is_absolute():
            raise ValueError("NHT render request path must be absolute.")
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(
                f"NHT render request target must be an ordinary file path: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target


@dataclass(frozen=True, slots=True)
class NHTRenderCommandRequest:
    """One shell-free NHT render invocation using observed and/or request cameras."""

    scene_path: Path
    output_directory: Path
    observed_camera_ids: tuple[str, ...] = ()
    arbitrary_cameras: NHTRenderRequest | None = None
    arbitrary_request_path: Path | None = None
    executable: str | Path = NHT_RENDER_COMMAND

    def __post_init__(self) -> None:
        for name, path in (
            ("scene_path", self.scene_path),
            ("output_directory", self.output_directory),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path.")
        if self.scene_path.name != "scene.json" or not self.scene_path.is_file():
            raise FileNotFoundError(
                f"NHT render scene.json does not exist: {self.scene_path}"
            )
        if self.output_directory.is_symlink():
            raise ValueError("NHT render output must not be a symbolic link.")
        if self.output_directory.exists() and not self.output_directory.is_dir():
            raise NotADirectoryError(
                f"NHT render output is an ordinary file: {self.output_directory}"
            )
        _validate_executable(self.executable)

        observed = tuple(self.observed_camera_ids)
        if any(_PORTABLE_ID.fullmatch(camera_id) is None for camera_id in observed):
            raise ValueError("Observed NHT camera IDs must be portable identifiers.")
        if len(observed) != len(set(observed)):
            raise ValueError("Observed NHT camera IDs must be unique.")
        object.__setattr__(self, "observed_camera_ids", observed)

        if (self.arbitrary_cameras is None) != (self.arbitrary_request_path is None):
            raise ValueError(
                "arbitrary_cameras and arbitrary_request_path must be provided together."
            )
        if self.arbitrary_request_path is not None:
            if (
                not isinstance(self.arbitrary_request_path, Path)
                or not self.arbitrary_request_path.is_absolute()
            ):
                raise ValueError(
                    "arbitrary_request_path must be an absolute pathlib.Path."
                )
            output = self.output_directory.resolve(strict=False)
            request_path = self.arbitrary_request_path.resolve(strict=False)
            if request_path == output or request_path.is_relative_to(output):
                raise ValueError(
                    "NHT request file must remain outside the replaceable render output."
                )

        arbitrary_ids = (
            tuple(camera.camera_id for camera in self.arbitrary_cameras.cameras)
            if self.arbitrary_cameras is not None
            else ()
        )
        if set(observed).intersection(arbitrary_ids):
            raise ValueError("Observed and arbitrary NHT camera IDs must not overlap.")
        if not observed and not arbitrary_ids:
            raise ValueError("NHT render command must select at least one camera.")

        output_resolved = self.output_directory.resolve(strict=False)
        export_root = self.scene_path.parent.resolve(strict=True)
        workspace = export_root.parent
        if (
            output_resolved == Path(output_resolved.anchor)
            or output_resolved.is_relative_to(workspace)
            or workspace.is_relative_to(output_resolved)
        ):
            raise ValueError(
                "NHT render output must not replace a protected scene path."
            )
        if self.arbitrary_request_path is not None:
            request_resolved = self.arbitrary_request_path.resolve(strict=False)
            if request_resolved.is_relative_to(workspace):
                raise ValueError(
                    "NHT request file must stay outside the reconstruction workspace."
                )

    @property
    def expected_camera_ids(self) -> tuple[str, ...]:
        """Return the exact result camera order expected from NHT."""
        arbitrary = (
            tuple(camera.camera_id for camera in self.arbitrary_cameras.cameras)
            if self.arbitrary_cameras is not None
            else ()
        )
        return (*self.observed_camera_ids, *arbitrary)

    def argv(self) -> tuple[str, ...]:
        """Return the exact public shell-free command argument vector."""
        arguments: list[str] = [str(self.executable), "--scene", str(self.scene_path)]
        for camera_id in self.observed_camera_ids:
            arguments.extend(("--camera-id", camera_id))
        if self.arbitrary_request_path is not None:
            arguments.extend(("--cameras", str(self.arbitrary_request_path)))
        arguments.extend(("--output", str(self.output_directory)))
        return tuple(arguments)


@dataclass(frozen=True, slots=True)
class NHTRenderRecord:
    """Validated RGB, alpha, depth, and preview files for one camera."""

    camera_id: str
    request_source: str
    width: int
    height: int
    rgb_path: Path
    rgb_preview_path: Path
    alpha_path: Path
    alpha_preview_path: Path
    depth_path: Path
    _arrays: NHTRenderArrays | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def arrays(self) -> NHTRenderArrays:
        """Return the immutable arrays scanned for this exact invocation."""
        if self._arrays is None:
            raise RuntimeError(
                "NHT render arrays are unavailable because this record was not "
                "produced by NHTRenderClient."
            )
        return self._arrays

    def _bind_arrays(self, arrays: NHTRenderArrays) -> None:
        """Bind the one validated payload loaded by ``NHTRenderClient``."""
        if not isinstance(arrays, NHTRenderArrays):
            raise TypeError("NHT record arrays must be NHTRenderArrays.")
        if self._arrays is not None:
            raise RuntimeError("NHT render arrays are already bound to this record.")
        if (arrays.width, arrays.height) != (self.width, self.height):
            raise ValueError("NHT render arrays disagree with the record resolution.")
        object.__setattr__(self, "_arrays", arrays)


@dataclass(frozen=True, slots=True, eq=False)
class NHTRenderArrays:
    """One read-only RGB/alpha/depth payload loaded and scanned exactly once."""

    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    _depth_maximum: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rgb = _render_array(
            self.rgb,
            name="NHT RGB",
            channels=3,
            unit_range=True,
        )
        height, width = rgb.shape[:2]
        alpha = _render_array(
            self.alpha,
            name="NHT alpha",
            channels=1,
            expected_shape=(height, width, 1),
            unit_range=True,
        )
        depth = _render_array(
            self.depth,
            name="NHT depth",
            channels=1,
            expected_shape=(height, width, 1),
            nonnegative=True,
        )
        for value in (rgb, alpha, depth):
            value.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(
            self,
            "_depth_maximum",
            float(depth.max(initial=np.float32(0.0))),
        )

    @property
    def width(self) -> int:
        """Return the validated render width."""
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        """Return the validated render height."""
        return int(self.rgb.shape[0])

    @property
    def byte_count(self) -> int:
        """Return bytes loaded from the three public array files."""
        return int(self.rgb.nbytes + self.alpha.nbytes + self.depth.nbytes)

    def metric_depth(
        self,
        *,
        nht_scene_units_per_metre: float,
    ) -> NDArray[np.float32]:
        """Convert the cached validated depth without reopening its public file."""
        from src.synthetic_data_generation.rendering.nht.depth import (
            _depth_scale,
            _validated_nht_depth_to_metric,
        )

        return _validated_nht_depth_to_metric(
            self.depth,
            scale=_depth_scale(nht_scene_units_per_metre),
            maximum=self._depth_maximum,
        )


@dataclass(frozen=True, slots=True)
class NHTRenderEvidence:
    """Measured repository-side work for one exact public render invocation."""

    invocation_index: int
    scene_validation_count: int
    scene_cache_hit: bool
    camera_count: int
    complete_payload_scan_count: int
    array_file_load_count: int
    preview_validation_count: int
    loaded_array_bytes: int
    subprocess_wall_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.invocation_index, bool) or self.invocation_index <= 0:
            raise ValueError("NHT invocation_index must be a positive integer.")
        if self.scene_validation_count not in {0, 1}:
            raise ValueError("One NHT invocation performs zero or one scene validation.")
        if self.scene_cache_hit != (self.scene_validation_count == 0):
            raise ValueError("NHT scene cache evidence is internally inconsistent.")
        if isinstance(self.camera_count, bool) or self.camera_count <= 0:
            raise ValueError("NHT evidence camera_count must be positive.")
        if self.complete_payload_scan_count != self.camera_count:
            raise ValueError("NHT must scan each complete camera payload exactly once.")
        if self.array_file_load_count != 3 * self.camera_count:
            raise ValueError("NHT must load exactly three public arrays per camera.")
        if self.preview_validation_count != 2 * self.camera_count:
            raise ValueError("NHT must validate exactly two previews per camera.")
        if isinstance(self.loaded_array_bytes, bool) or self.loaded_array_bytes < 0:
            raise ValueError("NHT loaded_array_bytes must be non-negative.")
        if (
            isinstance(self.subprocess_wall_seconds, bool)
            or not np.isfinite(self.subprocess_wall_seconds)
            or self.subprocess_wall_seconds < 0.0
        ):
            raise ValueError("NHT subprocess wall time must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class NHTRenderResult:
    """A complete validated ``nht_render_result_v1`` directory."""

    scene_id: str
    output_directory: Path
    records: tuple[NHTRenderRecord, ...]
    _evidence: NHTRenderEvidence | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def record(self, camera_id: str) -> NHTRenderRecord:
        """Return one camera result without selecting a fallback."""
        matches = [record for record in self.records if record.camera_id == camera_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown NHT render camera ID: {camera_id!r}.")
        return matches[0]

    @property
    def evidence(self) -> NHTRenderEvidence:
        """Return measured work for the invocation that produced this result."""
        if self._evidence is None:
            raise RuntimeError(
                "NHT render evidence is unavailable because this result was not "
                "produced by NHTRenderClient."
            )
        return self._evidence

    def _bind_evidence(self, evidence: NHTRenderEvidence) -> None:
        """Bind exact invocation evidence once after complete result validation."""
        if not isinstance(evidence, NHTRenderEvidence):
            raise TypeError("NHT result evidence must be NHTRenderEvidence.")
        if self._evidence is not None:
            raise RuntimeError("NHT render evidence is already bound to this result.")
        if evidence.camera_count != len(self.records):
            raise ValueError("NHT render evidence disagrees with the result inventory.")
        object.__setattr__(self, "_evidence", evidence)


def _render_array(
    value: NDArray[np.float32],
    *,
    name: str,
    channels: int,
    expected_shape: tuple[int, int, int] | None = None,
    unit_range: bool = False,
    nonnegative: bool = False,
) -> NDArray[np.float32]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}.")
    if array.ndim != 3 or array.shape[2] != channels:
        raise ValueError(f"{name} must have shape [H,W,{channels}], got {array.shape}.")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    if unit_range and (np.any(array < 0.0) or np.any(array > 1.0)):
        raise ValueError(f"{name} must stay in [0, 1].")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _validate_executable(executable: str | Path) -> None:
    if isinstance(executable, str):
        if executable != NHT_RENDER_COMMAND:
            raise ValueError("String render executable must be exactly nht-render.")
        return
    if not isinstance(executable, Path) or not executable.is_absolute():
        raise ValueError("Path render executable must be an absolute pathlib.Path.")
    if executable.name != NHT_RENDER_COMMAND:
        raise ValueError("Render executable basename must be nht-render.")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"nht-render executable is unavailable: {executable}")


__all__ = [
    "NHT_RENDER_COMMAND",
    "NHT_RENDER_REQUEST_SCHEMA",
    "NHT_RENDER_RESULT_SCHEMA",
    "NHTRenderCamera",
    "NHTRenderCommandRequest",
    "NHTRenderEvidence",
    "NHTRenderArrays",
    "NHTRenderRecord",
    "NHTRenderRequest",
    "NHTRenderResult",
]
