"""Explicit image sources for measured court-line inference."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.reconstruction.scene_export import (
    NHT_CAMERA_COORDINATE_CONVENTION,
    NHT_IMAGE_RESOLUTION_SEMANTICS,
    NHT_PIXEL_COORDINATE_CONVENTION,
    NHT_SCENE_COORDINATE_CONVENTION,
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht.client import NHTRenderClient
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHT_RENDER_REQUEST_SCHEMA,
    NHT_RENDER_RESULT_SCHEMA,
    NHTRenderCommandRequest,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

_INPUT_BATCH_SCHEMA = "alignment_line_input_batch_v1"
_NHT_RENDER_PROVENANCE_SCHEMA = "alignment_nht_rendered_rgb_v1"
_CAPTURED_PROVENANCE_SCHEMA = "alignment_captured_rgb_v1"


class CourtLineInputSource(StrEnum):
    """The image authority actually supplied to the court-line detector."""

    CAPTURED_RGB = "captured_rgb"
    NHT_RENDERED_RGB = "nht_rendered_rgb"


@dataclass(frozen=True, slots=True)
class AlignmentLineInputView:
    """One exact uint8 RGB detector input bound to its projection camera."""

    camera: SceneCamera
    image_rgb: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if not isinstance(self.camera, SceneCamera):
            raise TypeError("Alignment line input camera must be a SceneCamera.")
        image = np.asarray(self.image_rgb)
        expected_shape = (self.camera.height, self.camera.width, 3)
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError(
                "Alignment line input must be uint8 RGB at the camera resolution; "
                f"camera={self.camera.camera_id!r}, expected={expected_shape}, "
                f"actual={image.shape}/{image.dtype}."
            )
        image = np.array(image, dtype=np.uint8, order="C", copy=True)
        image.setflags(write=False)
        object.__setattr__(self, "image_rgb", image)

    @property
    def input_rgb_sha256(self) -> str:
        """Return a shape-bound digest of the exact detector RGB values."""
        return input_rgb_sha256(self.image_rgb)


@dataclass(frozen=True, slots=True)
class AlignmentLineInputBatch:
    """A fixed camera-ordered image batch with canonical source provenance."""

    source: CourtLineInputSource
    provenance: Mapping[str, object]
    views: tuple[AlignmentLineInputView, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, CourtLineInputSource):
            raise TypeError("Alignment line input source must be CourtLineInputSource.")
        provenance = _canonical_mapping(self.provenance, name="input provenance")
        views = tuple(self.views)
        if not views or any(
            not isinstance(view, AlignmentLineInputView) for view in views
        ):
            raise TypeError("Alignment line inputs must contain typed views.")
        camera_ids = tuple(view.camera.camera_id for view in views)
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Alignment line input camera IDs must be unique.")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "views", views)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return the exact input order."""
        return tuple(view.camera.camera_id for view in self.views)

    def image(self, camera_id: str) -> NDArray[np.uint8]:
        """Return one immutable RGB input without selecting a fallback."""
        matches = [
            view.image_rgb for view in self.views if view.camera.camera_id == camera_id
        ]
        if len(matches) != 1:
            raise KeyError(f"Unknown alignment line input camera ID: {camera_id!r}.")
        return matches[0]

    def cache_identity(self) -> dict[str, object]:
        """Identify the source, render settings, cameras, and exact RGB values."""
        return {
            "schema": _INPUT_BATCH_SCHEMA,
            "source": self.source.value,
            "provenance": _canonical_mapping(
                self.provenance,
                name="input provenance",
            ),
            "views": [
                {
                    "camera_id": view.camera.camera_id,
                    "source_frame_index": view.camera.source_frame_index,
                    "width": view.camera.width,
                    "height": view.camera.height,
                    "intrinsics": list(view.camera.intrinsics),
                    "camera_to_scene": view.camera.camera_to_scene.matrix().tolist(),
                    "input_rgb_sha256": view.input_rgb_sha256,
                }
                for view in self.views
            ],
        }


class AlignmentLineInputSource(Protocol):
    """Acquire detector RGBs for an exact set of public scene cameras."""

    def preflight(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> None:
        """Validate the source without publishing alignment outputs."""

    def load(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> AlignmentLineInputBatch:
        """Load the exact RGB inputs in camera order."""


@dataclass(frozen=True, slots=True)
class CapturedAlignmentLineInputSource:
    """Explicit non-production source retained for measured-source unit use."""

    def preflight(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> None:
        """Require every selected public captured image to be decodable."""
        _require_exact_scene_cameras(scene, cameras)
        for camera in cameras:
            _load_rgb_image(Path(camera.image_path), camera=camera, label="captured")

    def load(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> AlignmentLineInputBatch:
        """Load captured RGBs only when this source was explicitly selected."""
        self.preflight(scene, cameras)
        return AlignmentLineInputBatch(
            source=CourtLineInputSource.CAPTURED_RGB,
            provenance={
                "schema": _CAPTURED_PROVENANCE_SCHEMA,
                "scene_id": scene.scene_id,
                "scene_path": str(scene.scene_path),
                "image_authority": "StandardSceneExport.cameras[].image_path",
            },
            views=tuple(
                AlignmentLineInputView(
                    camera=camera,
                    image_rgb=_load_rgb_image(
                        Path(camera.image_path),
                        camera=camera,
                        label="captured",
                    ),
                )
                for camera in cameras
            ),
        )


@dataclass(frozen=True, slots=True)
class NHTRenderedAlignmentLineInputSource:
    """Render observed public cameras through the standalone NHT boundary."""

    client: NHTRenderClient
    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not callable(getattr(self.client, "render", None)):
            raise TypeError("NHT alignment input client must provide render().")
        if not isinstance(self.executable, (str, Path)) or not str(self.executable):
            raise TypeError("NHT render executable must be a non-empty path or name.")
        environment: dict[str, str] = {}
        for key, value in self.environment.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise TypeError(
                    "NHT render environment must contain trimmed non-empty strings."
                )
            environment[key] = value
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0.0
        ):
            raise ValueError("NHT render timeout must be positive and finite.")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def preflight(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> None:
        """Validate public scene/model/camera and executable authorities."""
        _require_exact_scene_cameras(scene, cameras)
        for path, label in (
            (scene.scene_path, "scene.json"),
            (scene.export_root / "cameras.json", "cameras.json"),
            (scene.checkpoint_path, "NHT checkpoint"),
            (scene.runtime_config_path, "NHT runtime config"),
        ):
            _require_ordinary_file(path, label=label)
        _resolve_executable(self.executable)

    def load(
        self,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> AlignmentLineInputBatch:
        """Render once and return only the validated public RGB previews."""
        self.preflight(scene, cameras)
        camera_ids = tuple(camera.camera_id for camera in cameras)
        with tempfile.TemporaryDirectory(prefix="tennis-lab-alignment-render-") as root:
            output_directory = Path(root) / "renders"
            request = NHTRenderCommandRequest(
                scene_path=scene.scene_path,
                output_directory=output_directory,
                observed_camera_ids=camera_ids,
                executable=self.executable,
            )
            result = self.client.render(
                request,
                environment=self.environment,
                timeout_seconds=self.timeout_seconds,
            )
            if result.scene_id != scene.scene_id:
                raise ValueError("NHT line-input render returned a different scene_id.")
            records = tuple(result.records)
            if tuple(record.camera_id for record in records) != camera_ids:
                raise ValueError(
                    "NHT line-input render changed the requested camera order."
                )
            views: list[AlignmentLineInputView] = []
            for camera, record in zip(cameras, records, strict=True):
                if record.request_source != "observed":
                    raise ValueError(
                        "Alignment line rendering must use observed scene cameras."
                    )
                if (record.width, record.height) != (camera.width, camera.height):
                    raise ValueError(
                        "NHT line-input render changed the camera resolution."
                    )
                views.append(
                    AlignmentLineInputView(
                        camera=camera,
                        image_rgb=_load_rgb_image(
                            record.rgb_preview_path,
                            camera=camera,
                            label="NHT rendered",
                        ),
                    )
                )
        return AlignmentLineInputBatch(
            source=CourtLineInputSource.NHT_RENDERED_RGB,
            provenance=self._provenance(scene=scene, cameras=cameras),
            views=tuple(views),
        )

    def _provenance(
        self,
        *,
        scene: StandardSceneExport,
        cameras: tuple[SceneCamera, ...],
    ) -> dict[str, object]:
        executable = _resolve_executable(self.executable)
        return {
            "schema": _NHT_RENDER_PROVENANCE_SCHEMA,
            "scene_id": scene.scene_id,
            "scene": _file_identity(scene.scene_path, root=scene.export_root),
            "cameras": _file_identity(
                scene.export_root / "cameras.json",
                root=scene.export_root,
            ),
            "checkpoint": _file_identity(
                scene.checkpoint_path,
                root=scene.export_root,
            ),
            "runtime_config": _file_identity(
                scene.runtime_config_path,
                root=scene.export_root,
            ),
            "renderer": {
                "request_schema": NHT_RENDER_REQUEST_SCHEMA,
                "result_schema": NHT_RENDER_RESULT_SCHEMA,
                "executable": _file_identity(executable, root=None),
                "configured_executable": str(self.executable),
                "environment": dict(sorted(self.environment.items())),
                "timeout_seconds": self.timeout_seconds,
                "request_source": "observed",
                "rgb_input": "validated nht-render rgb.png converted to RGB",
            },
            "camera_ids": [camera.camera_id for camera in cameras],
            "camera_coordinate_convention": NHT_CAMERA_COORDINATE_CONVENTION,
            "scene_coordinate_convention": NHT_SCENE_COORDINATE_CONVENTION,
            "pixel_coordinate_convention": NHT_PIXEL_COORDINATE_CONVENTION,
            "image_resolution_semantics": NHT_IMAGE_RESOLUTION_SEMANTICS,
        }


def input_rgb_sha256(image_rgb: NDArray[np.uint8]) -> str:
    """Hash exact HWC RGB values together with their shape and dtype contract."""
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2:] != (3,):
        raise ValueError("Court-line input digest requires uint8 HxWx3 RGB.")
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(b"court-line-input-rgb-uint8-hwc-v1\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _load_rgb_image(
    path: Path,
    *,
    camera: SceneCamera,
    label: str,
) -> NDArray[np.uint8]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"{label} image for {camera.camera_id!r} is not an ordinary absolute file: "
            f"{path}."
        )
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"Unable to decode {label} image for {camera.camera_id!r}: {path}."
        ) from error
    if rgb.shape != (camera.height, camera.width, 3):
        raise ValueError(
            f"{label} image shape disagrees for {camera.camera_id!r}: {rgb.shape}."
        )
    return rgb


def _require_exact_scene_cameras(
    scene: StandardSceneExport,
    cameras: tuple[SceneCamera, ...],
) -> None:
    if not cameras:
        raise ValueError("Alignment line input requires at least one camera.")
    for camera in cameras:
        if not isinstance(camera, SceneCamera):
            raise TypeError("Alignment line input cameras must be SceneCamera values.")
        if scene.camera(camera.camera_id) != camera:
            raise ValueError(
                "Alignment line input camera must be the exact StandardSceneExport "
                f"record: {camera.camera_id!r}."
            )


def _resolve_executable(value: str | Path) -> Path:
    text = str(value)
    candidate = Path(text)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        resolved = candidate.expanduser().resolve(strict=True)
    else:
        found = shutil.which(text)
        if found is None:
            raise FileNotFoundError(f"NHT render executable was not found: {text!r}.")
        resolved = Path(found).resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"NHT render executable is not a file: {resolved}.")
    return resolved


def _require_ordinary_file(path: Path, *, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not an ordinary absolute file: {path}.")


def _file_identity(path: Path, *, root: Path | None) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Provenance input is not a file: {resolved}.")
    reference = (
        resolved.relative_to(root.resolve(strict=True)).as_posix()
        if root is not None and resolved.is_relative_to(root.resolve(strict=True))
        else str(resolved)
    )
    return {
        "path": reference,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, object]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        decoded: Any = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite JSON mapping.") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise TypeError(f"{name} must be a string-keyed JSON mapping.")
    return cast(dict[str, object], decoded)


__all__ = [
    "AlignmentLineInputBatch",
    "AlignmentLineInputSource",
    "AlignmentLineInputView",
    "CapturedAlignmentLineInputSource",
    "CourtLineInputSource",
    "NHTRenderedAlignmentLineInputSource",
    "input_rgb_sha256",
]
