"""Renderer-independent contract for one captured static scene frame."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import SceneCamera

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SceneFrame:
    """RGB, camera-Z depth, and accumulated scene alpha for one camera."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    alpha: NDArray[np.float32]
    scene_fingerprint: str
    camera_id: str
    backend_id: str
    backend_version: str

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("Scene-frame RGB must have shape (H, W, 3).")
        if self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("Scene-frame depth shape must match RGB.")
        if self.alpha.shape != self.rgb.shape[:2]:
            raise ValueError("Scene-frame alpha shape must match RGB.")
        if self.rgb.dtype != np.uint8:
            raise ValueError("Scene-frame RGB must use uint8.")
        if self.depth.dtype != np.float32 or self.alpha.dtype != np.float32:
            raise ValueError("Scene-frame depth and alpha must use float32.")
        if np.isnan(self.depth).any() or bool(np.any(self.depth <= 0.0)):
            raise ValueError("Scene-frame depth must be positive or infinity.")
        if not np.isfinite(self.alpha).all() or bool(
            np.any((self.alpha < 0.0) | (self.alpha > 1.0))
        ):
            raise ValueError("Scene-frame alpha must lie in [0, 1].")
        if _SHA256_PATTERN.fullmatch(self.scene_fingerprint) is None:
            raise ValueError("Scene-frame fingerprint must be SHA-256.")
        for name, value in (
            ("camera_id", self.camera_id),
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
        ):
            if not value.strip():
                raise ValueError(f"Scene-frame {name} must not be empty.")
        self.rgb.setflags(write=False)
        self.depth.setflags(write=False)
        self.alpha.setflags(write=False)


@runtime_checkable
class SceneFrameRendererPort(Protocol):
    """Narrow port for rendering the reconstructed scene without primitives."""

    def render_scene_frame(
        self,
        *,
        scene_fingerprint: str,
        camera: SceneCamera,
    ) -> SceneFrame:
        """Render one accepted captured camera."""
        ...
