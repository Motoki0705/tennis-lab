"""Deterministic ground-court camera database with strict artifact identity."""

from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.camera_profiles import CameraSlotConfig
from src.utils.projection.camera_projector import make_look_at_camera
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

Array: TypeAlias = NDArray[Any]
POINTS = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(np.float64)
SEGMENTS = tuple((a, b) for a, b in COURT_SKELETON if a < 14 and b < 14)
SCHEMA = "court-line-db-v1"
AXES = "world: X across court, Y toward far baseline, Z up; camera: right/down/forward; pixels: integer centers"


@dataclass(frozen=True)
class DatabaseConfig:
    """Explicit synthetic prior; image_size is width, height."""

    count: int
    seed: int
    width: int
    height: int
    line_width: int
    camera: CameraSlotConfig

    def __post_init__(self) -> None:
        for name in ("count", "seed", "width", "height", "line_width"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0 or self.count < 1 or self.line_width < 1:
            raise ValueError("count/line_width must be positive and seed nonnegative")
        if min(self.width, self.height) < 32 or self.width % 16 or self.height % 16:
            raise ValueError("width/height must be multiples of 16, at least 32")
        if self.count * self.width * self.height > 20_000_000:
            raise ValueError("database exceeds baseline's 20 million pixel limit")
        CameraSlotConfig.from_mapping(asdict(self.camera))
        if (
            self.camera.height_m[0] <= 0
            or not 0 < self.camera.hfov_degrees[0] <= self.camera.hfov_degrees[1] < 179
        ):
            raise ValueError("camera height and hfov must be physically valid")
        if self.line_width > min(self.width, self.height) // 8:
            raise ValueError("line_width is too large")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DatabaseConfig:
        if set(raw) != {"count", "seed", "width", "height", "line_width", "camera"}:
            raise ValueError("database config keys do not match")
        return cls(**{**raw, "camera": CameraSlotConfig.from_mapping(raw["camera"])})

    def metadata(self) -> dict[str, Any]:
        geometry = json.dumps(
            {"points": POINTS.tolist(), "segments": SEGMENTS}, sort_keys=True
        )
        return {
            "schema": SCHEMA,
            "axes": AXES,
            "config": asdict(self),
            "source": "src.utils.schema.court + make_look_at_camera",
            "geometry_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                "".join(
                    inspect.getsource(function)
                    for function in (
                        make_look_at_camera,
                        generate,
                        render_lines,
                        descriptor,
                    )
                ).encode()
            ).hexdigest(),
            "descriptor": "numpy-HOG-v1-16block-8stride-8cell-9bins-L2Hys-no-spatial-interpolation",
            "opencv": cv2.__version__,
        }


def normalize_homography(matrix: Array) -> Array:
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-10:
        raise ValueError("invalid homography")
    h = h / h[2, 2]
    if abs(np.linalg.det(h)) < 1e-10:
        raise ValueError("singular homography")
    return h


def render_lines(h: Array, config: DatabaseConfig) -> Array:
    """Project canonical XY ground geometry; no net or camera-view relabeling."""
    h = normalize_homography(h)
    projected = np.column_stack((POINTS[:, :2], np.ones(14))) @ h.T
    if np.any(projected[:, 2] <= 1e-8):
        raise ValueError("ground court intersects or lies behind projection horizon")
    xy = projected[:, :2] / projected[:, 2:]
    if np.max(np.abs(xy)) > 1e6:
        raise ValueError("unbounded court projection")
    mask: Array = np.zeros((config.height, config.width), np.uint8)
    for a, b in SEGMENTS:
        cv2.line(
            mask,
            tuple(np.rint(xy[a]).astype(int)),
            tuple(np.rint(xy[b]).astype(int)),
            255,
            config.line_width,
            cv2.LINE_8,
        )
    if np.count_nonzero(mask) < 16:
        raise ValueError("camera produced insufficient visible court lines")
    return mask


def descriptor(mask: Array) -> Array:
    height, width = mask.shape
    gy, gx = np.gradient(mask.astype(np.float32) / 255.0)
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) % np.pi) * (9 / np.pi)
    lower = np.floor(orientation).astype(int) % 9
    fraction = orientation - np.floor(orientation)
    yy, xx = np.indices(mask.shape)
    cells: Array = np.zeros((height // 8, width // 8, 9), dtype=np.float32)
    np.add.at(cells, (yy // 8, xx // 8, lower), magnitude * (1 - fraction))
    np.add.at(cells, (yy // 8, xx // 8, (lower + 1) % 9), magnitude * fraction)
    blocks = np.concatenate(
        [
            cells[y : y + height // 8 - 1, x : x + width // 8 - 1]
            for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))
        ],
        axis=2,
    )
    blocks /= np.sqrt(np.sum(blocks**2, axis=2, keepdims=True) + 1e-10)
    blocks = np.minimum(blocks, 0.2)
    blocks /= np.sqrt(np.sum(blocks**2, axis=2, keepdims=True) + 1e-10)
    return np.asarray(blocks, dtype=np.float32).reshape(-1)


@dataclass
class LineDatabase:
    config: DatabaseConfig
    K: Array
    R: Array
    t: Array
    H: Array
    masks: Array
    descriptors: Array

    def validate(self) -> None:
        n, w, h = self.config.count, self.config.width, self.config.height
        expected = {
            "K": (n, 3, 3),
            "R": (n, 3, 3),
            "t": (n, 3),
            "H": (n, 3, 3),
            "masks": (n, h, w),
            "descriptors": (n, len(descriptor(np.zeros((h, w), np.uint8)))),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid database {name} shape or nonfinite values")
            dtype = (
                np.uint8
                if name == "masks"
                else np.float32
                if name == "descriptors"
                else np.float64
            )
            if value.dtype != dtype:
                raise ValueError(f"invalid database {name} dtype")
        if not np.allclose(
            self.R @ self.R.transpose(0, 2, 1), np.eye(3), atol=1e-6
        ) or not np.allclose(np.linalg.det(self.R), 1, atol=1e-6):
            raise ValueError("R must be proper world-to-camera rotations")
        for i in range(n):
            k = self.K[i]
            if k[0, 0] <= 0 or not np.allclose(
                k, [[k[0, 0], 0, w / 2], [0, k[0, 0], h / 2], [0, 0, 1]]
            ):
                raise ValueError("invalid intrinsic matrix")
            expected_h = normalize_homography(
                k @ np.column_stack((self.R[i, :, :2], self.t[i]))
            )
            if not np.allclose(self.H[i], expected_h):
                raise ValueError("H disagrees with K/R/t")
            if np.any((POINTS @ self.R[i].T + self.t[i])[:, 2] <= 0):
                raise ValueError("court behind camera")
            if not np.array_equal(
                self.masks[i], render_lines(self.H[i], self.config)
            ) or not np.array_equal(self.descriptors[i], descriptor(self.masks[i])):
                raise ValueError("raster or descriptor disagrees with camera")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=json.dumps(self.config.metadata(), sort_keys=True),
                **{
                    name: getattr(self, name)
                    for name in ("K", "R", "t", "H", "masks", "descriptors")
                },
            )

    @classmethod
    def load(cls, path: Path, *, expected_config: DatabaseConfig) -> LineDatabase:
        # Bound decompression before allocating arrays; this is a small CPU baseline.
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 1_000_000_000:
                raise ValueError("database archive exceeds 1 GB")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "metadata",
                "K",
                "R",
                "t",
                "H",
                "masks",
                "descriptors",
            }:
                raise ValueError("database archive keys do not match")
            if str(archive["metadata"].item()) != json.dumps(
                expected_config.metadata(), sort_keys=True
            ):
                raise ValueError("database config/source identity mismatch")
            result = cls(
                expected_config,
                **{
                    name: archive[name]
                    for name in ("K", "R", "t", "H", "masks", "descriptors")
                },
            )
        result.validate()
        return result


def generate(config: DatabaseConfig) -> LineDatabase:
    rng = np.random.default_rng(config.seed)
    slot = config.camera
    values: dict[str, list[Array]] = {
        name: [] for name in ("K", "R", "t", "H", "masks", "descriptors")
    }
    for _ in range(config.count):
        center = [
            rng.uniform(*bounds)
            for bounds in (slot.position_x_m, slot.position_y_m, slot.height_m)
        ]
        target = [
            rng.uniform(*bounds)
            for bounds in (slot.look_at_x_m, slot.look_at_y_m, slot.look_at_height_m)
        ]
        if np.linalg.norm(np.asarray(center) - target) < 1e-6:
            raise ValueError("camera center equals target")
        camera = make_look_at_camera(
            center,
            look_at=target,
            image_size=(config.width, config.height),
            hfov_deg=rng.uniform(*slot.hfov_degrees),
        )
        k = np.array(
            [[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]],
            dtype=np.float64,
        )
        r = camera.R.numpy().astype(np.float64)
        t = -r @ camera.C.numpy().astype(np.float64)
        if np.any((POINTS @ r.T + t)[:, 2] <= 0):
            raise ValueError("sampled court behind camera; adjust explicit prior")
        h = normalize_homography(k @ np.column_stack((r[:, :2], t)))
        mask = render_lines(h, config)
        for name, value in zip(
            values, (k, r, t, h, mask, descriptor(mask)), strict=True
        ):
            values[name].append(value)
    result = LineDatabase(
        config, **{name: np.stack(value) for name, value in values.items()}
    )
    result.validate()
    return result
