"""Shared rays for a GoPro-inspired equidistant view after pinhole composition."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .head_camera import HeadCameraConfig


@dataclass(frozen=True, slots=True)
class EquidistantLensConfig:
    """Reproducible approximation, not manufacturer-calibrated GoPro optics.

    Output uses r=f*theta and rectangular pixel centres. The enclosing pinhole
    and its explicit oversampling are common to Blender and NHT. All depth
    composition takes place there before this display projection is applied.
    """

    width: int = 640
    height: int = 360
    diagonal_fov_degrees: float = 159.0
    render_scale: int = 4
    warp_samples_per_axis: int = 2

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or x <= 0
            for x in (
                self.width,
                self.height,
                self.render_scale,
                self.warp_samples_per_axis,
            )
        ):
            raise ValueError(
                "Lens dimensions and render_scale must be positive integers"
            )
        if (
            isinstance(self.diagonal_fov_degrees, bool)
            or not np.isfinite(self.diagonal_fov_degrees)
            or not 0 < self.diagonal_fov_degrees < 180
        ):
            raise ValueError(
                "Equidistant diagonal FOV must be finite in (0,180) degrees"
            )
        half = np.deg2rad(self.diagonal_fov_degrees / 2)
        if self.render_scale < math.ceil(np.tan(half) / half):
            raise ValueError("render_scale undersamples the output optical-axis region")

    @property
    def focal_pixels(self) -> float:
        return float(
            np.hypot(self.width, self.height) / np.deg2rad(self.diagonal_fov_degrees)
        )

    @property
    def horizontal_fov_degrees(self) -> float:
        return float(np.rad2deg(self.width / self.focal_pixels))

    @property
    def vertical_fov_degrees(self) -> float:
        return float(np.rad2deg(self.height / self.focal_pixels))

    def pinhole_config(self, original: HeadCameraConfig) -> HeadCameraConfig:
        tangent_x = (
            np.tan(np.deg2rad(self.diagonal_fov_degrees / 2))
            * self.width
            / np.hypot(self.width, self.height)
        )
        return replace(
            original,
            width=self.width * self.render_scale,
            height=self.height * self.render_scale,
            horizontal_fov_degrees=float(np.rad2deg(2 * np.arctan(tangent_x))),
        )

    def project_directions(self, directions: NDArray) -> NDArray[np.float64]:
        rays = np.asarray(directions, np.float64)
        if (
            rays.shape[-1:] != (3,)
            or not np.isfinite(rays).all()
            or (rays[..., 2] <= 0).any()
        ):
            raise ValueError(
                "Equidistant projection requires finite forward camera rays"
            )
        radius = np.linalg.norm(rays[..., :2], axis=-1)
        theta = np.arctan2(radius, rays[..., 2])
        scale = np.divide(
            self.focal_pixels * theta,
            radius,
            out=np.zeros_like(radius),
            where=radius > 0,
        )
        return np.asarray(
            rays[..., :2] * scale[..., None] + [self.width / 2, self.height / 2],
            np.float64,
        )

    def pinhole_sampling_map(
        self, *, samples_per_axis: int = 1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return source array indices; raster pixel centres are (j+.5,i+.5)."""
        if type(samples_per_axis) is not int or samples_per_axis < 1:
            raise ValueError("Sampling density must be a positive integer")
        return _sampling_map(self, samples_per_axis)

    def metadata(self, original: HeadCameraConfig) -> dict[str, Any]:
        source = self.pinhole_config(original)
        return {
            "schema": "player_view_equidistant_lens_v1",
            "profile": "mission1-wide-inspired-approximation",
            "config": asdict(self),
            "projection": "equidistant r=f*theta; not a calibrated GoPro lens",
            "output_fov_degrees": {
                "horizontal": self.horizontal_fov_degrees,
                "vertical": self.vertical_fov_degrees,
                "diagonal": self.diagonal_fov_degrees,
            },
            "source_pinhole_config": asdict(source),
            "pixel_centres": "j+0.5,i+0.5",
            "composition_order": "common pinhole samples -> foreground/background composition -> linear RGB resampling -> display encoding",
            "resampling": "fixed output subpixel grid, bilinear in linear RGB, then average; never warp depth or coverage before composition",
            "depth_of_field": False,
            "reference_aperture_f_number": 2.8,
            "reference": "https://gopro.com/en/us/news/gopro-announces-three-cameras-mission-1-2026",
        }


@lru_cache(maxsize=4)
def _sampling_map(
    lens: EquidistantLensConfig,
    samples_per_axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    xx: NDArray[np.float64]
    yy: NDArray[np.float64]
    xx, yy = np.meshgrid(
        (np.arange(lens.width * samples_per_axis) + 0.5) / samples_per_axis
        - lens.width / 2,
        (np.arange(lens.height * samples_per_axis) + 0.5) / samples_per_axis
        - lens.height / 2,
    )
    radius = np.hypot(xx, yy)
    theta = radius / lens.focal_pixels
    tangent_scale = np.divide(
        np.tan(theta),
        radius,
        out=np.full_like(radius, 1 / lens.focal_pixels),
        where=radius > 0,
    )
    source = lens.pinhole_config(HeadCameraConfig())
    focal = source.intrinsic_matrix()[0, 0]
    x = focal * xx * tangent_scale + source.width / 2 - 0.5
    y = focal * yy * tangent_scale + source.height / 2 - 0.5
    if (
        x.min() < 0
        or y.min() < 0
        or x.max() > source.width - 1
        or y.max() > source.height - 1
    ):
        raise ValueError("Lens rays exceed the enclosing pinhole sample support")
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


def remap_linear_rgb(
    image: NDArray, lens: EquidistantLensConfig
) -> NDArray[np.float32]:
    """Resample already-composited linear RGB; no depth/alpha decision here."""
    source = lens.pinhole_config(HeadCameraConfig())
    if image.shape != (source.height, source.width, 3) or image.dtype.kind != "f":
        raise ValueError(
            "Lens remapping requires float linear RGB at the source pinhole resolution"
        )
    if not np.isfinite(image).all():
        raise ValueError("Lens remapping requires finite linear RGB")
    samples = lens.warp_samples_per_axis
    x, y = lens.pinhole_sampling_map(samples_per_axis=samples)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, source.width - 1), np.minimum(y0 + 1, source.height - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    result = (1 - wy) * ((1 - wx) * image[y0, x0] + wx * image[y0, x1]) + wy * (
        (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    )
    averaged = result.reshape(lens.height, samples, lens.width, samples, 3).mean((1, 3))
    return np.asarray(averaged, dtype=np.float32)


def lens_from_metadata(
    record: dict[str, Any], source: HeadCameraConfig
) -> EquidistantLensConfig:
    """Refuse mismatched render/display projections before sampling images."""
    if record.get("schema") != "player_view_equidistant_lens_v1":
        raise ValueError("Unsupported display lens schema")
    lens = EquidistantLensConfig(**record["config"])
    if asdict(lens.pinhole_config(source)) != asdict(source) or record.get(
        "source_pinhole_config"
    ) != asdict(source):
        raise ValueError("Display lens disagrees with the source pinhole camera")
    if record.get("depth_of_field") is not False:
        raise ValueError("Display lens requires depth of field to be disabled")
    return lens


def remap_srgb_bgr8(image: NDArray, lens: EquidistantLensConfig) -> NDArray[np.uint8]:
    """Decode a Standard/sRGB Blender PNG, warp in linear RGB, then encode once."""
    if image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit sRGB BGR image")
    encoded = np.asarray(image[..., ::-1], np.float32) / 255
    linear = np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )
    result = remap_linear_rgb(linear, lens)
    srgb = np.where(
        result <= 0.0031308, result * 12.92, 1.055 * result ** (1 / 2.4) - 0.055
    )
    return np.asarray(np.rint(np.clip(srgb[..., ::-1], 0, 1) * 255), dtype=np.uint8)
