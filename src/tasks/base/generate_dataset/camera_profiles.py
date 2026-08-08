"""Config-owned deterministic camera profiles shared by BLCS and PLCS."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    RigidTransform,
    SceneCamera,
)
from src.utils.projection.camera_projector import make_look_at_camera


def _range(value: object, *, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{name} must contain exactly two numeric values.")
    low, high = value
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
    ):
        raise TypeError(f"{name} must contain exactly two numeric values.")
    result = (float(low), float(high))
    if not all(math.isfinite(item) for item in result) or result[0] > result[1]:
        raise ValueError(f"{name} must be a finite increasing range.")
    return result


def _strict(value: object, *, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{name} keys do not match; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}."
        )
    return value


@dataclass(frozen=True, slots=True)
class CameraSlotConfig:
    """One camera slot whose entire sampling envelope comes from config."""

    slot_id: str
    position_x_m: tuple[float, float]
    position_y_m: tuple[float, float]
    height_m: tuple[float, float]
    look_at_x_m: tuple[float, float]
    look_at_y_m: tuple[float, float]
    look_at_height_m: tuple[float, float]
    hfov_degrees: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object) -> CameraSlotConfig:
        """Parse one slot and reject unknown/defaulted fields."""
        raw = _strict(
            value,
            name="camera slot",
            keys={
                "slot_id",
                "position_x_m",
                "position_y_m",
                "height_m",
                "look_at_x_m",
                "look_at_y_m",
                "look_at_height_m",
                "hfov_degrees",
            },
        )
        slot_id = raw["slot_id"]
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise TypeError("camera slot_id must be a non-empty string.")
        return cls(
            slot_id=slot_id,
            position_x_m=_range(raw["position_x_m"], name="position_x_m"),
            position_y_m=_range(raw["position_y_m"], name="position_y_m"),
            height_m=_range(raw["height_m"], name="height_m"),
            look_at_x_m=_range(raw["look_at_x_m"], name="look_at_x_m"),
            look_at_y_m=_range(raw["look_at_y_m"], name="look_at_y_m"),
            look_at_height_m=_range(raw["look_at_height_m"], name="look_at_height_m"),
            hfov_degrees=_range(raw["hfov_degrees"], name="hfov_degrees"),
        )


@dataclass(frozen=True, slots=True)
class CameraProfileConfig:
    """Strict authoritative camera profile mapping."""

    profile: str
    image_size: tuple[int, int]
    expected_camera_count: int
    slots: tuple[CameraSlotConfig, ...]

    @classmethod
    def from_mapping(cls, value: object) -> CameraProfileConfig:
        """Parse a profile with no Python fallback values."""
        raw = _strict(
            value,
            name="camera profile",
            keys={"profile", "image_size", "expected_camera_count", "slots"},
        )
        profile = raw["profile"]
        if profile not in {"default", "broadcast"}:
            raise ValueError("camera profile must be exactly 'default' or 'broadcast'.")
        image_size = raw["image_size"]
        if (
            not isinstance(image_size, Sequence)
            or isinstance(image_size, (str, bytes))
            or len(image_size) != 2
        ):
            raise TypeError("camera image_size must contain width and height.")
        width, height = image_size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 1
            or height <= 1
        ):
            raise ValueError(
                "camera image_size values must be integers greater than one."
            )
        expected = raw["expected_camera_count"]
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise ValueError("expected_camera_count must be a positive integer.")
        slots_raw = raw["slots"]
        if not isinstance(slots_raw, Sequence) or isinstance(slots_raw, (str, bytes)):
            raise TypeError("camera slots must be a sequence.")
        slots = tuple(CameraSlotConfig.from_mapping(item) for item in slots_raw)
        if len(slots) != expected:
            raise ValueError("camera slot count must equal expected_camera_count.")
        expected_for_profile = 6 if profile == "default" else 2
        if expected != expected_for_profile:
            raise ValueError(
                f"{profile} profile must declare {expected_for_profile} camera slots."
            )
        if len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("camera slot_id values must be unique.")
        return cls(
            profile=profile,
            image_size=(width, height),
            expected_camera_count=expected,
            slots=slots,
        )


@dataclass(frozen=True, slots=True)
class SampledCamera:
    """One sampled court-local camera and its scene-space equivalent."""

    slot_id: str
    court_local_center_m: tuple[float, float, float]
    court_local_look_at_m: tuple[float, float, float]
    hfov_degrees: float
    scene_camera: SceneCamera

    def to_metadata(self) -> dict[str, object]:
        """Return complete sampled parameters for dataset provenance."""
        return {
            "slot_id": self.slot_id,
            "court_local_center_m": list(self.court_local_center_m),
            "court_local_look_at_m": list(self.court_local_look_at_m),
            "hfov_degrees": self.hfov_degrees,
            "camera": self.scene_camera.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SampledCameraRig:
    """Deterministic sampled camera profile for one target court."""

    profile: str
    seed: int
    court_instance_id: str
    cameras: tuple[SampledCamera, ...]


def _sample(bounds: tuple[float, float], rng: random.Random) -> float:
    return bounds[0] if bounds[0] == bounds[1] else rng.uniform(*bounds)


def sample_camera_rig(
    config: CameraProfileConfig,
    *,
    seed: int,
    court: CourtInstance,
) -> SampledCameraRig:
    """Sample config-bounded cameras and transform them with one court binding."""
    rng = random.Random(seed)
    sampled: list[SampledCamera] = []
    for slot in config.slots:
        center = (
            _sample(slot.position_x_m, rng),
            _sample(slot.position_y_m, rng),
            _sample(slot.height_m, rng),
        )
        look_at = (
            _sample(slot.look_at_x_m, rng),
            _sample(slot.look_at_y_m, rng),
            _sample(slot.look_at_height_m, rng),
        )
        hfov = _sample(slot.hfov_degrees, rng)
        local = make_look_at_camera(
            center,
            look_at=look_at,
            image_size=config.image_size,
            hfov_deg=hfov,
        )
        camera_to_court = np.eye(4, dtype=np.float64)
        camera_to_court[:3, :3] = local.R.detach().cpu().numpy().astype(np.float64).T
        camera_to_court[:3, 3] = local.C.detach().cpu().numpy().astype(np.float64)
        camera_to_scene = court.scene_from_court.matrix() @ camera_to_court
        intrinsics = (
            local.f,
            0.0,
            local.cx,
            0.0,
            local.f,
            local.cy,
            0.0,
            0.0,
            1.0,
        )
        scene_camera = SceneCamera(
            camera_id=f"{court.court_instance_id}-{slot.slot_id}",
            source_frame_index=0,
            width=local.w,
            height=local.h,
            intrinsics=intrinsics,
            camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
            image_path=f"generated/{config.profile}/{slot.slot_id}.png",
        )
        sampled.append(
            SampledCamera(
                slot_id=slot.slot_id,
                court_local_center_m=center,
                court_local_look_at_m=look_at,
                hfov_degrees=hfov,
                scene_camera=scene_camera,
            )
        )
    return SampledCameraRig(
        profile=config.profile,
        seed=seed,
        court_instance_id=court.court_instance_id,
        cameras=tuple(sampled),
    )


def assert_projection_equivalent(
    sampled: SampledCamera,
    court: CourtInstance,
    points_court: NDArray[np.floating],
    *,
    atol: float,
) -> None:
    """Prove court-local and scene-space projection are numerically equivalent."""
    if atol <= 0.0 or not math.isfinite(atol):
        raise ValueError("Projection tolerance must be finite and positive.")
    local = make_look_at_camera(
        sampled.court_local_center_m,
        look_at=sampled.court_local_look_at_m,
        image_size=(sampled.scene_camera.width, sampled.scene_camera.height),
        hfov_deg=sampled.hfov_degrees,
    )
    expected_intrinsics = np.asarray(
        (
            local.f,
            0.0,
            local.cx,
            0.0,
            local.f,
            local.cy,
            0.0,
            0.0,
            1.0,
        ),
        dtype=np.float64,
    )
    actual_intrinsics = np.asarray(sampled.scene_camera.intrinsics, dtype=np.float64)
    if not np.allclose(actual_intrinsics, expected_intrinsics, atol=atol, rtol=0.0):
        raise ValueError("Generated camera intrinsics disagree with local authority.")
    camera_to_court = np.eye(4, dtype=np.float64)
    camera_to_court[:3, :3] = local.R.detach().cpu().numpy().astype(np.float64).T
    camera_to_court[:3, 3] = local.C.detach().cpu().numpy().astype(np.float64)
    expected_camera_to_scene = court.scene_from_court.matrix() @ camera_to_court
    if not np.allclose(
        sampled.scene_camera.camera_to_scene.matrix(),
        expected_camera_to_scene,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Generated scene camera disagrees with independent court-local authority."
        )
    scene_points = court.scene_from_court.apply(points_court)
    scene_pixels, _ = sampled.scene_camera.project_scene_points(scene_points)
    local_camera = SceneCamera(
        camera_id=sampled.scene_camera.camera_id,
        source_frame_index=0,
        width=sampled.scene_camera.width,
        height=sampled.scene_camera.height,
        intrinsics=sampled.scene_camera.intrinsics,
        camera_to_scene=RigidTransform.from_matrix(camera_to_court),
        image_path=sampled.scene_camera.image_path,
    )
    local_pixels, _ = local_camera.project_scene_points(points_court)
    if not np.allclose(local_pixels, scene_pixels, atol=atol, rtol=0.0):
        raise ValueError("Court-local and scene-space camera projections disagree.")


__all__ = [
    "CameraProfileConfig",
    "CameraSlotConfig",
    "SampledCamera",
    "SampledCameraRig",
    "assert_projection_equivalent",
    "sample_camera_rig",
]
