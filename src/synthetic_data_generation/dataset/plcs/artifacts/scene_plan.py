"""Versioned single/multi-person PLCS scene plans in court metres."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

PLCS_SCHEDULE_SCHEMA = "tennis_plcs_person_schedule_v1"
POSE_IDS = ("canonical", "ready", "forehand")
Mode = Literal["single", "multi"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _array_hash(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = (
        str(contiguous.dtype).encode()
        + json.dumps(contiguous.shape).encode()
        + contiguous.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(
    *,
    mode: Mode,
    seed: int,
    fps: float,
    identity_ids: tuple[str, ...],
    arrays: dict[str, NDArray[np.generic]],
) -> str:
    payload = {
        "schema": PLCS_SCHEDULE_SCHEMA,
        "mode": mode,
        "seed": seed,
        "fps": fps,
        "pose_ids": POSE_IDS,
        "identity_ids": identity_ids,
        "arrays": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _array_hash(array),
            }
            for name, array in sorted(arrays.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PLCSPersonSchedule:
    """Complete identity, placement, rotation, pose, and presence schedule."""

    schema: str
    schedule_fingerprint: str
    mode: Mode
    seed: int
    fps: float
    identity_ids: tuple[str, ...]
    instance_ids: IntArray
    positions_court_m: FloatArray
    velocities_court_mps: FloatArray
    yaw_radians: FloatArray
    pose_indices: IntArray
    present: BoolArray

    def __post_init__(self) -> None:
        if self.schema != PLCS_SCHEDULE_SCHEMA:
            raise ValueError("Unsupported PLCS person schedule schema.")
        if self.mode not in {"single", "multi"}:
            raise ValueError("mode must be single or multi.")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be positive and finite.")
        identities = tuple(self.identity_ids)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("identity_ids must be non-empty and unique.")
        if any(not value.strip() for value in identities):
            raise ValueError("identity_ids must not contain an empty ID.")

        positions = np.asarray(self.positions_court_m, dtype=np.float64)
        velocities = np.asarray(self.velocities_court_mps, dtype=np.float64)
        yaw = np.asarray(self.yaw_radians, dtype=np.float64)
        poses = np.asarray(self.pose_indices)
        present = np.asarray(self.present)
        instances = np.asarray(self.instance_ids)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("positions_court_m must have shape [T,N,3].")
        frame_count, person_count, _ = positions.shape
        if frame_count < 2 or person_count != len(identities):
            raise ValueError(
                "Schedule shape must match identities and have >=2 frames."
            )
        if velocities.shape != positions.shape:
            raise ValueError("velocities_court_mps must match positions.")
        expected_person_shape = (frame_count, person_count)
        if yaw.shape != expected_person_shape:
            raise ValueError("yaw_radians must have shape [T,N].")
        if poses.shape != expected_person_shape or not np.issubdtype(
            poses.dtype,
            np.integer,
        ):
            raise ValueError("pose_indices must be integer [T,N].")
        if present.shape != expected_person_shape or present.dtype != np.bool_:
            raise ValueError("present must be boolean [T,N].")
        if instances.shape != (person_count,) or not np.issubdtype(
            instances.dtype,
            np.integer,
        ):
            raise ValueError("instance_ids must be integer [N].")
        if not all(np.isfinite(array).all() for array in (positions, velocities, yaw)):
            raise ValueError("Schedule contains NaN or infinity.")
        if not np.array_equal(instances, np.arange(1, person_count + 1)):
            raise ValueError(
                "instance_ids must be contiguous one-based person columns."
            )
        if np.any(poses < 0) or np.any(poses >= len(POSE_IDS)):
            raise ValueError("pose_indices reference an unsupported pose.")
        if not present.all():
            raise ValueError("PLCS schedules do not silently drop frames.")
        if not np.allclose(positions[..., 2], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("positions_court_m encode ground footprints with z=0.")
        if np.any(np.abs(positions[..., 0]) > 4.115) or np.any(
            np.abs(positions[..., 1]) > 11.885
        ):
            raise ValueError("A person footprint leaves the singles court.")
        if person_count > 1:
            difference = positions[:, 0, :2] - positions[:, 1, :2]
            if np.any(np.linalg.norm(difference, axis=1) < 1.5):
                raise ValueError("Multi-person footprints collide.")

        arrays: dict[str, NDArray[np.generic]] = {
            "instance_ids": instances.astype(np.int64, copy=False),
            "positions_court_m": positions,
            "velocities_court_mps": velocities,
            "yaw_radians": yaw,
            "pose_indices": poses.astype(np.int64, copy=False),
            "present": present,
        }
        expected = _fingerprint(
            mode=self.mode,
            seed=self.seed,
            fps=float(self.fps),
            identity_ids=identities,
            arrays=arrays,
        )
        if self.schedule_fingerprint != expected:
            raise ValueError("PLCS schedule fingerprint differs.")
        object.__setattr__(self, "identity_ids", identities)
        for name, array in arrays.items():
            readonly = np.ascontiguousarray(array)
            readonly.setflags(write=False)
            object.__setattr__(self, name, readonly)

    @classmethod
    def create(
        cls,
        *,
        mode: Mode,
        seed: int,
        fps: float,
        identity_ids: tuple[str, ...],
        instance_ids: IntArray,
        positions_court_m: FloatArray,
        velocities_court_mps: FloatArray,
        yaw_radians: FloatArray,
        pose_indices: IntArray,
        present: BoolArray,
    ) -> Self:
        arrays: dict[str, NDArray[np.generic]] = {
            "instance_ids": instance_ids,
            "positions_court_m": positions_court_m,
            "velocities_court_mps": velocities_court_mps,
            "yaw_radians": yaw_radians,
            "pose_indices": pose_indices,
            "present": present,
        }
        return cls(
            schema=PLCS_SCHEDULE_SCHEMA,
            schedule_fingerprint=_fingerprint(
                mode=mode,
                seed=seed,
                fps=fps,
                identity_ids=identity_ids,
                arrays=arrays,
            ),
            mode=mode,
            seed=seed,
            fps=fps,
            identity_ids=identity_ids,
            instance_ids=instance_ids,
            positions_court_m=positions_court_m,
            velocities_court_mps=velocities_court_mps,
            yaw_radians=yaw_radians,
            pose_indices=pose_indices,
            present=present,
        )

    @property
    def frame_count(self) -> int:
        return int(self.positions_court_m.shape[0])

    @property
    def person_count(self) -> int:
        return int(self.positions_court_m.shape[1])


def build_person_schedule(
    *,
    mode: Mode,
    seed: int,
    frame_count: int = 12,
    fps: float = 30.0,
) -> PLCSPersonSchedule:
    """Build a bounded tennis movement with stable identities and pose indices."""
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be single or multi.")
    if isinstance(frame_count, bool) or frame_count < 6:
        raise ValueError("frame_count must be an integer of at least six.")
    rng = np.random.default_rng(seed)
    person_count = 1 if mode == "single" else 2
    time = np.linspace(0.0, 1.0, frame_count)
    positions: FloatArray = np.zeros((frame_count, person_count, 3), dtype=np.float64)
    phase = rng.uniform(-0.2, 0.2, size=person_count)
    amplitude = rng.uniform(1.5, 2.2, size=person_count)
    positions[:, 0, 0] = amplitude[0] * np.sin(np.pi * (time + phase[0]))
    positions[:, 0, 1] = -9.4 + 0.35 * np.sin(2.0 * np.pi * time)
    if person_count == 2:
        positions[:, 1, 0] = amplitude[1] * np.cos(np.pi * (time + phase[1]))
        positions[:, 1, 1] = 9.4 - 0.35 * np.cos(2.0 * np.pi * time)

    velocities = np.gradient(positions, 1.0 / fps, axis=0, edge_order=2)
    yaw: FloatArray = np.zeros((frame_count, person_count), dtype=np.float64)
    yaw[:, 0] = np.pi
    pose_indices: IntArray = np.ones((frame_count, person_count), dtype=np.int64)
    strike_start = frame_count // 3
    strike_stop = min(frame_count, strike_start + max(2, frame_count // 3))
    pose_indices[strike_start:strike_stop, 0] = 2
    if person_count == 2:
        shifted_start = min(frame_count - 2, strike_start + 2)
        shifted_stop = min(frame_count, shifted_start + max(2, frame_count // 3))
        pose_indices[shifted_start:shifted_stop, 1] = 2
    return PLCSPersonSchedule.create(
        mode=mode,
        seed=seed,
        fps=fps,
        identity_ids=tuple(f"person-{index:03d}" for index in range(person_count)),
        instance_ids=np.arange(1, person_count + 1, dtype=np.int64),
        positions_court_m=positions,
        velocities_court_mps=velocities,
        yaw_radians=yaw,
        pose_indices=pose_indices,
        present=np.ones((frame_count, person_count), dtype=np.bool_),
    )
