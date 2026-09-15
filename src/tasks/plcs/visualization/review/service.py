"""Catalog and world-joint reconstruction for the ACCAD review UI.

Every answer is derived read-only from the raw ``data/ACCAD/**/*_poses.npz``
archives. Joint positions use the repository's SMPL-H recipe
(:mod:`src.synthetic_data_generation.dataset.plcs.smplh`): shape blend, then
root-relative rigid joint transforms, then the AMASS root translation. The
result is the source frame of :class:`PLCSCoordinateContract`
(``plcs_amass_smplh_z_up_v1``: right-handed, ``+Z`` up, metres), which is the
world coordinate system the review UI draws.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Final, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from smplx.lbs import (  # type: ignore[import-untyped]
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)

from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSCoordinateContract,
)
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHModelData,
    load_smplh_model,
)
from src.utils.schema.player import (
    NUM_SMPLH_BODY_JOINTS,
    SMPLH_FULL_SKELETON,
    SMPLH_JOINT_NAMES,
)

POSE_WIDTH: Final = 156
JOINT_COUNT: Final = len(SMPLH_JOINT_NAMES)
MOTION_SUFFIX: Final = "_poses.npz"
GENDERS: Final = ("female", "male", "neutral")

FloatArray: TypeAlias = NDArray[np.float64]


def finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float or raise a typed error."""
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def display_name(motion: str) -> str:
    """Return the human motion name stored in one archive file name."""
    if not motion.endswith(MOTION_SUFFIX):
        raise ValueError(f"Motion file must end with {MOTION_SUFFIX!r}: {motion!r}")
    return motion[: -len(MOTION_SUFFIX)]


def world_contract() -> dict[str, object]:
    """Return the PLCS world-frame identity plus this viewer's applied offset."""
    return {**PLCSCoordinateContract().to_dict(), "root_translation_applied": True}


@dataclass(frozen=True, slots=True)
class MotionSummary:
    """Immutable metadata read from one ACCAD archive."""

    subject: str
    motion: str
    gender: str
    fps: float
    frame_count: int
    beta_count: int
    size_bytes: int
    revision: str

    @property
    def name(self) -> str:
        return display_name(self.motion)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.motion,
            "name": self.name,
            "gender": self.gender,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "size_bytes": self.size_bytes,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class SMPLHJointModel:
    """The only SMPL-H arrays needed to place joints, without pose directions."""

    gender: str
    template_vertices_m: FloatArray
    shape_directions: FloatArray
    joint_regressor: FloatArray
    parents: NDArray[np.int64]

    @property
    def beta_count(self) -> int:
        return int(self.shape_directions.shape[2])

    @classmethod
    def load(cls, model_root: Path, *, gender: str) -> SMPLHJointModel:
        """Load and validate one gender model through the production loader."""
        validated: SMPLHModelData = load_smplh_model(model_root, gender=gender)
        if validated.beta_count <= 0:
            raise ValueError("SMPL-H shape basis must not be empty.")
        return cls(
            gender=validated.gender,
            template_vertices_m=np.array(validated.template_vertices_m, copy=True),
            shape_directions=np.array(validated.shape_directions, copy=True),
            joint_regressor=np.array(validated.joint_regressor, copy=True),
            parents=np.array(validated.parents, copy=True),
        )


def world_joint_positions(
    model: SMPLHJointModel,
    poses: FloatArray,
    trans: FloatArray,
    betas: FloatArray,
) -> NDArray[np.float32]:
    """Return ``(T, 52, 3)`` SMPL-H joints in the AMASS world frame.

    ``poses`` keeps the exact AMASS ordering
    ``[global_orient, body, left_hand, right_hand]``; no pose mean is added, so
    the reconstruction matches the PLCS linear-blend skinning contract.
    """
    if poses.ndim != 2 or poses.shape[1] != POSE_WIDTH:
        raise ValueError(f"poses must have shape [T,{POSE_WIDTH}], got {poses.shape}.")
    frame_count = int(poses.shape[0])
    if frame_count == 0:
        raise ValueError("A motion must contain at least one frame.")
    if trans.shape != (frame_count, 3):
        raise ValueError(
            f"trans must have shape [{frame_count},3], got {trans.shape}."
        )
    if betas.shape != (model.beta_count,):
        raise ValueError(
            f"betas must have shape [{model.beta_count}] to match the SMPL-H "
            f"shape basis, got {betas.shape}."
        )
    if not np.isfinite(poses).all():
        raise ValueError("poses contains NaN or infinity.")
    if not np.isfinite(trans).all():
        raise ValueError("trans contains NaN or infinity.")
    if not np.isfinite(betas).all():
        raise ValueError("betas contains NaN or infinity.")

    template = torch.from_numpy(model.template_vertices_m)
    shapedirs = torch.from_numpy(model.shape_directions)
    regressor = torch.from_numpy(model.joint_regressor)
    parents = torch.from_numpy(model.parents)
    pose_tensor = torch.from_numpy(np.ascontiguousarray(poses))
    beta_tensor = torch.from_numpy(np.ascontiguousarray(betas))

    with torch.inference_mode():
        expanded = beta_tensor.unsqueeze(0).expand(frame_count, -1)
        shaped = template + blend_shapes(expanded, shapedirs)
        canonical = vertices2joints(regressor, shaped)
        rotations = batch_rodrigues(pose_tensor.reshape(-1, 3)).reshape(
            frame_count, JOINT_COUNT, 3, 3
        )
        joints, _ = batch_rigid_transform(
            rotations, canonical, parents, dtype=pose_tensor.dtype
        )
        world = joints + torch.from_numpy(np.ascontiguousarray(trans)).unsqueeze(1)

    result = world.numpy().astype(np.float32, copy=False)
    if result.shape != (frame_count, JOINT_COUNT, 3):
        raise RuntimeError(f"SMPL-H joint reconstruction produced {result.shape}.")
    if not np.isfinite(result).all():
        raise RuntimeError("SMPL-H joint reconstruction produced NaN or infinity.")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class LoadedMotion:
    """One sampled motion ready to serialize for the browser."""

    summary: MotionSummary
    stride: int
    joints: NDArray[np.float32]

    @property
    def sampled_frame_count(self) -> int:
        return int(self.joints.shape[0])


def sampled_frame_count(frame_count: int, stride: int) -> int:
    """Return how many frames ``range(0, frame_count, stride)`` selects."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    return (frame_count + stride - 1) // stride


class ReviewService:
    """Read-only catalog and SMPL-H reconstruction for one ACCAD root."""

    def __init__(
        self,
        accad_root: Path,
        smplh_root: Path,
        *,
        motion_cache_size: int = 4,
    ) -> None:
        root = Path(accad_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"ACCAD root is not a directory: {root}")
        models = Path(smplh_root).resolve(strict=True)
        if not models.is_dir():
            raise NotADirectoryError(f"SMPL-H model root is not a directory: {models}")
        self.root = root
        self.smplh_root = models
        self._lock = RLock()
        self._models: dict[str, SMPLHJointModel] = {}
        self._cached_motion = lru_cache(maxsize=motion_cache_size)(self._load_motion)
        self._cached_summary = lru_cache(maxsize=2048)(self._read_summary)

    # ---------------------------------------------------------------- catalog

    def catalog(self) -> dict[str, Any]:
        """Return every subject and motion file with cheap on-disk metadata."""
        subjects = []
        for subject_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            motions = [
                self._summary_for(path).to_dict()
                for path in sorted(subject_dir.glob(f"*{MOTION_SUFFIX}"))
                if path.is_file()
            ]
            if motions:
                subjects.append(
                    {
                        "id": subject_dir.name,
                        "motion_count": len(motions),
                        "motions": motions,
                    }
                )
        return {
            "root": str(self.root),
            "smplh_root": str(self.smplh_root),
            "world": world_contract(),
            "joints": {
                "count": JOINT_COUNT,
                "body_joint_count": NUM_SMPLH_BODY_JOINTS,
                "names": list(SMPLH_JOINT_NAMES),
                "edges": [list(edge) for edge in SMPLH_FULL_SKELETON],
            },
            "subjects": subjects,
        }

    def _summary_for(self, path: Path) -> MotionSummary:
        stat = path.stat()
        return self._cached_summary(str(path), stat.st_size, stat.st_mtime_ns)

    def _read_summary(
        self, path_str: str, size_bytes: int, mtime_ns: int
    ) -> MotionSummary:
        path = Path(path_str)
        with np.load(path, allow_pickle=False) as archive:
            gender = str(archive["gender"])
            fps = finite_float(archive["mocap_framerate"], name="mocap_framerate")
            frame_count = int(archive["poses"].shape[0])
            beta_count = int(archive["betas"].shape[0])
        if gender not in GENDERS:
            raise ValueError(f"Unsupported SMPL-H gender {gender!r} in {path.name}.")
        if fps <= 0.0:
            raise ValueError(f"mocap_framerate must be positive in {path.name}.")
        if frame_count <= 0 or beta_count <= 0:
            raise ValueError(f"Empty poses or betas in {path.name}.")
        revision = hashlib.sha256(f"{size_bytes}:{mtime_ns}".encode()).hexdigest()[:20]
        return MotionSummary(
            subject=path.parent.name,
            motion=path.name,
            gender=gender,
            fps=fps,
            frame_count=frame_count,
            beta_count=beta_count,
            size_bytes=size_bytes,
            revision=revision,
        )

    # ----------------------------------------------------------------- motion

    def motion_path(self, subject: str, motion: str) -> Path:
        """Resolve one motion inside the configured root, or raise."""
        for label, value in (("subject", subject), ("motion", motion)):
            if not value or Path(value).name != value or value in {".", ".."}:
                raise ValueError(f"{label} must be a single path component.")
        if not motion.endswith(MOTION_SUFFIX):
            raise ValueError(f"Motion file must end with {MOTION_SUFFIX!r}.")
        if "/" in motion or "\\" in motion:
            raise ValueError("Motion file must not contain a path separator.")
        path = (self.root / subject / motion).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Motion file escapes the configured ACCAD root.")
        return path

    def motion_meta(
        self, subject: str, motion: str, *, stride: int = 1
    ) -> dict[str, Any]:
        """Return identity, skeleton, and world-frame metadata for one motion."""
        if stride <= 0:
            raise ValueError("stride must be positive.")
        summary = self._summary_for(self.motion_path(subject, motion))
        return {
            **summary.to_dict(),
            "stride": stride,
            "source_frame_count": summary.frame_count,
            "sampled_frame_count": sampled_frame_count(summary.frame_count, stride),
            "duration_s": summary.duration_s,
            "world": world_contract(),
            "joints": {
                "count": JOINT_COUNT,
                "body_joint_count": NUM_SMPLH_BODY_JOINTS,
                "names": list(SMPLH_JOINT_NAMES),
                "edges": [list(edge) for edge in SMPLH_FULL_SKELETON],
            },
        }

    def motion_joints(
        self,
        subject: str,
        motion: str,
        *,
        stride: int = 1,
        revision: str | None = None,
    ) -> LoadedMotion:
        """Return sampled world joints, rejecting a stale catalog revision."""
        if stride <= 0:
            raise ValueError("stride must be positive.")
        return self._cached_motion(subject, motion, stride, revision)

    def _load_motion(
        self, subject: str, motion: str, stride: int, revision: str | None
    ) -> LoadedMotion:
        path = self.motion_path(subject, motion)
        summary = self._summary_for(path)
        if revision is not None and revision != summary.revision:
            raise RuntimeError("Motion changed on disk. Reload the catalog.")
        with np.load(path, allow_pickle=False) as archive:
            poses = np.asarray(archive["poses"], dtype=np.float64)[::stride]
            trans = np.asarray(archive["trans"], dtype=np.float64)[::stride]
            betas = np.asarray(archive["betas"], dtype=np.float64).reshape(-1)
        model = self._model(summary.gender)
        joints = world_joint_positions(model, poses, trans, betas)
        if self._summary_for(path).revision != summary.revision:
            raise RuntimeError("Motion changed while loading. Reload the catalog.")
        return LoadedMotion(summary=summary, stride=stride, joints=joints)

    def _model(self, gender: str) -> SMPLHJointModel:
        with self._lock:
            model = self._models.get(gender)
            if model is None:
                model = SMPLHJointModel.load(self.smplh_root, gender=gender)
                self._models[gender] = model
            return model


__all__ = [
    "JOINT_COUNT",
    "MOTION_SUFFIX",
    "LoadedMotion",
    "MotionSummary",
    "ReviewService",
    "SMPLHJointModel",
    "display_name",
    "finite_float",
    "sampled_frame_count",
    "world_contract",
    "world_joint_positions",
]
