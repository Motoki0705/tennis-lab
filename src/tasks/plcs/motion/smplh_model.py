"""Strict SMPL-H loading and bounded CUDA Gaussian linear-blend skinning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.plcs.motion.coordinates import (
    SMPLH_SURFACE_VERTEX_COUNT,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_VERTEX_COUNT = SMPLH_SURFACE_VERTEX_COUNT
_JOINT_COUNT = 52
_POSE_BLEND_WIDTH = (_JOINT_COUNT - 1) * 9
_REQUIRED_KEYS = {
    "v_template",
    "f",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "kintree_table",
    "weights",
}


def _finite_float(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return cast(FloatArray, np.ascontiguousarray(array))


def _integer(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> IntArray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return cast(IntArray, np.ascontiguousarray(array, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class SMPLHModelData:
    """Validated official SMPL-H arrays for one explicit gender."""

    model_path: Path
    gender: str
    template_vertices_m: FloatArray
    faces: IntArray
    shape_directions: FloatArray
    pose_directions: FloatArray
    joint_regressor: FloatArray
    parents: IntArray
    vertex_joint_weights: FloatArray

    def __post_init__(self) -> None:
        if self.gender not in {"female", "male", "neutral"}:
            raise ValueError("SMPL-H gender must be female, male, or neutral.")
        if self.model_path.name != "model.npz" or not self.model_path.is_file():
            raise FileNotFoundError(
                f"Expected an explicit licensed SMPL-H model.npz: {self.model_path}"
            )
        template = _finite_float(
            self.template_vertices_m,
            name="template_vertices_m",
            shape=(_VERTEX_COUNT, 3),
        )
        faces = _integer(self.faces, name="faces")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            raise ValueError("SMPL-H faces must have non-empty shape [F,3].")
        if np.any(faces < 0) or np.any(faces >= _VERTEX_COUNT):
            raise ValueError("SMPL-H faces contain an out-of-range vertex.")
        shapedirs = _finite_float(self.shape_directions, name="shape_directions")
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != (_VERTEX_COUNT, 3):
            raise ValueError("SMPL-H shapedirs must have shape [6890,3,B].")
        if shapedirs.shape[2] <= 0:
            raise ValueError("SMPL-H shapedirs must contain at least one beta basis.")
        posedirs = _finite_float(
            self.pose_directions,
            name="pose_directions",
            shape=(_POSE_BLEND_WIDTH, _VERTEX_COUNT * 3),
        )
        regressor = _finite_float(
            self.joint_regressor,
            name="joint_regressor",
            shape=(_JOINT_COUNT, _VERTEX_COUNT),
        )
        parents = _integer(self.parents, name="parents", shape=(_JOINT_COUNT,))
        if parents[0] != -1 or any(
            int(parent) < 0 or int(parent) >= index
            for index, parent in enumerate(parents[1:], start=1)
        ):
            raise ValueError("SMPL-H parents must define a forward acyclic tree.")
        weights = _finite_float(
            self.vertex_joint_weights,
            name="vertex_joint_weights",
            shape=(_VERTEX_COUNT, _JOINT_COUNT),
        )
        if np.any(weights < 0.0) or not np.allclose(
            weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError("SMPL-H LBS weights must be an explicit simplex.")
        for name, value in (
            ("template_vertices_m", template),
            ("faces", faces),
            ("shape_directions", shapedirs),
            ("pose_directions", posedirs),
            ("joint_regressor", regressor),
            ("parents", parents),
            ("vertex_joint_weights", weights),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def beta_count(self) -> int:
        """Return the exact shape-basis width of this licensed model."""
        return int(self.shape_directions.shape[2])


def load_smplh_model(model_root: str | Path, *, gender: str) -> SMPLHModelData:
    """Load one explicit ``smplh/<gender>/model.npz`` without adaptation."""
    normalized_gender = gender.strip().lower()
    if normalized_gender not in {"female", "male", "neutral"}:
        raise ValueError("SMPL-H gender must be female, male, or neutral.")
    root = Path(model_root).resolve()
    direct = root / normalized_gender / "model.npz"
    nested = root / "smplh" / normalized_gender / "model.npz"
    candidates = [path for path in (direct, nested) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Exactly one explicit SMPL-H model is required at "
            f"{direct} or {nested}; found={candidates}."
        )
    model_path = candidates[0]
    with np.load(model_path, allow_pickle=False) as archive:
        missing = _REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"SMPL-H archive is missing arrays: {sorted(missing)}.")
        kintree = _integer(
            archive["kintree_table"],
            name="kintree_table",
            shape=(2, _JOINT_COUNT),
        )
        if not np.array_equal(kintree[1], np.arange(_JOINT_COUNT)):
            raise ValueError("SMPL-H kintree joint IDs must be contiguous 0..51.")
        raw_parents = kintree[0].astype(np.uint64, copy=False)
        parents = raw_parents.astype(np.int64, copy=True)
        parents[0] = -1
        posedirs = np.asarray(archive["posedirs"], dtype=np.float64)
        if posedirs.shape != (_VERTEX_COUNT, 3, _POSE_BLEND_WIDTH):
            raise ValueError(
                "SMPL-H posedirs must have shape "
                f"{(_VERTEX_COUNT, 3, _POSE_BLEND_WIDTH)}, got {posedirs.shape}."
            )
        return SMPLHModelData(
            model_path=model_path,
            gender=normalized_gender,
            template_vertices_m=archive["v_template"],
            faces=archive["f"],
            shape_directions=archive["shapedirs"],
            pose_directions=posedirs.reshape(_VERTEX_COUNT * 3, _POSE_BLEND_WIDTH).T,
            joint_regressor=archive["J_regressor"],
            parents=parents,
            vertex_joint_weights=archive["weights"],
        )
