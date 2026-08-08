"""Lossless typed AMASS/ACCAD motion clips for PLCS production sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]


class MotionCategory(StrEnum):
    """The production motion-source categories required by PLCS."""

    RUNNING = "running"
    WALKING = "walking"
    GENERAL = "general"


_POSE_WIDTH = 156
_BODY_SLICE = slice(3, 66)
_RIGHT_HAND_SLICE = slice(66, 111)
_LEFT_HAND_SLICE = slice(111, 156)


def _readonly_float_array(
    value: object,
    *,
    name: str,
    shape_tail: tuple[int, ...],
) -> FloatArray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64, got {array.dtype}.")
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        expected = ("T", *shape_tail)
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return cast(FloatArray, result)


def _readonly_betas(value: object) -> FloatArray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"betas must use float32 or float64, got {array.dtype}.")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("betas must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("betas contains NaN or infinity.")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return cast(FloatArray, result)


def _trimmed(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True, slots=True)
class PLCSMotionClip:
    """Every source frame and SMPL-H component from one AMASS archive.

    The component arrays retain the source floating dtype and frame order.  No
    resampling, truncation, normalization, or pose repair is performed here.
    """

    source_path: str
    category: MotionCategory
    gender: str
    fps: float
    body_pose_axis_angle: FloatArray
    global_orient_axis_angle: FloatArray
    left_hand_pose_axis_angle: FloatArray
    right_hand_pose_axis_angle: FloatArray
    root_translation_m: FloatArray
    betas: FloatArray
    frame_count: int = field(init=False)

    def __post_init__(self) -> None:
        source_path = _trimmed(self.source_path, name="source_path")
        if not Path(source_path).is_absolute():
            raise ValueError("source_path must be an absolute path.")
        try:
            category = MotionCategory(self.category)
        except ValueError as error:
            raise ValueError(
                "category must be running, walking, or general."
            ) from error
        gender = _trimmed(self.gender, name="gender").lower()
        if gender not in {"female", "male", "neutral"}:
            raise ValueError("gender must be female, male, or neutral.")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be numeric.")
        fps = float(self.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive.")

        body = _readonly_float_array(
            self.body_pose_axis_angle,
            name="body_pose_axis_angle",
            shape_tail=(63,),
        )
        global_orient = _readonly_float_array(
            self.global_orient_axis_angle,
            name="global_orient_axis_angle",
            shape_tail=(3,),
        )
        left_hand = _readonly_float_array(
            self.left_hand_pose_axis_angle,
            name="left_hand_pose_axis_angle",
            shape_tail=(45,),
        )
        right_hand = _readonly_float_array(
            self.right_hand_pose_axis_angle,
            name="right_hand_pose_axis_angle",
            shape_tail=(45,),
        )
        translation = _readonly_float_array(
            self.root_translation_m,
            name="root_translation_m",
            shape_tail=(3,),
        )
        frame_count = int(body.shape[0])
        if frame_count == 0:
            raise ValueError("A PLCS motion clip must contain at least one frame.")
        arrays = (global_orient, left_hand, right_hand, translation)
        if any(int(array.shape[0]) != frame_count for array in arrays):
            raise ValueError("All pose and translation arrays must have the same T.")
        dtypes = {
            body.dtype,
            global_orient.dtype,
            left_hand.dtype,
            right_hand.dtype,
            translation.dtype,
        }
        if len(dtypes) != 1:
            raise TypeError(
                "All per-frame source arrays must retain one floating dtype."
            )

        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "gender", gender)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "body_pose_axis_angle", body)
        object.__setattr__(self, "global_orient_axis_angle", global_orient)
        object.__setattr__(self, "left_hand_pose_axis_angle", left_hand)
        object.__setattr__(self, "right_hand_pose_axis_angle", right_hand)
        object.__setattr__(self, "root_translation_m", translation)
        object.__setattr__(self, "betas", _readonly_betas(self.betas))
        object.__setattr__(self, "frame_count", frame_count)

    @classmethod
    def from_amass_arrays(
        cls,
        *,
        source_path: str | Path,
        category: MotionCategory | str,
        gender: str,
        fps: float,
        poses: object,
        trans: object,
        betas: object,
    ) -> PLCSMotionClip:
        """Split one untouched AMASS pose matrix into the SMPL-H components."""
        pose_array = _readonly_float_array(
            poses,
            name="poses",
            shape_tail=(_POSE_WIDTH,),
        )
        translation = _readonly_float_array(
            trans,
            name="trans",
            shape_tail=(3,),
        )
        if pose_array.shape[0] != translation.shape[0]:
            raise ValueError("AMASS poses and trans must contain the same T.")
        try:
            typed_category = MotionCategory(category)
        except ValueError as error:
            raise ValueError(
                "category must be running, walking, or general."
            ) from error
        clip = cls(
            source_path=str(Path(source_path).resolve()),
            category=typed_category,
            gender=gender,
            fps=fps,
            body_pose_axis_angle=pose_array[:, _BODY_SLICE],
            global_orient_axis_angle=pose_array[:, :3],
            left_hand_pose_axis_angle=pose_array[:, _LEFT_HAND_SLICE],
            right_hand_pose_axis_angle=pose_array[:, _RIGHT_HAND_SLICE],
            root_translation_m=translation,
            betas=cast(FloatArray, np.asarray(betas)),
        )
        reconstructed = clip.full_pose_axis_angle()
        if reconstructed.dtype != pose_array.dtype or not np.array_equal(
            reconstructed,
            pose_array,
        ):
            raise RuntimeError("AMASS pose component split was not lossless.")
        return clip

    def full_pose_axis_angle(self) -> FloatArray:
        """Reconstruct the exact 156-value AMASS/SMPL-H pose rows."""
        result = np.concatenate(
            (
                self.global_orient_axis_angle,
                self.body_pose_axis_angle,
                self.right_hand_pose_axis_angle,
                self.left_hand_pose_axis_angle,
            ),
            axis=1,
        )
        result.setflags(write=False)
        return result

    def metadata(self) -> dict[str, object]:
        """Return source provenance without an artifact identity digest."""
        return {
            "source_path": self.source_path,
            "category": self.category.value,
            "gender": self.gender,
            "native_fps": self.fps,
            "frame_count": self.frame_count,
            "pose_dtype": str(self.body_pose_axis_angle.dtype),
            "beta_count": int(self.betas.shape[0]),
        }


def infer_accad_category(path: Path) -> MotionCategory:
    """Classify one ACCAD path into the explicit production vocabulary."""
    text = "/".join(part.lower() for part in path.parts)
    if "running" in text or "sprint" in text or "run " in text:
        return MotionCategory.RUNNING
    if "walking" in text or "walk" in text:
        return MotionCategory.WALKING
    return MotionCategory.GENERAL


def load_amass_motion_clip(
    path: str | Path,
    *,
    category: MotionCategory | str | None = None,
) -> PLCSMotionClip:
    """Load one full AMASS archive without pickle or frame selection."""
    source = Path(path).resolve()
    if source.suffix != ".npz" or not source.is_file():
        raise FileNotFoundError(f"AMASS motion archive does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"AMASS archive is missing fields: {sorted(missing)}.")
        raw_gender = archive["gender"]
        if raw_gender.ndim != 0:
            raise ValueError("AMASS gender must be a scalar.")
        gender_value = raw_gender.item()
        if isinstance(gender_value, bytes):
            gender = gender_value.decode("utf-8")
        else:
            gender = str(gender_value)
        raw_fps = archive["mocap_framerate"]
        if raw_fps.ndim != 0:
            raise ValueError("AMASS mocap_framerate must be a scalar.")
        return PLCSMotionClip.from_amass_arrays(
            source_path=source,
            category=infer_accad_category(source) if category is None else category,
            gender=gender,
            fps=float(raw_fps.item()),
            poses=archive["poses"],
            trans=archive["trans"],
            betas=archive["betas"],
        )


@dataclass(frozen=True, slots=True)
class ACCADMotionLibrary:
    """Deterministic category-indexed ACCAD source inventory."""

    files_by_category: Mapping[MotionCategory, tuple[Path, ...]]

    def __post_init__(self) -> None:
        expected = set(MotionCategory)
        if set(self.files_by_category) != expected:
            raise ValueError(
                "ACCAD library must explicitly index running, walking, and general."
            )
        normalized: dict[MotionCategory, tuple[Path, ...]] = {}
        for category in MotionCategory:
            files = tuple(
                sorted(
                    Path(path).resolve() for path in self.files_by_category[category]
                )
            )
            if not files:
                raise ValueError(
                    f"ACCAD category {category.value!r} has no motion files."
                )
            if len(files) != len(set(files)):
                raise ValueError(
                    f"ACCAD category {category.value!r} contains duplicates."
                )
            invalid = [
                str(path)
                for path in files
                if path.suffix != ".npz" or not path.is_file()
            ]
            if invalid:
                raise FileNotFoundError(f"Invalid ACCAD motion files: {invalid}.")
            normalized[category] = files
        object.__setattr__(self, "files_by_category", normalized)

    @classmethod
    def from_root(cls, root: str | Path) -> ACCADMotionLibrary:
        """Index every ACCAD ``*_poses.npz`` once in stable path order."""
        directory = Path(root).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"ACCAD root does not exist: {directory}")
        grouped: dict[MotionCategory, list[Path]] = {
            category: [] for category in MotionCategory
        }
        for path in sorted(directory.rglob("*_poses.npz")):
            grouped[infer_accad_category(path)].append(path.resolve())
        return cls(
            files_by_category={
                category: tuple(paths) for category, paths in grouped.items()
            }
        )

    @classmethod
    def from_category_paths(
        cls,
        paths: Mapping[MotionCategory | str, Sequence[str | Path]],
    ) -> ACCADMotionLibrary:
        """Build an explicit config-selected category inventory."""
        grouped: dict[MotionCategory, tuple[Path, ...]] = {}
        for key, values in paths.items():
            category = MotionCategory(key)
            expanded: list[Path] = []
            for value in values:
                candidate = Path(value).resolve()
                if candidate.is_file():
                    expanded.append(candidate)
                elif candidate.is_dir():
                    expanded.extend(sorted(candidate.rglob("*_poses.npz")))
                else:
                    raise FileNotFoundError(
                        f"Configured ACCAD source does not exist: {candidate}"
                    )
            grouped[category] = tuple(expanded)
        return cls(files_by_category=grouped)

    def select(self, category: MotionCategory | str, *, seed: int) -> PLCSMotionClip:
        """Select one full clip deterministically, with no cross-category fallback."""
        typed_category = MotionCategory(category)
        return load_amass_motion_clip(
            self.select_path(typed_category, seed=seed),
            category=typed_category,
        )

    def select_path(self, category: MotionCategory | str, *, seed: int) -> Path:
        """Resolve a source deterministically without loading its archive.

        Stage owners use this split boundary to cache an exact source across
        preflight and execution.  Selection never opens an NPZ and therefore
        cannot accidentally double the measured clip-load count.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        typed_category = MotionCategory(category)
        files = self.files_by_category[typed_category]
        return files[
            random.Random(f"{seed}:{typed_category.value}").randrange(len(files))
        ]


__all__ = [
    "ACCADMotionLibrary",
    "MotionCategory",
    "PLCSMotionClip",
    "infer_accad_category",
    "load_amass_motion_clip",
]
