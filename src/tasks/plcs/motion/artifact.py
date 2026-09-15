"""Atomic, pickle-free persistence for canonical PLCS motion clips."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import numpy as np

from src.tasks.plcs.motion.contracts import (
    COCO17_MOTION_SCHEMA_VERSION,
    Coco17MotionClip,
    MotionSourceKind,
)

_ARRAY_KEYS = {
    "metadata_json",
    "timestamps_s",
    "joints_3d_m",
    "root_translation_m",
    "root_rotation",
    "joint_confidence",
    "frame_valid",
}


def save_motion_clip(clip: Coco17MotionClip, path: str | Path) -> Path:
    """Atomically save one ``.motion.npz`` artifact without object arrays."""
    if not isinstance(clip, Coco17MotionClip):
        raise TypeError("clip must be Coco17MotionClip.")
    destination = Path(path)
    if not destination.name.endswith(".motion.npz"):
        raise ValueError("Canonical motion artifacts must end in '.motion.npz'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        clip.metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(metadata_json),
                timestamps_s=clip.timestamps_s,
                joints_3d_m=clip.joints_3d_m,
                root_translation_m=clip.root_translation_m,
                root_rotation=clip.root_rotation,
                joint_confidence=clip.joint_confidence,
                frame_valid=clip.frame_valid,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_motion_clip(path: str | Path) -> Coco17MotionClip:
    """Load and fully validate one canonical motion artifact."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Canonical motion artifact not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != _ARRAY_KEYS:
            raise ValueError(
                f"{source}: artifact keys must be exactly {sorted(_ARRAY_KEYS)}, "
                f"got {sorted(archive.files)}."
            )
        raw_metadata = archive["metadata_json"]
        if raw_metadata.ndim != 0:
            raise ValueError(f"{source}: metadata_json must be scalar.")
        decoded = json.loads(str(raw_metadata.item()))
        if not isinstance(decoded, dict):
            raise ValueError(f"{source}: metadata_json must encode an object.")
        expected_metadata = {
            "schema_version",
            "coordinate_system",
            "keypoint_schema",
            "source_id",
            "source_path",
            "source_kind",
            "category",
            "gender",
            "native_fps",
            "frame_count",
            "provenance",
        }
        if set(decoded) != expected_metadata:
            raise ValueError(f"{source}: invalid metadata fields.")
        if decoded["schema_version"] != COCO17_MOTION_SCHEMA_VERSION:
            raise ValueError(f"{source}: unsupported motion schema version.")
        if decoded["coordinate_system"] != "right_handed_z_up_m":
            raise ValueError(f"{source}: unsupported coordinate system.")
        if decoded["keypoint_schema"] != "coco17":
            raise ValueError(f"{source}: unsupported keypoint schema.")
        metadata_frame_count = decoded["frame_count"]
        if type(metadata_frame_count) is not int or metadata_frame_count <= 0:
            raise ValueError(f"{source}: frame_count metadata must be a positive int.")
        clip = Coco17MotionClip(
            source_id=cast(str, decoded["source_id"]),
            source_path=cast(str, decoded["source_path"]),
            source_kind=cast(MotionSourceKind, decoded["source_kind"]),
            category=cast(str, decoded["category"]),
            gender=cast(str, decoded["gender"]),
            fps=cast(float, decoded["native_fps"]),
            timestamps_s=np.asarray(archive["timestamps_s"]),
            joints_3d_m=np.asarray(archive["joints_3d_m"]),
            root_translation_m=np.asarray(archive["root_translation_m"]),
            root_rotation=np.asarray(archive["root_rotation"]),
            joint_confidence=np.asarray(archive["joint_confidence"]),
            frame_valid=np.asarray(archive["frame_valid"]),
            provenance=cast(dict[str, object], decoded["provenance"]),
        )
    if clip.frame_count != metadata_frame_count:
        raise ValueError(
            f"{source}: metadata frame_count does not match stored arrays."
        )
    return clip


__all__ = ["load_motion_clip", "save_motion_clip"]
