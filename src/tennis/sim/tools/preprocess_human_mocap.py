from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis.geometry.skeleton import VITPOSE_17_NAMES


try:  # pragma: no cover - optional dependency for BVH parsing
    from npybvh import Bvh as NpyBvh
except ImportError as exc:  # pragma: no cover - surfaced with actionable hint
    NPYBVH_IMPORT_ERROR: Exception | None = exc
    NpyBvh = None  # type: ignore[assignment]
else:  # pragma: no cover
    NPYBVH_IMPORT_ERROR = None


VITPOSE_CORE_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


_BVH_NAME_HINTS: dict[str, tuple[str, ...]] = {
    # Head / face
    "nose": ("Head", "HeadTop", "Head_End", "HeadNub"),
    # Upper body
    "left_shoulder": ("LeftShoulder", "LShoulder", "LeftArm"),
    "right_shoulder": ("RightShoulder", "RShoulder", "RightArm"),
    "left_elbow": ("LeftForeArm", "LeftElbow"),
    "right_elbow": ("RightForeArm", "RightElbow"),
    "left_wrist": ("LeftHand", "LHand"),
    "right_wrist": ("RightHand", "RHand"),
    # Lower body
    "left_hip": ("LeftUpLeg", "LHip", "LeftHip"),
    "right_hip": ("RightUpLeg", "RHip", "RightHip"),
    "left_knee": ("LeftLeg", "LeftKnee"),
    "right_knee": ("RightLeg", "RightKnee"),
    "left_ankle": ("LeftFoot", "LAnkle", "LeftAnkle"),
    "right_ankle": ("RightFoot", "RAnkle", "RightAnkle"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BVH mocap files to VitPose-17 3D skeleton clips stored as NPZ. "
            "This is a best-effort generic converter; joint name mapping may "
            "need adjustment for your dataset."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root directory containing input .bvh files",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for converted .npz clips",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Minimum number of frames required to keep a clip (default: 30)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help=(
            "Optional limit on how many BVH files to process (0 = no limit). "
            "Files are discovered with raw-root.rglob('*.bvh')."
        ),
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=30.0,
        help=(
            "Default FPS to use when the BVH reader does not expose frame_time "
            "or FPS meta-data (default: 30.0)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress information.",
    )
    return parser.parse_args()


def _ensure_npybvh_available() -> None:
    if NpyBvh is None or NPYBVH_IMPORT_ERROR is not None:  # pragma: no cover
        msg = (
            "npybvh is required to parse BVH files. "
            "Install it with: pip install git+https://github.com/dabeschte/npybvh.git"
        )
        raise RuntimeError(msg) from NPYBVH_IMPORT_ERROR


def _build_joint_index_map(joint_names: Sequence[str]) -> dict[str, int | None]:
    lower_names = [name.lower() for name in joint_names]
    index_map: dict[str, int | None] = {name: None for name in VITPOSE_17_NAMES}

    for vit_name, hints in _BVH_NAME_HINTS.items():
        found: int | None = None
        for hint in hints:
            h = hint.lower()
            for idx, jn in enumerate(lower_names):
                if h in jn:
                    found = idx
                    break
            if found is not None:
                break
        index_map[vit_name] = found

    return index_map


def _check_core_joints(index_map: dict[str, int | None], path: Path) -> bool:
    missing_core = [name for name in VITPOSE_CORE_JOINTS if index_map.get(name) is None]
    if missing_core:
        sys.stderr.write(
            f"[human-mocap] Skipping {path} because core joints are missing: "
            f"{', '.join(missing_core)}\n",
        )
        return False
    return True


def _build_vitpose_sequence(
    all_positions: np.ndarray,
    joint_names: Sequence[str],
    path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    if all_positions.ndim != 3:
        sys.stderr.write(
            f"[human-mocap] Unexpected all_positions shape for {path}: "
            f"{all_positions.shape} (expected [T, J, 3])\n",
        )
        return None

    index_map = _build_joint_index_map(joint_names)
    if not _check_core_joints(index_map, path):
        return None

    T, J_src, C = all_positions.shape
    if C != 3:
        sys.stderr.write(
            f"[human-mocap] Expected 3D positions for {path}, got C={C}\n",
        )
        return None

    joints = np.full((T, len(VITPOSE_17_NAMES), 3), np.nan, dtype=np.float32)

    for vit_idx, vit_name in enumerate(VITPOSE_17_NAMES):
        src_idx = index_map.get(vit_name)
        if src_idx is None:
            continue
        if not (0 <= src_idx < J_src):
            continue
        joints[:, vit_idx, :] = all_positions[:, src_idx, :]

    # Fill face landmarks from nose/head if missing.
    nose_idx = VITPOSE_17_NAMES.index("nose")
    nose_positions = joints[:, nose_idx, :].copy()

    for face_name in ("left_eye", "right_eye", "left_ear", "right_ear"):
        face_idx = VITPOSE_17_NAMES.index(face_name)
        if np.isnan(joints[:, face_idx, :]).all():
            joints[:, face_idx, :] = nose_positions

    left_hip_idx = VITPOSE_17_NAMES.index("left_hip")
    right_hip_idx = VITPOSE_17_NAMES.index("right_hip")
    left_hip = joints[:, left_hip_idx, :]
    right_hip = joints[:, right_hip_idx, :]

    valid_mask = ~np.isnan(left_hip).any(axis=1) & ~np.isnan(right_hip).any(axis=1)
    if not np.any(valid_mask):
        sys.stderr.write(
            f"[human-mocap] All frames invalid for {path} due to NaNs in hips.\n",
        )
        return None

    joints = joints[valid_mask]
    pelvis = 0.5 * (joints[:, left_hip_idx, :] + joints[:, right_hip_idx, :])
    pelvis = pelvis - pelvis[0]
    joints_rel = joints - pelvis[:, None, :]

    return joints_rel.astype(np.float32, copy=False), pelvis.astype(np.float32, copy=False)


def _load_bvh_positions(anim: Any, default_fps: float) -> tuple[np.ndarray, float, list[str]]:
    try:
        nframes = int(anim.nframes)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"BVH object lacks nframes attribute: {exc}"
        raise RuntimeError(msg) from exc

    if nframes <= 0:
        raise RuntimeError("BVH file has zero frames")

    joint_names = list(anim.joint_names())
    num_joints = len(joint_names)
    positions_all = np.empty((nframes, num_joints, 3), dtype=np.float32)

    for frame_idx in range(nframes):
        positions, _rotations = anim.frame_pose(frame_idx)
        arr = np.asarray(positions, dtype=np.float32)
        if arr.shape != (num_joints, 3):
            msg = (
                "frame_pose() returned unexpected shape "
                f"{arr.shape} (expected ({num_joints}, 3))"
            )
            raise RuntimeError(msg)
        positions_all[frame_idx] = arr

    fps: float
    frame_time = getattr(anim, "frame_time", None)
    if frame_time is not None:
        try:
            fps = float(1.0 / float(frame_time))
        except Exception:  # pragma: no cover - fall back
            fps = float(default_fps)
    else:
        fps = float(default_fps)

    return positions_all, fps, joint_names


def _process_single_bvh(
    in_path: Path,
    out_root: Path,
    min_frames: int,
    default_fps: float,
    verbose: bool,
) -> bool:
    try:
        anim = NpyBvh()  # type: ignore[operator]
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"[human-mocap] Failed to create Bvh() for {in_path}: {exc}\n")
        return False

    try:
        anim.parse_file(str(in_path))
        positions_all, fps, joint_names = _load_bvh_positions(anim, default_fps)
        joints_rel, pelvis = _build_vitpose_sequence(positions_all, joint_names, in_path) or (
            None,
            None,
        )
        if joints_rel is None or pelvis is None:
            return False
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"[human-mocap] Error while parsing {in_path}: {exc}\n")
        return False

    if joints_rel.shape[0] < int(min_frames):
        if verbose:
            sys.stderr.write(
                f"[human-mocap] Skipping {in_path} because it has only "
                f"{joints_rel.shape[0]} frames (< {min_frames}).\n",
            )
        return False

    rel = in_path.relative_to(in_path.parents[0])
    # Mirror directory structure under out_root but switch extension to .npz
    out_path = out_root / rel
    out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        joints=joints_rel.astype(np.float32, copy=False),
        pelvis=pelvis.astype(np.float32, copy=False),
        fps=float(fps),
    )

    if verbose:
        sys.stdout.write(f"[human-mocap] Wrote {out_path}\n")
    return True


def main() -> int:
    args = _parse_args()
    _ensure_npybvh_available()

    raw_root = args.raw_root
    out_root = args.out_root
    min_frames = int(args.min_frames)
    max_files = int(args.max_files or 0)
    default_fps = float(args.default_fps)
    verbose: bool = bool(args.verbose)

    if not raw_root.exists():
        sys.stderr.write(f"[human-mocap] raw-root not found: {raw_root}\n")
        return 1

    bvh_paths = list(sorted(raw_root.rglob("*.bvh")))
    if not bvh_paths:
        sys.stderr.write(f"[human-mocap] No .bvh files found under {raw_root}\n")
        return 1

    if max_files > 0:
        bvh_paths = bvh_paths[:max_files]

    out_root.mkdir(parents=True, exist_ok=True)

    converted = 0
    for path in bvh_paths:
        if _process_single_bvh(path, out_root, min_frames, default_fps, verbose):
            converted += 1

    sys.stdout.write(
        f"[human-mocap] Converted {converted} / {len(bvh_paths)} BVH files to NPZ under {out_root}\n",
    )
    return 0 if converted > 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
