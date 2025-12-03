#!/usr/bin/env python
"""Analyze AMASS ACCAD subset under data/ACCAD for PLCS data generation.

- Recursively find all `*_poses.npz`
- For each file, read:
    - poses shape -> num_frames, pose_dim
    - trans shape
    - betas shape
    - gender
    - mocap_framerate
- Infer simple `subject` / `category` from folder names
- Print:
    - Overall stats
    - Per-category stats (total sequences / minutes / frames)

Usage:
    python analyze_accad.py
    python analyze_accad.py --root data/ACCAD
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MotionSeq:
    """Metadata for a single AMASS motion sequence."""

    path: Path
    subject: str | None
    category: str | None
    num_frames: int
    pose_dim: int
    fps: float
    duration_sec: float
    gender: str
    has_dmpls: bool


def infer_subject_category(parent_name: str) -> tuple[str | None, str | None]:
    """Try to infer (subject, category) from a folder name like:
        - 'Female1Walking_c3d'
        - 'Male2General_c3d'
        - 'MartialArtsWalksTurns_c3d'
        - 's001'
    If we can't infer, return (None, base_name_without_suffix).
    """
    name = parent_name

    # Strip trailing '_c3d' if present
    if name.endswith("_c3d"):
        name_core = name[:-4]
    else:
        name_core = name

    # Case 1: Female1Walking, Male2Running, etc.
    for prefix in ("Female", "Male"):
        if name_core.startswith(prefix):
            # subject is like 'Female1', 'Male2'
            i = len(prefix)
            # consume digits after 'Female'/'Male'
            while i < len(name_core) and name_core[i].isdigit():
                i += 1
            subject = name_core[:i] if i > len(prefix) else name_core
            category = name_core[i:] or None
            return subject, category

    # Case 2: s001, s007, etc. (treat as subject only)
    if name_core.startswith("s") and name_core[1:].isdigit():
        return name_core, None

    # Fallback: no explicit subject, treat whole as category
    return None, name_core or None


def analyze_accad(root: Path) -> list[MotionSeq]:
    npz_paths = sorted(root.rglob("*_poses.npz"))
    if not npz_paths:
        raise SystemExit(f"No *_poses.npz files found under: {root}")

    sequences: list[MotionSeq] = []

    for path in npz_paths:
        rel = path.relative_to(root)
        parent_name = rel.parent.name

        subject, category = infer_subject_category(parent_name)

        bdata = np.load(path, allow_pickle=True)

        if "poses" not in bdata or "trans" not in bdata:
            print(f"[WARN] {rel} missing 'poses' or 'trans'; skipping")
            continue

        poses = bdata["poses"]
        trans = bdata["trans"]

        num_frames, pose_dim = poses.shape
        if trans.shape[0] != num_frames:
            print(
                f"[WARN] {rel} poses ({num_frames}) and trans ({trans.shape[0]}) "
                "have different T; using min(T)"
            )
            num_frames = min(num_frames, trans.shape[0])

        # fps
        if "mocap_framerate" in bdata:
            fps = float(bdata["mocap_framerate"])
        else:
            fps = 60.0  # sensible default
        duration_sec = num_frames / fps

        # gender
        gender_raw = bdata["gender"] if "gender" in bdata else "unknown"
        try:
            gender = str(gender_raw.item())
        except Exception:
            gender = str(gender_raw)

        has_dmpls = "dmpls" in bdata

        seq = MotionSeq(
            path=rel,
            subject=subject,
            category=category,
            num_frames=num_frames,
            pose_dim=pose_dim,
            fps=fps,
            duration_sec=duration_sec,
            gender=gender,
            has_dmpls=has_dmpls,
        )
        sequences.append(seq)

    return sequences


def print_summary(sequences: list[MotionSeq]) -> None:
    print(f"\nTotal sequences: {len(sequences)}")

    total_frames = sum(s.num_frames for s in sequences)
    total_duration_sec = sum(s.duration_sec for s in sequences)
    pose_dims = sorted({s.pose_dim for s in sequences})
    fps_set = sorted({s.fps for s in sequences})

    print(f"Total frames:   {total_frames}")
    print(f"Total duration: {total_duration_sec / 60.0:.2f} min")
    print(f"Pose dims:      {pose_dims}")
    print(f"FPS set:        {fps_set}")

    # Category-wise stats
    cat_stats = defaultdict(lambda: {"count": 0, "frames": 0, "sec": 0.0})
    for s in sequences:
        key = s.category or "(no_category)"
        cs = cat_stats[key]
        cs["count"] += 1
        cs["frames"] += s.num_frames
        cs["sec"] += s.duration_sec

    print("\nPer-category summary (category, #seq, minutes, #frames):")
    for cat, cs in sorted(cat_stats.items(), key=lambda kv: kv[0]):
        print(
            f"  {cat:20s}  "
            f"{cs['count']:4d} seq   "
            f"{cs['sec'] / 60.0:7.2f} min   "
            f"{cs['frames']:7d} frames"
        )

    # Optional: show a few example sequences
    print("\nExample sequences:")
    for s in sequences[:10]:
        print(
            f"  {s.path} | subj={s.subject or '-'} "
            f"| cat={s.category or '-'} | T={s.num_frames} | fps={s.fps}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/ACCAD"),
        help="Root directory of ACCAD-AMASS (default: data/ACCAD)",
    )
    args = parser.parse_args()

    sequences = analyze_accad(args.root)
    print_summary(sequences)


if __name__ == "__main__":
    main()
