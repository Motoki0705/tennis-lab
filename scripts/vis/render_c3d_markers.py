#!/usr/bin/env python
"""Render 3DTennisDS C3D markers as a 3D point cloud animation and save a video.

Usage:
    python tools/render_c3d_markers.py \
        data/raw/3dtennisds/tp1/bh/tp1_bh_s1.c3d \
        --mode skeleton \
        --out outputs/vis/tp1_bh_s1_skeleton.mp4 \
        --fps 50 \
        --stride 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.artist import Artist

# ---- スケルトン定義（Plug-in Gait 想定。必要ならラベル名を調整してください） ----
JOINT_MARKERS: dict[str, list[str]] = {
    # 体幹まわり
    "pelvis": ["LASI", "RASI"],
    "spine": ["T10", "C7", "STRN"],
    "head": ["LFHD", "LBHD", "RFHD", "RBHD"],
    # 左上肢
    "l_shoulder": ["LSHO"],
    "l_elbow": ["LELB"],
    "l_wrist": ["LWRA", "LWRB"],
    # 右上肢
    "r_shoulder": ["RSHO"],
    "r_elbow": ["RELB"],
    "r_wrist": ["RWRA", "RWRB"],
    # 左下肢
    "l_hip": ["LTHI"],
    "l_knee": ["LKNE"],
    "l_ankle": ["LANK"],
    "l_heel": ["LHEE"],
    "l_toe": ["LTOE"],
    # 右下肢
    "r_hip": ["RTHI"],
    "r_knee": ["RKNE"],
    "r_ankle": ["RANK"],
    "r_heel": ["RHEE"],
    "r_toe": ["RTOE"],
    # ラケット（オマケ）
    "racket_handle": ["Dol"],  # racket: Dol
    "racket_head": ["RH2", "RH3", "RH4", "RH5"],  # racket:RH*
}


BONES: list[tuple[str, str]] = [
    ("pelvis", "l_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("l_ankle", "l_heel"),
    ("l_ankle", "l_toe"),
    ("pelvis", "r_hip"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("r_ankle", "r_heel"),
    ("r_ankle", "r_toe"),
    ("pelvis", "spine"),
    ("spine", "head"),
    ("spine", "l_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("spine", "r_shoulder"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    # ラケット（右手に繋ぐ）
    ("r_wrist", "racket_handle"),
    ("racket_handle", "racket_head"),
]


def load_c3d_points(c3d_path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load marker coordinates and labels from a C3D file.

    Args:
        c3d_path (Path): Path to the input C3D file.

    Returns:
        tuple[np.ndarray, list[str], float]: A tuple containing:
            - np.ndarray: Marker coordinates with shape (T, M, 3), where
              T is the number of frames and M is the number of markers.
            - list[str]: Marker labels in the same order as the second axis.
            - float: Frame rate [Hz] stored in the C3D file.

    """
    c3d = ezc3d.c3d(str(c3d_path))

    # points: shape (4, M, T)
    #   0: x, 1: y, 2: z, 3: residual (negative values usually mean "gap")
    points = c3d["data"]["points"]
    xyz = points[:3, :, :]  # (3, M, T)
    residual = points[3, :, :]  # (M, T)

    # (coord, marker, frame) -> (frame, marker, coord)
    coords = np.transpose(xyz, (2, 1, 0)).astype(np.float32)  # (T, M, 3)

    # Mask out invalid samples (residual < 0)
    invalid_mask = residual < 0
    if invalid_mask.any():
        coords[invalid_mask.T] = np.nan  # transpose to (T, M)

    # Marker labels
    raw_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    labels: list[str] = []
    for raw in raw_labels:
        s = str(raw).strip()
        base = s.split(":")[-1].strip()
        labels.append(base)

    # Frame rate: header["points"]["frameRate"] or POINT/RATE
    try:
        header = c3d["header"]
        points_header = header.get("points", {})
        if "frameRate" in points_header:
            frame_rate = float(points_header["frameRate"])
        else:
            rate_param = c3d["parameters"]["POINT"]["RATE"]["value"]
            frame_rate = float(rate_param[0])
    except Exception:
        frame_rate = 100.0

    return coords, labels, frame_rate


def set_axes_equal_3d(ax: plt.Axes, coords: np.ndarray) -> None:
    """Set equal scaling on a 3D Axes based on the given coordinates.

    The function computes a bounding cube that encloses all valid points in the
    provided coordinates and applies the same range to x, y, and z limits so
    that the 3D markers are not distorted.

    Args:
        ax (plt.Axes): Matplotlib 3D axes on which to set limits.
        coords (np.ndarray): Marker coordinates with shape (T, M, 3).

    Returns:
        None: This function does not return anything.

    """
    flat = coords.reshape(-1, 3)
    flat = flat[~np.isnan(flat).any(axis=1)]

    if flat.size == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        return

    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()

    span *= 1.05

    x_center, y_center, z_center = center
    half = span / 2.0

    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)
    ax.set_zlim(z_center - half, z_center + half)


def compute_joint_coords(
    coords: np.ndarray,
    labels: list[str],
) -> dict[str, np.ndarray]:
    """Compute joint coordinates from marker coordinates and labels.

    For each joint, this function looks up the corresponding marker labels,
    averages their coordinates (if multiple markers are provided), and returns
    a time series of joint positions.

    Args:
        coords (np.ndarray): Marker coordinates with shape (T, M, 3).
        labels (list[str]): Marker labels corresponding to axis 1 of coords.

    Returns:
        dict[str, np.ndarray]: A mapping from joint name to joint coordinates
        with shape (T, 3). Joints for which none of the configured marker
        labels are found are omitted from the result.

    """
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    joint_coords: dict[str, np.ndarray] = {}

    for joint_name, marker_names in JOINT_MARKERS.items():
        indices = [
            label_to_index[m_name]
            for m_name in marker_names
            if m_name in label_to_index
        ]
        if not indices:
            continue

        # Average over selected markers for this joint: (T, len(indices), 3) -> (T, 3)
        joint_coords[joint_name] = np.nanmean(coords[:, indices, :], axis=1)

    return joint_coords


def render_c3d_markers(
    c3d_path: Path,
    out_path: Path,
    fps: int = 50,
    stride: int = 1,
    unit_scale: float = 0.001,
    mode: str = "markers",
) -> None:
    """Render a C3D as a 3D animation (markers or skeleton) and save to video.

    Args:
        c3d_path (Path): Path to the input C3D file.
        out_path (Path): Path to the output video file (e.g. .mp4).
        fps (int): Frames per second for the output video.
        stride (int): Use every N-th frame for rendering (>= 1).
        unit_scale (float): Scale factor applied to coordinates.
        mode (str): Visualization mode. Either "markers" or "skeleton".

    Returns:
        None: This function does not return anything.

    Raises:
        ValueError: If an unknown visualization mode is specified or if
            no joints can be computed from the marker labels in skeleton
            mode.

    """
    print(f"[INFO] Loading C3D: {c3d_path}")
    coords, labels, frame_rate = load_c3d_points(c3d_path)
    print(f"[INFO] Original frame rate: {frame_rate} Hz")
    print(f"[INFO] Frames: {coords.shape[0]}, markers: {coords.shape[1]}")

    if stride > 1:
        coords = coords[::stride]

    coords = coords * unit_scale

    T, _, _ = coords.shape
    print(f"[INFO] After stride: {T} frames")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    artists: list[Artist] = []

    if mode == "markers":
        # ---- マーカー点群モード ----
        set_axes_equal_3d(ax, coords)

        first_valid_idx = 0
        if np.isnan(coords[0]).all():
            for i in range(T):
                if not np.isnan(coords[i]).all():
                    first_valid_idx = i
                    break

        xs = coords[first_valid_idx, :, 0]
        ys = coords[first_valid_idx, :, 1]
        zs = coords[first_valid_idx, :, 2]

        scatter = ax.scatter(xs, ys, zs, s=10)
        artists.append(scatter)

        def init() -> tuple[Artist, ...]:
            """Initialize the markers animation by clearing all points.

            Returns:
                tuple[Artist, ...]: A tuple containing the scatter artist.

            """
            scatter._offsets3d = ([], [], [])
            return (scatter,)

        def update(frame_idx: int) -> tuple[Artist, ...]:
            """Update the markers scatter plot for a given frame index.

            Args:
                frame_idx (int): Index of the frame to render.

            Returns:
                tuple[Artist, ...]: A tuple containing the updated scatter
                artist.

            """
            frame = coords[frame_idx]
            xs_frame = frame[:, 0]
            ys_frame = frame[:, 1]
            zs_frame = frame[:, 2]

            scatter._offsets3d = (xs_frame, ys_frame, zs_frame)
            ax.set_title(f"{c3d_path.name} (frame {frame_idx + 1}/{T})")
            return (scatter,)

    elif mode == "skeleton":
        # ---- スケルトンモード ----
        joint_coords = compute_joint_coords(coords, labels)
        if not joint_coords:
            raise ValueError(
                "No joints could be computed from marker labels. "
                "Please adjust JOINT_MARKERS mapping.",
            )

        joint_names = sorted(joint_coords.keys())
        joint_positions = np.stack(
            [joint_coords[name] for name in joint_names],
            axis=1,
        )  # (T, J, 3)

        set_axes_equal_3d(ax, joint_positions)

        xs0 = joint_positions[0, :, 0]
        ys0 = joint_positions[0, :, 1]
        zs0 = joint_positions[0, :, 2]

        joint_scatter = ax.scatter(xs0, ys0, zs0, s=20)
        artists.append(joint_scatter)

        # Prepare lines for bones
        bone_lines: list[Artist] = []
        name_to_index = {name: idx for idx, name in enumerate(joint_names)}
        for j1, j2 in BONES:
            if j1 not in name_to_index or j2 not in name_to_index:
                continue
            (line,) = ax.plot([], [], [], linewidth=2)
            bone_lines.append(line)
            artists.append(line)

        def init() -> tuple[Artist, ...]:
            """Initialize the skeleton animation artists.

            Returns:
                tuple[Artist, ...]: A tuple containing scatter and line artists.

            """
            joint_scatter._offsets3d = ([], [], [])
            for line_artist in bone_lines:
                line_artist.set_data([], [])
                line_artist.set_3d_properties([])
            return tuple(artists)

        def update(frame_idx: int) -> tuple[Artist, ...]:
            """Update the skeleton artists for a given frame index.

            Args:
                frame_idx (int): Index of the frame to render.

            Returns:
                tuple[Artist, ...]: A tuple containing the updated artists.

            """
            frame_joints = joint_positions[frame_idx]
            xs_frame = frame_joints[:, 0]
            ys_frame = frame_joints[:, 1]
            zs_frame = frame_joints[:, 2]

            joint_scatter._offsets3d = (xs_frame, ys_frame, zs_frame)

            # Update each bone line
            bone_idx = 0
            for j1, j2 in BONES:
                if j1 not in name_to_index or j2 not in name_to_index:
                    continue
                i1 = name_to_index[j1]
                i2 = name_to_index[j2]
                line_artist = bone_lines[bone_idx]
                xs_line = [frame_joints[i1, 0], frame_joints[i2, 0]]
                ys_line = [frame_joints[i1, 1], frame_joints[i2, 1]]
                zs_line = [frame_joints[i1, 2], frame_joints[i2, 2]]
                line_artist.set_data(xs_line, ys_line)
                line_artist.set_3d_properties(zs_line)
                bone_idx += 1

            ax.set_title(f"{c3d_path.name} (frame {frame_idx + 1}/{T})")
            return tuple(artists)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    print(f"[INFO] Creating animation (fps={fps}, mode={mode})...")
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=False,
    )

    print(f"[INFO] Saving to: {out_path}")
    writer_class = animation.FFMpegWriter
    writer = writer_class(fps=fps, bitrate=2000)

    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print("[INFO] Done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the C3D marker renderer.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(
        description="Render C3D markers as a 3D animation video.",
    )
    parser.add_argument("c3d_path", type=Path, help="Path to input .c3d file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output video path. Default: outputs/vis/<c3d_stem>_markers_or_skeleton.mp4"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output video FPS (default: 50).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for rendering (use every N-th frame, default: 1).",
    )
    parser.add_argument(
        "--mode",
        choices=("markers", "skeleton"),
        default="markers",
        help=(
            "Visualization mode. 'markers' renders all markers as a point "
            "cloud, 'skeleton' renders a simplified skeleton."
        ),
    )
    parser.add_argument(
        "--no-unit-scale",
        action="store_true",
        help=("If set, do NOT scale units (keep raw C3D units, usually millimetres)."),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the C3D marker renderer CLI.

    Returns:
        None: This function does not return anything.

    Raises:
        FileNotFoundError: If the input C3D file does not exist.
        ValueError: If an unknown visualization mode is specified in
            the command-line arguments.

    """
    args = parse_args()
    c3d_path: Path = args.c3d_path

    if not c3d_path.is_file():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    if args.out is None:
        out_dir = Path("outputs/vis")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_markers" if args.mode == "markers" else "_skeleton"
        out_name = c3d_path.stem + suffix + ".mp4"
        out_path = out_dir / out_name
    else:
        out_path = args.out

    unit_scale = 1.0 if args.no_unit_scale else 0.001

    render_c3d_markers(
        c3d_path=c3d_path,
        out_path=out_path,
        fps=args.fps,
        stride=max(1, args.stride),
        unit_scale=unit_scale,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
