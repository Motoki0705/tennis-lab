"""CLI tool to render augmented TennisSceneWindowDataset samples to video.

This script instantiates ``TennisSceneWindowDataset`` from a dataset config
YAML (e.g. ``configs/datasets/tennis_multi_cam_3d_pose_sim.yaml``), applies on-the-fly
camera sampling and 2D augmentation (as configured), and uses
``src.visualize.tennis_render`` to render pose sequences as videos for
visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.datasets.tennis import TennisSceneWindowDataset
from src.visualize.tennis_render import render_pose2d_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render augmented TennisSceneWindowDataset windows as videos for visual inspection."
        )
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/datasets/tennis_multi_cam_3d_pose_sim.yaml",
        help=(
            "Path to dataset config YAML (used to instantiate "
            "TennisSceneWindowDataset)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tennis_augmented_viz",
        help="Directory where rendered videos will be saved.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of dataset windows to render.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting dataset index to visualize.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help=(
            "Camera index to visualize. If negative, the first camera with any "
            "players in the window is chosen automatically."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible augmentation.",
    )
    return parser.parse_args()


def _load_dataset_from_config(
    dataset_cfg_path: str,
    split: str,
) -> TennisSceneWindowDataset:
    cfg = OmegaConf.load(dataset_cfg_path)
    cfg_container = OmegaConf.to_container(cfg, resolve=True) or {}
    if not isinstance(cfg_container, dict):
        msg = f"Dataset config root must be a mapping: {dataset_cfg_path}"
        raise SystemExit(msg)
    cfg_dict: dict[str, Any] = cfg_container

    root = cfg_dict.get("root", "data/tennis_autogen")
    name = cfg_dict.get("name") or cfg_dict.get("dataset_name")
    if not name:
        msg = (
            "Dataset config must define 'name' (or 'dataset_name') "
            f"to locate the dataset. Config: {dataset_cfg_path}"
        )
        raise SystemExit(msg)
    window_T = int(cfg_dict.get("window_T", 10))
    max_cameras = int(cfg_dict.get("max_cameras", 4))
    max_players = int(cfg_dict.get("max_players", 20))
    num_joints = int(cfg_dict.get("num_joints", 20))
    use_memmap = bool(cfg_dict.get("use_memmap", False))

    min_cameras_val = cfg_dict.get("min_cameras")
    min_cameras = int(min_cameras_val) if min_cameras_val is not None else None
    augment_2d = bool(cfg_dict.get("augment_2d", False))

    return TennisSceneWindowDataset(
        dataset_root=root,
        dataset_name=name,
        split=split,
        window_T=window_T,
        max_cameras=max_cameras,
        max_players=max_players,
        num_joints=num_joints,
        use_memmap=use_memmap,
        min_cameras=min_cameras,
        augment_2d=augment_2d,
    )


def _denormalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map normalized coordinates in [-1,1] to pixel coordinates."""
    pts = points.astype(np.float32).copy()
    if width <= 0 or height <= 0:
        return pts
    pts[..., 0] = (pts[..., 0] + 1.0) * 0.5 * float(width)
    pts[..., 1] = (pts[..., 1] + 1.0) * 0.5 * float(height)
    return pts


def _select_camera_index(sample: dict[str, torch.Tensor], explicit_idx: int) -> int:
    keypoints_2d = sample["keypoints_2d"]
    player_mask = sample["player_mask"]
    T, V, M, _, _ = keypoints_2d.shape
    if explicit_idx >= 0:
        if explicit_idx >= V:
            msg = f"Requested camera_index={explicit_idx} but only V={V} cameras are present"
            raise ValueError(msg)
        return explicit_idx

    # Auto-select: first camera that sees any player in the window.
    mask_any = player_mask.any(dim=0).any(dim=1)  # [V]
    for v in range(V):
        if bool(mask_any[v]):
            return v
    # Fallback: camera 0.
    return 0


def _render_sample_to_video(
    sample: dict[str, torch.Tensor],
    out_path: Path,
    camera_index: int,
    fps: float,
) -> None:
    keypoints_2d = sample["keypoints_2d"]  # [T, V, M, J, 2]
    player_mask = sample["player_mask"]  # [T, V, M]
    court_2d = sample["court_2d"]  # [V, 20, 2]
    image_size = sample["image_size"]  # [V, 2]

    T, V, M, J, _ = keypoints_2d.shape
    v = _select_camera_index(sample, camera_index)

    width = int(image_size[v, 0].item())
    height = int(image_size[v, 1].item())
    if width <= 0 or height <= 0:
        # Fallback to a sane default if image_size is missing.
        width, height = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        msg = f"Failed to open VideoWriter for path: {out_path}"
        raise RuntimeError(msg)

    try:
        k2d = keypoints_2d[:, v].numpy()  # [T, M, J, 2]
        pm = player_mask[:, v].numpy()  # [T, M]
        court = court_2d[v].numpy()  # [20, 2]
        court_pix = _denormalize_points(court, width, height)

        for t in range(T):
            players_pose: list[np.ndarray] = []
            players_racket: list[np.ndarray] = []
            for m in range(M):
                if not bool(pm[t, m]):
                    continue
                pts = k2d[t, m]  # [J, 2]
                pose2d = pts[:17, :]
                racket2d = pts[17:20, :]
                pose_pix = _denormalize_points(pose2d, width, height)
                racket_pix = _denormalize_points(racket2d, width, height)
                players_pose.append(pose_pix)
                players_racket.append(racket_pix)

            frame = render_pose2d_frame(
                width=width,
                height=height,
                court_points=court_pix,
                court_visibility=None,
                player_poses=players_pose,
                player_pose_visibility=None,
                racket_points=players_racket,
                racket_visibility=None,
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    """Render dataset samples with augmentation applied for quick inspection."""
    args = _parse_args()

    torch.manual_seed(int(args.seed))

    dataset = _load_dataset_from_config(args.dataset_config, args.split)
    out_dir = Path(args.output_dir)

    start = int(args.start_index)
    num = int(args.num_samples)
    if start < 0:
        raise SystemExit("start_index must be non-negative")
    if num <= 0:
        raise SystemExit("num_samples must be positive")

    max_index = min(start + num, len(dataset))
    if start >= len(dataset):
        msg = f"start_index={start} is out of range for dataset length {len(dataset)}"
        raise SystemExit(msg)

    for idx in range(start, max_index):
        sample = dataset[idx]
        scene_id = int(sample["scene_id"].item())
        t_start = int(sample["t_start"].item())
        t_end = int(sample["t_end"].item())
        out_path = (
            out_dir
            / f"{args.split}_idx{idx:06d}_scene{scene_id}_t{t_start:04d}-{t_end:04d}.mp4"
        )
        _render_sample_to_video(
            sample=sample,
            out_path=out_path,
            camera_index=int(args.camera_index),
            fps=float(args.fps),
        )
        print(f"[tennis-viz] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
