"""Render videos from v2 GT decomposition (canonical/root_trans/root_rot).

- Load TennisSceneWindowDataset (memmap recommended).
- Reconstruct global pose from canonical/root_trans/root_rot.
- Project reconstructed 3D poses to 2D with camera params.
- Build a simulator-like scene dict and call render_video().

Used to visually check that the v2 decomposition is correct.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from src.datasets.tennis import TennisSceneWindowDataset
from src.tennis.geometry.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)
from src.visualize.tennis_multi_cam_3d_pose import render_video


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct global pose from v2 GT and render videos.",
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default="configs/datasets/tennis_multi_cam_3d_pose_sim.yaml",
        help="Path to dataset config YAML.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tennis_v2_decomposition_viz",
        help="Directory to save rendered videos.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of windows to render.",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting dataset index.",
    )
    p.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help=(
            "Camera index to visualize. If negative, choose the first camera "
            "that sees any player."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS for output videos.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for camera sampling.",
    )
    return p.parse_args()


def _load_dataset_from_config(path: str, split: str) -> TennisSceneWindowDataset:
    cfg = OmegaConf.load(path)
    container = OmegaConf.to_container(cfg, resolve=True) or {}
    if not isinstance(container, Mapping):
        raise SystemExit(f"Dataset config root must be a mapping: {path}")
    cfg_dict: Mapping[str, Any] = container

    root = cfg_dict.get("root", "data/tennis_autogen")
    name = cfg_dict.get("name") or cfg_dict.get("dataset_name")
    if not name:
        raise SystemExit(
            f"Dataset config must define 'name' (or 'dataset_name'): {path}"
        )
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
        dataset_name=str(name),
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
    pts = points.astype(np.float32).copy()
    if width <= 0 or height <= 0:
        return pts
    pts[..., 0] = (pts[..., 0] + 1.0) * 0.5 * float(width)
    pts[..., 1] = (pts[..., 1] + 1.0) * 0.5 * float(height)
    return pts


def _select_camera_index(sample: dict[str, torch.Tensor], explicit_idx: int) -> int:
    keypoints_2d = sample["keypoints_2d"]  # [T,V,M,J,2]
    player_mask = sample["player_mask"]  # [T,V,M]
    _, V, M, _, _ = keypoints_2d.shape
    if explicit_idx >= 0:
        if explicit_idx >= V:
            raise ValueError(
                f"Requested camera_index={explicit_idx} but only V={V} cameras"
            )
        return explicit_idx
    mask_any = player_mask.any(dim=0).any(dim=1)  # [V]
    for v in range(V):
        if bool(mask_any[v]):
            return v
    return 0


def _reconstruct_global_from_v2(
    canonical: Tensor,
    root_trans: Tensor,
    root_rot: Tensor,
) -> Tensor:
    """Reconstruct global pose from v2 components.

    canonical: [T,M,J,3], root_trans: [T,M,3], root_rot: [T,M,2].
    Returns: [T,M,J,3].
    """
    if canonical.ndim != 4 or root_trans.ndim != 3 or root_rot.ndim != 3:
        raise ValueError("Unexpected shapes for v2 components")

    cos_theta = root_rot[..., 0]
    sin_theta = root_rot[..., 1]

    x_can = canonical[..., 0]
    y_can = canonical[..., 1]
    z_can = canonical[..., 2]

    # R(+theta) * canonical -> root-relative
    x_rel = cos_theta[..., None] * x_can - sin_theta[..., None] * y_can
    y_rel = sin_theta[..., None] * x_can + cos_theta[..., None] * y_can
    z_rel = z_can
    rel = torch.stack([x_rel, y_rel, z_rel], dim=-1)

    return rel + root_trans[:, :, None, :]


def _denorm_pose3d(pose_norm: Tensor) -> Tensor:
    scales = pose_norm.new_tensor(
        [HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST],
        dtype=pose_norm.dtype,
    )
    return pose_norm * scales


def _project_world_points(
    cam_C: Tensor,
    cam_R: Tensor,
    cam_intr: Tensor,
    xyz_world: Tensor,
) -> tuple[Tensor, Tensor]:
    rel = xyz_world - cam_C.view(1, 3)
    Xc = rel @ cam_R.t()
    z = Xc[:, 2]
    mask = z > 1e-6
    z_safe = torch.where(mask, z, torch.ones_like(z))
    f = cam_intr[0]
    cx = cam_intr[1]
    cy = cam_intr[2]
    u = f * (Xc[:, 0] / z_safe) + cx
    v = f * (-Xc[:, 1] / z_safe) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, mask


def _build_scene_from_sample(
    sample: dict[str, torch.Tensor],
    global_recon: Tensor,
    camera_index: int,
    fps: float,
) -> tuple[dict[str, Any], int, int]:
    keypoints_2d = sample["keypoints_2d"]  # [T,V,M,J,2] (shape referenceのみ使用)
    player_mask = sample["player_mask"]  # [T,V,M]
    court_2d = sample["court_2d"]  # [V,20,2]
    camera_C = sample["camera_C"]  # [V,3]
    camera_R = sample["camera_R"]  # [V,3,3]
    camera_intr = sample["camera_intr"]  # [V,3]
    image_size = sample["image_size"]  # [V,2]

    T, V, M, J, _ = keypoints_2d.shape
    if J < 20:
        raise ValueError("Expected at least 20 joints (17 body + 3 racket)")
    if global_recon.shape[:3] != (T, M, J):
        raise ValueError("global_recon shape does not match sample keypoints shape")

    v_idx = _select_camera_index(sample, camera_index)
    size_tensor = image_size[v_idx]
    width = int(size_tensor[0].item())
    height = int(size_tensor[1].item())
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    court = court_2d[v_idx].numpy()
    court_px = _denormalize_points(court, width, height)

    cam_C = camera_C[v_idx]
    cam_R = camera_R[v_idx]
    cam_intr = camera_intr[v_idx]

    frames: list[dict[str, Any]] = []
    for t in range(T):
        players_joints: list[list[list[float]]] = []
        players_joints_vis: list[list[int]] = []
        players_racket: list[list[list[float]]] = []
        players_racket_vis: list[list[int]] = []

        for m in range(M):
            if not bool(player_mask[t, v_idx, m]):
                continue

            # global_recon: 正規化コート座標 → ワールド座標 → ピクセルに再投影
            pose_norm = global_recon[t, m]  # [J,3]
            pose_world = _denorm_pose3d(pose_norm)
            uv, vis = _project_world_points(cam_C, cam_R, cam_intr, pose_world)
            uv_np = uv.detach().cpu().numpy().astype("float32")
            vis_np = vis.detach().cpu().numpy().astype("uint8")

            body_uv = uv_np[:17]
            racket_uv = uv_np[17:20]
            body_vis = vis_np[:17].tolist()
            racket_vis = vis_np[17:20].tolist()

            players_joints.append(body_uv.tolist())
            players_joints_vis.append(body_vis)
            players_racket.append(racket_uv.tolist())
            players_racket_vis.append(racket_vis)

        cam_payload = {
            "court_keypoints_2d": {
                "points": court_px.tolist(),
                "visibility": [1] * int(court_px.shape[0]),
            },
            "player_keypoints_2d": {
                "joints": players_joints,
                "visibility": players_joints_vis,
            },
            "racket_keypoints_2d": {
                "points": players_racket,
                "visibility": players_racket_vis,
            },
        }
        frames.append({"cam_0": cam_payload})

    scene = {
        "fps": float(fps),
        "cameras": [{"image_size": [width, height]}],
        "frames": frames,
    }
    return scene, width, height


def _render_sample(
    sample: dict[str, torch.Tensor],
    out_path: Path,
    camera_index: int,
    fps: float,
) -> None:
    canonical = sample.get("canonical_pose_gt")
    root_trans = sample.get("root_trans_gt")
    root_rot = sample.get("root_rot_gt")
    global_gt = sample.get("global_pose_gt")
    if canonical is None or root_trans is None or root_rot is None:
        raise SystemExit(
            "Sample is missing v2 GT tensors (canonical/root_trans/root_rot)"
        )

    global_recon = _reconstruct_global_from_v2(canonical, root_trans, root_rot)

    # Optional: print reconstruction error vs stored global_pose_gt if available.
    if global_gt is not None and global_gt.shape == global_recon.shape:
        diff = (global_recon - global_gt).abs()
        max_err = float(diff.max().item())
        mean_err = float(diff.mean().item())
        print(f"[v2-decomp] recon error (max={max_err:.4f}, mean={mean_err:.4f})")

    scene, width, height = _build_scene_from_sample(
        sample, global_recon, camera_index, fps
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_video(
        scene, str(out_path), camera_index=0, width=width, height=height, fps=int(fps)
    )


def main() -> int:
    """Render tennis v2 decomposition videos.

    Returns:
        int: Exit code (0 for success).

    Raises:
        SystemExit: If start_index is negative or num_samples is not positive.

    """
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
        raise SystemExit(
            f"start_index={start} is out of range for dataset length {len(dataset)}"
        )

    for idx in range(start, max_index):
        sample = dataset[idx]
        scene_id = int(sample["scene_id"].item())
        t_start = int(sample["t_start"].item())
        t_end = int(sample["t_end"].item())
        out_path = (
            out_dir
            / f"{args.split}_idx{idx:06d}_scene{scene_id}_t{t_start:04d}-{t_end:04d}.mp4"
        )
        print(f"[v2-decomp] Rendering index={idx} -> {out_path}")
        _render_sample(
            sample=sample,
            out_path=out_path,
            camera_index=int(args.camera_index),
            fps=float(args.fps),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
