"""Utilities to render 2D debug visualizations from training batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.training.utils.tennis_projection import (
    denorm_pose3d,
    norm_to_px,
    project_world_points,
)


def render_debug_images_with_cameras(
    batch: Mapping[str, Tensor],
    pose_pred: Tensor | None,
    exist_conf: Tensor | None,
    exist_threshold: float,
    render_pose2d_frame: Any,
) -> tuple[Tensor | None, Tensor | None]:
    """Render GT/pred images using calibrated cameras, if available."""
    required = {"camera_C", "camera_R", "camera_intr", "image_size"}
    if not required.issubset(batch.keys()):
        return None, None

    keypoints_2d = batch.get("keypoints_2d")
    player_mask = batch.get("player_mask")
    court_2d = batch.get("court_2d")
    camera_C = batch.get("camera_C")
    camera_R = batch.get("camera_R")
    camera_intr = batch.get("camera_intr")
    image_size = batch.get("image_size")

    if (
        keypoints_2d is None
        or player_mask is None
        or court_2d is None
        or pose_pred is None
        or camera_C is None
        or camera_R is None
        or camera_intr is None
        or image_size is None
    ):
        return None, None

    if keypoints_2d.ndim != 6 or player_mask.ndim != 4 or pose_pred.ndim != 5:
        return None, None

    B, T, V, M, J, _ = keypoints_2d.shape
    if B == 0 or T == 0 or V == 0:
        return None, None

    b_idx = 0
    t_idx = 0
    v_idx = 0

    size_tensor = image_size[b_idx, v_idx]
    width = int(size_tensor[0].item())
    height = int(size_tensor[1].item())
    if width <= 0 or height <= 0:
        return None, None

    def _select_cam(tensor: Tensor) -> Tensor:
        if tensor.ndim == 3:
            return tensor[b_idx, v_idx]
        if tensor.ndim == 4:
            return tensor[b_idx, v_idx]
        return tensor[v_idx]

    cam_C = _select_cam(camera_C).to(device=pose_pred.device, dtype=pose_pred.dtype)
    cam_R = _select_cam(camera_R).to(device=pose_pred.device, dtype=pose_pred.dtype)
    cam_intr = _select_cam(camera_intr).to(
        device=pose_pred.device,
        dtype=pose_pred.dtype,
    )

    court = court_2d[b_idx, v_idx] if court_2d.ndim == 4 else court_2d[v_idx]
    court_px = norm_to_px(court, width, height)
    court_vis = [1] * int(court_px.shape[0])

    kp = keypoints_2d[b_idx, t_idx, v_idx]
    mask = player_mask[b_idx, t_idx, v_idx]
    player_pose_list_gt: list[np.ndarray] = []
    racket_list_gt: list[np.ndarray] = []
    for m in range(M):
        if not bool(mask[m].item()):
            continue
        pts_px = norm_to_px(kp[m], width, height)
        player_pose_list_gt.append(pts_px[:17])
        racket_list_gt.append(pts_px[17:])

    img_gt_np = render_pose2d_frame(
        width=width,
        height=height,
        court_points=court_px,
        court_visibility=court_vis,
        player_poses=player_pose_list_gt,
        player_pose_visibility=None,
        racket_points=racket_list_gt,
        racket_visibility=None,
    )
    img_gt = (
        torch.from_numpy(img_gt_np)
        .permute(2, 0, 1)
        .to(device=keypoints_2d.device, dtype=torch.float32)
        / 255.0
    )

    pose_slice = pose_pred[b_idx, :, t_idx]
    pose_world = denorm_pose3d(pose_slice).detach()

    if exist_conf is not None and exist_conf.shape[0] > b_idx:
        exist_mask = exist_conf[b_idx, :, 0] >= exist_threshold
    else:
        exist_mask = torch.ones(
            pose_world.shape[0],
            dtype=torch.bool,
            device=pose_pred.device,
        )

    player_pose_list_pred: list[np.ndarray] = []
    pose_vis_list: list[list[int]] = []
    racket_list_pred: list[np.ndarray] = []
    racket_vis_list: list[list[int]] = []
    for q in range(pose_world.shape[0]):
        if not bool(exist_mask[q].item()):
            continue
        uv, vis = project_world_points(cam_C, cam_R, cam_intr, pose_world[q])
        uv_np = uv.detach().float().cpu().numpy().astype("float32")
        vis_np = vis.detach().cpu().numpy().astype("uint8")
        player_pose_list_pred.append(uv_np[:17])
        racket_list_pred.append(uv_np[17:])
        pose_vis_list.append(vis_np[:17].tolist())
        racket_vis_list.append(vis_np[17:].tolist())

    img_pred_np = render_pose2d_frame(
        width=width,
        height=height,
        court_points=court_px,
        court_visibility=court_vis,
        player_poses=player_pose_list_pred,
        player_pose_visibility=pose_vis_list if pose_vis_list else None,
        racket_points=racket_list_pred,
        racket_visibility=racket_vis_list if racket_vis_list else None,
    )
    img_pred = (
        torch.from_numpy(img_pred_np)
        .permute(2, 0, 1)
        .to(device=keypoints_2d.device, dtype=torch.float32)
        / 255.0
    )

    return img_gt, img_pred


def render_debug_images_naive(
    batch: Mapping[str, Tensor],
    pose_pred: Tensor | None,
    render_pose2d_frame: Any,
) -> tuple[Tensor | None, Tensor | None]:
    """Render fallback 2D debug images without camera information."""
    keypoints_2d = batch.get("keypoints_2d")
    player_mask = batch.get("player_mask")
    court_2d = batch.get("court_2d")

    if (
        keypoints_2d is None
        or player_mask is None
        or court_2d is None
        or pose_pred is None
    ):
        return None, None

    if keypoints_2d.ndim != 6 or pose_pred.ndim != 5 or court_2d.ndim != 4:
        return None, None

    B, T, V, M, _, _ = keypoints_2d.shape
    if B == 0 or T == 0 or V == 0 or M == 0:
        return None, None

    H, W = 288, 512

    kp = keypoints_2d[0, 0, 0]
    mask = player_mask[0, 0, 0]
    court = court_2d[0, 0]

    court_px = norm_to_px(court, W, H)
    player_pose_list_gt: list[np.ndarray] = []
    racket_list_gt: list[np.ndarray] = []
    for m in range(M):
        if not bool(mask[m].item()):
            continue
        pts_px = norm_to_px(kp[m], W, H)
        player_pose_list_gt.append(pts_px[:17])
        racket_list_gt.append(pts_px[17:])

    court_vis = [1] * int(court_px.shape[0])
    img_gt_np = render_pose2d_frame(
        width=W,
        height=H,
        court_points=court_px,
        court_visibility=court_vis,
        player_poses=player_pose_list_gt,
        player_pose_visibility=None,
        racket_points=racket_list_gt,
        racket_visibility=None,
    )
    img_gt = (
        torch.from_numpy(img_gt_np)
        .permute(2, 0, 1)
        .to(device=keypoints_2d.device, dtype=torch.float32)
        / 255.0
    )

    pose_slice = pose_pred[0, :, 0]
    player_pose_list_pred: list[np.ndarray] = []
    racket_list_pred: list[np.ndarray] = []
    for pts3d in pose_slice:
        coords = pts3d[:, :2].detach().float().cpu().numpy().astype("float32")
        coords_px = np.empty_like(coords)
        coords_px[..., 0] = (coords[..., 0] * 0.1 + 0.5) * float(W - 1)
        coords_px[..., 1] = (coords[..., 1] * 0.1 + 0.5) * float(H - 1)
        player_pose_list_pred.append(coords_px[:17])
        racket_list_pred.append(coords_px[17:])

    img_pred_np = render_pose2d_frame(
        width=W,
        height=H,
        court_points=court_px,
        court_visibility=court_vis,
        player_poses=player_pose_list_pred,
        player_pose_visibility=None,
        racket_points=racket_list_pred,
        racket_visibility=None,
    )
    img_pred = (
        torch.from_numpy(img_pred_np)
        .permute(2, 0, 1)
        .to(device=keypoints_2d.device, dtype=torch.float32)
        / 255.0
    )

    return img_gt, img_pred
