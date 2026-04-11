"""Training visualisation helpers for court detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# Colour palette for 7 seg classes (BGR for cv2)
_SEG_PALETTE = np.array([
    [0, 0, 0],
    [255, 100, 100],
    [100, 100, 255],
    [100, 255, 100],
    [255, 255, 100],
    [255, 100, 255],
    [100, 255, 255],
], dtype=np.uint8)


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ``[3, H, W]`` ImageNet tensor to ``[H, W, 3]`` uint8 BGR."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map label mask ``[H, W]`` to a coloured BGR image."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(len(_SEG_PALETTE)):
        vis[mask == label] = _SEG_PALETTE[label]
    return vis


def save_seg_vis(
    img_tensor: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_logits: torch.Tensor,
    path: Path,
) -> None:
    """Save a 3-panel seg visualisation (input | GT | pred)."""
    img_bgr = _denormalize_image(img_tensor)
    gt_np = gt_mask.cpu().numpy().astype(np.uint8)
    pred_np = pred_logits.argmax(0).cpu().numpy().astype(np.uint8)

    h, w = img_bgr.shape[:2]
    gt_vis = _colorize_mask(gt_np)
    pred_vis = _colorize_mask(pred_np)

    if gt_vis.shape[:2] != (h, w):
        gt_vis = cv2.resize(gt_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_vis.shape[:2] != (h, w):
        pred_vis = cv2.resize(pred_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    panel = np.concatenate([img_bgr, gt_vis, pred_vis], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def save_kp_vis(
    img_tensor: torch.Tensor,
    gt_heatmap: torch.Tensor,
    pred_heatmap: torch.Tensor,
    path: Path,
) -> None:
    """Save a 3-panel kp visualisation (input | GT heatmap | pred heatmap)."""
    img_bgr = _denormalize_image(img_tensor)

    gt_max = gt_heatmap.max(0).values.cpu().numpy()
    pred_max = torch.sigmoid(pred_heatmap).max(0).values.cpu().numpy()

    gt_cm = cv2.applyColorMap((gt_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pred_cm = cv2.applyColorMap((pred_max * 255).astype(np.uint8), cv2.COLORMAP_JET)

    h, w = img_bgr.shape[:2]
    if gt_cm.shape[:2] != (h, w):
        gt_cm = cv2.resize(gt_cm, (w, h))
    if pred_cm.shape[:2] != (h, w):
        pred_cm = cv2.resize(pred_cm, (w, h))

    panel = np.concatenate([img_bgr, gt_cm, pred_cm], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def save_line_vis(
    img_tensor: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_logits: torch.Tensor,
    path: Path,
) -> None:
    """Save a 4-panel line visualisation (input | GT | pred mask | overlay)."""
    img_bgr = _denormalize_image(img_tensor)
    gt_np = (gt_mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_prob = torch.sigmoid(pred_logits).squeeze(0).cpu().numpy()
    pred_np = (pred_prob > 0.5).astype(np.uint8) * 255

    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    color = np.zeros_like(overlay)
    color[..., 2] = pred_np
    color[..., 1] = pred_np // 4
    overlay = cv2.addWeighted(overlay, 0.72, color, 0.55, 0.0)

    h, w = img_bgr.shape[:2]
    if gt_bgr.shape[:2] != (h, w):
        gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_bgr.shape[:2] != (h, w):
        pred_bgr = cv2.resize(pred_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    panel = np.concatenate([img_bgr, gt_bgr, pred_bgr, overlay], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)
