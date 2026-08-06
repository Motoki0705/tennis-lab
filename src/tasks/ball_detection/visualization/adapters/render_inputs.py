"""Adapter that converts training batch tensors into render_animation_frames inputs.

This module bridges the gap between the training pipeline (batched tensors, ImageNet
normalisation) and the clip renderer (lists of RGB numpy arrays with pixel-space
coordinates).  It also houses a helper for computing Motion Difference Decomposition
(MDD) frames directly from model-input tensors so that the MDD panel in the 2x2 grid
can be populated without requiring a separate ``ClipSequence`` or ``Predictor`` object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_detection.data.components.augmentation import (
    denormalize_tensor_images_imagenet,
)
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.utils.data.augmentation import tensor_images_to_uint8_rgb
from src.utils.data.heatmaps import heatmaps_to_argmax


def build_mdd_frames_from_images(
    images_btchw: Tensor,
    model_io: BallModelIOAdapter,
) -> list[np.ndarray]:
    """Compute per-frame MDD RGB visualisation from a ``(B, T, C, H, W)`` tensor.

    MDD channels are derived by the same canonical model-I/O adapter used for
    inference.

    Args:
        images_btchw: ``(B, T, C, H, W)`` float tensor in ImageNet-normalised space
            *before* any mode conversion.  The function uses only ``B=sample_idx``
            slice via the caller (see :func:`build_render_animation_inputs`).
        model_io: The selected canonical ball model-I/O adapter.

    Returns:
        List of T ``(H, W, 3)`` uint8 RGB arrays.
    """
    h = images_btchw.shape[3]
    w = images_btchw.shape[4]

    with torch.no_grad():
        features = model_io.mdd_features(images_btchw)

    # Use the first sample in the batch.
    brighten = features[0, 0].clamp(0.0, 1.0).cpu().numpy()  # (T, H, W)
    darken = features[0, 1].clamp(0.0, 1.0).cpu().numpy()  # (T, H, W)

    mdd_frames: list[np.ndarray] = []
    for t in range(brighten.shape[0]):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (darken[t] * 255.0).astype(np.uint8)  # red  = darken
        rgb[..., 1] = (brighten[t] * 255.0).astype(np.uint8)  # green = brighten
        mdd_frames.append(rgb)
    return mdd_frames


def build_render_animation_inputs(
    *,
    images_btchw: Tensor,
    pred_heatmaps_bthw: Tensor,
    peak_threshold: float,
    normalize_cfg: dict[str, Any] | None = None,
    model_io: BallModelIOAdapter,
    sample_idx: int = 0,
    clip_label: str = "train",
) -> dict[str, Any]:
    """Build the keyword arguments expected by ``render_animation_frames``.

    Converts batched training tensors (ImageNet-normalised images, predicted
    heatmaps in [0, 1]) into the frame lists / coordinate sequences that the
    renderer needs.

    Args:
        images_btchw: ``(B, T, C, H, W)`` float tensor (ImageNet-normalised).
        pred_heatmaps_bthw: ``(B, T, Hh, Ww)`` float tensor in [0, 1].
        peak_threshold: Confidence threshold above which a detection is drawn.
        normalize_cfg: Dict with ``enabled``, ``mean``, ``std`` keys (from
            ``data.augmentation.normalize_imagenet`` config section).  Used to
            undo ImageNet normalisation before converting to uint8.
        model_io: Bound adapter used for canonical MDD construction.
        sample_idx: Which element of the batch to visualise (default 0).
        clip_label: Human-readable label placed in the rendered header.

    Returns:
        Dict with all keyword arguments for ``render_animation_frames``.
        The caller can pass it directly as ``render_animation_frames(**inputs)``.
    """
    if normalize_cfg is None:
        raise ValueError("normalize_cfg is required for render input adaptation.")
    cfg = normalize_cfg
    _, t, _, h, w = images_btchw.shape

    # ------------------------------------------------------------------ images
    frames_tensor = images_btchw[sample_idx].detach().cpu()  # (T, C, H, W)
    if bool(cfg["enabled"]):
        frames_tensor = denormalize_tensor_images_imagenet(
            frames_tensor,
            mean=cfg["mean"],
            std=cfg["std"],
        )
    frames_rgb: list[np.ndarray] = list(tensor_images_to_uint8_rgb(frames_tensor))

    # --------------------------------------------------------------- heatmaps
    heatmaps_t = pred_heatmaps_bthw[sample_idx].detach().cpu()  # (T, Hh, Ww)
    # heatmaps_to_argmax wants (T, Hh, Ww) → returns (T, 2) normalized, (T,) values
    coords_normalized, confidences = heatmaps_to_argmax(heatmaps_t)  # (T, 2), (T,)

    image_height, image_width = h, w
    pred_coords_px: list[tuple[float, float]] = []
    pred_visibility: list[bool] = []
    pred_confidences: list[float] = []

    for t_idx in range(t):
        x_norm = float(coords_normalized[t_idx, 0].item())
        y_norm = float(coords_normalized[t_idx, 1].item())
        conf = float(confidences[t_idx].item())
        x_px = x_norm * max(image_width - 1, 0)
        y_px = y_norm * max(image_height - 1, 0)
        pred_coords_px.append((x_px, y_px))
        pred_confidences.append(conf)
        pred_visibility.append(conf >= peak_threshold)

    # ----------------------------------------------------------------- heatmap panels (resized to frame size)
    pred_heatmaps_np: list[np.ndarray] = []
    for t_idx in range(t):
        hm = heatmaps_t[t_idx].numpy()  # (Hh, Ww)
        pred_heatmaps_np.append(hm)

    # --------------------------------------------------------------- MDD frames
    mdd_frames_rgb = build_mdd_frames_from_images(
        images_btchw[sample_idx : sample_idx + 1],  # keep batch dim → (1, T, C, H, W)
        model_io=model_io,
    )

    # --------------------------------------------------------------- frame names
    frame_names = [f"t{t_idx:02d}" for t_idx in range(t)]

    return {
        "frames_rgb": frames_rgb,
        "frame_names": frame_names,
        "mdd_frames_rgb": mdd_frames_rgb,
        "pred_coords_px": pred_coords_px,
        "pred_visibility": pred_visibility,
        "pred_confidences": pred_confidences,
        "pred_heatmaps": pred_heatmaps_np,
        "peak_threshold": peak_threshold,
        "clip_label": clip_label,
    }


__all__ = [
    "build_mdd_frames_from_images",
    "build_render_animation_inputs",
]
