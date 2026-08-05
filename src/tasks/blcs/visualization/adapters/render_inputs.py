"""Render input adapters for BLCS training qualitative visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def batch_to_trajectory_arrays(
    batch: dict[str, Any],
    output: dict[str, Any],
    *,
    sample_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract GT and predicted trajectory arrays from a training batch and model output.

    Args:
        batch: Training batch dict. Must contain ``"position_3d"`` (B, T, 3).
        output: Model output dict. Must contain ``"position"`` (B, T, 3).
        sample_idx: Which sample within the batch to extract (default 0).

    Returns:
        Tuple of ``(gt_positions, pred_positions)``, each shaped ``(T, 3)`` as
        float32 numpy arrays.  If ``batch["ball_mask"]`` is present only the
        valid (non-padded) frames are returned; both arrays are trimmed to the
        same index set.

    """
    gt_raw = batch["position_3d"]
    pred_raw = output["position"]

    # Support both Tensor and numpy
    if hasattr(gt_raw, "detach"):
        gt_np: np.ndarray = gt_raw.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_raw, dtype=np.float32)

    if hasattr(pred_raw, "detach"):
        pred_np: np.ndarray = pred_raw.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_raw, dtype=np.float32)

    gt: np.ndarray = gt_np[sample_idx].astype(np.float32)  # (T, 3)
    pred: np.ndarray = pred_np[sample_idx].astype(np.float32)  # (T, 3)

    # Trim to valid frames if ball_mask is available
    ball_mask = batch.get("ball_mask")
    if ball_mask is not None:
        if hasattr(ball_mask, "detach"):
            mask_np: np.ndarray = ball_mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(ball_mask, dtype=np.float32)

        # Normalise to (B, T). Handles single-view (B, T), (B, T, 1), (B, 1, T)
        # and multi-view (B, N, T) masks (a frame is valid if any view is).
        if mask_np.ndim == 3:
            if mask_np.shape[-1] == 1:  # (B, T, 1)
                mask_np = mask_np.squeeze(-1)
            elif mask_np.shape[1] == 1:  # (B, 1, T)
                mask_np = mask_np.squeeze(1)
            else:  # (B, N, T) multi-view -> collapse the view axis
                mask_np = mask_np.max(axis=1)

        row: np.ndarray = mask_np[sample_idx]  # (T,)
        valid_idx = np.where(row > 0)[0]

        if len(valid_idx) > 0:
            gt = gt[valid_idx]
            pred = pred[valid_idx]

    return gt, pred
