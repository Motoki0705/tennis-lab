"""Adapter: convert a training batch + model output into PoseRenderScene pairs.

Usage::

    gt_scene, pred_scene = batch_to_pose_render_scenes(batch, out, sample_idx=0)
    view = "3d" if (gt_scene.canonical_pose_3d is not None
                    and pred_scene.canonical_pose_3d is not None) else "2d_topdown"
    anim = PLCSSceneRenderer().create_comparison_animation(gt_scene, pred_scene, view=view)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.tasks.plcs.model_io import PLCSDecodedPrediction
from src.tasks.plcs.visualization.contracts import PoseRenderScene
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
from src.utils.tensor_utils import to_numpy


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a tensor or array-like to a float32 numpy array."""
    array: np.ndarray = to_numpy(t, dtype=np.float32)
    return array


def _ensure_time_dim(arr: np.ndarray, ndim_without_time: int) -> np.ndarray:
    """Insert a leading time dimension when the array has no T axis.

    ``ndim_without_time`` is the expected number of dims for a *single* frame.
    If ``arr.ndim == ndim_without_time`` we add a leading dim to get shape
    ``(1, ...)``.
    """
    if arr.ndim == ndim_without_time:
        return arr[np.newaxis]
    return arr


def batch_to_pose_render_scenes(
    batch: dict[str, Any],
    output: PLCSDecodedPrediction,
    *,
    sample_idx: int = 0,
) -> tuple[PoseRenderScene, PoseRenderScene]:
    """Build a (gt_scene, pred_scene) pair from a training batch and model output.

    The function extracts sample ``sample_idx`` from the batch dimension ``B``
    and ensures all arrays have a leading time dimension ``T``.

    GT canonical_pose_3d strategy
    ------------------------------
    The dataset always includes ``batch["human_kp_3d"]`` — world-coordinate
    COCO-17 joints (T, 17, 3).  We invert the placement transform
    (``world_pose_to_canonical_pose``) to recover the canonical representation
    so that :meth:`PLCSSceneRenderer._compute_world_pose` can reconstruct the
    same world pose.

    Pred canonical_pose_3d strategy
    --------------------------------
    If the model outputs ``"canonical_pose"`` it is used directly.  Otherwise
    ``canonical_pose_3d`` is ``None`` and callers should use ``view="2d_topdown"``.

    Args:
        batch: Collated training batch (keys: ``position``, ``rotation``,
            ``human_kp_3d`` (optional), …).  Tensors have batch-leading dim B.
        output: Model output dict (keys: ``position``, ``rotation``,
            ``canonical_pose`` (optional)).  Tensors have batch-leading dim B.
        sample_idx: Which sample in the batch to render (default 0).

    Returns:
        ``(gt_scene, pred_scene)`` — two :class:`PoseRenderScene` instances.
    """
    # ---- GT ----------------------------------------------------------------
    gt_pos_raw = _to_numpy(batch["position"])[sample_idx]  # ([T], 3)
    gt_rot_raw = _to_numpy(batch["rotation"])[sample_idx]  # ([T], 2)

    gt_pos = _ensure_time_dim(gt_pos_raw, ndim_without_time=1)  # (T, 3)
    gt_rot = _ensure_time_dim(gt_rot_raw, ndim_without_time=1)  # (T, 2)
    T = gt_pos.shape[0]

    gt_canonical: np.ndarray | None = None
    human_kp_3d = batch.get("human_kp_3d")
    if human_kp_3d is not None:
        kp3d = _to_numpy(human_kp_3d)[sample_idx]  # ([T], 17, 3)
        kp3d = _ensure_time_dim(kp3d, ndim_without_time=2)  # (T, 17, 3)
        # Convert world joints → canonical (local yaw-zero) space.
        # We use the torch helper which accepts arbitrary shapes (..., J, 3).
        pos_t = torch.from_numpy(gt_pos).float()  # (T, 3)
        rot_t = torch.from_numpy(gt_rot).float()  # (T, 2)
        kp3d_t = torch.from_numpy(kp3d).float()  # (T, 17, 3)
        gt_canonical = _to_numpy(
            world_pose_to_canonical_pose(kp3d_t, pos_t, rot_t)
        )  # (T, 17, 3)

    meta: dict[str, Any] = {"num_frames": T}
    gt_scene = PoseRenderScene(
        position=gt_pos,
        rotation=gt_rot,
        canonical_pose_3d=gt_canonical,
        meta=meta,
    )

    # ---- Pred --------------------------------------------------------------
    pred_pos_raw = _to_numpy(output.position)[sample_idx]  # ([T], 3)
    pred_rot_raw = _to_numpy(output.rotation)[sample_idx]  # ([T], 2)

    pred_pos = _ensure_time_dim(pred_pos_raw, ndim_without_time=1)  # (T, 3)
    pred_rot = _ensure_time_dim(pred_rot_raw, ndim_without_time=1)  # (T, 2)
    T_pred = pred_pos.shape[0]

    pred_canonical: np.ndarray | None = None
    canonical_pose_out = output.canonical_pose
    if canonical_pose_out is not None:
        cp = _to_numpy(canonical_pose_out)[sample_idx]  # ([T], J, 3)
        pred_canonical = _ensure_time_dim(cp, ndim_without_time=2)  # (T, J, 3)

    pred_meta: dict[str, Any] = {"num_frames": T_pred}
    pred_scene = PoseRenderScene(
        position=pred_pos,
        rotation=pred_rot,
        canonical_pose_3d=pred_canonical,
        meta=pred_meta,
    )

    return gt_scene, pred_scene
