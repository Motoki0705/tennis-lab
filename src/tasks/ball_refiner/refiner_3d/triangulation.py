"""The sole task boundary: consume #935 BallGMM2D, never raw detections."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_2d.distribution import FLOAT_DTYPES, BallGMM2D
from src.tasks.base.model_io.tensors import TensorSpec
from src.utils.geometry.probabilistic_triangulation import CameraGMM


def frame_observations(
    distribution: BallGMM2D, source_size_wh: Tensor, *, frame: int
) -> CameraGMM:
    """Treat B as synchronized cameras in caller-declared order; detach to CPU.

    Synchronization and camera identity belong to the caller. Uses the existing
    pixel_moments transformation; neither averaging nor top-1 selection occurs.
    """
    if not 0 <= frame < distribution.means.shape[1]:
        raise ValueError("Frame index is outside the refiner sequence")
    TensorSpec((distribution.means.shape[0], 2), FLOAT_DTYPES).validate("source_size_wh", source_size_wh)
    if source_size_wh.device != distribution.means.device:
        raise ValueError("Source image sizes must share the original GMM device")
    # Promote the source Cholesky factor BEFORE either matrix product/scaling.
    # Float32 D*Sigma*D can round the two off-diagonals differently on a
    # non-square image. This avoids the roundoff, without symmetrizing/jittering.
    frame_distribution = BallGMM2D(
        distribution.means[:, frame:frame + 1].detach().to(dtype=torch.float64, device="cpu"),
        distribution.scale_tril[:, frame:frame + 1].detach().to(dtype=torch.float64, device="cpu"),
        distribution.mixture_logits[:, frame:frame + 1].detach().to(dtype=torch.float64, device="cpu"),
        distribution.presence_logits[:, frame:frame + 1].detach().to(dtype=torch.float64, device="cpu"),
    )
    means, covariance = frame_distribution.pixel_moments(source_size_wh.detach().to(dtype=torch.float64, device="cpu"))
    return CameraGMM(
        means[:, 0].numpy(),
        covariance[:, 0].numpy(),
        distribution.weights[:, frame].detach().cpu().numpy().astype(np.float64),
        distribution.presence_probability[:, frame]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    )
