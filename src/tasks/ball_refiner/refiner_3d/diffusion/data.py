"""Validated synthetic rallies to full-GMM training windows, including padding."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import TrainingBatch
from src.tasks.ball_refiner.refiner_3d.diffusion.model import MixtureCondition


def rally_window(
    arrays: dict[str, NDArray[Any]], record: dict[str, Any], *, start: int,
    frames: int, allow_nonconverged: bool,
) -> TrainingBatch:
    """Requires arrays already verified by SyntheticDataset.load.

    Nonconvergence never removes a frame/component. The caller must explicitly
    permit flagged numerical inputs for a plumbing experiment. Missing flags are
    rejected rather than interpreting a historical dataset as converged.
    """
    if frames < 3 or start < 0 or start >= record['frames'] - 2:
        raise ValueError('Need >=3 real frames and a valid window start')
    if 'integration_converged' not in arrays:
        raise ValueError('Reintegrate historical data to obtain convergence flags')
    end = min(start + frames, record['frames'])
    length = end - start
    if not allow_nonconverged and not arrays['integration_converged'][start:end].all():
        raise ValueError('Nonconverged input requires explicit plumbing permission')

    def temporal(key: str, *, axis: int = 0) -> torch.Tensor:
        value = torch.from_numpy(arrays[key].copy()).movedim(axis, 0)[start:end]
        padded = torch.zeros((frames, *value.shape[1:]), dtype=value.dtype)
        padded[:length] = value
        return padded.movedim(0, axis).unsqueeze(0)

    padding = torch.arange(frames)[None] >= length
    condition = MixtureCondition(
        temporal('gmm3d_means_m').float(), temporal('gmm3d_covariance_m2').float(),
        temporal('gmm3d_weights').float(), temporal('gmm3d_camera_subsets').bool(),
        temporal('prior_only_probability').float(), temporal('timestamps_seconds').float(), padding,
    )
    # BallGMM2D performs the exact #935 source-size conversion in float64.
    distribution = BallGMM2D(*(torch.from_numpy(arrays[key][:, start:end].copy()) for key in (
        'gmm2d_means_uv', 'gmm2d_scale_tril_uv', 'gmm2d_mixture_logits', 'gmm2d_presence_logits')))
    means, covariance = distribution.pixel_moments(torch.from_numpy(arrays['source_size_wh'].astype(np.float64)))

    def camera_padding(value: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros((value.shape[0], frames, *value.shape[2:]), dtype=torch.float32)
        padded[:, :length] = value.float()
        return padded[None]

    matrix = arrays['camera_estimated_K'] @ np.concatenate((arrays['camera_estimated_R'], arrays['camera_estimated_t'][..., None]), axis=-1)
    return TrainingBatch(
        condition, temporal('positions_3d_m').float(), temporal('event_labels').bool(),
        temporal('free_flight_mask').bool(), torch.from_numpy(matrix)[None].float(),
        camera_padding(means), camera_padding(covariance),
        camera_padding(distribution.weights), camera_padding(distribution.presence_probability),
        torch.tensor([record['physics']['gravity']], dtype=torch.float32),
    )
