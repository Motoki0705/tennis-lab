"""Monte Carlo highest-density regions of the complete conditional mixture on R²."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_2d.distribution import (
    BallGMM2D,
    conditional_log_density,
)


@dataclass(frozen=True)
class HDRResult:
    """CPU tensors (B,T,L); area is UV² and excludes no off-image Gaussian tails."""

    log_threshold: Tensor
    covered: Tensor
    area_uv2: Tensor
    area_mc_standard_error_uv2: Tensor


def validate_hdr_settings(levels: tuple[float, ...], samples: int, seed: int, chunk_size: int) -> None:
    if not levels or tuple(sorted(set(levels))) != levels or not all(math.isfinite(p) and 0 < p < 1 for p in levels):
        raise ValueError("HDR levels must be distinct, increasing probabilities in (0,1)")
    if samples < 32 or chunk_size < 1 or not 0 <= seed < 2**63 - 1:
        raise ValueError("Require samples>=32, positive chunk_size, and seed in [0,2**63-1)")


def _sample_log_density(
    means: Tensor, tril: Tensor, logits: Tensor, samples: int, generator: torch.Generator,
) -> Tensor:
    # A fixed number of CPU uniforms per frame makes the random stream independent
    # of execution device/chunk size. Box-Muller avoids normal-generator cache effects.
    uniform = torch.rand((len(means), samples, 3), generator=generator, dtype=torch.float64).to(means.device)
    cdf = logits.softmax(-1).cumsum(-1)
    cdf[:, -1] = 1
    component = torch.searchsorted(cdf.contiguous(), uniform[..., 0].contiguous(), right=True)
    radius = torch.sqrt(-2 * torch.log1p(-uniform[..., 1]))
    phase = 2 * math.pi * uniform[..., 2]
    noise = torch.stack((radius * phase.cos(), radius * phase.sin()), dim=-1)
    frame = torch.arange(len(means), device=means.device)[:, None]
    point = means[frame, component] + (tril[frame, component] @ noise[..., None]).squeeze(-1)
    return conditional_log_density(point, means[:, None], tril[:, None], logits[:, None])


def highest_density_regions(
    prediction: BallGMM2D, target_uv: Tensor, *, levels: tuple[float, ...],
    samples: int, seed: int, chunk_size: int,
) -> HDRResult:
    """Fit density quantiles, then estimate area with independent mixture draws.

    For threshold c, area(HDR) = E_p[1(p(X)>=c)/p(X)]. The reported area SE
    conditions on the estimated threshold; it does not include threshold uncertainty.
    Neither the target nor existence probability affects the conditional region.
    """
    validate_hdr_settings(levels, samples, seed, chunk_size)
    if (target_uv.shape != (*prediction.presence_logits.shape, 2)
            or target_uv.device != prediction.means.device or not target_uv.is_floating_point()
            or not bool(torch.isfinite(target_uv).all())):
        raise ValueError("HDR targets must be finite floating B,T,2 on the prediction device")
    b, t, k, _ = prediction.means.shape
    means = prediction.means.detach().reshape(-1, k, 2).double()
    tril = prediction.scale_tril.detach().reshape(-1, k, 2, 2).double()
    logits = prediction.mixture_logits.detach().reshape(-1, k).double()
    target = target_uv.detach().reshape(-1, 2).double()
    quantiles = 1 - torch.tensor(levels, device=means.device, dtype=torch.float64)
    fit_rng = torch.Generator().manual_seed(seed)
    area_rng = torch.Generator().manual_seed(seed + 1)
    thresholds, coverage, areas, errors = [], [], [], []
    for start in range(0, b * t, chunk_size):
        mu, scale, weights = (value[start:start + chunk_size] for value in (means, tril, logits))
        density = _sample_log_density(mu, scale, weights, samples, fit_rng)
        threshold = torch.quantile(density, quantiles, dim=1).T
        target_density = conditional_log_density(target[start:start + chunk_size], mu, scale, weights)
        area_density = _sample_log_density(mu, scale, weights, samples, area_rng)
        contribution = torch.where(area_density[..., None] >= threshold[:, None],
                                   (-area_density).exp()[..., None], 0)
        thresholds.append(threshold.cpu())
        coverage.append((target_density[:, None] >= threshold).cpu())
        areas.append(contribution.mean(1).cpu())
        errors.append((contribution.std(1, unbiased=True) / math.sqrt(samples)).cpu())
    result = HDRResult(*(torch.cat(value).reshape(b, t, len(levels)) for value in (thresholds, coverage, areas, errors)))
    if any(not bool(torch.isfinite(value).all()) for value in (result.log_threshold, result.area_uv2, result.area_mc_standard_error_uv2)):
        raise ValueError("Nonfinite Monte Carlo HDR estimate")
    return result
