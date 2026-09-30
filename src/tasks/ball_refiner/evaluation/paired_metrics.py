"""Frame-aligned, explicitly uncalibrated distribution comparison metrics."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.data.targets import ClipTargets, TargetReason
from src.tasks.ball_refiner.evaluation.hdr import highest_density_regions
from src.tasks.ball_refiner.refiner_2d.distribution import (
    BallGMM2D,
    conditional_log_density,
)

Array: TypeAlias = NDArray[np.generic]


def spatial_reference(target: ClipTargets) -> NDArray[np.bool_]:
    """Estimated labels are diagnostics only, never observed training labels."""
    return np.isin(target.reason, [TargetReason.OBSERVED, TargetReason.OCCLUSION_ESTIMATED, TargetReason.INTERPOLATED])


def gmm_rows(
    prediction: BallGMM2D, target: ClipTargets, size_wh: tuple[int, int], *,
    levels: tuple[float, ...], samples: int, seed: int, chunk_size: int,
    clip_id: str, condition: str, device: torch.device,
) -> dict[str, Array]:
    located = spatial_reference(target)
    n = len(target.pts)
    output: dict[str, Array] = {key: np.full(n, np.nan) for key in ('error_px', 'nll_uv', 'nll_px', 'presence_nll')}
    output['coverage'] = np.full((n, len(levels)), np.nan)
    output['area_px2'] = np.full((n, len(levels)), np.nan)
    if located.any():
        selected = BallGMM2D(**{f.name: getattr(prediction, f.name)[:, located].to(device) for f in fields(prediction)})
        uv = torch.from_numpy(target.uv[located])[None].to(device)
        rng_seed = int.from_bytes(hashlib.sha256(f'{seed}:{clip_id}:{condition}'.encode()).digest()[:8], 'little') % (2**63 - 1)
        hdr = highest_density_regions(selected, uv, levels=levels, samples=samples, seed=rng_seed, chunk_size=chunk_size)
        scale = np.array([size_wh[0] - 1, size_wh[1] - 1], np.float64)
        nll = -conditional_log_density(uv, selected.means, selected.scale_tril, selected.mixture_logits)
        means = selected.means[0, torch.arange(int(located.sum()), device=device), selected.mixture_logits[0].argmax(-1)]
        output['error_px'][located] = np.linalg.norm((means.cpu().numpy() - target.uv[located]) * scale, axis=-1)
        output['nll_uv'][located] = nll[0].cpu().numpy()
        output['nll_px'][located] = output['nll_uv'][located] + np.log(scale).sum()
        output['coverage'][located] = hdr.covered[0].numpy()
        output['area_px2'][located] = hdr.area_uv2[0].numpy() * scale.prod()
    known = target.presence_valid
    output['presence_nll'][known] = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.presence_logits[0, known], torch.from_numpy(target.presence[known].astype(np.float32)), reduction='none',
    ).numpy()
    return output


def heatmap_rows(
    heatmaps: NDArray[np.float32], target: ClipTargets, size_wh: tuple[int, int],
    endpoint_factor: tuple[float, float], *, levels: tuple[float, ...], uniform_weight: float,
) -> dict[str, Array]:
    """Integrate a piecewise-constant native-grid density over source UV [0,1]².

    Internal cell edges bisect the endpoint-aligned native grid. Boundary cells
    reach the source image edges. Equal-density HDR ties include all tied cells,
    so a uniform/zero heatmap has full-image coverage at every nominal mass.
    The explicit uniform component also defines a completely zero heatmap.
    """
    if heatmaps.ndim != 3 or heatmaps.shape[0] != len(target.pts) or min(heatmaps.shape[1:]) < 2:
        raise ValueError('Heatmaps must have shape (frames,H>=2,W>=2)')
    if not np.isfinite(heatmaps).all() or ((heatmaps < 0) | (heatmaps > 1)).any():
        raise ValueError('Expected finite native sigmoid heatmaps')
    if not 0 < uniform_weight < 1 or not levels or not all(0 < x < 1 for x in levels):
        raise ValueError('Explicit uniform weight and HDR levels must lie in (0,1)')
    n, height, width = heatmaps.shape
    edges = []
    for count, factor in zip((width, height), endpoint_factor, strict=True):
        if not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError('Stored/source endpoint mapping must lie in (0,1]')
        centers = np.linspace(0, factor, count)
        edges.append(np.r_[0, (centers[1:] + centers[:-1]) / 2, 1])
    areas = np.outer(np.diff(edges[1]), np.diff(edges[0])).reshape(-1)
    located = spatial_reference(target)
    out: dict[str, Array] = {key: np.full(n, np.nan) for key in ('nll_uv', 'nll_px')}
    out['coverage'] = np.full((n, len(levels)), np.nan)
    out['area_px2'] = np.full((n, len(levels)), np.nan)
    scale = (size_wh[0] - 1) * (size_wh[1] - 1)
    for frame in np.flatnonzero(located):
        uv = target.uv[frame].astype(np.float64)
        if not np.isfinite(uv).all() or ((uv < 0) | (uv > 1)).any():
            raise ValueError('Spatial reference must be finite and inside source UV')
        values = heatmaps[frame].reshape(-1).astype(np.float64)
        # The zero-map case explicitly declares complete spatial ignorance.
        integral = float(values @ areas)
        density = (1 - uniform_weight) * values / integral + uniform_weight if integral > 0 else np.ones_like(values)
        x, y = [min(np.searchsorted(edge, value, side='right') - 1, len(edge) - 2)
                for edge, value in zip(edges, uv, strict=True)]
        actual = density[y * width + x]
        out['nll_uv'][frame] = -np.log(actual)
        out['nll_px'][frame] = -np.log(actual) + np.log(scale)
        order = np.argsort(-density, kind='stable')
        mass = np.cumsum((density * areas)[order])
        for index, level in enumerate(levels):
            threshold = density[order[min(int(np.searchsorted(mass, level)), len(order) - 1)]]
            out['coverage'][frame, index] = actual >= threshold
            out['area_px2'][frame, index] = areas[density >= threshold].sum() * scale
    return out


def strata(target: ClipTargets, gap: NDArray[np.bool_], condition: str) -> dict[str, NDArray[np.bool_]]:
    if condition not in ('observed', 'evidence_gap'):
        raise ValueError('Unknown evidence condition')
    active = gap if condition == 'evidence_gap' else np.ones(len(gap), np.bool_)
    return {name: active & (target.reason == reason) for name, reason in (
        ('observed', TargetReason.OBSERVED), ('occlusion_estimated_reference', TargetReason.OCCLUSION_ESTIMATED),
        ('interpolated_reference', TargetReason.INTERPOLATED), ('absent', TargetReason.OUT_OF_FRAME),
        ('unresolved', TargetReason.UNRESOLVED), ('no_instance_unknown', TargetReason.NO_INSTANCE),
        ('unreviewed_unknown', TargetReason.UNREVIEWED), ('multiple_instances_unknown', TargetReason.MULTIPLE_INSTANCES),
    )}


def summarize(rows: list[dict[str, Array]], levels: tuple[float, ...]) -> dict[str, Any]:
    """Every quantity carries its own denominator; unavailable values remain null."""
    joined = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    result: dict[str, Any] = {'frames': len(joined['error_px'])}
    for name in ('error_px', 'nll_uv', 'nll_px', 'presence_nll'):
        values = joined[name][np.isfinite(joined[name])]
        result[f'{name}_frames'] = len(values)
        result[f'mean_{name}'] = float(values.mean()) if len(values) else None
        if name == 'error_px':
            for key, q in [('median_error_px', .5), ('p95_error_px', .95)]:
                result[key] = float(np.quantile(values, q)) if len(values) else None
            result['recall_20px'] = float((values <= 20).mean()) if len(values) else None
    for column in ('coverage', 'area_px2'):
        for i, level in enumerate(levels):
            values = joined[column][:, i]
            values = values[np.isfinite(values)]
            result[f'{column}_{level:g}_frames'] = len(values)
            result[f'{column}_{level:g}'] = float(values.mean()) if len(values) else None
    return result
