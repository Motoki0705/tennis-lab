"""Decode detector evidence without an observation threshold or trajectory gate."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
    BallModelIOError,
)
from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic


def decode_candidates(
    heatmaps: Tensor, *, config: BallCandidateConfig, subpixel_refine: bool,
) -> BallCandidates:
    """Decode (B,T,H,W) probability maps, retaining even very weak local peaks.

    The shared contrastive NMS defines peak existence. A uniform multi-cell
    map has no peaks when nms_kernel > 1; its dense evidence is still retained
    in BallPrediction. Patches use integer lattice centres, not resampling.
    All returned tensors are float32 / int64 / bool on CPU.
    """
    if heatmaps.ndim != 4 or min(heatmaps.shape) <= 0 or not heatmaps.is_floating_point():
        raise BallModelIOError("Candidate heatmaps must be nonempty floating (B,T,H,W).")
    if not bool(torch.isfinite(heatmaps).all()) or bool(((heatmaps < 0) | (heatmaps > 1)).any()):
        raise BallModelIOError("Candidate heatmaps must be finite probabilities in [0,1].")
    maps = heatmaps.detach().to(dtype=torch.float32)
    b, t, h, w = maps.shape
    coords, scores, valid = heatmaps_to_peaks(
        maps, threshold=0.0, nms_kernel=config.nms_kernel, max_peaks=config.max_candidates,
    )
    cells = (coords * coords.new_tensor([w - 1, h - 1])).round().long()
    if subpixel_refine:
        coords = refine_peaks_log_parabolic(maps, coords)
    # The common decoder caps K at H*W; our public contract always reserves K.
    padding = config.max_candidates - scores.shape[-1]
    coords = F.pad(coords, (0, 0, 0, padding))
    cells = F.pad(cells, (0, 0, 0, padding))
    scores = F.pad(scores, (0, padding))
    valid = F.pad(valid, (0, padding), value=False)
    coords = torch.where(valid[..., None], coords, 0.0)
    cells = torch.where(valid[..., None], cells, 0)

    radius = config.patch_size // 2
    offset = torch.arange(-radius, radius + 1, device=maps.device)
    x = cells[..., 0, None, None] + offset[None, :]
    y = cells[..., 1, None, None] + offset[:, None]
    patch_valid = valid[..., None, None] & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    # Advanced indexing broadcasts to B,T,K,P,P without materialising all patches.
    bi = torch.arange(b, device=maps.device)[:, None, None, None, None]
    ti = torch.arange(t, device=maps.device)[None, :, None, None, None]
    patches = maps[bi, ti, y.clamp(0, h - 1), x.clamp(0, w - 1)]
    patches = torch.where(patch_valid, patches, 0.0)
    return BallCandidates(
        coords=coords.cpu(), scores=scores.cpu(), valid=valid.cpu(), cells=cells.cpu(),
        patches=patches.cpu(), patch_valid=patch_valid.cpu(), config=config,
    )
