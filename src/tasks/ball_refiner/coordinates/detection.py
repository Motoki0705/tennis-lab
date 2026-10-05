"""Convert detector heatmaps to the coordinate-only input boundary."""

import torch
from torch import Tensor

from src.utils.data.heatmaps import heatmaps_to_argmax


def heatmaps_to_coordinates(heatmaps: Tensor, missing: Tensor, *, image_size_wh: tuple[int, int]) -> Tensor:
    """Return one source-pixel coordinate per frame; peak scores are discarded.

    Missingness is explicit, owned by the detector/caller. No refiner confidence
    threshold, court input, local probability patch, or candidate set is created.
    """
    if heatmaps.ndim != 4 or min(heatmaps.shape[-2:]) < 2:
        raise ValueError("Heatmaps must have shape B,T,H>=2,W>=2")
    if missing.shape != heatmaps.shape[:2] or missing.dtype != torch.bool or missing.device != heatmaps.device:
        raise ValueError("Missing mask must be a colocated B,T boolean tensor")
    if min(image_size_wh) < 2 or not heatmaps.is_floating_point() or not torch.isfinite(heatmaps).all():
        raise ValueError("Require finite floating heatmaps and valid source image size")
    coordinates, _ = heatmaps_to_argmax(heatmaps)
    scale = heatmaps.new_tensor(image_size_wh) - 1
    return torch.where(missing[..., None], 0, coordinates * scale)
