"""Identity renaming across overlapping windows of the same observations."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def stitch_overlap_ids(
    previous: Tensor, current: Tensor, *, next_identity: int
) -> tuple[Tensor, int]:
    """Map clip IDs once from identical overlapping (view,time,local-slot) observations.

    Inputs cover the same overlap and contain -1 for missing/non-target objects.
    Returned mapping[k] is a scene ID; callers apply it to the complete new window.
    Unmatched clip IDs receive new scene IDs. A slot must denote the same raw
    observation at both sides of the overlap, not an independently reordered row.
    """
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("Overlap IDs must have the same (V,T,P) shape")
    local_ids = current[current >= 0].unique()
    old_ids = previous[previous >= 0].unique()
    width = int(local_ids.max().item()) + 1 if len(local_ids) else 0
    mapping = current.new_full((width,), -1)
    if len(local_ids) and len(old_ids):
        counts = torch.stack(
            [
                torch.stack(
                    [((current == local) & (previous == old)).sum() for old in old_ids]
                )
                for local in local_ids
            ]
        )
        rows, columns = linear_sum_assignment(-counts.cpu().numpy())
        for row, column in zip(rows, columns, strict=True):
            if counts[int(row), int(column)] > 0:
                mapping[local_ids[int(row)]] = old_ids[int(column)]
    for local in local_ids:
        if mapping[local] < 0:
            mapping[local] = next_identity
            next_identity += 1
    return mapping, next_identity
