"""The 32 pose-conditioned variants and four distinct pose-free variants."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from .config import MDDPoseConfig


@dataclass(frozen=True)
class CoordinateVariant:
    name: str
    config: MDDPoseConfig


def coordinate_variants(base: MDDPoseConfig) -> tuple[CoordinateVariant, ...]:
    variants = []
    compressions = ("conv2d", "average", "unshuffle", "haar")
    for compression, pooling, readout in product(
        compressions, ("deepsets", "attention", "hierarchical", "gnn"), ("query", "pose"),
    ):
        variants.append(CoordinateVariant(
            f"{compression}-{pooling}-{readout}",
            replace(base, compression=compression, pose_pooling=pooling, readout=readout),
        ))
    for compression in compressions:
        variants.append(CoordinateVariant(
            f"{compression}-query_only", replace(base, compression=compression, pose_pooling=None, readout="query_only"),
        ))
    return tuple(variants)
