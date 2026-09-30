"""Explicit native visible-part descriptors; no flattened embedding proxy.

KPR's mean Euclidean distance over common visible parts is used. A pair without
common visible parts has no appearance evidence (valid=False), not a fake match.
See kpr_vendor/NOTICE.md for encoder provenance and the run-9 protocol for the
explicit 1-distance adapter to existing tracker/association score scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NativeParts:
    embeddings: NDArray[np.float32]  # N,P,E; normalized independently per visible part
    visible: NDArray[np.bool_]  # N,P

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3 or min(self.embeddings.shape[1:]) < 1 \
                or self.visible.shape != self.embeddings.shape[:2] \
                or self.embeddings.dtype != np.float32 or self.visible.dtype != np.bool_ \
                or not np.isfinite(self.embeddings).all():
            raise ValueError('Native part embeddings require finite float32 (N,P,E) and boolean visibility (N,P)')
        if not np.allclose(np.linalg.norm(self.embeddings, axis=-1)[self.visible], 1, atol=1e-4):
            raise ValueError('Visible part embeddings must have unit norm')

    def take(self, rows: NDArray[np.int64]) -> NativeParts:
        return NativeParts(self.embeddings[rows], self.visible[rows])


def part_distance(left: NativeParts, right: NativeParts) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if left.embeddings.shape[1:] != right.embeddings.shape[1:]:
        raise ValueError('Native part dimensions differ')
    dot = np.einsum('npe,mpe->nmp', left.embeddings.astype(np.float64), right.embeddings.astype(np.float64))
    squared = (np.square(left.embeddings.astype(np.float64)).sum(-1)[:, None]
               + np.square(right.embeddings.astype(np.float64)).sum(-1)[None] - 2 * dot)
    common = left.visible[:, None] & right.visible[None]
    count = common.sum(-1)
    distance = (np.sqrt(np.maximum(squared, 0)) * common).sum(-1) / np.maximum(count, 1)
    return distance, count > 0


def mean_parts(parts: NativeParts) -> NativeParts:
    count = parts.visible.sum(0)
    vectors = (parts.embeddings.astype(np.float64) * parts.visible[..., None]).sum(0)
    norm = np.linalg.norm(vectors, axis=-1)
    visible = count > 0
    if (visible & (norm <= 1e-12)).any():
        raise ValueError('Visible part samples average to zero')
    vectors /= np.maximum(norm[:, None], 1e-12)
    return NativeParts(vectors[None].astype(np.float32), visible[None])


def update_parts(previous: NativeParts, current: NativeParts, alpha: float) -> NativeParts:
    if previous.embeddings.shape != current.embeddings.shape or len(current.embeddings) != 1 or not 0 <= alpha <= 1:
        raise ValueError('Part EMA needs compatible single observations and alpha in [0,1]')
    vectors = previous.embeddings.copy()
    fresh = current.visible & ~previous.visible
    common = current.visible & previous.visible
    vectors[fresh] = current.embeddings[fresh]
    vectors[common] = alpha * vectors[common] + (1 - alpha) * current.embeddings[common]
    norm = np.linalg.norm(vectors, axis=-1)
    visible = previous.visible | current.visible
    if (visible & (norm <= 1e-12)).any():
        raise ValueError('Part EMA cancelled to zero')
    vectors /= np.maximum(norm[..., None], 1e-12)
    return NativeParts(vectors, visible)
