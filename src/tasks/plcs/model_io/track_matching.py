"""Exact cosine clustering with camera exclusivity and transitive identities."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch import Tensor


@lru_cache(maxsize=32)
def _partition_constraints(count: int) -> tuple[tuple[tuple[int, int], ...], LinearConstraint]:
    pairs = tuple(combinations(range(count), 2))
    index = {pair: i for i, pair in enumerate(pairs)}
    rows, columns, coefficients = [], [], []
    row = 0
    for i, j, k in combinations(range(count), 3):
        edges = (index[i, j], index[i, k], index[j, k])
        for negative in range(3):
            for position, edge in enumerate(edges):
                rows.append(row)
                columns.append(edge)
                coefficients.append(-1. if position == negative else 1.)
            row += 1
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(row, len(pairs))).tocsc()
    return pairs, LinearConstraint(matrix, np.full(row, -np.inf), np.ones(row))


def match_track_embeddings(embeddings: Tensor, valid: Tensor, *, threshold: float) -> Tensor:
    """Return scene IDs (V,P). Unmatched valid people retain singleton identities.

    A cluster contains at most one track per camera; all cluster pair scores
    exceed threshold. The solver maximizes total cosine surplus over threshold.
    """
    if embeddings.ndim != 3 or valid.shape != embeddings.shape[:-1] or valid.dtype != torch.bool or not -1 < threshold < 1:
        raise ValueError("Expected (V,P,D) embeddings, boolean (V,P), threshold in (-1,1)")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Non-finite Re-ID embeddings")
    result = torch.full(valid.shape, -1, dtype=torch.int64)
    positions = valid.cpu().nonzero()
    count = len(positions)
    if not count:
        return result
    vectors = embeddings.detach().cpu()[valid.cpu()].double()
    if not torch.allclose(vectors.norm(dim=-1), torch.ones(count, dtype=torch.float64), atol=1e-4, rtol=1e-4):
        raise ValueError("Re-ID matching requires unit-normalized embeddings")
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    if count > 1:
        similarities = (vectors @ vectors.T).numpy()
        pairs, constraints = _partition_constraints(count)
        scores = np.asarray([similarities[i, j] - threshold for i, j in pairs])
        allowed = np.asarray([positions[i, 0] != positions[j, 0] and scores[e] > 0 for e, (i, j) in enumerate(pairs)], dtype=np.float64)
        solution = milp(-scores, integrality=np.ones(len(pairs)), bounds=Bounds(np.zeros(len(pairs)), allowed), constraints=constraints, options={"time_limit": 10.})
        if not solution.success or solution.x is None:
            raise RuntimeError(f"PLCS multi-view matching failed: {solution.message}")
        for (i, j), chosen in zip(pairs, solution.x, strict=True):
            if chosen > .5:
                parent[root(j)] = root(i)
    groups: dict[int, int] = {}
    for i, (view, slot) in enumerate(positions.tolist()):
        label = root(i)
        if label not in groups:
            groups[label] = len(groups)
        result[view, slot] = groups[label]
    return result
