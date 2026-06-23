"""Deterministic seeding helpers.

Consolidates the ``random.seed`` / ``np.random.seed`` / ``torch.manual_seed``
triad that was re-implemented as a private ``_seed_everything`` in multiple
dataset-generation scripts, plus the worker-aware per-sample RNG used by
stochastic dataset augmentation.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and Torch (CPU) RNGs.

    This is the lightweight, dependency-free seeding used by data-generation
    scripts. Training entry points that need full Lightning determinism should
    continue to use ``lightning.pytorch.seed_everything``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sample_rng(sample_idx: int) -> random.Random:
    """Return a deterministic per-sample, worker-aware :class:`random.Random`.

    Combines the dataloader worker's base seed (``torch.initial_seed()``) with
    the worker id and ``sample_idx`` so augmentation is reproducible for a given
    (seed, worker, sample) while staying decorrelated across workers.
    """
    worker_info = get_worker_info()
    base_seed = int(torch.initial_seed())
    if worker_info is not None:
        base_seed += int(worker_info.id) * 1_000_003
    return random.Random(base_seed + int(sample_idx))


__all__ = ["seed_everything", "make_sample_rng"]
