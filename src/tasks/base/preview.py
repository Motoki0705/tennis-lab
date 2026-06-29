"""Shared helpers for Hydra-driven dataset preview scripts.

These parse the common ``cfg.preview`` / ``cfg.data.split`` config conventions
used by the per-task ``preview_heatmaps`` / ``preview_augmentation`` scripts.
They depend only on the OmegaConf config shape, not on any task's domain types,
so the ball- and court-detection scripts share a single implementation.
"""

from __future__ import annotations

from omegaconf import DictConfig

__all__ = ["resolve_sample_indices", "resolve_split_file"]


def resolve_split_file(cfg: DictConfig, split_name: str) -> str:
    """Return the split-file path for ``split_name`` from ``cfg.data.split``."""
    split_cfg = cfg.data.split
    key = f"{split_name}_file"
    if key not in split_cfg:
        available = ", ".join(sorted(split_cfg.keys()))
        raise ValueError(f"Unknown preview.split={split_name!r}. Available: {available}")
    return str(split_cfg[key])


def resolve_sample_indices(
    cfg: DictConfig,
    dataset_size: int,
    *,
    min_samples: int = 0,
) -> list[int]:
    """Return validated preview sample indices.

    Uses ``cfg.preview.sample_indices`` when non-empty, otherwise the first
    ``max(cfg.preview.max_samples, min_samples)`` indices (clamped to
    ``dataset_size``). ``min_samples`` defaults to ``0`` (no floor); pass ``1``
    to guarantee at least one sample. Raises ``IndexError`` if any resolved
    index is out of range.
    """
    explicit = [int(value) for value in cfg.preview.sample_indices]
    if explicit:
        sample_indices = explicit
    else:
        count = max(int(cfg.preview.max_samples), min_samples)
        sample_indices = list(range(min(count, dataset_size)))
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= dataset_size:
            raise IndexError(
                f"preview sample_index={sample_index} is out of range for "
                f"dataset size {dataset_size}."
            )
    return sample_indices
