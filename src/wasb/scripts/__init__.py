"""WASB script entrypoints (Hydra-based).

Subpackages:
    - `src.wasb.scripts.train`
    - `src.wasb.scripts.generate_dataset`
    - `src.wasb.scripts.visualize`
"""

from __future__ import annotations

from src.wasb.scripts.generate_dataset import clip_sampling

__all__ = ["clip_sampling"]
