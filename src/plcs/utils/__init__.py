"""Utility functions for PLCS.

Note:
    For geometry constants (HALF_DOUBLES_WIDTH, HALF_LENGTH, NUM_COURT_KP, etc.),
    import directly from src.utils.geometry.
"""

from src.plcs.utils.config import load_config, merge_configs

__all__ = [
    "load_config",
    "merge_configs",
]
