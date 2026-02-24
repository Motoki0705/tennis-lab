"""Codec for converting metadata dictionaries to BLCS-specific types.

This module provides lightweight conversion functions from generic metadata
dictionaries (decoded by src.utils.data.npz_meta.decode_meta) into strongly-typed
BLCS metadata classes (BLCSSceneMeta, RallySceneMeta).

Note: This module does not import torch, lightning, or heavy dataset classes,
keeping it lightweight to avoid circular imports.
"""

from __future__ import annotations

from typing import Any

from src.tasks.blcs.data.types import BLCSSceneMeta, RallySceneMeta


def parse_blcs_scene_meta(meta: dict[str, Any]) -> BLCSSceneMeta | None:
    """Convert a metadata dictionary into BLCSSceneMeta when possible.

    Args:
        meta: Decoded metadata dictionary (from decode_meta).

    Returns:
        BLCSSceneMeta instance if conversion succeeds, None otherwise.
    """
    if not isinstance(meta, dict):
        return None
    try:
        return BLCSSceneMeta.from_dict(meta)
    except (KeyError, TypeError, ValueError):
        return None


def parse_rally_scene_meta(meta: dict[str, Any]) -> RallySceneMeta | None:
    """Convert a metadata dictionary into RallySceneMeta when possible.

    Args:
        meta: Decoded metadata dictionary (from decode_meta).

    Returns:
        RallySceneMeta instance if conversion succeeds, None otherwise.
    """
    if not isinstance(meta, dict):
        return None
    try:
        return RallySceneMeta.from_dict(meta)
    except (KeyError, TypeError, ValueError):
        return None
