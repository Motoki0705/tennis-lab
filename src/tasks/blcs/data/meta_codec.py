"""Codec for converting decoded BLCS metadata dictionaries into typed objects."""

from __future__ import annotations

from typing import Any

from src.tasks.blcs.data.types import BLCSSceneMeta


def parse_blcs_scene_meta(meta: dict[str, Any]) -> BLCSSceneMeta | None:
    """Convert a metadata dictionary into rally-native BLCSSceneMeta."""
    if not isinstance(meta, dict):
        return None
    try:
        return BLCSSceneMeta.from_dict(meta)
    except (KeyError, TypeError, ValueError):
        return None
