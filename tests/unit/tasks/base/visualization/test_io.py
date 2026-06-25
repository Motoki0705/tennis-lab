"""Unit tests for visualization scene-IO camera resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.tasks.base.visualization.io import BaseSceneBundle, resolve_cameras

pytestmark = pytest.mark.unit


def _all_cams() -> list[int]:
    return [0, 1, 2]


def test_resolve_all_expands_via_callback() -> None:
    assert resolve_cameras(3, 0, "all", _all_cams) == [0, 1, 2]


def test_resolve_explicit_list() -> None:
    assert resolve_cameras(3, 0, [1, 2], _all_cams) == [1, 2]


def test_resolve_none_falls_back_to_single_camera() -> None:
    assert resolve_cameras(3, 2, None, _all_cams) == [2]


def test_resolve_empty_list_falls_back_to_single_camera() -> None:
    # an empty list is falsy -> falls through to [camera]
    assert resolve_cameras(3, 1, [], _all_cams) == [1]


def test_out_of_range_high_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_cameras(3, 0, [3], _all_cams)


def test_out_of_range_negative_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_cameras(3, 0, [-1], _all_cams)


def test_fallback_camera_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_cameras(2, 5, None, _all_cams)


def test_all_returning_empty_raises() -> None:
    with pytest.raises(ValueError, match="No cameras selected"):
        resolve_cameras(3, 0, "all", list)  # list() -> []


def test_scene_bundle_dataclass_frozen() -> None:
    bundle = BaseSceneBundle(cameras=[0, 1], fps=30.0)
    assert bundle.cameras == [0, 1]
    assert bundle.fps == 30.0
    with pytest.raises(FrozenInstanceError):
        bundle.fps = 60.0  # type: ignore[misc]
