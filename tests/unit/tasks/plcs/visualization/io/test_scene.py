"""Boundary tests for required PLCS visualization timing metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.tasks.plcs.visualization.io.scene as scene_io


def test_scene_bundle_requires_explicit_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = SimpleNamespace(num_cameras=1, meta={})
    monkeypatch.setattr(scene_io, "load_scene", lambda path: scene)

    with pytest.raises(KeyError, match="fps"):
        scene_io.load_scene_bundle(Path("scene"), 0, None)


def test_scene_bundle_rejects_nonpositive_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = SimpleNamespace(num_cameras=1, meta={"fps": 0.0})
    monkeypatch.setattr(scene_io, "load_scene", lambda path: scene)

    with pytest.raises(ValueError, match="meta.fps must be positive"):
        scene_io.load_scene_bundle(Path("scene"), 0, None)
