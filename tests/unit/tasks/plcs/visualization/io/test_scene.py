"""Boundary tests for required PLCS visualization timing metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.tasks.plcs.visualization.io.scene as scene_io
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract

PHYSICAL_V1_COURT = resolve_court_keypoint_contract("physical_v1")


def test_scene_bundle_requires_explicit_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = SimpleNamespace(num_cameras=1, meta={})
    monkeypatch.setattr(
        scene_io,
        "load_scene",
        lambda path, *, court_keypoint_contract: scene,
    )

    with pytest.raises(KeyError, match="fps"):
        scene_io.load_scene_bundle(
            Path("scene"),
            0,
            None,
            court_keypoint_contract=PHYSICAL_V1_COURT,
        )


def test_scene_bundle_rejects_nonpositive_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = SimpleNamespace(num_cameras=1, meta={"fps": 0.0})
    monkeypatch.setattr(
        scene_io,
        "load_scene",
        lambda path, *, court_keypoint_contract: scene,
    )

    with pytest.raises(ValueError, match="meta.fps must be positive"):
        scene_io.load_scene_bundle(
            Path("scene"),
            0,
            None,
            court_keypoint_contract=PHYSICAL_V1_COURT,
        )
