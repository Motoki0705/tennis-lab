from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.tennis.sim.assets import TennisAssetLibrary


def test_asset_library_samples_sequence() -> None:
    if not Path("data/raw/3dtennisds").exists():
        pytest.skip("3DTennisDS assets missing")
    pytest.importorskip("ezc3d")
    lib = TennisAssetLibrary("data/raw/3dtennisds", min_frames=10, max_files=1)
    sample = lib.sample_sequence(frames_total=5, target_fps=60.0, rng=random.Random(0))
    assert sample.joints.shape == (5, 17, 3)
    assert sample.racket.shape == (5, 3, 3)
    assert sample.pelvis.shape == (5, 3)
