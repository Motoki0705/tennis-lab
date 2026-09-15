"""Resume safety and order-independent extraction seeds."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.plcs.motion.reproducibility import (
    content_digest,
    publish_run,
    seed_motion,
)


def test_seed_is_stable_across_resume_and_clip_order() -> None:
    before = torch.are_deterministic_algorithms_enabled()
    try:
        seed_motion(42, "clip:a", True)
        expected = (random.random(), np.random.random(), torch.rand(1).item())
        seed_motion(42, "clip:b", True)
        seed_motion(42, "clip:a", True)
        assert (random.random(), np.random.random(), torch.rand(1).item()) == expected
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)


def test_changed_reproduction_identity_never_overwrites_record(tmp_path: Path) -> None:
    original = {
        "sha256": content_digest({"weights": "first"}),
        "config": {"run": {"seed": 42}},
    }
    publish_run(tmp_path, original)
    assert (tmp_path / "config.yaml").is_file()
    path = tmp_path / "reproducibility.json"
    old = path.read_bytes()
    with pytest.raises(RuntimeError, match="changed"):
        publish_run(tmp_path, {"sha256": content_digest({"weights": "second"})})
    assert path.read_bytes() == old
    publish_run(tmp_path, original)


def test_legacy_output_requires_new_directory(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no reproducibility identity"):
        publish_run(tmp_path, {"sha256": "new"})
