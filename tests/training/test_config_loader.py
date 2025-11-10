from __future__ import annotations

import os

import pytest
from omegaconf import DictConfig

from src.training.utils.config import load_cfg


def test_load_cfg_merges_includes_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFG_DEBUG_MINIMAL", "1")
    cfg = load_cfg("configs/scene_model.yaml", overrides=["dataset.window.size=5"])
    assert isinstance(cfg, DictConfig)
    assert cfg.dataset.window.size == 5
    assert cfg.debug.minimal is True
    assert "training" in cfg and "model" in cfg
    monkeypatch.delenv("CFG_DEBUG_MINIMAL", raising=False)
