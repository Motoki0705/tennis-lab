"""Unit tests for BasePredictor's static/instance helper methods."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.base.inference.predictor import BasePredictor

pytestmark = pytest.mark.unit


class _Predictor(BasePredictor):
    """Minimal concrete predictor to access non-abstract helpers."""

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu", **kwargs):  # pragma: no cover
        raise NotImplementedError

    def predict(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BasePredictor()  # type: ignore[abstract]


def test_ensure_checkpoint_accepts_single_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("x")
    result = _Predictor._ensure_checkpoint(ckpt)
    assert result == [ckpt]
    # string form too
    assert _Predictor._ensure_checkpoint(str(ckpt)) == [ckpt]


def test_ensure_checkpoint_accepts_iterable(tmp_path: Path) -> None:
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_text("x")
    b.write_text("y")
    assert _Predictor._ensure_checkpoint([a, b]) == [a, b]


def test_ensure_checkpoint_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _Predictor._ensure_checkpoint(tmp_path / "nope.ckpt")


def test_ensure_checkpoint_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _Predictor._ensure_checkpoint([])


def test_resolve_device_cpu() -> None:
    dev = _Predictor._resolve_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


def test_to_device_preserves_none() -> None:
    a = torch.zeros(2)
    moved = _Predictor._to_device(torch.device("cpu"), a, None, torch.ones(3))
    assert moved[0].device.type == "cpu"
    assert moved[1] is None
    assert moved[2].device.type == "cpu"


def test_denormalize_coords_scales() -> None:
    p = _Predictor.__new__(_Predictor)  # bypass __init__ (none defined, but be safe)
    coords = torch.ones(2, 3)
    out = p._denormalize_coords(coords, [2.0, 3.0, 4.0])
    assert out.shape == (2, 3)
    assert out[0].tolist() == [2.0, 3.0, 4.0]
