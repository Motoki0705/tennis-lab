"""Unit tests for BasePredictor's static/instance helper methods."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.base.inference.predictor import BasePredictor
from src.utils.configuration import PathResolver, RuntimePathRoots

pytestmark = pytest.mark.unit


class _Predictor(BasePredictor):
    """Minimal concrete predictor to access non-abstract helpers."""

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu", **kwargs):  # pragma: no cover
        raise NotImplementedError

    def predict(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _resolver(checkpoint_root: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots.from_mapping(
            {
                "project_root": str(checkpoint_root),
                "data_root": "data",
                "checkpoint_root": str(checkpoint_root),
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "external",
            },
            repository_root=checkpoint_root,
        )
    )

def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BasePredictor()  # type: ignore[abstract]


def test_removed_checkpoint_compatibility_loader_is_unavailable() -> None:
    assert "_load_single_lightning_checkpoint" not in BasePredictor.__dict__
    with pytest.raises(AttributeError, match="_load_single_lightning_checkpoint"):
        _ = _Predictor._load_single_lightning_checkpoint  # type: ignore[attr-defined]


def test_ensure_checkpoint_accepts_single_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("x")
    resolver = _resolver(tmp_path)
    result = _Predictor._ensure_checkpoint(ckpt, resolver=resolver)
    assert result == [ckpt]
    # string form too
    assert _Predictor._ensure_checkpoint(str(ckpt), resolver=resolver) == [ckpt]


def test_ensure_checkpoint_accepts_iterable(tmp_path: Path) -> None:
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_text("x")
    b.write_text("y")
    assert _Predictor._ensure_checkpoint([a, b], resolver=_resolver(tmp_path)) == [a, b]


def test_ensure_checkpoint_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _Predictor._ensure_checkpoint(
            tmp_path / "nope.ckpt", resolver=_resolver(tmp_path)
        )


def test_ensure_checkpoint_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _Predictor._ensure_checkpoint([], resolver=_resolver(tmp_path))


def test_to_device_preserves_none() -> None:
    a = torch.zeros(2)
    moved = _Predictor._to_device(torch.device("cpu"), a, None, torch.ones(3))
    assert isinstance(moved[0], torch.Tensor)
    assert moved[0].device.type == "cpu"
    assert moved[1] is None
    assert isinstance(moved[2], torch.Tensor)
    assert moved[2].device.type == "cpu"


def test_denormalize_coords_scales() -> None:
    p = _Predictor.__new__(_Predictor)  # bypass __init__ (none defined, but be safe)
    coords = torch.ones(2, 3)
    out = p._denormalize_coords(coords, [2.0, 3.0, 4.0])
    assert out.shape == (2, 3)
    assert out[0].tolist() == [2.0, 3.0, 4.0]
