from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.wasb.inference.trajectory_completion import (
    BiLSTMCompleter,
    IterativeRefinementCompleter,
    PhysicsInterpolator,
    TransformerCompleter,
    build_completer,
)
from src.wasb.training import TrajectoryLightningModule


def test_build_completer_physics() -> None:
    completer = build_completer(method="physics")
    assert isinstance(completer, PhysicsInterpolator)


@pytest.mark.parametrize(
    ("cls", "method"),
    [
        (TransformerCompleter, "transformer"),
        (BiLSTMCompleter, "bilstm"),
        (IterativeRefinementCompleter, "refiner"),
    ],
)
def test_build_completer_checkpoint_uses_lightning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cls,
    method: str,
) -> None:
    called: dict[str, object] = {}

    def fake_load_from_checkpoint(class_, checkpoint_path: str, map_location=None, **_kwargs):
        called["checkpoint_path"] = checkpoint_path
        called["map_location"] = map_location
        model = torch.nn.Identity()
        return SimpleNamespace(model=model, num_steps=2)

    monkeypatch.setattr(
        TrajectoryLightningModule,
        "load_from_checkpoint",
        classmethod(fake_load_from_checkpoint),
    )

    ckpt = tmp_path / "dummy.ckpt"
    ckpt.write_bytes(b"")

    completer = build_completer(method=method, checkpoint_path=ckpt, device="cpu")
    assert isinstance(completer, cls)
    assert called["checkpoint_path"] == str(ckpt)


def test_transformer_complete_fills_missing() -> None:
    model = torch.nn.Identity()
    completer = TransformerCompleter(model=model, device="cpu", score_threshold=0.5)

    xy = np.array([[10.0, 20.0], [0.0, 0.0], [30.0, 40.0]], dtype=np.float32)
    vis = np.array([True, False, True])
    score = np.array([0.9, 0.0, 0.9], dtype=np.float32)

    result = completer.complete(xy, vis, score)
    assert result.xy.shape == (3, 2)
    assert result.visibility.tolist() == [1, 2, 1]
