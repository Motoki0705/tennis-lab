"""Unit tests for compile-aware staged batch calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

import src.tasks.ball_detection.training.staged_calibration as calibration_module
from src.tasks.ball_detection.training.staged_calibration import probe_batch_size_by_t
from src.tasks.base.model_io import BoundModelIO

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs"


def test_probe_compiles_model_with_shared_training_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_staged", overrides=["model=stunet"])
    model = nn.Linear(2, 2)
    model_io = BoundModelIO(model=model, adapter=cast(Any, object()))
    compile_calls: list[dict[str, object]] = []

    def fake_compile(callable_: Any, **kwargs: object) -> Any:
        compile_calls.append(dict(kwargs))
        return callable_

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        calibration_module,
        "build_ball_detection_pair",
        lambda _: model_io,
    )
    monkeypatch.setattr(calibration_module, "_fits", lambda *args: True)

    result = probe_batch_size_by_t(
        config,
        [1],
        device=torch.device("cpu"),
        token_budget=1,
        safety=1.0,
    )

    assert result == {1: 1}
    assert compile_calls == [
        {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
        }
    ]
