"""Tests for the strict court-alignment model-only checkpoint contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from src.tasks.court_alignment.models.checkpoint import (
    load_court_alignment_model_checkpoint,
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


def _payload(model: nn.Module) -> dict[str, Any]:
    return {
        "epoch": 47,
        "global_step": 12_288,
        "optimizer_states": [{"ignored": True}],
        "lr_schedulers": [{"ignored": True}],
        "state_dict": {
            f"model.{key}": value.clone() for key, value in model.state_dict().items()
        },
    }


def _save(path: Path, payload: object) -> Path:
    torch.save(payload, path)
    return path.resolve()


def test_loads_historical_model_prefix_and_ignores_training_state(
    tmp_path: Path,
) -> None:
    source = _model()
    target = _model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
        for parameter in target.parameters():
            parameter.zero_()
    checkpoint_path = _save(tmp_path / "historical.ckpt", _payload(source))

    metadata = load_court_alignment_model_checkpoint(target, checkpoint_path)

    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    assert metadata["epoch"] == 47
    assert metadata["global_step"] == 12_288
    assert metadata["state_dict_key_count"] == len(source.state_dict())


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda _: [], TypeError, "root must be a mapping"),
        (
            lambda payload: {**payload, "state_dict": []},
            TypeError,
            "state_dict.*mapping",
        ),
        (
            lambda payload: {
                **payload,
                "state_dict": {
                    **payload["state_dict"],
                    "loss_fn.weight": torch.ones(1),
                },
            },
            ValueError,
            "mixed or invalid prefixes",
        ),
        (
            lambda payload: {
                **payload,
                "state_dict": dict(list(payload["state_dict"].items())[1:]),
            },
            RuntimeError,
            "missing=",
        ),
        (
            lambda payload: {
                **payload,
                "state_dict": {
                    **payload["state_dict"],
                    "model.extra": torch.ones(1),
                },
            },
            RuntimeError,
            "unexpected=",
        ),
        (
            lambda payload: {
                **payload,
                "state_dict": {
                    **payload["state_dict"],
                    "model.0.weight": torch.ones(9, 9),
                },
            },
            RuntimeError,
            "shape_mismatches=",
        ),
    ],
)
def test_rejects_invalid_or_inexact_historical_payloads(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    error_type: type[Exception],
    message: str,
) -> None:
    checkpoint_path = _save(
        tmp_path / "invalid.ckpt",
        mutation(_payload(_model())),
    )

    with pytest.raises(error_type, match=message):
        load_court_alignment_model_checkpoint(_model(), checkpoint_path)
