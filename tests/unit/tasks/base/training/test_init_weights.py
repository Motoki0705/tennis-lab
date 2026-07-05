"""Unit tests for weight-only fine-tune loading in BaseTrainingRunner."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.tasks.base.training.runner import BaseTrainingRunner


class _TinyModule(pl.LightningModule):
    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 2)


def _runner() -> BaseTrainingRunner:
    return BaseTrainingRunner()


def _cfg(init_weights: str | None, resume: str | None = None) -> OmegaConf:
    return OmegaConf.create({"run": {"init_weights": init_weights, "resume": resume}})


def test_noop_when_unset() -> None:
    module = _TinyModule()
    before = module.linear.weight.clone()
    _runner().maybe_load_init_weights(_cfg(None), module)
    torch.testing.assert_close(module.linear.weight, before)


def test_loads_weights_from_checkpoint(tmp_path) -> None:
    source = _TinyModule()
    ckpt = tmp_path / "src.ckpt"
    torch.save({"state_dict": source.state_dict()}, ckpt)

    target = _TinyModule()
    assert not torch.allclose(target.linear.weight, source.linear.weight)
    _runner().maybe_load_init_weights(_cfg(str(ckpt)), target)
    torch.testing.assert_close(target.linear.weight, source.linear.weight)


def test_mutually_exclusive_with_resume(tmp_path) -> None:
    ckpt = tmp_path / "src.ckpt"
    torch.save({"state_dict": _TinyModule().state_dict()}, ckpt)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _runner().maybe_load_init_weights(_cfg(str(ckpt), resume=str(ckpt)), _TinyModule())


def test_raises_when_checkpoint_does_not_match(tmp_path) -> None:
    # Checkpoint keys are unrelated to the model -> nothing loads.
    ckpt = tmp_path / "src.ckpt"
    torch.save({"state_dict": {"unrelated.weight": torch.zeros(3)}}, ckpt)
    with pytest.raises(RuntimeError, match="does not match"):
        _runner().maybe_load_init_weights(_cfg(str(ckpt)), _TinyModule())
