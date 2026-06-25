"""Unit tests for BaseLightningModule pure helpers.

Covers ``_concat_padded`` (time-axis padding), config-driven step estimation,
optimizer/scheduler construction, and test-prediction persistence. Heavy
training (a real ``training_step`` over a model) is deferred to the integration
smoke suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from src.tasks.base.training.lightning_module import (
    BaseLightningModule,
    _concat_padded,
)

pytestmark = pytest.mark.unit


class _TinyModule(BaseLightningModule):
    """Concrete module with a single trainable parameter for optimizer tests."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.lin = nn.Linear(2, 2)


# ---------------------------------------------------------------------------
# _concat_padded
# ---------------------------------------------------------------------------


def test_concat_padded_single_chunk_passthrough() -> None:
    chunk = np.ones((2, 3))
    assert _concat_padded([chunk]) is chunk


def test_concat_padded_pads_time_axis() -> None:
    a = np.ones((2, 3, 4))  # T=3
    b = np.ones((1, 5, 4))  # T=5
    out = _concat_padded([a, b])
    assert out.shape == (3, 5, 4)  # batch concat, T padded to 5
    # padded region of `a` (time steps 3,4) is zero
    assert (out[0:2, 3:, :] == 0).all()
    # original content preserved
    assert (out[0:2, :3, :] == 1).all()


def test_concat_padded_low_dim_arrays_concatenated() -> None:
    a = np.array([1.0, 2.0])  # ndim 1
    b = np.array([3.0])
    out = _concat_padded([a, b])
    assert out.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# _estimate_total_steps
# ---------------------------------------------------------------------------


def test_estimate_total_steps_from_steps_per_epoch_attr() -> None:
    m = _TinyModule({"training": {"max_epochs": 4}})
    m.steps_per_epoch = 50
    assert m._estimate_total_steps() == 200


def test_estimate_total_steps_from_config_steps_per_epoch() -> None:
    m = _TinyModule({"training": {"max_epochs": 3, "steps_per_epoch": 10}})
    assert m._estimate_total_steps() == 30


def test_estimate_total_steps_from_num_samples() -> None:
    m = _TinyModule(
        {
            "training": {"max_epochs": 2},
            "data": {"num_samples_per_epoch": 100, "batch_size": 10},
        }
    )
    # steps_per_epoch = 100 // 10 = 10 -> * 2 epochs
    assert m._estimate_total_steps() == 20


def test_estimate_total_steps_default_fallback() -> None:
    m = _TinyModule({"training": {"max_epochs": 1}})
    # default num_samples=10000, batch_size=64 -> 156 steps * 1
    assert m._estimate_total_steps() == (10000 // 64) * 1


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_step_interval_no_warmup() -> None:
    m = _TinyModule({"training": {"max_epochs": 2, "steps_per_epoch": 5}})
    cfg = m.configure_optimizers()
    assert isinstance(cfg["optimizer"], AdamW)
    assert cfg["lr_scheduler"]["interval"] == "step"
    assert isinstance(cfg["lr_scheduler"]["scheduler"], CosineAnnealingLR)


def test_configure_optimizers_epoch_warmup_uses_sequential() -> None:
    m = _TinyModule({"training": {"max_epochs": 10, "warmup_epochs": 3}})
    cfg = m.configure_optimizers()
    assert cfg["lr_scheduler"]["interval"] == "epoch"
    assert isinstance(cfg["lr_scheduler"]["scheduler"], SequentialLR)


def test_configure_optimizers_respects_lr_and_betas() -> None:
    m = _TinyModule(
        {
            "training": {
                "max_epochs": 1,
                "learning_rate": 5e-4,
                "weight_decay": 0.01,
                "steps_per_epoch": 1,
                "optimizer": {"betas": [0.8, 0.95]},
            }
        }
    )
    opt = m.configure_optimizers()["optimizer"]
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(5e-4)
    assert group["weight_decay"] == pytest.approx(0.01)
    assert group["betas"] == (0.8, 0.95)


def test_optimizer_betas_none_when_unset() -> None:
    m = _TinyModule({"training": {"max_epochs": 1}})
    assert m.optimizer_betas is None


# ---------------------------------------------------------------------------
# test_prediction_payload defaults / collection / saving
# ---------------------------------------------------------------------------


def test_default_payload_empty_and_collect_noop() -> None:
    m = _TinyModule({})
    assert m.test_prediction_payload(batch=None, result={}) == {}
    # collecting an empty payload must not create a buffer
    m.collect_test_predictions(batch=None, result={})
    assert not getattr(m, "_test_pred_arrays", None)


def test_save_test_predictions_writes_npz(tmp_path: Path, monkeypatch) -> None:
    class _PredModule(_TinyModule):
        def test_prediction_payload(self, batch, result):
            return {"position": result["position"]}

    m = _PredModule({})
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    m._reset_test_prediction_buffer()

    # two batches of differing time length to exercise padding
    m.collect_test_predictions(None, {"position": torch.zeros(2, 3, 3)})
    m.collect_test_predictions(None, {"position": torch.ones(1, 5, 3)})

    npz_path = m.save_test_predictions(metrics={"mae": 0.1})
    assert npz_path is not None
    assert npz_path.name == "pred_test.npz"
    loaded = np.load(npz_path)
    assert loaded["position"].shape == (3, 5, 3)  # batch 2+1, T padded to 5
    assert loaded["scene_ids"].shape == (3,)
    metrics = (npz_path.parent / "metrics.json").read_text()
    assert "mae" in metrics


def test_save_test_predictions_none_when_empty(tmp_path: Path, monkeypatch) -> None:
    m = _TinyModule({})
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    assert m.save_test_predictions() is None  # nothing collected
