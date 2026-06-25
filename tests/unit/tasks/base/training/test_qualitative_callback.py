"""Unit tests for QualitativeLoggingCallback selection/gating logic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.base.training.qualitative_callback import (
    QualitativeLoggingCallback,
    _detach_to_cpu,
)

pytestmark = pytest.mark.unit


def test_init_clamps_minimums() -> None:
    cb = QualitativeLoggingCallback(every_n_epochs=0, num_samples=0)
    assert cb.every_n_epochs == 1
    assert cb.num_samples == 1


def test_select_random_subset_bounded() -> None:
    cb = QualitativeLoggingCallback(num_samples=3, selection_mode="random")
    selected = cb._select_batch_indices(total=10)
    assert len(selected) == 3
    assert selected <= set(range(10))


def test_select_random_caps_at_total() -> None:
    cb = QualitativeLoggingCallback(num_samples=20, selection_mode="random")
    selected = cb._select_batch_indices(total=5)
    assert selected == set(range(5))


def test_select_zero_total_empty() -> None:
    cb = QualitativeLoggingCallback(num_samples=4)
    assert cb._select_batch_indices(total=0) == set()


def test_fixed_indices_mode() -> None:
    cb = QualitativeLoggingCallback(
        selection_mode="fixed_indices", selected_indices=[0, 2, 4]
    )
    assert cb._select_batch_indices(total=6) == {0, 2, 4}


def test_fixed_indices_requires_list() -> None:
    cb = QualitativeLoggingCallback(selection_mode="fixed_indices", selected_indices=None)
    with pytest.raises(ValueError, match="non-empty selected_indices"):
        cb._select_batch_indices(total=6)


def test_fixed_indices_out_of_range_raises() -> None:
    cb = QualitativeLoggingCallback(
        selection_mode="fixed_indices", selected_indices=[0, 7]
    )
    with pytest.raises(ValueError, match="out-of-range"):
        cb._select_batch_indices(total=5)


def test_unknown_selection_mode_raises() -> None:
    cb = QualitativeLoggingCallback(selection_mode="bogus")
    with pytest.raises(ValueError, match="must be 'random' or"):
        cb._select_batch_indices(total=5)


class _Trainer:
    def __init__(self, *, enabled_epoch: int = 0, sanity: bool = False) -> None:
        self.current_epoch = enabled_epoch
        self.sanity_checking = sanity


def test_should_log_respects_enabled_flag() -> None:
    cb = QualitativeLoggingCallback(enabled=False)
    assert cb._should_log(_Trainer()) is False


def test_should_log_skips_sanity_check() -> None:
    cb = QualitativeLoggingCallback(enabled=True)
    assert cb._should_log(_Trainer(sanity=True)) is False


def test_should_log_every_n_epochs() -> None:
    cb = QualitativeLoggingCallback(enabled=True, every_n_epochs=3)
    assert cb._should_log(_Trainer(enabled_epoch=0)) is True
    assert cb._should_log(_Trainer(enabled_epoch=3)) is True
    assert cb._should_log(_Trainer(enabled_epoch=1)) is False


def test_detach_to_cpu_recurses_structures() -> None:
    t = torch.ones(2, requires_grad=True)
    data = {"a": t, "b": [t, t], "c": ("x", 3)}
    out = _detach_to_cpu(data)
    assert out["a"].requires_grad is False
    assert out["a"].device.type == "cpu"
    assert isinstance(out["b"], list)
    assert isinstance(out["c"], tuple)
    assert out["c"] == ("x", 3)


def test_detach_to_cpu_passthrough_non_tensor() -> None:
    assert _detach_to_cpu(5) == 5
    assert _detach_to_cpu("hello") == "hello"
    arr = np.zeros(3)
    assert _detach_to_cpu(arr) is arr
