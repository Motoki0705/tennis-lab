"""A fresh process must continue all model/augmentation RNG streams."""

import random

import numpy as np
import pytest
import torch

from src.tasks.base.training.random_state import (
    capture_random_state,
    restore_random_state,
)


def _draw():
    return random.random(), np.random.normal(size=3), torch.randn(4)


def test_saved_process_rng_continues_next_draw_without_reseeding(tmp_path):
    _draw()
    state = capture_random_state()
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    state = torch.load(path, weights_only=True)
    expected = _draw()
    random.seed(999)
    np.random.seed(998)
    torch.manual_seed(997)
    restore_random_state(state)
    actual = _draw()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


def test_missing_rng_stream_is_rejected():
    state = capture_random_state()
    del state["numpy"]
    with pytest.raises(KeyError, match="numpy"):
        restore_random_state(state)
