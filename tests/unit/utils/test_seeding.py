"""Unit tests for :mod:`src.utils.seeding`."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seeding import make_sample_rng, seed_everything


class TestSeedEverything:
    def test_python_random_is_reproducible(self) -> None:
        seed_everything(123)
        a = [random.random() for _ in range(5)]
        seed_everything(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_is_reproducible(self) -> None:
        seed_everything(7)
        a = np.random.rand(5)
        seed_everything(7)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_is_reproducible(self) -> None:
        seed_everything(99)
        a = torch.randn(5)
        seed_everything(99)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        seed_everything(1)
        a = torch.randn(10)
        seed_everything(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestMakeSampleRng:
    def test_returns_random_instance(self) -> None:
        torch.manual_seed(0)
        assert isinstance(make_sample_rng(0), random.Random)

    def test_same_sample_idx_is_deterministic(self) -> None:
        torch.manual_seed(0)
        a = make_sample_rng(5).random()
        torch.manual_seed(0)
        b = make_sample_rng(5).random()
        assert a == b

    def test_different_sample_idx_decorrelated(self) -> None:
        torch.manual_seed(0)
        a = make_sample_rng(1).random()
        torch.manual_seed(0)
        b = make_sample_rng(2).random()
        assert a != b
