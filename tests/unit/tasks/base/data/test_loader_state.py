"""Shuffle/iterator RNG continues at the next epoch independently of the model."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.tasks.base.data.loader_state import LoaderRNGDataModule


def _epoch(data):
    return list(
        DataLoader(
            list(range(11)),
            shuffle=True,
            generator=data.loader_generator("train"),
            batch_size=None,
        )
    )


def test_fresh_loader_continues_shuffled_epochs_and_preserves_model_rng():
    data = LoaderRNGDataModule(loader_seed=123)
    first = _epoch(data)
    checkpoint = data.state_dict()
    before = torch.get_rng_state().clone()
    second = _epoch(data)
    assert first != second and sorted(second) == list(range(11))
    assert torch.equal(torch.get_rng_state(), before)
    fresh = LoaderRNGDataModule(loader_seed=456)
    fresh.load_state_dict(checkpoint)
    assert _epoch(fresh) == second
    torch.testing.assert_close(fresh.state_dict(), data.state_dict(), rtol=0, atol=0)


def test_evaluation_iterator_does_not_advance_train_order():
    data = LoaderRNGDataModule(loader_seed=123)
    other = LoaderRNGDataModule(loader_seed=123)
    list(DataLoader(list(range(11)), generator=data.loader_generator("val")))
    assert _epoch(data) == _epoch(other)


def test_missing_loader_stage_is_rejected():
    data = LoaderRNGDataModule(loader_seed=123)
    checkpoint = data.state_dict()
    del checkpoint["loader_rng"]["train"]
    with pytest.raises(ValueError, match="stages"):
        data.load_state_dict(checkpoint)


def test_loader_accepts_the_full_unsigned_torch_seed_range():
    seed = 2**64 - 1
    assert _epoch(LoaderRNGDataModule(loader_seed=seed)) == _epoch(
        LoaderRNGDataModule(loader_seed=seed)
    )
