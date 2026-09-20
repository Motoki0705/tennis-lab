"""Domain balancing changes exposure, with auditable epoch-local randomness."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.slcs.data.sampling import DomainBalancedSampler, DomainSamplingConfig


def test_weights_length_and_epoch_replay() -> None:
    videos = ["a"] * 90 + ["b"] * 10
    mapping = {"a": "major", "b": "minor"}
    sampler = DomainBalancedSampler(videos, mapping, seed=42)
    assert len(sampler) == 100
    assert sampler.weights[:90].sum().item() == pytest.approx(1)
    assert sampler.weights[90:].sum().item() == pytest.approx(1)
    initial = list(sampler)
    assert initial == list(sampler)
    assert len(set(initial)) < 100  # replacement, not a permutation
    state = torch.random.get_rng_state()
    sampler.set_epoch(8)
    later = list(sampler)
    assert later != initial
    assert torch.equal(state, torch.random.get_rng_state())
    reloaded = DomainBalancedSampler(videos, mapping, seed=42)
    reloaded.set_epoch(8)
    assert list(reloaded) == later
    assert list(DomainBalancedSampler(videos, mapping, seed=43)) != initial


@pytest.mark.parametrize(
    "videos,mapping,match",
    [
        ([], {"a": "one"}, "non-empty train"),
        (["a"], {}, "requires video_domains"),
        (["a", "b"], {"a": "one"}, "Missing train"),
        (["a"], {"a": "one", "b": "two"}, "no retained train"),
        (["a"], {"a": ""}, "domains must"),
        (["a"], {"a": " one"}, "domains must"),
        (["a"], {"a": 2}, "domains must"),
    ],
)
def test_invalid_domain_mapping(
    videos: list[str], mapping: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DomainBalancedSampler(videos, mapping, seed=42)


def test_extra_video_in_existing_domain_is_allowed() -> None:
    sampler = DomainBalancedSampler(["a"], {"a": "one", "unused": "one"}, seed=42)
    assert list(sampler) == [0]
    with pytest.raises(ValueError, match="non-negative"):
        sampler.set_epoch(-1)
    assert DomainSamplingConfig(enabled=False, video_domains={}).enabled is False


def test_real_rgb_window_budget_retains_thirty_batches() -> None:
    videos = ["meiji_video"] * 426 + ["shanghai"] * 25 + ["washington"] * 15
    sampler = DomainBalancedSampler(
        videos,
        {"meiji_video": "meiji", "shanghai": "broadcast", "washington": "broadcast"},
        seed=42,
    )
    loader = DataLoader(
        TensorDataset(torch.arange(466)), batch_size=16, sampler=sampler, drop_last=False
    )
    assert len(loader) == 30
    assert len(loader) * 60 == 1800
    assert sampler.weights[:426].sum().item() == pytest.approx(1)
    assert sampler.weights[426:].sum().item() == pytest.approx(1)
