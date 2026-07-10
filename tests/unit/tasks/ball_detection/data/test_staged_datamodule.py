"""Unit tests for staged datamodule split gating and T-distribution config."""

from __future__ import annotations

from typing import Any, cast

import pytest
from omegaconf import OmegaConf

import src.tasks.ball_detection.data.staged_datamodule as staged_module
from src.tasks.ball_detection.data.components.staged_sampler import (
    ConcatVariableTDataset,
    FixedTDataset,
    accumulation_for,
)
from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule


class _TaggedDataset:
    """Minimal dataset that exposes which staged source/split produced a sample."""

    def __init__(self, source: str, split: str, n: int) -> None:
        self.source = source
        self.split = split
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        return {"source": self.source, "split": self.split, "index": index}


class _FakeSourceDataModule:
    def __init__(self, source: str, config: Any) -> None:
        self.source = source
        self.config = config
        self.train_dataset: _TaggedDataset | None = None
        self.val_dataset: _TaggedDataset | None = None
        self.test_dataset: _TaggedDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = _TaggedDataset(self.source, "train", 8)
            self.val_dataset = _TaggedDataset(self.source, "val", 3)
        if stage in (None, "validate"):
            self.val_dataset = _TaggedDataset(self.source, "val", 3)
        if stage in (None, "test"):
            self.test_dataset = _TaggedDataset(self.source, "test", 2)


def _fake_source_class(source: str) -> type[_FakeSourceDataModule]:
    class FakeSourceDataModule(_FakeSourceDataModule):
        def __init__(self, config: Any) -> None:
            super().__init__(source, config)

    return FakeSourceDataModule


@pytest.fixture()
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        staged_module,
        "TrackNetDataModule",
        _fake_source_class("tracknet"),
    )
    monkeypatch.setattr(
        staged_module,
        "WebBallDataModule",
        _fake_source_class("web"),
    )


def _config(
    *,
    t_distribution: str = "variable",
    t_max: int = 4,
    web_enabled: bool = True,
    web_splits: list[str] | None = None,
) -> Any:
    return OmegaConf.create(
        {
            "data": {
                "t_max": t_max,
                "t_distribution": t_distribution,
                "t1_prob": 0.5,
                "val_num_frames": min(2, t_max),
                "num_workers": 0,
                "pin_memory": False,
                "effective_batch_size": 8,
                "batch_size_by_t": {1: 8, 2: 4, 3: 3, 4: 2},
                "sources": {
                    "tracknet": {
                        "enabled": True,
                        "splits": ["train", "val", "test"],
                    },
                    "web": {
                        "enabled": web_enabled,
                        "splits": web_splits or ["train", "val", "test"],
                    },
                },
            }
        }
    )


def _concat_sources(dataset: Any) -> list[str]:
    if isinstance(dataset, FixedTDataset):
        dataset = dataset.base
    concat = cast(ConcatVariableTDataset, dataset)
    return [child.source for child in concat.datasets]


def test_source_splits_gate_web_to_train_only(fake_sources: None) -> None:
    _ = fake_sources
    datamodule = StagedBallDataModule(_config(web_splits=["train"]))

    datamodule.setup(stage=None)

    assert _concat_sources(datamodule.train_dataset) == ["tracknet", "web"]
    assert _concat_sources(datamodule.val_dataset) == ["tracknet"]
    assert _concat_sources(datamodule.test_dataset) == ["tracknet"]
    assert datamodule.train_dataset is not None
    assert datamodule.train_dataset[8]["source"] == "web"
    assert len(datamodule.val_dataset or []) == 3
    assert len(datamodule.test_dataset or []) == 2


def test_fixed_t_distribution_samples_only_t_max(fake_sources: None) -> None:
    _ = fake_sources
    datamodule = StagedBallDataModule(
        _config(t_distribution="fixed", t_max=4, web_enabled=False)
    )
    datamodule.setup(stage="fit")

    loader = datamodule.train_dataloader()
    batches = list(loader.batch_sampler)
    expected_accumulate = accumulation_for(effective_batch=8, physical_batch=2)
    expected_groups = len(datamodule.train_dataset or []) // (2 * expected_accumulate)

    assert datamodule.t_probs == {4: 1.0}
    assert len(batches) == expected_groups * expected_accumulate
    assert len(batches) > 0
    assert all(len(batch) == 2 for batch in batches)
    assert {t for batch in batches for _, t in batch} == {4}


def test_fixed_t_distribution_allows_t_max_one() -> None:
    datamodule = StagedBallDataModule(
        _config(t_distribution="fixed", t_max=1, web_enabled=False)
    )

    assert datamodule.t_probs == {1: 1.0}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"web_splits": ["train", "holdout"]}, "unknown split"),
        ({"t_distribution": "constant"}, "data.t_distribution"),
    ],
)
def test_invalid_staged_config_raises(
    override: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        StagedBallDataModule(_config(**override))
