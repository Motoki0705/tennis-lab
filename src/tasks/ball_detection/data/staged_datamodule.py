"""Staged multi-frame DataModule for the #579 training schedule.

Combines TrackNet and (optionally) the unified Web store into one training
stream. Windows are built at ``data.t_max`` frames; the train loader draws a clip
length ``T <= t_max`` per optimizer-step group via
:class:`VariableTBatchSampler`, while val/test run at a fixed ``data.val_num_frames``
(default 1) for a stable, comparable monitor. Phases 1/2 set ``t_max=1`` so the
exact same machinery degenerates to plain single-frame batches.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.tasks.ball_detection.data.components.staged_sampler import (
    ConcatVariableTDataset,
    FixedTDataset,
    VariableTBatchSampler,
    linear_decreasing_t_probs,
)
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.ball_detection.data.types import BallDetectionSample
from src.tasks.ball_detection.data.web_datamodule import WebBallDataModule

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)

_VALID_T_DISTRIBUTIONS = {"variable", "fixed"}
_VALID_SPLITS = frozenset({"train", "val", "test"})

# Shared per-frame fields forwarded verbatim to every source sub-config.
_SHARED_DATA_KEYS = (
    "image_size",
    "heatmap_size",
    "sigma_ratio",
    "max_instances",
    "num_workers",
    "pin_memory",
    "augmentation",
)


class StagedBallDataModule(pl.LightningDataModule):
    """Mixed TrackNet+Web datamodule with a variable-T training schedule."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}
        data_cfg = self.config.get("data", {}) or {}

        self.t_max = int(data_cfg.get("t_max", 1))
        if self.t_max < 1:
            raise ValueError("data.t_max must be >= 1.")
        self.val_num_frames = int(data_cfg.get("val_num_frames", 1))
        if not 1 <= self.val_num_frames <= self.t_max:
            raise ValueError("data.val_num_frames must be in [1, t_max].")
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.sampling_seed = int(data_cfg.get("seed", 1234))
        self.val_batch_size = int(data_cfg.get("val_batch_size", 8))

        self.t_distribution = str(data_cfg.get("t_distribution", "variable")).lower()
        if self.t_distribution not in _VALID_T_DISTRIBUTIONS:
            raise ValueError(
                "data.t_distribution must be one of "
                f"{sorted(_VALID_T_DISTRIBUTIONS)}, got {self.t_distribution!r}."
            )
        self.t1_prob: float | None
        if self.t_distribution == "variable":
            self.t1_prob = float(data_cfg.get("t1_prob", 0.5))
            self.t_probs = linear_decreasing_t_probs(self.t_max, self.t1_prob)
        else:
            self.t1_prob = None
            self.t_probs = {self.t_max: 1.0}
            message = (
                "[staged] data.t_distribution=fixed; train sampler uses only "
                f"T={self.t_max}. data.t1_prob is ignored."
            )
            LOGGER.info(message)

        sources_cfg = data_cfg.get("sources", {}) or {}
        self.source_splits = self._parse_source_splits(sources_cfg)
        self.enabled_sources = [
            name
            for name in ("tracknet", "web")
            if bool((sources_cfg.get(name, {}) or {}).get("enabled", False))
        ]
        if not self.enabled_sources:
            raise ValueError("At least one of data.sources.{tracknet,web} must be enabled.")

        # B(T) physical batch table + effective batch size. Defaults here are a
        # safe fallback; the runner overrides them from OOM calibration.
        self.batch_size_by_t = {
            int(t): int(b)
            for t, b in (data_cfg.get("batch_size_by_t", {1: 4}) or {1: 4}).items()
        }
        self.effective_batch_size = int(
            data_cfg.get("effective_batch_size") or self.batch_size_by_t.get(1, 4)
        )

        self._submodules: dict[str, pl.LightningDataModule] = {}
        self.train_dataset: Dataset[BallDetectionSample] | None = None
        self.val_dataset: Dataset[BallDetectionSample] | None = None
        self.test_dataset: Dataset[BallDetectionSample] | None = None

    def _parse_source_splits(self, sources_cfg: Any) -> dict[str, frozenset[str]]:
        parsed: dict[str, frozenset[str]] = {}
        for source_name, source_cfg in (sources_cfg or {}).items():
            source = str(source_name)
            if source not in {"tracknet", "web"}:
                raise ValueError(f"Unknown staged source: {source!r}.")
            raw_splits = (source_cfg or {}).get("splits", tuple(_VALID_SPLITS))
            if isinstance(raw_splits, str):
                raise ValueError(
                    f"data.sources.{source}.splits must be a list of split names, "
                    f"got {raw_splits!r}."
                )
            splits = frozenset(str(split) for split in raw_splits)
            unknown = sorted(splits - _VALID_SPLITS)
            if unknown:
                raise ValueError(
                    f"data.sources.{source}.splits contains unknown split(s): "
                    f"{unknown}. Valid splits are {sorted(_VALID_SPLITS)}."
                )
            if not splits:
                raise ValueError(f"data.sources.{source}.splits must not be empty.")
            parsed[source] = splits
        for source in ("tracknet", "web"):
            parsed.setdefault(source, _VALID_SPLITS)
        return parsed

    # ------------------------------------------------------------------
    def set_batch_plan(
        self, batch_size_by_t: Mapping[int, int], effective_batch_size: int
    ) -> None:
        """Inject the calibrated B(T) table + EBS before ``setup``/fit."""
        self.batch_size_by_t = {int(t): int(b) for t, b in batch_size_by_t.items()}
        self.effective_batch_size = int(effective_batch_size)

    def _sub_config(self, source_name: str) -> DictConfig:
        data_cfg = self.config.get("data", {}) or {}
        source_cfg = dict((data_cfg.get("sources", {}) or {}).get(source_name, {}) or {})
        source_cfg.pop("enabled", None)
        source_cfg.pop("splits", None)
        merged: dict[str, Any] = {"batch_size": 1}
        for key in _SHARED_DATA_KEYS:
            if key in data_cfg:
                merged[key] = OmegaConf.to_container(
                    OmegaConf.create({key: data_cfg[key]}), resolve=True
                )[key]
        merged.update(source_cfg)
        return OmegaConf.create(
            {"data": merged, "model": {"num_frames": self.t_max}}
        )

    def _build_submodules(self) -> None:
        if self._submodules:
            return
        builders: dict[str, type[pl.LightningDataModule]] = {
            "tracknet": TrackNetDataModule,
            "web": WebBallDataModule,
        }
        for name in self.enabled_sources:
            self._submodules[name] = builders[name](self._sub_config(name))

    def setup(self, stage: str | None = None) -> None:
        """Build per-source datasets and concatenate them per split."""
        self._build_submodules()
        for module in self._submodules.values():
            module.setup(stage=stage)

        if stage in (None, "fit"):
            self.train_dataset = self._concat("train", "train_dataset")
            self.val_dataset = self._fixed_t(self._concat("val", "val_dataset"))
        if stage in (None, "validate"):
            self.val_dataset = self._fixed_t(self._concat("val", "val_dataset"))
        if stage in (None, "test"):
            self.test_dataset = self._fixed_t(self._concat("test", "test_dataset"))

    def _concat(self, split: str, attr: str) -> Dataset[BallDetectionSample]:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Unknown staged split: {split!r}. Valid splits are {sorted(_VALID_SPLITS)}."
            )
        datasets = [
            getattr(module, attr)
            for name, module in self._submodules.items()
            if split in self.source_splits[name]
            if getattr(module, attr, None) is not None
        ]
        if not datasets:
            raise RuntimeError(f"No source produced a {split!r} {attr!r}.")
        return ConcatVariableTDataset(datasets)

    def _fixed_t(
        self, dataset: Dataset[BallDetectionSample]
    ) -> Dataset[BallDetectionSample]:
        if self.val_num_frames == self.t_max:
            return dataset
        return FixedTDataset(dataset, self.val_num_frames)

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        sampler = VariableTBatchSampler(
            num_samples=len(self.train_dataset),
            t_probs=self.t_probs,
            batch_size_by_t=self.batch_size_by_t,
            effective_batch=self.effective_batch_size,
            seed=self.sampling_seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["StagedBallDataModule"]
