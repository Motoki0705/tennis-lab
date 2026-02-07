"""PLCS training runners using BaseTrainingRunner."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.plcs.data.datamodule import (
    PLCSDataModule,
    PLCSMultiViewDataModule,
    PLCSMultiViewSequenceDataModule,
    PLCSSequenceDataModule,
)
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.lightning_module_kp3d import PLCSKeypoint3DLightningModule
from src.plcs.training.multiview_lightning_module import PLCSMultiViewLightningModule
from src.plcs.training.sequence_lightning_module import PLCSSequenceLightningModule

_RUNNER_REGISTRY: dict[str, type[BaseTrainingRunner]] = {}


def _register(name: str):  # noqa: ANN202
    """Decorator to register a runner class under *name*."""

    def _wrap(cls: type[BaseTrainingRunner]) -> type[BaseTrainingRunner]:
        _RUNNER_REGISTRY[name] = cls
        return cls

    return _wrap


def select_runner(config: Any) -> BaseTrainingRunner:
    """Return the appropriate PLCS training runner for *config*.

    Selection is based on ``config.data.mode``:

    * ``"multiview"`` / ``"multiview_sequence"`` → :class:`PLCSMultiViewTrainingRunner`
    * anything else (``"frame"`` / ``"sequence"`` / ``"kp3d"``) → :class:`PLCSTrainingRunner`
    """
    mode = str(getattr(config.data, "mode", "frame"))
    if mode in _RUNNER_REGISTRY:
        return _RUNNER_REGISTRY[mode]()
    if mode.startswith("multiview"):
        return PLCSMultiViewTrainingRunner()
    return PLCSTrainingRunner()


@_register("frame")
@_register("sequence")
class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS single-view model (frame and sequence modes).

    Supports two modes based on config.data.mode:
    - "frame" (default): Frame-based PLCS training.
    - "sequence": Sequence-based PLCS training with temporal modeling.
    """

    def _is_sequence_mode(self, config: Any) -> bool:
        """Check if running in sequence mode."""
        return str(getattr(config.data, "mode", "frame")) == "sequence"

    def _is_kp3d_model(self, config: Any) -> bool:
        """Check if running keypoint-3D frame model."""
        return str(getattr(config.model, "name", "")) == "plcs_kp3d"

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build PLCS data module based on config.data.mode."""
        if self._is_sequence_mode(config):
            return PLCSSequenceDataModule(config)
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build PLCS lightning module based on config.data.mode."""
        if self._is_sequence_mode(config):
            return PLCSSequenceLightningModule(config)
        if self._is_kp3d_model(config):
            return PLCSKeypoint3DLightningModule(config)
        return PLCSLightningModule(config)


@_register("multiview")
@_register("multiview_sequence")
class PLCSMultiViewTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS multi-view model.

    Supports two modes based on config.data.mode:
    - "multiview" (default): Frame-based multi-view PLCS training.
    - "multiview_sequence": Sequence-based multi-view PLCS training.
    """

    def _is_sequence_mode(self, config: Any) -> bool:
        """Check if running in multiview_sequence mode."""
        mode = str(getattr(config.data, "mode", "multiview"))
        return mode == "multiview_sequence"

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build PLCS multi-view data module based on config.data.mode."""
        if self._is_sequence_mode(config):
            return PLCSMultiViewSequenceDataModule(config)
        return PLCSMultiViewDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build PLCS multi-view lightning module."""
        return PLCSMultiViewLightningModule(config)
