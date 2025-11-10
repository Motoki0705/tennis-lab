"""Utilities for loading and instantiating training configs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

INCLUDE_KEY = "includes"
_TRUTHY = {"1", "true", "yes", "on"}


def _container(cfg: DictConfig | None) -> dict:
    return {} if cfg is None else dict(OmegaConf.to_container(cfg, resolve=True))


def _bool_from_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUTHY


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def load_cfg(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load and merge the hierarchical YAML config structure."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        msg = f"Config file not found: {cfg_path}"
        raise FileNotFoundError(msg)
    base = OmegaConf.load(cfg_path)
    include_map = _container(base.get(INCLUDE_KEY))
    merged: DictConfig = OmegaConf.create({})
    for key, rel in include_map.items():
        include_path = _resolve_path(cfg_path.parent / rel)
        part = OmegaConf.load(include_path)
        merged = OmegaConf.merge(merged, OmegaConf.create({key: part}))
    cleaned = {k: v for k, v in base.items() if k != INCLUDE_KEY}
    merged = OmegaConf.merge(merged, OmegaConf.create(cleaned))
    if _bool_from_env("CFG_DEBUG_MINIMAL"):
        merged = OmegaConf.merge(merged, OmegaConf.create({"debug": {"minimal": True}}))
    norm_overrides = [o for o in overrides or [] if o]
    if norm_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(norm_overrides))
    return merged


@dataclass(slots=True)
class ConfigLoader:
    """Factory helpers that build training objects from the DictConfig."""

    cfg: DictConfig

    def build_datamodule(self):
        from src.training.scene_model.datamodule import DancetrackDataModule

        dataset_cfg = self.cfg.get("dataset")
        debug_cfg = self.cfg.get("debug")
        return DancetrackDataModule(dataset_cfg, debug_cfg)

    def build_lit_module(self):
        from src.training.scene_model.lightning import SceneModelLightningModule

        return SceneModelLightningModule(self.cfg)

    def build_callbacks(self) -> list[Callback]:
        from src.training.scene_model.callbacks import build_callbacks

        return build_callbacks(self.cfg.get("logging"))

    def build_logger(self) -> Logger:
        from src.training.scene_model.callbacks import build_logger

        experiment_name = self.cfg.get("experiment_name")
        return build_logger(self.cfg.get("logging"), experiment_name)

    def build_trainer(
        self,
        logger: Logger | bool | None = None,
        callbacks: Iterable[Callback] | None = None,
    ) -> Trainer:
        trainer_cfg = _container(self.cfg.get("training")).get("trainer", {})
        pl_logger = logger if logger is not None else self.build_logger()
        callback_list = list(callbacks) if callbacks is not None else self.build_callbacks()
        return Trainer(logger=pl_logger, callbacks=callback_list, **trainer_cfg)
