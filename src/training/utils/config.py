"""Utilities for loading and instantiating training configs."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

if TYPE_CHECKING:
    from src.training.scene_model.datamodule import DancetrackDataModule
    from src.training.scene_model.lightning import SceneModelLightningModule
    from src.training.tennis_multi_cam_3d_pose.datamodule import TennisPoseDataModule
    from src.training.tennis_multi_cam_3d_pose.lightning import TennisDetrModule
    from src.training.tennis_multi_cam_3d_pose.lightning_v2 import TennisDetrV2Module
    from src.training.tennis_multi_cam_3d_pose.lightning_v2_5 import (
        TennisDetrV25Module,
    )
    from src.training.tennis_multi_cam_3d_pose.lightning_v3 import TennisDetrV3Module

INCLUDE_KEY = "includes"
_TRUTHY = {"1", "true", "yes", "on"}

_DATAMODULE_REGISTRY: dict[
    str,
    Callable[
        [DictConfig | None, DictConfig | None],
        DancetrackDataModule | TennisPoseDataModule,
    ],
] = {}
_LIGHTNING_REGISTRY: dict[
    tuple[str, str],
    Callable[
        [DictConfig],
        SceneModelLightningModule
        | TennisDetrModule
        | TennisDetrV2Module
        | TennisDetrV25Module
        | TennisDetrV3Module,
    ],
] = {}


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
    _logger: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the ConfigLoader after dataclass creation."""
        import logging

        self._logger = logging.getLogger(__name__)
        self._logger.debug("ConfigLoader initialized with task=%s", self._task())

    def _task(self) -> str:
        """Return the configured task name with a backward-compatible default."""
        task = self.cfg.get("task")
        return str(task) if task else "scene_model"

    def _experiment_name(self) -> str:
        experiment_name = self.cfg.get("experiment_name")
        return str(experiment_name or "").lower()

    def _lightning_key(self) -> tuple[str, str]:
        task = self._task()
        if task == "tennis_multi_cam_3d_pose":
            experiment_name = self._experiment_name()
            if "v3" in experiment_name:
                variant = "v3"
            elif "v2_5" in experiment_name:
                variant = "v2_5"
            elif "v2" in experiment_name:
                variant = "v2"
            else:
                variant = "v1"
        else:
            variant = "default"
        return task, variant

    def build_datamodule(self) -> DancetrackDataModule | TennisPoseDataModule:
        """Construct the LightningDataModule declared by the current task.

        Defaults to the existing SceneModel DataModule for backward compatibility.
        For the tennis_multi_cam_3d_pose task, P0 provides only scaffolding and
        raises a descriptive error until subsequent phases (P1+) are implemented.
        """
        task = self._task()
        dataset_cfg = self.cfg.get("dataset")
        debug_cfg = self.cfg.get("debug")
        dataset_keys = list(dataset_cfg.keys()) if dataset_cfg else []

        builder = _DATAMODULE_REGISTRY.get(task)
        if builder is None:
            experiment_name = self._experiment_name()
            msg = f"Unsupported task={task} (experiment_name={experiment_name}) for datamodule"
            self._logger.error(msg)
            raise NotImplementedError(msg)
        datamodule = builder(dataset_cfg, debug_cfg)
        self._logger.info(
            "DataModule built for task=%s (dataset_keys=%s)", task, dataset_keys
        )
        return datamodule

    def build_lit_module(
        self,
    ) -> (
        SceneModelLightningModule
        | TennisDetrModule
        | TennisDetrV2Module
        | TennisDetrV25Module
        | TennisDetrV3Module
    ):
        """Instantiate the LightningModule for the configured task.

        Defaults to SceneModel. For tennis_multi_cam_3d_pose, supports both v1 and v2
        based on the training._target_ configuration.
        """
        task = self._task()
        experiment_name = self._experiment_name()
        training_cfg = self.cfg.get("training", {})
        target = str(training_cfg.get("_target_", ""))
        key = self._lightning_key()
        builder = _LIGHTNING_REGISTRY.get(key)
        if builder is None:
            msg = (
                "Unsupported LightningModule selection for task="
                f"{task} (experiment_name={experiment_name}, target={target})"
            )
            self._logger.error(msg)
            raise NotImplementedError(msg)
        self._logger.info(
            "Building LightningModule for task=%s (variant=%s)", task, key[1]
        )
        module = builder(self.cfg)
        self._logger.info("LightningModule built for task=%s", task)
        return module

    def build_callbacks(self) -> list[Callback]:
        """Create callbacks (checkpoints, LR monitor, etc.) from config."""
        from src.training.scene_model.callbacks import build_callbacks

        callbacks = build_callbacks(self.cfg.get("logging"))
        self._logger.info("Constructed %d callbacks", len(callbacks))
        return callbacks

    def build_logger(self) -> Logger:
        """Create the experiment logger defined under ``cfg.logging``."""
        from src.training.scene_model.callbacks import build_logger

        experiment_name = self.cfg.get("experiment_name")
        logger = build_logger(self.cfg.get("logging"), experiment_name)
        self._logger.info("Logger built for experiment=%s", experiment_name)
        return logger

    def build_trainer(
        self,
        logger: Logger | bool | None = None,
        callbacks: Iterable[Callback] | None = None,
    ) -> Trainer:
        """Build a Lightning Trainer, optionally overriding logger/callbacks."""
        trainer_cfg = _container(self.cfg.get("training")).get("trainer", {})
        pl_logger = logger if logger is not None else self.build_logger()
        trainer = Trainer(
            logger=pl_logger, callbacks=list(callbacks or []), **trainer_cfg
        )
        self._logger.info(
            "Trainer built with logger=%s, callbacks=%d",
            pl_logger,
            len(trainer.callbacks),
        )
        return trainer


def _register_default_builders() -> None:
    def _build_scene_datamodule(
        dataset_cfg: DictConfig | None, debug_cfg: DictConfig | None
    ) -> Any:
        from src.training.scene_model.datamodule import DancetrackDataModule

        return DancetrackDataModule(dataset_cfg, debug_cfg)

    def _build_tennis_datamodule(
        dataset_cfg: DictConfig | None, debug_cfg: DictConfig | None
    ) -> Any:
        from src.training.tennis_multi_cam_3d_pose.datamodule import (
            TennisPoseDataModule,
        )

        return TennisPoseDataModule(dataset_cfg, debug_cfg)

    _DATAMODULE_REGISTRY["scene_model"] = _build_scene_datamodule
    _DATAMODULE_REGISTRY["tennis_multi_cam_3d_pose"] = _build_tennis_datamodule

    def _build_scene_module(cfg: DictConfig) -> Any:
        from src.training.scene_model.lightning import SceneModelLightningModule

        return SceneModelLightningModule(cfg)

    def _build_tennis_v1_module(cfg: DictConfig) -> Any:
        from src.training.tennis_multi_cam_3d_pose.lightning import TennisDetrModule

        return TennisDetrModule(cfg)

    def _build_tennis_v2_module(cfg: DictConfig) -> Any:
        from src.training.tennis_multi_cam_3d_pose.lightning_v2 import (
            TennisDetrV2Module,
        )

        return TennisDetrV2Module(cfg)

    def _build_tennis_v25_module(cfg: DictConfig) -> Any:
        from src.training.tennis_multi_cam_3d_pose.lightning_v2_5 import (
            TennisDetrV25Module,
        )

        return TennisDetrV25Module(cfg)

    def _build_tennis_v3_module(cfg: DictConfig) -> Any:
        from src.training.tennis_multi_cam_3d_pose.lightning_v3 import (
            TennisDetrV3Module,
        )

        return TennisDetrV3Module(cfg)

    _LIGHTNING_REGISTRY[("scene_model", "default")] = _build_scene_module
    _LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v1")] = _build_tennis_v1_module
    _LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v2")] = _build_tennis_v2_module
    _LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v2_5")] = _build_tennis_v25_module
    _LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v3")] = _build_tennis_v3_module


_register_default_builders()
