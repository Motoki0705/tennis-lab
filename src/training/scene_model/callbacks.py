"""Callback builders shared across the training entrypoints."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger


def _as_dict(cfg: DictConfig | None) -> dict[str, Any]:
    return {} if cfg is None else dict(OmegaConf.to_container(cfg, resolve=True))


def build_logger(
    logging_cfg: DictConfig | None, experiment_name: str | None = None
) -> Logger:
    """Instantiate the TensorBoard logger described in the config."""
    cfg = _as_dict(logging_cfg).get("logger", {})
    save_dir = cfg.get("save_dir", "runs")
    name = experiment_name or cfg.get("name", "scene_model")
    version = cfg.get("version")
    default_hp_metric = cfg.get("default_hp_metric", False)
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        default_hp_metric=default_hp_metric,
    )


def _build_checkpoint(cfg: dict[str, Any]) -> ModelCheckpoint:
    kwargs: dict[str, Any] = {
        "monitor": cfg.get("monitor", "val/total_loss"),
        "mode": cfg.get("mode", "min"),
        "save_top_k": cfg.get("save_top_k", 1),
        "save_last": cfg.get("save_last", True),
    }
    every_n_epochs = cfg.get("every_n_epochs")
    if every_n_epochs is not None:
        kwargs["every_n_epochs"] = every_n_epochs
    filename = cfg.get("filename")
    if filename is not None:
        kwargs["filename"] = filename
    return ModelCheckpoint(**kwargs)


def build_callbacks(logging_cfg: DictConfig | None) -> list[Callback]:
    """Create the callback collection declared in the logging config."""
    cfg = _as_dict(logging_cfg).get("callbacks", {})
    callbacks: list[Callback] = []
    checkpoint_cfg = cfg.get("checkpoint")
    if isinstance(checkpoint_cfg, dict):
        callbacks.append(_build_checkpoint(checkpoint_cfg))
    lr_cfg = cfg.get("lr_monitor")
    if isinstance(lr_cfg, dict):
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_cfg.get("logging_interval", "epoch")
            )
        )
    return callbacks
