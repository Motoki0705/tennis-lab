"""Shared helpers for training CLI entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Sequence

try:
    from lightning.pytorch.utilities.seed import seed_everything
except ImportError:  # pragma: no cover - fallback for Fabric-only installs
    try:
        from lightning_fabric.utilities.seed import seed_everything
    except ImportError:  # pragma: no cover - legacy PyTorch Lightning
        from pytorch_lightning.utilities.seed import seed_everything

from src.training.utils.config import ConfigLoader, load_cfg


def run_training_from_config(
    *,
    config_path: str,
    overrides: Sequence[str] | None,
    required_task: str | None = None,
    usage_error_message: str | None = None,
    use_explicit_logger: bool = False,
    catch_all_train_errors: bool = False,
    handle_notimplemented_as_usage_error: bool = False,
) -> int:
    """Run a training loop using a YAML config path.

    This function centralizes common CLI wiring (config loading, seeding,
    ConfigLoader usage) while allowing task-specific behavior via flags.
    """
    try:
        cfg = load_cfg(config_path, overrides or [])
    except FileNotFoundError as exc:
        sys.stderr.write(f"[config-error] {exc}\n")
        return 2

    if required_task is not None:
        task = str(cfg.get("task") or "").strip().lower()
        if task != required_task:
            msg = usage_error_message or (
                f"[usage-error] cfg.task must be '{required_task}' for this CLI.\n"
            )
            sys.stderr.write(msg)
            return 2

    seed_value = cfg.get("seed") or cfg.get("training", {}).get("seed")
    if seed_value is not None:
        seed_everything(int(seed_value), workers=True)

    loader = ConfigLoader(cfg)
    try:
        datamodule = loader.build_datamodule()
        lit_module = loader.build_lit_module()
        if use_explicit_logger:
            logger = loader.build_logger()
            callbacks = loader.build_callbacks()
            trainer = loader.build_trainer(logger=logger, callbacks=callbacks)
        else:
            trainer = loader.build_trainer()
    except NotImplementedError as exc:
        if handle_notimplemented_as_usage_error:
            sys.stderr.write(f"[train-error] {exc}\n")
            return 2
        if catch_all_train_errors:
            sys.stderr.write(f"[train-error] {exc}\n")
            return 1
        raise
    except Exception as exc:  # pragma: no cover - surfacing errors with context
        if catch_all_train_errors:
            sys.stderr.write(f"[train-error] {exc}\n")
            return 1
        raise

    try:
        trainer.fit(lit_module, datamodule=datamodule)
    except Exception as exc:  # pragma: no cover - trainer/fit errors
        if catch_all_train_errors:
            sys.stderr.write(f"[train-error] {exc}\n")
            return 1
        raise

    return 0
