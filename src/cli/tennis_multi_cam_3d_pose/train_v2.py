"""CLI entrypoint for training the Tennis Pose v2 stack."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

try:
    from lightning.pytorch.utilities.seed import seed_everything
except ImportError:
    try:  # pragma: no cover - fallback for Fabric-only installs
        from lightning_fabric.utilities.seed import (
            seed_everything,
        )
    except ImportError:  # pragma: no cover - fallback for legacy PyTorch Lightning
        from pytorch_lightning.utilities.seed import (
            seed_everything,
        )

from src.training.utils.config import ConfigLoader, load_cfg


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the tennis pose v2 trainer.

    Args:
        argv (Sequence[str] | None): Optional sequence of raw CLI arguments.
            If ``None``, defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(description="Train the Tennis Pose v2 stack")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the top-level YAML config (e.g. "
            "configs/tennis_multi_cam_3d_pose_v2.yaml)"
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dot notation (e.g. training.trainer.max_epochs=2)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Tennis Pose v2 training loop using the provided config."""
    args = _parse_args(argv)
    try:
        cfg = load_cfg(args.config, args.overrides)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[config-error] {exc}\n")
        return 2

    task = str(cfg.get("task") or "").strip().lower()
    if task != "tennis_multi_cam_3d_pose":
        sys.stderr.write(
            "[usage-error] cfg.task must be 'tennis_multi_cam_3d_pose' for this CLI. "
            "Pass --set task=tennis_multi_cam_3d_pose or use configs/tennis_multi_cam_3d_pose_v2.yaml.\n"
        )
        return 2

    seed_value = cfg.get("seed") or cfg.get("training", {}).get("seed")
    if seed_value is not None:
        seed_everything(int(seed_value), workers=True)

    loader = ConfigLoader(cfg)
    try:
        datamodule = loader.build_datamodule()
        lit_module = loader.build_lit_module()
        logger = loader.build_logger()
        callbacks = loader.build_callbacks()
        trainer = loader.build_trainer(logger=logger, callbacks=callbacks)
    except NotImplementedError as exc:
        sys.stderr.write(f"[train-error] {exc}\n")
        return 2

    trainer.fit(lit_module, datamodule=datamodule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
