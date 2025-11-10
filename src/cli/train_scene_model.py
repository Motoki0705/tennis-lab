"""CLI entrypoint for training the SceneModel on DanceTrack."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pytorch_lightning.utilities.seed import seed_everything

from src.training.utils.config import ConfigLoader, load_cfg


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SceneModel on DanceTrack")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level YAML config (e.g. configs/scene_model.yaml)",
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
    """CLI entrypoint that wires configs, modules, and the Lightning trainer."""
    args = _parse_args(argv)
    try:
        cfg = load_cfg(args.config, args.overrides)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[config-error] {exc}\n")
        return 2
    loader = ConfigLoader(cfg)
    seed_value = cfg.get("seed") or cfg.get("training", {}).get("seed")
    if seed_value is not None:
        seed_everything(int(seed_value), workers=True)
    try:
        datamodule = loader.build_datamodule()
        module = loader.build_lit_module()
        trainer = loader.build_trainer()
        trainer.fit(module, datamodule=datamodule)
    except Exception as exc:  # pragma: no cover - surfacing errors with context
        sys.stderr.write(f"[train-error] {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
