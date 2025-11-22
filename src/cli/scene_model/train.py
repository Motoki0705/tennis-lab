"""CLI entrypoint for training the SceneModel on DanceTrack."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.cli.tools.train_utils import run_training_from_config


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
    return run_training_from_config(
        config_path=args.config,
        overrides=args.overrides,
        required_task=None,
        use_explicit_logger=False,
        catch_all_train_errors=True,
        handle_notimplemented_as_usage_error=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
