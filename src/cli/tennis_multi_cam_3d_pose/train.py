"""CLI entrypoint for training the Tennis Pose stack."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.cli.tools.train_utils import run_training_from_config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the tennis pose trainer.

    Args:
        argv (Sequence[str] | None): Optional sequence of raw CLI arguments.
            If ``None``, defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(description="Train the Tennis Pose stack")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the top-level YAML config (e.g. "
            "configs/tennis_multi_cam_3d_pose.yaml)"
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
    """Run the Tennis Pose training loop using the provided config."""
    args = _parse_args(argv)
    usage_msg = (
        "[usage-error] cfg.task must be 'tennis_multi_cam_3d_pose' for this CLI. "
        "Pass --set task=tennis_multi_cam_3d_pose or use configs/tennis_multi_cam_3d_pose.yaml.\n"
    )
    return run_training_from_config(
        config_path=args.config,
        overrides=args.overrides,
        required_task="tennis_multi_cam_3d_pose",
        usage_error_message=usage_msg,
        use_explicit_logger=True,
        catch_all_train_errors=False,
        handle_notimplemented_as_usage_error=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
