"""CLI entry point for the monocular location+rotation trainer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.cli.tools.train_utils import run_training_from_config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level YAML config (e.g. configs/tennis_mono_locrot.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tennis_mono_locrot training pipeline.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments, defaults to ``sys.argv``.

    Returns:
        int: Exit status propagated from the training helper.

    """
    args = _parse_args(argv)
    usage_msg = (
        "[usage-error] cfg.task must be 'tennis_mono_locrot' for this CLI. "
        "Pass --set task=tennis_mono_locrot or use configs/tennis_mono_locrot.yaml.\n"
    )
    return run_training_from_config(
        config_path=args.config,
        overrides=args.overrides,
        required_task="tennis_mono_locrot",
        usage_error_message=usage_msg,
        use_explicit_logger=True,
        catch_all_train_errors=False,
        handle_notimplemented_as_usage_error=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
