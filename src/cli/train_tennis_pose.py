"""CLI entrypoint (scaffold) for training the Tennis Pose stack.

P0 provides a safe CLI that loads the config and validates `cfg.task`.
Training hooks are wired in later phases (P1+). This avoids breaking the repo
while establishing a consistent UX with the SceneModel trainer.
"""

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

from src.training.utils.config import load_cfg


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tennis Pose stack (scaffold)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the top-level YAML config (e.g. configs/tennis_pose.yaml)",
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
    """P0 scaffold: Load config, validate task, and exit with guidance.

    In later phases, this mirrors the SceneModel training flow and calls
    ConfigLoader to build the datamodule, module, and trainer.
    """
    args = _parse_args(argv)
    try:
        cfg = load_cfg(args.config, args.overrides)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[config-error] {exc}\n")
        return 2

    task = str(cfg.get("task") or "").strip().lower()
    if task != "tennis_pose":
        sys.stderr.write(
            "[usage-error] cfg.task must be 'tennis_pose' for this CLI. "
            "Pass --set task=tennis_pose or use configs/tennis_pose.yaml.\n"
        )
        return 2

    seed_value = cfg.get("seed") or cfg.get("training", {}).get("seed")
    if seed_value is not None:
        seed_everything(int(seed_value), workers=True)

    # Scaffold milestone message to avoid failing until P1+ is implemented.
    sys.stdout.write(
        "[tennis-pose] P0 scaffold OK: config loaded and task validated.\n"
        "Next steps: implement P1 (geometry/sim), P2 (dataset), P3 (model).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

