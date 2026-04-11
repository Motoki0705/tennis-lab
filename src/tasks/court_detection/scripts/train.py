"""Unified court detection training script.

Supports three tasks via ``--task``:

* ``seg`` — Court cell segmentation (CE + Dice, 7 classes).
* ``kp``  — Court keypoint heatmap regression (Focal BCE, 14 channels).
* ``line`` — Court white-line segmentation (BCE + Dice, 1 channel).

Usage::

    # Segmentation
    python -m src.tasks.court_detection.scripts.train --task seg

    # Keypoint
    python -m src.tasks.court_detection.scripts.train --task kp

    # White-line segmentation
    python -m src.tasks.court_detection.scripts.train --task line

    # Dry run
    python -m src.tasks.court_detection.scripts.train --task seg --dry-run
"""

from __future__ import annotations

import argparse

from src.tasks.court_detection.training.train import train_kp, train_line, train_seg


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train a court detection model.")
    parser.add_argument(
        "--task", required=True, choices=["seg", "kp", "line"],
        help="Task to train: 'seg', 'kp', or 'line'.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check.")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path.")
    parser.add_argument("--save-train-vis", action="store_true", help="Save train visualisations.")
    parser.add_argument("--save-vis-every", type=int, default=5, help="Save vis every N epochs.")
    parser.add_argument("--save-vis-max-samples", type=int, default=8, help="Max vis samples.")
    args = parser.parse_args()

    kwargs = {
        "dry_run": args.dry_run,
        "resume": args.resume,
        "save_train_vis": args.save_train_vis,
        "save_vis_every": args.save_vis_every,
        "save_vis_max_samples": args.save_vis_max_samples,
    }

    if args.task == "seg":
        results = train_seg(**kwargs)
    elif args.task == "line":
        results = train_line(**kwargs)
    else:
        results = train_kp(**kwargs)

    print(f"[train] Results: {results}")


if __name__ == "__main__":
    main()
