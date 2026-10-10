"""Evaluate a Court checkpoint on synthetic test and real validation separately."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.court_detection.evaluation.ablation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for name in ("data-root", "checkpoint-root", "external-asset-root"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    overrides = {
        name: str(getattr(args, name).resolve())
        for name in ("data_root", "checkpoint_root", "external_asset_root")
        if getattr(args, name) is not None
    }
    evaluate_checkpoint(
        args.checkpoint, args.output, device=args.device, path_overrides=overrides
    )
    print(args.output / "evaluation.json")


if __name__ == "__main__":
    main()
