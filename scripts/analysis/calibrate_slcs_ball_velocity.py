"""Calibrate velocity weight on one fresh-model, train-only minibatch.

CUDA execution requires the shared training queue. No checkpoint is loaded.
"""

import argparse
from pathlib import Path

from src.tasks.slcs.training.velocity_calibration import calibrate_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--velocity-scale-mps", type=float, required=True)
    parser.add_argument("--gradient-ratio", type=float, default=0.1)
    args = parser.parse_args()
    print(calibrate_training_run(**vars(args)))


if __name__ == "__main__":
    main()
