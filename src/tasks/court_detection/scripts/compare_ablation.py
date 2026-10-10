"""Create matched quantitative tables and same-image qualitative comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.court_detection.evaluation.comparison import compare_evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(compare_evaluations(args.evaluations, args.output))


if __name__ == "__main__":
    main()
