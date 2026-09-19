"""Compare two saved SLCS condition directories using an explicit train-only speed threshold."""

import argparse
from pathlib import Path

from src.tasks.slcs.evaluation.ball_transition_comparison import (
    save_ball_transition_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fast-speed-mps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(save_ball_transition_comparison(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
