"""Render the same evaluated clip range across explicit model/T selections."""

import argparse
from pathlib import Path

from src.tasks.ball_detection.visualization.presentation.temporal_heatmap import (
    render_temporal_heatmap_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=480)
    args = parser.parse_args()
    render_temporal_heatmap_comparison(args.evaluations, clip_id=args.clip_id, start=args.start, stop=args.stop,
                                      output=args.output, panel_width=args.panel_width)


if __name__ == "__main__":
    main()
