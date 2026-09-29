"""CPU-only, fair full-frame person source comparison on the four fixed dev clips."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from person_selection_cpu import replay  # type: ignore[import-not-found]  # sibling CLI

from src.tasks.player_detection.evaluation.fullframe_sources import (
    SOURCES,
    summarize_fullframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('sources', 'tracks'))
    parser.add_argument('--ft-progress', type=Path)
    parser.add_argument('--coco-inference', type=Path)
    parser.add_argument('--max-cameras', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.phase == 'sources':
        if args.ft_progress is None or args.coco_inference is None:
            parser.error('sources requires --ft-progress and --coco-inference')
        summarize_fullframe(args.ft_progress, args.coco_inference, args.report)
    else:
        replay(args.report, names=SOURCES, max_cameras=args.max_cameras)


if __name__ == '__main__':
    main()
