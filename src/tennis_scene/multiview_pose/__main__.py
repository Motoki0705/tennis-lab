"""Run framewise 3D pose on saved, synchronized all-person observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.multiview_pose.inputs import PinnedCourtSide, read_saved_inputs
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.storage import publish_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-store",
        required=True,
        type=Path,
        help="Saved DINO, ViTPose and local calibration store",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Separate output clip store"
    )
    parser.add_argument(
        "--side-store",
        required=True,
        type=Path,
        help="Ball-based video court side store",
    )
    parser.add_argument(
        "--side-digest", required=True, help="Pinned side decision SHA-256"
    )
    parser.add_argument(
        "--court-store",
        required=True,
        type=Path,
        help="Shared court store named by the pinned side",
    )
    parser.add_argument(
        "--drift-store",
        type=Path,
        help="Drift store when the pinned decision names drift evidence",
    )
    args = parser.parse_args()
    side = PinnedCourtSide.load(
        args.side_store, args.side_digest, args.court_store, args.drift_store
    )
    inputs = read_saved_inputs(args.clip_store, side)
    result = reconstruct(inputs.observations, DEFAULT_CONFIG)
    ref = publish_candidates(
        args.output, args.clip_store, inputs, result, DEFAULT_CONFIG
    )
    print(args.output / ref.path)


if __name__ == "__main__":
    main()
