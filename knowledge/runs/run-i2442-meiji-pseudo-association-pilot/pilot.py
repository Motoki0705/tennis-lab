"""Reproduce the bounded four-clip calibration and three-clip doubles pilot."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.tasks.player_association.pseudo_annotation.calibration import (
    prepare_calibration,
)
from src.tasks.player_association.pseudo_annotation.extract import load_plan
from src.tasks.player_association.pseudo_annotation.frozen_run import FrozenRun
from src.tasks.player_association.pseudo_annotation.job import Packet
from src.tasks.player_association.pseudo_annotation.prepare import prepare_baseline
from src.tasks.player_association.pseudo_annotation.runner import (
    accepted_labels,
    run_annotation,
)
from src.tasks.player_association.pseudo_annotation.sampling import (
    CONFIDENT,
    SINGLE_VIEW,
    STOPPED,
)

CALIBRATION = {
    "video_000/clip_000": 480,
    "video_000/clip_007": 320,
    "video_001/clip_001": 753,
    "video_002/clip_013": 260,
}
DOUBLES = {"clip_020": STOPPED, "clip_032": SINGLE_VIEW, "clip_025": CONFIDENT}


def trial(scope: Path, prepare: Callable[[Path], Packet]) -> None:
    packet, run = scope / "packet", scope / "run"
    if not packet.exists():
        prepare(packet)
    if run.exists():
        accepted_labels(packet, run)
    else:
        receipt = run_annotation(packet, run)
        print(scope, receipt["wall_seconds"], receipt["usage"], flush=True)


def singles(dataset: Path, root: Path) -> None:
    def annotate(clip_id: str, start: int) -> None:
        trial(
            root / "3cam" / clip_id.replace("/", "-"),
            lambda folder: prepare_calibration(
                dataset, clip_id, range(start, start + 12), folder
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(annotate, clip, start) for clip, start in CALIBRATION.items()
        ]
        for future in futures:
            future.result()


def doubles(snapshot: Path, selection: Path, root: Path) -> None:
    frozen = FrozenRun(snapshot)
    plan = load_plan(selection, frozen)

    def annotate(clip_name: str, stratum: str, *, hinted: bool = False) -> None:
        clip = f"video_000/{clip_name}"
        frames = plan["clips"][clip]["sampled"][stratum]
        arm = "hinted" if hinted else "blind"
        trial(
            root / "4cam" / f"{clip_name}-{arm}",
            lambda folder: prepare_baseline(
                frozen, plan, clip, frames, folder, show_prior_association=hinted
            ),
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(annotate, clip, stratum) for clip, stratum in DOUBLES.items()
        ]
        for future in futures:
            future.result()
    annotate("clip_032", SINGLE_VIEW, hinted=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    groups = parser.add_subparsers(dest="trial", required=True)
    groups.add_parser("3cam").add_argument("dataset", type=Path)
    group = groups.add_parser("4cam")
    group.add_argument("snapshot", type=Path)
    group.add_argument("selection", type=Path)
    args = parser.parse_args()
    if args.trial == "3cam":
        singles(args.dataset, args.pilot_root)
    else:
        doubles(args.snapshot, args.selection, args.pilot_root)
