"""Reconstruct Meiji players with fixed camera and ball artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from prepare import ReplayRun, prepare_clip

from src.tennis_scene.generate_dataset.manifest import (
    DatasetClipRecord,
    load_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("cpu", "cpu-report", "gpu", "publish", "verify", "restore")
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--comparison", type=Path, required=True, help="#2440 replay directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clip", action="append", help="A clip_XXX name. With no --clip, use all 69."
    )
    parser.add_argument(
        "--workers", type=int, choices=(1, 2), default=1, help="CPU phase only"
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve(strict=True)
    manifest = load_dataset_manifest(dataset)
    names = tuple(f"clip_{i:03d}" for i in range(69))
    keys = tuple(f"video_000/{name}" for name in names)
    if manifest.dataset_id != "meiji_4cam" or set(manifest.clips) != set(keys):
        parser.error("dataset must be the complete i2363-all-v1 Meiji 4cam video")
    if args.clip and (
        set(args.clip) - set(names) or len(set(args.clip)) != len(args.clip)
    ):
        parser.error("--clip requires distinct clip names from the archived video")
    records = [
        manifest.clips[key]
        for key in keys
        if not args.clip or manifest.clips[key].clip_name in args.clip
    ]
    run = ReplayRun.bind(dataset, args.output.resolve(), args.comparison.resolve())
    if args.phase != "cpu":
        from finish import complete_phase

        complete_phase(args.phase, run, records)
        return

    run_cpu(run, records, args.workers)


def run_cpu(run: ReplayRun, records: list[DatasetClipRecord], workers: int) -> None:
    errors: list[Exception] = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        pending = {
            executor.submit(prepare_clip, run, record): record.clip_id
            for record in records
        }
        for future in as_completed(pending):
            try:
                print_result(future.result())
            except Exception as error:
                error.add_note(f"Clip: {pending[future]}")
                print(f"{pending[future]} CPU failed: {error}", flush=True)
                errors.append(error)
    if errors:
        raise ExceptionGroup("CPU reconstruction failed for one or more clips", errors)


def print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['clip_id']} {result['mode']} "
        f"{result['before']['joint_player_frames']} -> {result['cpu_joint_player_frames']} "
        f"joint player-frames ({result['seconds']['cpu']:.2f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
