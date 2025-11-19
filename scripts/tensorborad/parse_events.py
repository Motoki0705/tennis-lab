#!/usr/bin/env python
r"""Utility to parse TensorBoard events.out.tfevents files.

This script reads scalar summaries from a specified TensorBoard event file and
exports them to a CSV file for further analysis.

Example:
    python scripts/tensorborad/parse_events.py \
        --event-file runs/Nov19/events.out.tfevents.12345 \
        --output outputs/events_scalars.csv \
        --tags train/loss val/loss

"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Parse a TensorBoard events.out.tfevents file and export scalar "
            "summaries to CSV."
        )
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        required=True,
        help=(
            "Path to a TensorBoard events.out.tfevents file (typically under runs/)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. If omitted, '<event-file>.scalars.csv' will be "
            "used in the same directory."
        ),
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Scalar tags to export. If omitted, all scalar tags found in the "
            "event file will be exported."
        ),
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List available scalar tags in the event file and exit.",
    )
    return parser.parse_args()


def load_event_accumulator(event_file: Path) -> event_accumulator.EventAccumulator:
    """Load a TensorBoard event file.

    Args:
        event_file (Path): Path to the event file.

    Returns:
        event_accumulator.EventAccumulator: An initialized EventAccumulator.

    Raises:
        FileNotFoundError: If the event file does not exist.

    """
    if not event_file.is_file():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    accumulator.Reload()
    return accumulator


def get_scalar_tags(
    accumulator: event_accumulator.EventAccumulator,
    selected_tags: Sequence[str] | None = None,
) -> list[str]:
    """Return scalar tags to use, optionally filtering by selection.

    Args:
        accumulator (event_accumulator.EventAccumulator): Loaded EventAccumulator.
        selected_tags (Sequence[str] | None): Optional list of tags requested by the
            user.

    Returns:
        list[str]: List of scalar tags to use.

    Raises:
        ValueError: If a requested tag does not exist in the event file.

    """
    all_scalar_tags = accumulator.Tags().get("scalars", [])

    if not all_scalar_tags:
        return []

    if selected_tags is None or not selected_tags:
        return list(all_scalar_tags)

    missing_tags = sorted({tag for tag in selected_tags if tag not in all_scalar_tags})
    if missing_tags:
        raise ValueError(
            "Requested tags not found in event file: " + ", ".join(missing_tags)
        )

    return list(selected_tags)


def export_scalars_to_csv(
    accumulator: event_accumulator.EventAccumulator,
    tags: Iterable[str],
    output_path: Path,
) -> None:
    """Export scalar summaries for the given tags to a CSV file.

    Args:
        accumulator (event_accumulator.EventAccumulator): Loaded EventAccumulator.
        tags (Iterable[str]): Iterable of scalar tags to export.
        output_path (Path): Destination CSV file path.

    Returns:
        None: This function does not return anything; it writes to a CSV file.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tag", "step", "wall_time", "value"])

        for tag in tags:
            for scalar in accumulator.Scalars(tag):
                writer.writerow([tag, scalar.step, scalar.wall_time, scalar.value])


def main() -> None:
    """Entry point for the CLI utility."""
    args = parse_args()

    accumulator = load_event_accumulator(args.event_file)
    scalar_tags = get_scalar_tags(accumulator, args.tags)

    if not scalar_tags:
        print(f"No scalar tags found in event file: {args.event_file}")
        return

    if args.list_tags:
        print(f"Scalar tags in {args.event_file}:")
        for tag in scalar_tags:
            print(f"- {tag}")
        return

    if args.output is not None:
        output_path = args.output
    else:
        # e.g. events.out.tfevents.12345 -> events.out.tfevents.12345.scalars.csv
        output_path = args.event_file.parent / f"{args.event_file.name}.scalars.csv"

    export_scalars_to_csv(accumulator, scalar_tags, output_path)

    print(f"Exported scalar summaries for {len(scalar_tags)} tags to: {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
