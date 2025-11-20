#!/usr/bin/env python
r"""Collect all TensorBoard events.out.tfevents.* under runs/ and generate CSV + summary.

For each directory containing an ``events.out.tfevents.*`` file, this script:
- Exports scalar summaries to ``<event_file>.scalars.csv`` in the same directory.
- Generates a Markdown summary table ``<event_file>.summary.md`` in the same directory.
- Skips processing if both files already exist for a given event file.

Usage:
    python scripts/tensorboard/collect_and_summarize.py [--runs-dir runs/]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


@dataclass
class TagStats:
    """Running statistics for a single scalar tag."""

    count: int = 0
    first_step: int | None = None
    first_value: float | None = None
    last_step: int | None = None
    last_value: float | None = None
    min_value: float | None = None
    min_step: int | None = None
    max_value: float | None = None
    max_step: int | None = None
    sum_value: float = 0.0
    sum_sq_value: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect all TensorBoard events.out.tfevents.* under runs/ and generate "
            "CSV + Markdown summary. Skips if CSV/summary already exist."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory under which to search for events files (default: runs/).",
    )
    return parser.parse_args()


def _update_stats(stats: TagStats, step: int, value: float) -> None:
    """Update running statistics for a single observation."""
    stats.count += 1

    if stats.count == 1:
        stats.first_step = step
        stats.first_value = value

    stats.last_step = step
    stats.last_value = value

    if stats.min_value is None or value < stats.min_value:
        stats.min_value = value
        stats.min_step = step

    if stats.max_value is None or value > stats.max_value:
        stats.max_value = value
        stats.max_step = step

    stats.sum_value += value
    stats.sum_sq_value += value * value


def load_event_accumulator(event_file: Path) -> event_accumulator.EventAccumulator:
    """Load a TensorBoard event file."""
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
) -> list[str]:
    """Return all scalar tags found in the event file."""
    all_scalar_tags = accumulator.Tags().get("scalars", [])
    return list(all_scalar_tags)


def export_scalars_to_csv(
    accumulator: event_accumulator.EventAccumulator,
    tags: Iterable[str],
    output_path: Path,
) -> None:
    """Export scalar summaries for the given tags to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tag", "step", "wall_time", "value"])

        for tag in tags:
            for scalar in accumulator.Scalars(tag):
                writer.writerow([tag, scalar.step, scalar.wall_time, scalar.value])


def infer_direction(tag: str) -> str:
    """Heuristically infer whether lower or higher is better for a tag."""
    lowered = tag.lower()
    keywords_min = ["loss", "error", "err", "l1", "l2"]
    if any(k in lowered for k in keywords_min):
        return "min"
    return "max"


def iter_summary_rows(
    stats_by_tag: dict[str, TagStats],
) -> Iterable[dict]:
    """Yield per-tag summary rows with derived metrics."""
    for tag in sorted(stats_by_tag.keys()):
        s = stats_by_tag[tag]
        if s.count == 0:
            continue

        mean = s.sum_value / s.count
        var = s.sum_sq_value / s.count - mean * mean
        std = math.sqrt(var) if var > 0.0 else 0.0

        direction = infer_direction(tag)
        if direction == "min":
            best_value = s.min_value
            best_step = s.min_step
        else:
            best_value = s.max_value
            best_step = s.max_step

        yield {
            "tag": tag,
            "count": s.count,
            "first_step": s.first_step,
            "first_value": s.first_value,
            "last_step": s.last_step,
            "last_value": s.last_value,
            "min_value": s.min_value,
            "max_value": s.max_value,
            "mean": mean,
            "std": std,
            "direction": direction,
            "best_value": best_value,
            "best_step": best_step,
        }


def format_markdown_table(rows: Iterable[dict]) -> str:
    """Format summary rows as a Markdown table."""
    rows = list(rows)
    if not rows:
        return "No scalar data to summarize."

    headers = [
        "tag",
        "count",
        "first_step",
        "first_value",
        "last_step",
        "last_value",
        "min",
        "max",
        "mean",
        "std",
        "best",
        "best_step",
        "dir",
    ]

    def row_to_strs(r: dict) -> list[str]:
        return [
            r["tag"],
            f"{r['count']}",
            f"{r['first_step']}",
            f"{r['first_value']:.4g}" if r["first_value"] is not None else "-",
            f"{r['last_step']}",
            f"{r['last_value']:.4g}" if r["last_value"] is not None else "-",
            f"{r['min_value']:.4g}" if r["min_value"] is not None else "-",
            f"{r['max_value']:.4g}" if r["max_value"] is not None else "-",
            f"{r['mean']:.4g}",
            f"{r['std']:.3g}",
            f"{r['best_value']:.4g}" if r["best_value"] is not None else "-",
            f"{r['best_step']}",
            r["direction"],
        ]

    body = [row_to_strs(r) for r in rows]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in body:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)


def load_tag_stats_from_csv(csv_path: Path) -> dict[str, TagStats]:
    """Load scalar CSV and compute per-tag statistics."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    stats_by_tag: dict[str, TagStats] = defaultdict(TagStats)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"tag", "step", "wall_time", "value"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain columns: tag, step, wall_time, value. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            tag = row["tag"]
            if tag is None:
                continue

            try:
                step = int(row["step"])
                value = float(row["value"])
            except (TypeError, ValueError):
                continue

            _update_stats(stats_by_tag[tag], step, value)

    return stats_by_tag


def process_event_file(event_file: Path) -> None:
    """Process a single event file: generate CSV and Markdown summary if missing."""
    csv_path = event_file.with_name(f"{event_file.name}.scalars.csv")
    summary_path = event_file.with_name(f"{event_file.name}.summary.md")

    if csv_path.exists() and summary_path.exists():
        print(f"SKIP: {event_file} (CSV and summary already exist)")
        return

    print(f"Processing: {event_file}")

    accumulator = load_event_accumulator(event_file)
    scalar_tags = get_scalar_tags(accumulator)

    if not scalar_tags:
        print(f"  No scalar tags found in event file: {event_file}")
        return

    # Export CSV if missing
    if not csv_path.exists():
        export_scalars_to_csv(accumulator, scalar_tags, csv_path)
        print(f"  Exported CSV: {csv_path}")
    else:
        print(f"  CSV already exists: {csv_path}")

    # Generate Markdown summary if missing
    if not summary_path.exists():
        stats_by_tag = load_tag_stats_from_csv(csv_path)
        rows = list(iter_summary_rows(stats_by_tag))
        table_md = format_markdown_table(rows)

        summary_md = f"# TensorBoard Summary\n\n**Event file:** `{event_file}`\n\n"
        summary_md += f"**Generated:** {Path.cwd()}\n\n"
        summary_md += "## Scalar Summary\n\n"
        summary_md += table_md + "\n"

        summary_path.write_text(summary_md, encoding="utf-8")
        print(f"  Generated summary: {summary_path}")
    else:
        print(f"  Summary already exists: {summary_path}")


def find_all_event_files(root: Path) -> list[Path]:
    """Recursively find all events.out.tfevents.* files under root."""
    return sorted(root.rglob("events.out.tfevents.*"))


def main() -> None:
    """Entry point for the batch collection CLI."""
    args = parse_args()
    runs_dir = args.runs_dir.resolve()

    if not runs_dir.is_dir():
        print(f"Runs directory not found: {runs_dir}")
        return

    event_files = find_all_event_files(runs_dir)
    if not event_files:
        print(f"No TensorBoard event files found under: {runs_dir}")
        return

    print(f"Found {len(event_files)} event file(s) under: {runs_dir}\n")

    for event_file in event_files:
        try:
            process_event_file(event_file)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  ERROR processing {event_file}: {e}")

    print("\nDone.")


if __name__ == "__main__":  # pragma: no cover
    main()
