"""Render the CPU target coverage summary without reading any source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary: Path, output: Path) -> None:
    rows = json.loads(summary.read_text())["rows"]
    matplotlib.rcParams["svg.hashsalt"] = "ball-temporal-target-survey"
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for axis, split in zip(axes, ("train", "val", "test"), strict=True):
        for context, label, color in (("clip", "Safe clip context", "#2471a3"), ("play", "Play-only context", "#ba4a00")):
            selected = sorted((r for r in rows if r["split"] == split and r["context"] == context), key=lambda r: r["frames"])
            x = [r["frames"] for r in selected]
            y = [100 * r["observed_coverage_full32"] for r in selected]
            axis.plot(x, y, marker="o", color=color, label=label)
            for length, coverage in zip(x, y, strict=True):
                if length in (128, 256):
                    axis.annotate(f"{coverage:.1f}%", (length, coverage), xytext=(0, 8 if context == "clip" else -14),
                                  textcoords="offset points", ha="center", fontsize=8, color=color)
        axis.set_xscale("log", base=2)
        axis.set_xticks([32, 64, 128, 256], labels=["32", "64", "128", "256"])
        axis.set_ylim(0, 105)
        axis.set_title({"train": "Train", "val": "Validation", "test": "Test"}[split])
        axis.set_xlabel("Input length T (frame_step=1)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Unique observed teacher coverage (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Central 16 targets / all original 32-frame window teachers", fontsize=11)
    figure.tight_layout()
    figure.savefig(output, metadata={"Date": None})
    plt.close(figure)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
