"""Plot measured per-update training scalars for this short capacity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    events.Reload()
    panels = (
        ("Training total loss", ("train/loss",)),
        (
            "Dense task losses",
            tuple(f"train/loss_{k}" for k in ("kp", "seg", "line", "semantic_line")),
        ),
        ("Pose loss (synthetic training only)", ("train/loss_direct_pose",)),
        ("Learning rate (32-step pilot schedule)", ("lr-AdamW",)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, (title, tags) in zip(axes.flat, panels, strict=True):
        for tag in tags:
            scalars = events.Scalars(tag)
            assert len(scalars) == 32
            ax.plot([v.step for v in scalars], [v.value for v in scalars], label=tag)
        ax.set_title(title)
        ax.set_xlabel("Native TensorBoard step (0 through 31)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        if "loss" in title.lower():
            ax.set_yscale("log")
    fig.suptitle("L4 timing/capacity probe: 32 updates; no convergence conclusion")
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "event_path": args.events.as_posix(),
                "event_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest(),
                "tags": [tag for _, tags in panels for tag in tags],
                "scope": "Per-update training losses and LR only. Real validation loss has a different supervision scope and is not plotted as a matching training objective.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
