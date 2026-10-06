"""Plot saved camera coverage on a verified common input sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def common_coverage(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[int], list[int]]:
    if baseline["manifest_sha256"] != candidate["manifest_sha256"]:
        raise ValueError(
            "Cannot compare camera coverage from different input manifests"
        )
    if baseline["input_images"] != candidate["input_images"]:
        raise ValueError("Input counts disagree")
    baseline_input = set(baseline["registered_images"]) | set(
        baseline["missing_images"]
    )
    candidate_input = set(candidate["registered_images"]) | set(
        candidate["missing_images"]
    )
    if (
        baseline_input != candidate_input
        or len(baseline_input) != baseline["input_images"]
    ):
        raise ValueError("Input image names disagree")
    indices = [
        sorted(
            int(Path(name).stem.rsplit("_", 1)[-1])
            for name in audit["registered_images"]
        )
        for audit in (baseline, candidate)
    ]
    return indices[0], indices[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_audit.read_text())
    candidate = json.loads(args.candidate_audit.read_text())
    first, second = common_coverage(baseline, candidate)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 2.8), constrained_layout=True)
    ax.eventplot(
        [first, second],
        lineoffsets=[1, 0],
        linelengths=0.5,
        colors=["#475569", "#2563eb"],
    )
    ax.set_yticks(
        [1, 0],
        [
            f"SIFT: {len(first)}/{baseline['input_images']}",
            f"VidMap: {len(second)}/{candidate['input_images']}",
        ],
    )
    ax.set_xlabel(f"Input frame index (same frozen {baseline['input_images']}-image sequence)")
    ax.set_title(
        "Each mark is a camera pose actually present in the saved reconstruction"
    )
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
