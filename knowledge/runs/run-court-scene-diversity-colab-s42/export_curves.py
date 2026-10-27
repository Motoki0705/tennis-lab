"""Export final-model fit history without the discarded scene2 attempt tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TAGS = (
    "train/loss",
    "val/loss",
    "val/kp_mean_dist",
    "val/seg_miou",
    "val/line_dice",
    "val/semantic_line_miou",
)
RESUME_UPDATE = 26528


def select_points(
    before: list[list[float]], resumed: list[list[float]] | None = None
) -> tuple[list[list[float]], list[list[float]]]:
    """Inputs use optimizer updates (TensorBoard's zero-based step plus one)."""
    discarded = [] if resumed is None else [p for p in before if p[0] > RESUME_UPDATE]
    retained = (
        before if resumed is None else [p for p in before if p[0] <= RESUME_UPDATE]
    )
    if resumed is not None:
        if any(p[0] < RESUME_UPDATE for p in resumed):
            raise ValueError("Resumed history precedes the restored checkpoint")
        retained = retained + resumed
    points = {int(p[0]): float(p[1]) for p in retained}
    if any(not math.isfinite(value) for value in points.values()):
        raise ValueError("Nonfinite curve value")
    return [[step, value] for step, value in sorted(points.items())], discarded


def load_event(path: Path) -> dict[str, list[list[float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    return {
        tag: [[item.step + 1, float(item.value)] for item in events.Scalars(tag)]
        for tag in TAGS
    }


def event_file(directory: Path) -> Path:
    found = list(directory.glob("events.out.tfevents.*"))
    if len(found) != 1:
        raise ValueError(f"Expected one final immutable event file: {directory}")
    return found[0]


def export(source_root: Path, repository: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_map: dict[str, Any] = {}
    for condition in range(1, 5):
        before_path = (
            source_root / "scene-1-final.tfevents"
            if condition == 1
            else event_file(
                source_root
                / "curve-sources"
                / ("scene-2-before" if condition == 2 else f"scene-{condition}")
            )
        )
        paths = [before_path]
        if condition == 2:
            paths.append(event_file(source_root / "curve-sources/scene-2-resumed"))
        attempts = [load_event(path) for path in paths]
        series, discarded = {}, {}
        for tag in TAGS:
            points, tail = select_points(
                attempts[0][tag], attempts[1][tag] if condition == 2 else None
            )
            series[tag] = points
            if tail:
                discarded[tag] = tail
        if [p[0] for p in series["val/loss"]] != list(range(3316, 33161, 3316)):
            raise ValueError("Validation updates differ from the ten fixed checkpoints")
        sources = [
            {
                "path_relative_to_transfer": str(path.relative_to(source_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "drive_file": (
                    "gdrive:tennis_lab/outputs/court_detection/train/"
                    f"scene-diversity-6000-s384-colab/scene-{condition}-s42/"
                    f"logs/version_{index if condition == 2 else 0}/"
                    + (
                        "events.out.tfevents.1792933574.d6ea976442b5.45568.0"
                        if condition == 1
                        else path.name
                    )
                ),
            }
            for index, path in enumerate(paths)
        ]
        output = (
            repository
            / "knowledge/runs"
            / (f"run-court-scene-diversity-colab-scene-{condition}-s42")
        )
        payload = {
            "schema": "court_scene_diversity_fit_curves_v1",
            "condition": condition,
            "step_conversion": "optimizer_update = tensorboard_step + 1",
            "validation_domain": "fit-time aggregate synthetic500 + real2211",
            "kp_coordinate_units": "post-preprocessing model pixels, longside512",
            "resume_update": RESUME_UPDATE if condition == 2 else None,
            "sources": sources,
            "series": series,
            "discarded_old_attempt_tail": discarded,
            "discarded_tail_is_not_final_model_history": True,
        }
        (output / "curves.json").write_text(json.dumps(payload, indent=2) + "\n")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        for ax, tag in zip(axes.flat, TAGS, strict=True):
            points = series[tag]
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=1)
            if tag.endswith("loss"):
                ax.set_yscale("log")
            if condition == 2:
                ax.axvline(
                    RESUME_UPDATE, color="darkred", linestyle="--", linewidth=0.8
                )
            ax.set(title=tag, xlabel="optimizer updates")
            ax.grid(alpha=0.25)
        fig.suptitle(
            f"Scene {condition}: fit-time aggregate validation (not domain-separated final eval)"
            + (
                "\nResume at 26528; discarded old tail excluded"
                if condition == 2
                else ""
            )
        )
        fig.savefig(output / "curves.png", dpi=110)
        plt.close(fig)
        source_map[str(condition)] = sources
    target = repository / "knowledge/runs/run-court-scene-diversity-colab-s42"
    (target / "curve-sources.json").write_text(json.dumps(source_map, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    export(arguments.source_root.resolve(), arguments.repository.resolve())
