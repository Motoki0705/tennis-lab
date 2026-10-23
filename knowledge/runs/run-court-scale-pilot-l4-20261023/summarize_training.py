"""Read exact-update checkpoints and native TensorBoard step timestamps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    events = EventAccumulator(
        str(args.run_root / "logs/version_0"), size_guidance={"scalars": 0}
    )
    events.Reload()
    tags = events.Tags()["scalars"]
    assert "train/loss" in tags
    losses = events.Scalars("train/loss")
    assert len(losses) == args.expected_step
    assert [item.step for item in losses] == list(
        range(losses[0].step, losses[0].step + args.expected_step)
    )
    assert all(math.isfinite(item.value) for item in losses)
    profile = json.loads((args.evidence_root / "training-io.json").read_text())
    command = json.loads((args.evidence_root / "train-timing.json").read_text())
    calls = profile["rclone_calls"]
    assert all(call["success"] for call in calls)
    assert not any(
        call["operation"] in {"delete", "deletefile", "purge"} for call in calls
    )
    deltas: list[dict[str, Any]] = []
    for event_index, (previous, current) in enumerate(
        zip(losses[:-1], losses[1:], strict=True), start=2
    ):
        start, end = previous.wall_time, current.wall_time
        io = sum(
            max(
                0,
                min(end, call["started_unix"] + call["wall_seconds"])
                - max(start, call["started_unix"]),
            )
            for call in calls
        )
        deltas.append(
            {
                "event_step": current.step,
                "event_index": event_index,
                "wall_seconds": end - start,
                "rclone_overlap_seconds": io,
                "excluding_rclone_seconds": end - start - io,
            }
        )
    steady = [item for item in deltas if 6 <= item["event_index"] < args.expected_step]
    assert len(steady) >= 2 and all(
        item["excluding_rclone_seconds"] > 0 for item in steady
    )
    checkpoint = (
        args.run_root
        / f"logs/version_0/checkpoints/update-{args.expected_step:08d}.ckpt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert payload["global_step"] == args.expected_step
    condition = json.loads((args.run_root / "data-condition.json").read_text())
    assert payload["hyper_parameters"]["data_condition"] == condition
    summary = {
        "schema": "court_pilot_training_summary_v1",
        "global_step": payload["global_step"],
        "scheduler_last_epoch": payload["lr_schedulers"][0]["last_epoch"],
        "data_condition": condition,
        "first_logged_update_wall_seconds": losses[0].wall_time
        - datetime.fromisoformat(command["started_at"]).timestamp(),
        "first_update_scope": "Imports, data/model setup, lazy compilation, first update, and its artifact sync; not isolated compilation time",
        "steady_event_steps": [item["event_step"] for item in steady],
        "steady_delta_median_seconds": statistics.median(
            item["wall_seconds"] for item in steady
        ),
        "steady_delta_mean_seconds": statistics.mean(
            item["wall_seconds"] for item in steady
        ),
        "steady_excluding_rclone_median_seconds": statistics.median(
            item["excluding_rclone_seconds"] for item in steady
        ),
        "step_measurement": "Native train/loss event wall-time differences; excludes first five updates and the final interval. Includes loader, model, metrics and logging; rclone overlap is also reported separately.",
        "step_deltas": deltas,
        "command_wall_seconds": command["wall_seconds"],
        "gpu_peak_used_mib": command["gpu_peak_used_mib"],
        "data_preflight": profile["data_preflight"],
        "sync_tree_seconds": [
            call["wall_seconds"] for call in calls if call["operation"] == "copy"
        ],
        "checkpoint_saves": profile["checkpoint_saves"],
        "all_logged_losses_finite": all(
            math.isfinite(item.value)
            for tag in tags
            if tag.startswith("train/loss")
            for item in events.Scalars(tag)
        ),
        "checkpoint_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(checkpoint.parent.glob("*.ckpt"))
        ],
        "scalar_tags": tags,
        "accuracy_interpretation": "Timing/capacity pilot only; 32 updates do not establish convergence or scaling accuracy.",
    }
    assert summary["scheduler_last_epoch"] == args.expected_step
    assert summary["all_logged_losses_finite"]
    target = args.evidence_root / "training-summary.json"
    assert not target.exists()
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "global_step",
                    "steady_delta_median_seconds",
                    "steady_excluding_rclone_median_seconds",
                    "command_wall_seconds",
                    "gpu_peak_used_mib",
                    "all_logged_losses_finite",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
