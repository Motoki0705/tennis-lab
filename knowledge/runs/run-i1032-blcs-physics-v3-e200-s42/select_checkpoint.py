"""Freeze the completed run's validation-only checkpoint choice before test."""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.training_run.resolve(strict=True)
    logs = run / "logs/version_0"
    last_path = logs / "checkpoints/last.ckpt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if last["epoch"] != 199 or last["global_step"] != 9000:
        raise ValueError("Expected a completed 200-epoch, 9000-update run")
    callbacks = [
        state
        for key, state in last["callbacks"].items()
        if str(key).startswith("ModelCheckpoint") and "val/position_error_m" in str(key)
    ]
    if len(callbacks) != 1:
        raise ValueError("Expected exactly one validation-position checkpoint selector")
    state = callbacks[0]
    checkpoint = Path(state["best_model_path"]).resolve(strict=True)
    if checkpoint.parent != last_path.parent or checkpoint == last_path:
        raise ValueError("Selected checkpoint must be a saved best candidate in this run")
    score = float(state["best_model_score"])
    if not math.isfinite(score):
        raise ValueError("Selected validation score is nonfinite")
    del last, state, callbacks
    gc.collect()
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_epoch = int(selected["epoch"])
    selected_step = int(selected["global_step"])
    if not all(torch.isfinite(value).all() for value in selected["state_dict"].values()):
        raise ValueError("Selected checkpoint contains nonfinite tensors")
    del selected
    gc.collect()
    accumulator = EventAccumulator(str(logs), size_guidance={"scalars": 0}).Reload()
    validations = accumulator.Scalars("val/position_error_m")
    if len(validations) != 40:
        raise ValueError("Expected 40 validation measurements")
    best_event = min(validations, key=lambda event: event.value)
    if best_event.step + 1 != selected_step or not math.isclose(best_event.value, score, abs_tol=1e-9):
        raise ValueError("Checkpoint selector and recorded validation minimum disagree")
    if selected_step != (selected_epoch + 1) * 45:
        raise ValueError("Selected epoch and optimizer step disagree")
    epochs = accumulator.Scalars("epoch")
    if int(epochs[-1].value) != 199:
        raise ValueError("TensorBoard does not confirm completion")
    metrics = {}
    for tag in accumulator.Tags()["scalars"]:
        if tag.startswith("val/"):
            series = accumulator.Scalars(tag)
            chosen = [event for event in series if event.step == best_event.step]
            if len(chosen) != 1:
                raise ValueError(f"Missing/duplicate validation metric: {tag}")
            metrics[tag] = {"selected": chosen[0].value, "final": series[-1].value}
    receipt = {
        "source": str(checkpoint),
        "sha256": dual_sha256(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "selected_epoch_zero_based": selected_epoch,
        "selected_optimizer_step": selected_step,
        "completed_epochs": 200,
        "optimizer_updates": 9000,
        "validation_count": len(validations),
        "selection_monitor": "val/position_error_m",
        "selection_mode": "min",
        "selection_score_m": score,
        "selection_split": "val",
        "selection_used_test": False,
        "selection_frozen_at_utc": datetime.now(UTC).isoformat(),
        "last_epoch_finished_utc": datetime.fromtimestamp(epochs[-1].wall_time, UTC).isoformat(),
        "training_commit": "cc863cacdd347027acd0a4700706fec6cca86f81",
        "dataset_manifest_sha256": "36c1c978867016173a32e0f75bae06165bb557bfd981a32db27bec2a291e9544",
        "validation_metrics": metrics,
        "tennis_scene_integration": "deferred by user; checkpoint and knowledge only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
