"""Checkpoint-bound evaluation reports and immutable predictions."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.evaluation.evaluator import evaluate
from src.tasks.ball_refiner_3d.evaluation.physics import (
    integrated_metrics,
    physics_metrics,
)
from src.tasks.ball_refiner_3d.model_io.checkpoint import (
    checkpoint_flight_clock,
    load_checkpoint,
)
from src.tasks.ball_refiner_3d.physics.targets import FlightClock
from src.tasks.ball_refiner_3d.visualization.plots import plot_predictions
from src.utils.io import write_json_atomic


def evaluate_checkpoint(
    checkpoint: Path,
    prediction_dir: Path,
    test: list[PreparedRally],
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Bind each saved evaluation to its checkpoint and the fixed test inputs."""

    selected, metadata = load_checkpoint(checkpoint, device)
    if selected.config.physics_heads and checkpoint_flight_clock(
        metadata
    ) != FlightClock.of([r.source.physics for r in test]):
        raise ValueError("Checkpoint flight clock differs from the evaluation data")
    report, predictions = evaluate(
        selected, test, device, seed=seed, batch_size=batch_size, physics=True
    )
    report.update(common_metadata)
    report.update(
        checkpoint_kind=checkpoint.stem,
        checkpoint_step=metadata["step"],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation_rmse=metadata["validation_rmse"],
        gan_weight=metadata["gan_weight"],
        reconstruction_weight=metadata["reconstruction_weight"],
    )
    prediction_dir.mkdir()
    with (prediction_dir / "pred_test.npz").open("xb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **predictions)
    unit = report["unit"]
    metrics = {
        f"test_rmse_{unit}": report["all"]["rmse"],
        f"test_missing_rmse_{unit}": report["missing"]["rmse"],
        f"test_event_rmse_{unit}": report["event"]["rmse"],
        "test_frame_missing_rate": report["frame_missing_rate"],
        "inference_ms_per_frame": report["milliseconds_per_frame"],
        "checkpoint_step": metadata["step"],
        "test_event_brier": report["event_probability"]["brier"],
        "test_event_soft_ce": report["event_probability"]["soft_cross_entropy"],
        **physics_metrics(report["physics"]),
        **(integrated_metrics(report) if selected.config.physics_heads else {}),
    }
    if "best_step" in common_metadata:
        metrics["best_step"] = common_metadata["best_step"]
    write_json_atomic(prediction_dir / "metrics.json", metrics)
    write_json_atomic(prediction_dir / "diagnostic_metrics.json", report)
    plot_predictions(
        predictions,
        prediction_dir / "examples.png",
        dimensions=selected.config.dimensions,
    )
    if os.environ.get("TENNIS_REPRO_DIR"):
        shutil.copytree(
            prediction_dir,
            Path(os.environ["TENNIS_REPRO_DIR"]) / prediction_dir.name,
            dirs_exist_ok=True,
        )
    return metrics
