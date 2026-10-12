"""The shared coordinate epoch: input, optional augmentation, update and logging."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.augmentation.video import VideoAugmenter
from src.tasks.ball_detection.model_io.predict import CoordinateModel
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.tasks.ball_detection.training.runtime.execution import (
    CoordinateRuntime,
    coordinate_train_step,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
)
from src.tasks.ball_detection.training.schedules import FineTuneSchedule
from src.tasks.ball_detection.visualization.rendering.coordinate_preview import (
    save_query_preview,
)


@dataclass(frozen=True)
class EpochResult:
    global_step: int
    mean_loss: float
    seconds: float


def train_epoch(model: CoordinateModel, optimizer: torch.optim.Optimizer, loader: DataLoader[Any],
                device: torch.device, runtime: CoordinateRuntime, *, epoch: int, global_step: int,
                window_budget: int, output: Path, log_every: int,
                augmenter: VideoAugmenter | None = None, schedule: FineTuneSchedule | None = None,
                preview: bool = False) -> EpochResult:
    if schedule is not None:
        if not isinstance(model, DeepMDDQueryDetector):
            raise ValueError("Encoder fine-tuning requires the deep query model")
        model.freeze_encoder(epoch < schedule.freeze_epochs)
    if augmenter is not None and not isinstance(model, DeepMDDQueryDetector):
        raise ValueError("Video augmentation currently requires the pose-free deep query model")
    model.train()
    begin = time.perf_counter()
    previous_end = begin
    loss_sum, observed, windows = 0., 0, 0
    input_wait, reader_wait = 0., 0.
    for updates, batch in enumerate(coordinate_batches(loader, device, prefetch=runtime.image_prefetch), 1):
        waited = time.perf_counter() - previous_end
        input_wait += waited
        reader_wait += batch["_reader_wait_seconds"] if runtime.image_prefetch else waited
        if schedule is not None:
            schedule.apply(optimizer, step=global_step, frozen=epoch < schedule.freeze_epochs)
        audit: dict[str, torch.Tensor] = {}
        if augmenter is not None:
            rgb, batch, audit = augmenter(image_input(batch, device), batch, epoch=epoch)
            batch = dict(batch, rgb=rgb)
        loss, grad, prediction = coordinate_train_step(model, optimizer, batch, device, runtime)
        count = int(batch["position_valid"].sum())
        observed += count
        loss_sum += float(loss) * count
        windows += len(batch["timestamps"])
        global_step += 1
        if updates == 1 and preview:
            target = dict(batch, uv=batch["uv"].detach().cpu(), position_valid=batch["position_valid"].cpu())
            save_query_preview(output / "previews" / f"train-epoch-{epoch:03d}.gif", image_input(batch, device),
                               prediction.cpu(), target, audit.get("artificially_occluded"))
        if global_step % log_every == 0 or windows == window_budget:
            elapsed = time.perf_counter() - begin
            record: dict[str, Any] = dict(epoch=epoch, global_step=global_step, windows=windows,
                train_loss=loss_sum / observed, grad_norm=float(grad), learning_rate=optimizer.param_groups[-1]["lr"],
                train_seconds=elapsed, windows_per_second=windows / elapsed,
                mean_input_ready_wait_seconds=input_wait / updates, mean_reader_wait_seconds=reader_wait / updates,
                input_ready_wait_includes_decode=runtime.image_prefetch)
            if schedule is not None:
                record.update(encoder_frozen=epoch < schedule.freeze_epochs,
                              encoder_learning_rate=optimizer.param_groups[0]["lr"],
                              augmentation={key: value.detach().cpu().tolist() for key, value in audit.items()})
            if device.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({key: value for key, value in record.items() if key != "augmentation"}), flush=True)
        previous_end = time.perf_counter()
    if windows != window_budget or observed == 0:
        raise ValueError("Coordinate epoch did not consume its declared supervised window budget")
    return EpochResult(global_step, loss_sum / observed, time.perf_counter() - begin)
