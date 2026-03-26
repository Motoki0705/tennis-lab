"""Train a supervised ball detection model with a manual PyTorch loop.

Example commands:
    `uv run python -m src.tasks.ball_detection.scripts.train`
    `uv run python -m src.tasks.ball_detection.scripts.train run.dry_run=true`
    `uv run python -m src.tasks.ball_detection.scripts.train training.trainer.max_epochs=1`
    `uv run python -m src.tasks.ball_detection.scripts.train training.semi_supervised.num_semi_phases=1`

Config entry point: `src/tasks/ball_detection/configs/train.yaml`
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.tasks.ball_detection.data.argumentation import BallDetectionArgumentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import BallDetectionBatch
from src.tasks.ball_detection.models import build_ball_detection_model
from src.tasks.ball_detection.training.losses import BallDetectionFocalLoss
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.ball_detection.training.pseudo_labeling import generate_phase_pseudo_labels

if TYPE_CHECKING:
    from torch.amp.grad_scaler import GradScaler


@dataclass(slots=True)
class CheckpointState:
    """Serializable training state for resume and evaluation.

    Attributes:
        epoch: Last completed epoch.
        phase: Last completed semi-supervised phase.
        phase_epoch: Epoch index within the current phase.
        best_monitor: Best monitored score seen so far.
        best_metrics: Metric dictionary from the best checkpoint epoch.
        history: Per-epoch metric history.
        pseudo_summary: Latest pseudo-label generation summary.
        optimizer_state_dict: Optional optimizer state for resume.
        scheduler_state_dict: Optional scheduler state for resume.
        scaler_state_dict: Optional grad-scaler state for resume.
    """

    epoch: int
    phase: int
    phase_epoch: int
    best_monitor: float
    best_metrics: dict[str, float]
    history: list[dict[str, float]]
    pseudo_summary: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class VisualizationSequence:
    """Contiguous sequence to visualize with labels in resized image space."""

    name: str
    frames_rgb: list[np.ndarray]
    coords_image: list[tuple[float, float]]
    visibility: list[float]


@dataclass(frozen=True, slots=True)
class VisualizationPrediction:
    """One frame prediction for visualization output."""

    confidence: float
    visible: bool
    x_image: float
    y_image: float

def train(config: DictConfig) -> dict[str, float]:
    """Train the detector with supervised or phase-based semi-supervised scheduling."""
    _seed_everything(int(config.run.seed))
    _configure_runtime(config)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(config, output_dir)
    history_path = output_dir / "history.jsonl"

    device = _resolve_device(config)
    val_loader, test_loader = _build_eval_dataloaders(config)

    raw_model = build_ball_detection_model(config).to(device)
    model = _maybe_compile(raw_model, device=device)
    loss_fn = BallDetectionFocalLoss(dict(config.get("loss", {}) or {})).to(device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_enabled = bool(config.training.checkpoint.get("enabled", True))
    save_last = bool(config.training.checkpoint.get("save_last", True))
    state = CheckpointState(
        epoch=0,
        phase=0,
        phase_epoch=0,
        best_monitor=_initial_best_monitor(config),
        best_metrics={},
        history=[],
        pseudo_summary={},
    )
    resume_path = _resolve_resume_path(config)
    if resume_path is not None:
        bootstrap_optimizer = _build_optimizer(config, model)
        bootstrap_scheduler = _build_scheduler(
            config,
            bootstrap_optimizer,
            steps_per_epoch=1,
            max_epochs=1,
            base_lr=float(bootstrap_optimizer.param_groups[0]["lr"]),
        )
        bootstrap_scaler = _build_grad_scaler(config, device=device)
        state = _load_checkpoint(
            path=resume_path,
            model=raw_model,
            optimizer=bootstrap_optimizer,
            scheduler=bootstrap_scheduler,
            scaler=bootstrap_scaler,
            device=device,
        )
        model = _maybe_compile(raw_model, device=device)
        _write_jsonl_history(history_path, state.history)
    elif history_path.exists():
        history_path.unlink()

    if bool(config.run.get("dry_run", False)):
        train_loader = _build_train_dataloader(config)
        dry_metrics = _run_dry_run(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            device=device,
            config=config,
        )
        _write_json(output_dir / "dry_run_metrics.json", dry_metrics)
        return dry_metrics

    phase_plan = _build_phase_plan(config)
    max_epochs = sum(phase_length for _, phase_length in phase_plan)
    check_val_every = max(int(config.training.trainer.get("check_val_every_n_epoch", 1)), 1)
    early_stopping_cfg = config.training.get("early_stopping", {})
    patience = int(early_stopping_cfg.get("patience", 0))
    last_metrics: dict[str, float] = {}
    for phase_index, phase_length in phase_plan:
        phase_checkpoint_dir = _phase_checkpoint_dir(checkpoint_dir, phase_index)
        phase_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        phase_start_epoch = _phase_start_epoch(phase_plan, phase_index)
        phase_end_epoch = phase_start_epoch + phase_length - 1
        phase_completed = state.phase > phase_index or (
            state.phase == phase_index and state.phase_epoch >= phase_length
        )
        if phase_completed:
            continue

        if state.phase == phase_index and state.phase_epoch > 0:
            phase_best_monitor = float(state.best_monitor)
            phase_best_metrics = dict(state.best_metrics)
        else:
            phase_best_monitor = _initial_best_monitor(config)
            phase_best_metrics = {}
            state.best_monitor = phase_best_monitor
            state.best_metrics = dict(phase_best_metrics)
        epochs_without_improvement = 0

        pseudo_manifest_paths: list[Path] = []
        if phase_index > 0:
            pseudo_summary = _ensure_phase_pseudo_labels(
                model=model,
                device=device,
                config=config,
                output_dir=output_dir,
                phase_index=phase_index,
                dry_run=bool(config.run.get("dry_run", False)),
            )
            state.pseudo_summary = dict(pseudo_summary)
            pseudo_manifest_paths.append(Path(str(pseudo_summary["manifest_path"])))

        train_loader = _build_train_dataloader(
            config,
            pseudo_manifest_paths=pseudo_manifest_paths,
        )
        phase_lr = _phase_learning_rate(config, phase_index)
        optimizer = _build_optimizer(config, model, learning_rate=phase_lr)
        scheduler = _build_scheduler(
            config,
            optimizer,
            steps_per_epoch=len(train_loader),
            max_epochs=phase_length,
            base_lr=phase_lr,
        )
        scaler = _build_grad_scaler(config, device=device)
        if state.phase == phase_index and state.epoch >= phase_start_epoch and state.optimizer_state_dict is not None:
            optimizer.load_state_dict(state.optimizer_state_dict)
            if state.scheduler_state_dict is not None:
                scheduler.load_state_dict(state.scheduler_state_dict)
            if scaler is not None and state.scaler_state_dict is not None:
                scaler.load_state_dict(state.scaler_state_dict)

        start_phase_epoch = state.phase_epoch + 1 if state.phase == phase_index else 1
        for phase_epoch in range(start_phase_epoch, phase_length + 1):
            epoch = phase_start_epoch + phase_epoch - 1
            train_metrics = _train_epoch(
                loader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                total_epochs=max_epochs,
            )
            epoch_metrics = dict(train_metrics)
            epoch_metrics["phase"] = float(phase_index)
            epoch_metrics["phase_epoch"] = float(phase_epoch)

            should_validate = (epoch % check_val_every == 0) or epoch == max_epochs
            if should_validate:
                val_metrics = _evaluate(
                    loader=val_loader,
                    model=model,
                    loss_fn=loss_fn,
                    device=device,
                    config=config,
                    prefix="val",
                    total_epochs=max_epochs,
                    epoch=epoch,
                )
                epoch_metrics.update(val_metrics)
                _save_epoch_visualizations(
                    model=model,
                    device=device,
                    config=config,
                    output_dir=output_dir,
                    epoch=epoch,
                    phase_index=phase_index,
                    pseudo_manifest_paths=pseudo_manifest_paths,
                )

            state.history.append({"epoch": float(epoch), **epoch_metrics})
            _append_jsonl(history_path, {"epoch": float(epoch), **epoch_metrics})
            _print_epoch_summary(epoch_metrics, epoch=epoch, total_epochs=max_epochs)
            last_metrics = epoch_metrics

            monitor_name = str(config.training.checkpoint.get("monitor", "val/loss"))
            improved = should_validate and monitor_name in epoch_metrics and _is_improved(
                epoch_metrics[monitor_name],
                phase_best_monitor,
                mode=str(config.training.checkpoint.get("mode", "min")),
                min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
            )
            if improved:
                phase_best_monitor = float(epoch_metrics[monitor_name])
                phase_best_metrics = {key: float(value) for key, value in epoch_metrics.items()}
                epochs_without_improvement = 0
            elif should_validate:
                epochs_without_improvement += 1

            state.phase = phase_index
            phase_should_stop = (
                bool(early_stopping_cfg.get("enabled", False))
                and patience > 0
                and epochs_without_improvement >= patience
            )
            state.phase_epoch = phase_length if phase_should_stop else phase_epoch
            state.epoch = epoch
            state.best_monitor = phase_best_monitor
            state.best_metrics = dict(phase_best_metrics)
            state.optimizer_state_dict = optimizer.state_dict()
            state.scheduler_state_dict = scheduler.state_dict()
            state.scaler_state_dict = scaler.state_dict() if scaler is not None else None

            checkpoint_state = {
                "epoch": state.epoch,
                "phase": state.phase,
                "phase_epoch": state.phase_epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": state.optimizer_state_dict,
                "scheduler_state_dict": state.scheduler_state_dict,
                "scaler_state_dict": state.scaler_state_dict,
                "best_monitor": state.best_monitor,
                "best_metrics": state.best_metrics,
                "history": state.history,
                "pseudo_summary": state.pseudo_summary,
                "config": _to_plain_python(config),
            }
            if checkpoint_enabled and save_last:
                _save_last_checkpoint(phase_checkpoint_dir, checkpoint_state)
            if checkpoint_enabled and should_validate and monitor_name in epoch_metrics:
                checkpoint_state["monitor_value"] = float(epoch_metrics[monitor_name])
                _save_ranked_checkpoint(
                    checkpoint_dir=phase_checkpoint_dir,
                    state=checkpoint_state,
                    epoch=epoch,
                    config=config,
                )

            if phase_should_stop:
                break

    best_path = _select_best_checkpoint(_phase_checkpoint_dir(checkpoint_dir, state.phase))
    if best_path is not None:
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(best_state["model_state_dict"])
        model = _maybe_compile(raw_model, device=device)

    test_metrics = _evaluate(
        loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        device=device,
        config=config,
        prefix="test",
        total_epochs=max_epochs,
        epoch=max_epochs,
    )
    _write_json(output_dir / "test_metrics.json", test_metrics)
    summary = {**last_metrics, **test_metrics}
    if best_path is not None:
        summary["best_checkpoint"] = str(best_path)
    return summary


def _build_train_dataloader(
    config: DictConfig,
    *,
    pseudo_manifest_paths: list[Path] | None = None,
) -> DataLoader[BallDetectionBatch]:
    """Build the phase-specific training dataloader."""
    data_cfg = config.get("data", {}) or {}
    split_cfg = data_cfg.get("split", {}) or {}
    data_dir = data_cfg.get("data_dir", "data/tennis")
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = num_workers > 0

    train_dataset = BallDetectionDataset(
        data_dir=data_dir,
        split_file=str(split_cfg.get("train_file", "train.txt")),
        config=config,
        argumentation=BallDetectionArgumentation(dict(data_cfg.get("augmentation", {}) or {})),
        pseudo_manifest_paths=pseudo_manifest_paths,
    )
    generator = torch.Generator()
    generator.manual_seed(int(config.run.seed))
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
        generator=generator,
    )


def _build_eval_dataloaders(
    config: DictConfig,
) -> tuple[DataLoader[BallDetectionBatch], DataLoader[BallDetectionBatch]]:
    """Build validation and test dataloaders."""
    data_cfg = config.get("data", {}) or {}
    split_cfg = data_cfg.get("split", {}) or {}
    data_dir = data_cfg.get("data_dir", "data/tennis")
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = num_workers > 0

    val_dataset = BallDetectionDataset(
        data_dir=data_dir,
        split_file=str(split_cfg.get("val_file", "val.txt")),
        config=config,
        argumentation=None,
    )
    test_dataset = BallDetectionDataset(
        data_dir=data_dir,
        split_file=str(split_cfg.get("test_file", "test.txt")),
        config=config,
        argumentation=None,
    )
    eval_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
    }
    val_loader: DataLoader[BallDetectionBatch] = DataLoader(val_dataset, **eval_kwargs)
    test_loader: DataLoader[BallDetectionBatch] = DataLoader(test_dataset, **eval_kwargs)
    return val_loader, test_loader


def _build_phase_plan(config: DictConfig) -> list[tuple[int, int]]:
    """Build the phase schedule from config."""
    semi_cfg = config.training.get("semi_supervised", {})
    num_semi_phases = int(semi_cfg.get("num_semi_phases", 0))
    if num_semi_phases <= 0:
        return [(0, int(config.training.trainer.max_epochs))]
    phase0_epochs = int(semi_cfg.get("phase0_epochs", config.training.trainer.max_epochs))
    phase_epochs = int(semi_cfg.get("phase_epochs", config.training.trainer.max_epochs))
    return [(0, phase0_epochs)] + [(phase_idx, phase_epochs) for phase_idx in range(1, num_semi_phases + 1)]


def _phase_start_epoch(phase_plan: list[tuple[int, int]], phase_index: int) -> int:
    """Return the first global epoch index for the requested phase."""
    epoch = 1
    for current_phase, phase_length in phase_plan:
        if current_phase == phase_index:
            return epoch
        epoch += phase_length
    raise ValueError(f"Unknown phase index: {phase_index}")


def _phase_learning_rate(config: DictConfig, phase_index: int) -> float:
    """Resolve the learning rate for one phase."""
    training_cfg = config.training
    semi_cfg = training_cfg.get("semi_supervised", {})
    base_lr = float(training_cfg.get("learning_rate", 1.0e-3))
    min_lr = float(training_cfg.get("min_lr", 1.0e-6))
    phase_lr_decay = float(semi_cfg.get("phase_lr_decay", 1.0))
    return max(base_lr * (phase_lr_decay ** phase_index), min_lr)


def _ensure_phase_pseudo_labels(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    output_dir: Path,
    phase_index: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Generate or reuse pseudo labels for the requested phase."""
    pseudo_root = output_dir / "pseudo_label"
    manifest_path = pseudo_root / f"phase_{phase_index:02d}" / "manifest.jsonl"
    if manifest_path.exists():
        summary_path = manifest_path.parent / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "phase": f"phase_{phase_index:02d}",
            "manifest_path": str(manifest_path),
        }
    return generate_phase_pseudo_labels(
        model=model,
        device=device,
        config=config,
        label_root=pseudo_root,
        phase_index=phase_index,
        dry_run=dry_run,
    )


def _train_epoch(
    *,
    loader: DataLoader[BallDetectionBatch],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    device: torch.device,
    config: DictConfig,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """Run one training epoch."""
    trainer_cfg = config.training.trainer
    max_batches = 1 if bool(config.run.get("fast_dev_run", False)) else None
    grad_clip_val = float(trainer_cfg.get("gradient_clip_val", 0.0))

    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        total=_resolve_progress_total(loader, max_batches=max_batches),
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["heatmaps"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(config, device=device):
            logits = model(_to_model_input(images)).squeeze(1)
            if logits.shape != targets.shape:
                logits = F.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = loss_fn(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_val > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()
        scheduler.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    progress.close()
    mean_loss = total_loss / max(total_samples, 1)
    current_lr = float(optimizer.param_groups[0]["lr"])
    return {
        "train/loss": mean_loss,
        "train/lr": current_lr,
        "epoch": float(epoch),
    }


@torch.no_grad()
def _evaluate(
    *,
    loader: DataLoader[BallDetectionBatch],
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    config: DictConfig,
    prefix: str,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """Evaluate the model on one split."""
    max_batches = 1 if bool(config.run.get("fast_dev_run", False)) else None
    metrics_cfg = config.get("metrics", {}) or {}
    metric = BallDetectionMetrics(
        peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
        ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
    ).to(device)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc=f"{prefix.capitalize()} {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
        total=_resolve_progress_total(loader, max_batches=max_batches),
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["heatmaps"].to(device=device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device=device, dtype=torch.float32, non_blocking=True)
        visibility = batch["visibility"].to(device=device, dtype=torch.float32, non_blocking=True)
        original_size = batch["original_size"].to(device=device, dtype=torch.float32, non_blocking=True)
        heatmap_size = batch["heatmap_size"].to(device=device, dtype=torch.float32, non_blocking=True)

        with _autocast_context(config, device=device):
            logits = model(_to_model_input(images)).squeeze(1)
            if logits.shape != targets.shape:
                logits = F.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = loss_fn(logits, targets)
        probs = torch.sigmoid(logits)
        metric.update(probs, coords, visibility, original_size, heatmap_size)

        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    progress.close()

    values = metric.compute()
    summary = {f"{prefix}/loss": total_loss / max(total_samples, 1)}
    summary.update({f"{prefix}/{name}": float(value.detach().cpu().item()) for name, value in values.items()})
    return summary


def _run_dry_run(
    *,
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[BallDetectionBatch],
    device: torch.device,
    config: DictConfig,
) -> dict[str, float]:
    """Run one forward/backward pass to validate shapes and connectivity."""
    model.train()
    batch = next(iter(train_loader))
    images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
    targets = batch["heatmaps"].to(device=device, dtype=torch.float32, non_blocking=True)

    with _autocast_context(config, device=device):
        logits = model(_to_model_input(images)).squeeze(1)
        if logits.shape != targets.shape:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = loss_fn(logits, targets)
    loss.backward()
    return {
        "dry_run/loss": float(loss.detach().cpu().item()),
        "dry_run/batch_size": float(images.shape[0]),
    }


def _save_epoch_visualizations(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    output_dir: Path,
    epoch: int,
    phase_index: int,
    pseudo_manifest_paths: Sequence[Path],
) -> None:
    """Render qualitative videos for the current validation epoch."""
    vis_cfg = config.training.get("visualization", {}) or {}
    if not bool(vis_cfg.get("enabled", True)):
        return

    epoch_dir = output_dir / "vis" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    train_sequence = _load_split_visualization_sequence(
        config=config,
        split_file=str(config.data.split.get("train_file", "train.txt")),
        name="train",
    )
    _write_visualization_video(
        sequence=train_sequence,
        predictions=_infer_visualization_predictions(
            model=model,
            device=device,
            config=config,
            frames_rgb=train_sequence.frames_rgb,
        ),
        output_path=epoch_dir / "train.mp4",
        config=config,
    )

    val_sequence = _load_split_visualization_sequence(
        config=config,
        split_file=str(config.data.split.get("val_file", "val.txt")),
        name="val",
    )
    _write_visualization_video(
        sequence=val_sequence,
        predictions=_infer_visualization_predictions(
            model=model,
            device=device,
            config=config,
            frames_rgb=val_sequence.frames_rgb,
        ),
        output_path=epoch_dir / "val.mp4",
        config=config,
    )

    if phase_index <= 0 or not pseudo_manifest_paths:
        return
    pseudo_sequence = _load_pseudo_visualization_sequence(
        config=config,
        manifest_path=pseudo_manifest_paths[0],
    )
    if pseudo_sequence is None:
        return
    _write_visualization_video(
        sequence=pseudo_sequence,
        predictions=_infer_visualization_predictions(
            model=model,
            device=device,
            config=config,
            frames_rgb=pseudo_sequence.frames_rgb,
        ),
        output_path=epoch_dir / "pseudo.mp4",
        config=config,
    )


def _load_split_visualization_sequence(
    *,
    config: DictConfig,
    split_file: str,
    name: str,
) -> VisualizationSequence:
    """Load one contiguous sequence from a supervised split for visualization."""
    dataset = BallDetectionDataset(
        data_dir=str(config.data.get("data_dir", "data/tennis")),
        split_file=split_file,
        config=config,
        argumentation=None,
    )
    if not dataset.windows:
        raise RuntimeError(f"No windows available for visualization split={name}.")
    return _window_to_visualization_sequence(
        dataset=dataset,
        window=dataset.windows[0],
        max_frames=int(config.training.visualization.get("num_frames", 126)),
        name=name,
    )


def _load_pseudo_visualization_sequence(
    *,
    config: DictConfig,
    manifest_path: Path,
) -> VisualizationSequence | None:
    """Load one contiguous sequence from the current phase pseudo manifest."""
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        first_record = next((line.strip() for line in handle if line.strip()), "")
    if not first_record:
        return None

    entry = json.loads(first_record)
    image_dir = Path(str(entry["image_dir"])).expanduser()
    label_csv = Path(str(entry["label_csv"])).expanduser()
    frame_count = int(entry["frame_count"])
    original_width = int(entry["original_width"])
    original_height = int(entry["original_height"])

    dataset = BallDetectionDataset(
        data_dir=str(config.data.get("data_dir", "data/tennis")),
        split_file=str(config.data.split.get("train_file", "train.txt")),
        config=config,
        argumentation=None,
    )
    labels = dataset._read_label_csv(label_csv)
    max_frames = min(int(config.training.visualization.get("num_frames", 126)), frame_count)
    frame_names = tuple(f"{frame_index:06d}.jpg" for frame_index in range(frame_count))
    return _build_visualization_sequence(
        dataset=dataset,
        clip_dir=image_dir,
        frame_names=frame_names,
        labels=labels,
        original_size=(original_width, original_height),
        start_index=0,
        max_frames=max_frames,
        name="pseudo",
    )


def _window_to_visualization_sequence(
    *,
    dataset: BallDetectionDataset,
    window: Any,
    max_frames: int,
    name: str,
) -> VisualizationSequence:
    """Convert one dataset window into a longer contiguous visualization sequence."""
    return _build_visualization_sequence(
        dataset=dataset,
        clip_dir=window.clip_dir,
        frame_names=window.frame_names,
        labels=window.labels,
        original_size=window.original_size,
        start_index=window.start_index,
        max_frames=max_frames,
        name=name,
    )


def _build_visualization_sequence(
    *,
    dataset: BallDetectionDataset,
    clip_dir: Path,
    frame_names: Sequence[str],
    labels: dict[str, Any],
    original_size: tuple[int, int],
    start_index: int,
    max_frames: int,
    name: str,
) -> VisualizationSequence:
    """Build a contiguous labeled visualization sequence from raw frame files."""
    image_h, image_w = dataset.image_size
    original_w, original_h = original_size
    end_index = min(len(frame_names), start_index + max(max_frames, dataset.num_frames))
    if end_index - start_index < dataset.num_frames:
        raise RuntimeError(
            f"Visualization sequence {name} is shorter than model.num_frames={dataset.num_frames}."
        )

    frames_rgb: list[np.ndarray] = []
    coords_image: list[tuple[float, float]] = []
    visibility: list[float] = []
    for frame_name in frame_names[start_index:end_index]:
        frame = dataset._load_frame(clip_dir / frame_name)
        frames_rgb.append(frame)
        label = labels.get(frame_name)
        if label is not None and float(label.visibility) > 0:
            coords_image.append(
                (
                    float(label.x) * image_w / max(original_w, 1),
                    float(label.y) * image_h / max(original_h, 1),
                )
            )
            visibility.append(1.0)
        else:
            coords_image.append((0.0, 0.0))
            visibility.append(0.0)
    return VisualizationSequence(
        name=name,
        frames_rgb=frames_rgb,
        coords_image=coords_image,
        visibility=visibility,
    )


@torch.no_grad()
def _infer_visualization_predictions(
    *,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    frames_rgb: Sequence[np.ndarray],
) -> list[VisualizationPrediction]:
    """Infer frame-level predictions by averaging overlapping window heatmaps."""
    num_frames = int(config.model.num_frames)
    if len(frames_rgb) < num_frames:
        raise RuntimeError(
            f"Visualization requires at least {num_frames} frames, got {len(frames_rgb)}."
        )

    heatmap_h = int(config.data.heatmap_size[0])
    heatmap_w = int(config.data.heatmap_size[1])
    image_h = int(config.data.image_size[0])
    image_w = int(config.data.image_size[1])
    batch_size = int(config.training.visualization.get("inference_batch_size", 8))
    threshold = float(
        config.training.visualization.get(
            "confidence_threshold",
            config.metrics.get("peak_threshold", 0.5),
        )
    )

    starts = list(range(0, len(frames_rgb) - num_frames + 1))
    heatmap_sums = np.zeros((len(frames_rgb), heatmap_h, heatmap_w), dtype=np.float32)
    heatmap_counts = np.zeros(len(frames_rgb), dtype=np.int32)

    model.eval()
    for batch_starts in _batched_values(starts, batch_size):
        batch_inputs = []
        for start in batch_starts:
            window = np.stack(
                [
                    frame.transpose(2, 0, 1)
                    for frame in frames_rgb[start : start + num_frames]
                ]
            )
            batch_inputs.append(window)
        inputs = torch.from_numpy(np.stack(batch_inputs)).to(device=device, dtype=torch.float32)
        with _autocast_context(config, device=device):
            logits = model(_to_model_input(inputs)).squeeze(1)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        for batch_index, start in enumerate(batch_starts):
            for offset in range(num_frames):
                frame_index = start + offset
                heatmap_sums[frame_index] += probs[batch_index, offset]
                heatmap_counts[frame_index] += 1

    predictions: list[VisualizationPrediction] = []
    for frame_index in range(len(frames_rgb)):
        support_count = int(heatmap_counts[frame_index])
        if support_count <= 0:
            predictions.append(
                VisualizationPrediction(
                    confidence=0.0,
                    visible=False,
                    x_image=0.0,
                    y_image=0.0,
                )
            )
            continue
        avg = heatmap_sums[frame_index] / float(support_count)
        peak = float(avg.max())
        peak_y, peak_x = np.unravel_index(int(avg.argmax()), avg.shape)
        visible = peak >= threshold
        predictions.append(
            VisualizationPrediction(
                confidence=peak,
                visible=visible,
                x_image=float(peak_x * image_w / max(heatmap_w, 1)) if visible else 0.0,
                y_image=float(peak_y * image_h / max(heatmap_h, 1)) if visible else 0.0,
            )
        )
    return predictions


def _write_visualization_video(
    *,
    sequence: VisualizationSequence,
    predictions: Sequence[VisualizationPrediction],
    output_path: Path,
    config: DictConfig,
) -> None:
    """Write a qualitative MP4 with GT and predicted ball positions."""
    if len(sequence.frames_rgb) != len(predictions):
        raise ValueError("Visualization frames and predictions must have the same length.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_h = int(config.data.image_size[0])
    image_w = int(config.data.image_size[1])
    fps = float(config.training.visualization.get("fps", 12))
    codec = str(config.training.visualization.get("codec", "mp4v"))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (image_w, image_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open visualization writer: {output_path}")

    try:
        for frame_index, (frame_rgb, prediction) in enumerate(zip(sequence.frames_rgb, predictions, strict=True)):
            frame_bgr = cv2.cvtColor(
                np.clip(frame_rgb * 255.0, 0.0, 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            gt_visible = sequence.visibility[frame_index] > 0
            gt_x, gt_y = sequence.coords_image[frame_index]
            if gt_visible:
                cv2.circle(frame_bgr, (int(round(gt_x)), int(round(gt_y))), 5, (0, 255, 0), 2)
            if prediction.visible:
                cv2.circle(
                    frame_bgr,
                    (int(round(prediction.x_image)), int(round(prediction.y_image))),
                    5,
                    (0, 0, 255),
                    2,
                )
            cv2.putText(
                frame_bgr,
                f"{sequence.name} frame={frame_index:03d} conf={prediction.confidence:.2f}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                "GT=green Pred=red",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()


def _batched_values(values: Sequence[int], batch_size: int) -> list[list[int]]:
    """Split integer values into fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return [list(values[index : index + batch_size]) for index in range(0, len(values), batch_size)]


def _build_optimizer(
    config: DictConfig,
    model: nn.Module,
    *,
    learning_rate: float | None = None,
) -> Optimizer:
    """Build the configured optimizer."""
    training_cfg = config.get("training", {}) or {}
    optimizer_cfg = training_cfg.get("optimizer", {}) or {}
    resolved_lr = float(learning_rate if learning_rate is not None else training_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1.0e-4))
    betas_cfg = optimizer_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=resolved_lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return Adam(model.parameters(), lr=resolved_lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer name: {name}")


def _build_scheduler(
    config: DictConfig,
    optimizer: Optimizer,
    *,
    steps_per_epoch: int,
    max_epochs: int | None = None,
    base_lr: float | None = None,
) -> LambdaLR:
    """Build a per-step warmup + cosine scheduler."""
    training_cfg = config.get("training", {}) or {}
    resolved_max_epochs = int(max_epochs if max_epochs is not None else training_cfg.get("trainer", {}).get("max_epochs", 1))
    total_steps = max(steps_per_epoch * resolved_max_epochs, 1)
    warmup_steps = min(max(int(training_cfg.get("warmup_steps", 0)), 0), total_steps - 1)
    resolved_base_lr = float(
        base_lr if base_lr is not None else training_cfg.get("learning_rate", 1.0e-3)
    )
    min_lr = float(training_cfg.get("min_lr", 1.0e-6))
    min_lr_scale = min_lr / max(resolved_base_lr, 1.0e-12)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0e-8)
        if total_steps <= warmup_steps + 1:
            return min_lr_scale
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_grad_scaler(config: DictConfig, *, device: torch.device) -> GradScaler | None:
    """Create a GradScaler only when fp16 CUDA training is active."""
    precision = str(config.training.trainer.get("precision", "") or "").lower()
    if device.type != "cuda" or precision not in {"16", "16-mixed", "fp16", "float16"}:
        return None
    return torch.amp.GradScaler(device="cuda")


def _autocast_context(config: DictConfig, *, device: torch.device) -> Any:
    """Return the autocast context configured for the current device."""
    precision = str(config.training.trainer.get("precision", "") or "").lower()
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resolve_device(config: DictConfig) -> torch.device:
    """Resolve the training device from runtime config."""
    requested_gpus = int(config.run.get("gpus", 1))
    if requested_gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_compile(model: nn.Module, *, device: torch.device) -> nn.Module:
    """Compile the model on CUDA when supported."""
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, backend="inductor")
    except Exception as exc:
        print(f"[compile] falling back to eager mode: {exc}")
        return model


def _to_model_input(images: Tensor) -> Tensor:
    """Convert `(B, T, C, H, W)` images to `(B, C, T, H, W)`."""
    if images.ndim != 5:
        raise ValueError(
            "Expected images with shape (B, T, C, H, W), "
            f"got {tuple(images.shape)}."
        )
    return images.permute(0, 2, 1, 3, 4).contiguous()


def _resolve_progress_total(loader: DataLoader[Any], *, max_batches: int | None) -> int | None:
    """Resolve a stable tqdm total for full and fast-dev runs."""
    try:
        total = len(loader)
    except TypeError:
        return max_batches
    if max_batches is None:
        return total
    return min(total, max_batches)


def _save_last_checkpoint(checkpoint_dir: Path, state: dict[str, Any]) -> None:
    """Persist the rolling last checkpoint."""
    torch.save(state, checkpoint_dir / "last.pt")


def _save_ranked_checkpoint(
    *,
    checkpoint_dir: Path,
    state: dict[str, Any],
    epoch: int,
    config: DictConfig,
) -> None:
    """Save a ranked checkpoint and prune to the configured top-k for one phase."""
    filename_template = str(config.training.checkpoint.get("filename", "ball-detection-{epoch:02d}"))
    checkpoint_path = checkpoint_dir / f"{filename_template.format(epoch=epoch)}.pt"
    torch.save(state, checkpoint_path)

    top_k = int(config.training.checkpoint.get("save_top_k", 1))
    if top_k <= 0:
        return
    mode = str(config.training.checkpoint.get("mode", "min"))
    pattern = "*.pt"
    ranked_paths = [
        path
        for path in checkpoint_dir.glob(pattern)
        if path.name not in {"last.pt", "best.pt", "best_v1.pt"}
    ]
    ranked_paths.sort(
        key=lambda path: _checkpoint_monitor_value(path),
        reverse=(mode == "max"),
    )
    survivors = ranked_paths[:top_k]
    for path in ranked_paths[top_k:]:
        path.unlink(missing_ok=True)

    alias_paths = [checkpoint_dir / "best.pt", checkpoint_dir / "best_v1.pt"]
    for alias_idx, alias_path in enumerate(alias_paths):
        if alias_idx < len(survivors):
            alias_state = torch.load(survivors[alias_idx], map_location="cpu", weights_only=False)
            torch.save(alias_state, alias_path)
        elif alias_path.exists():
            alias_path.unlink(missing_ok=True)


def _select_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the phase-best checkpoint path if it exists."""
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return best_path
    return None


def _checkpoint_monitor_value(path: Path) -> float:
    """Read the stored monitor value from a checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return float(state.get("monitor_value", state.get("best_monitor", float("inf"))))


def _phase_checkpoint_dir(checkpoint_root: Path, phase_index: int) -> Path:
    """Return the checkpoint directory for one phase."""
    return checkpoint_root / f"phase_{phase_index:02d}"


def _load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    device: torch.device,
) -> CheckpointState:
    """Restore model and optimizer state from a manual training checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler_state = state.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = state.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return CheckpointState(
        epoch=int(state.get("epoch", 0)),
        phase=int(state.get("phase", 0)),
        phase_epoch=int(state.get("phase_epoch", 0)),
        best_monitor=float(state.get("best_monitor", float("inf"))),
        best_metrics={str(k): float(v) for k, v in dict(state.get("best_metrics", {})).items()},
        history=[
            {str(k): float(v) for k, v in dict(item).items()}
            for item in list(state.get("history", []))
        ],
        pseudo_summary=dict(state.get("pseudo_summary", {})),
        optimizer_state_dict=dict(state["optimizer_state_dict"]) if "optimizer_state_dict" in state else None,
        scheduler_state_dict=dict(state["scheduler_state_dict"]) if "scheduler_state_dict" in state else None,
        scaler_state_dict=dict(state["scaler_state_dict"]) if state.get("scaler_state_dict") is not None else None,
    )


def _resolve_resume_path(config: DictConfig) -> Path | None:
    """Resolve an optional resume checkpoint path."""
    resume = config.run.get("resume")
    if resume in (None, ""):
        return None
    path = Path(str(resume)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def _initial_best_monitor(config: DictConfig) -> float:
    """Return the initial best monitor sentinel for checkpoint comparison."""
    mode = str(config.training.checkpoint.get("mode", "min"))
    return float("inf") if mode == "min" else float("-inf")


def _is_improved(current: float, best: float, *, mode: str, min_delta: float) -> bool:
    """Check whether the current monitor value improved."""
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible supervised training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_runtime(config: DictConfig) -> None:
    """Apply matmul and TF32 runtime settings from the config."""
    training_cfg = config.get("training", {}) or {}
    matmul_precision = training_cfg.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    if torch.cuda.is_available():
        allow_tf32 = bool(training_cfg.get("allow_tf32", True))
        fp32_mode = "tf32" if allow_tf32 else "ieee"
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = fp32_mode
        else:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = fp32_mode
        elif hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = fp32_mode
        else:
            torch.backends.cudnn.allow_tf32 = allow_tf32


def _save_config_snapshot(config: DictConfig, output_dir: Path) -> None:
    """Persist the resolved Hydra config for later inspection."""
    from omegaconf import OmegaConf

    OmegaConf.save(config=config, f=output_dir / "config.yaml")


def _to_plain_python(value: Any) -> Any:
    """Convert OmegaConf containers and paths into JSON-safe objects."""
    if isinstance(value, DictConfig):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, float] | list[dict[str, float]]) -> None:
    """Write a compact JSON artifact."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, float]) -> None:
    """Append one JSON object to a JSONL history file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_jsonl_history(path: Path, history: list[dict[str, float]]) -> None:
    """Rewrite a JSONL history file from the in-memory history list."""
    with path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _print_epoch_summary(metrics: dict[str, float], *, epoch: int, total_epochs: int) -> None:
    """Print one compact epoch summary line."""
    ordered_keys = [
        "train/loss",
        "train/lr",
        "val/loss",
        "val/f1",
        "val/precision",
        "val/recall",
        "val/mean_distance_px",
    ]
    parts = [f"epoch={epoch}/{total_epochs}"]
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")
    print(" ".join(parts))


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entrypoint for supervised ball detection training."""
    train(config)


if __name__ == "__main__":
    main()
