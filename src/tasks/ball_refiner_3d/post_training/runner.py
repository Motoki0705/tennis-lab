"""Frozen-base post-training, validation selection and held-out test prediction."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.inference.segmentation import DEFAULT_PICKING
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner
from src.tasks.ball_refiner_3d.post_training.dataset import (
    Clip,
    PreparedDataset,
    sha256_file,
    verify_file,
)
from src.tasks.ball_refiner_3d.post_training.fusion import (
    FUSION_MODES,
    FusionMode,
    MotionRefiner,
)
from src.tasks.ball_refiner_3d.post_training.labels import (
    MATCH_TOLERANCE,
    UNCERTAIN_RADIUS,
)
from src.tasks.ball_refiner_3d.post_training.motion import (
    COORDINATE_SCALE,
    PLAYER_FEATURES,
)
from src.tasks.ball_refiner_3d.post_training.scoring import (
    Decision,
    report,
    score_clip,
    summary,
)
from src.utils.io import write_json_atomic

FUSION_CHECKPOINT = "ball_refiner_3d.motion_fusion.v1"
FPS_TOLERANCE_HZ = 0.06
TIMING_REPEATS = 3
Mode: TypeAlias = FusionMode | Literal["ball-only"]
Model: TypeAlias = RegressionRefiner | MotionRefiner


@dataclass(frozen=True, slots=True)
class PostTrainingConfig:
    mode: Mode
    seed: int
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.mode not in (*FUSION_MODES, "ball-only")
            or self.seed < 0
            or self.epochs < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not math.isfinite(self.gradient_clip)
            or self.gradient_clip <= 0
        ):
            raise ValueError("Invalid post-training configuration")


class FrameInputs(NamedTuple):
    coordinates: Tensor
    missing: Tensor
    padding: Tensor
    players: Tensor
    valid: Tensor

    @classmethod
    def from_clip(cls, clip: Clip) -> FrameInputs:
        m = clip.motion
        missing = torch.from_numpy(~m.ball_valid[None])
        return cls(
            torch.from_numpy((m.ball_m / COORDINATE_SCALE)[None]),
            missing,
            torch.zeros_like(missing),
            torch.from_numpy(m.players[None]),
            torch.from_numpy(m.root_valid[None]),
        )

    def to(self, device: torch.device, *, with_motion: bool) -> FrameInputs:
        return FrameInputs(
            self.coordinates.to(device),
            self.missing.to(device),
            self.padding.to(device),
            self.players.to(device) if with_motion else self.players,
            self.valid.to(device) if with_motion else self.valid,
        )

    def forward(self, model: Model) -> RefinerOutput:
        if isinstance(model, MotionRefiner):
            output: RefinerOutput = model(*self)
        else:
            output = model(self.coordinates, self.missing, self.padding)
        return output


@dataclass(frozen=True, slots=True)
class TrainingClip:
    clip: Clip
    frames: FrameInputs
    target: Tensor
    valid: Tensor

    @classmethod
    def prepare(cls, clip: Clip, sigma: float) -> TrainingClip:
        target, valid = clip.labels.targets(sigma)
        if not valid.any():
            raise ValueError(f"No supervised frame/class pairs for {clip.clip_id}")
        return cls(
            clip,
            FrameInputs.from_clip(clip),
            torch.from_numpy(target),
            torch.from_numpy(valid),
        )

    def loss(self, logits: Tensor) -> Tensor:
        valid = self.valid.to(logits.device)
        target = self.target.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits[0][valid], target[valid])


def _restore_fusion(model: MotionRefiner, path: Path, identity: dict[str, Any]) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != FUSION_CHECKPOINT or payload["identity"] != identity:
        raise ValueError(
            "Fusion weights do not match the base, features and run settings"
        )
    model.fusion.load_state_dict(payload["fusion"], strict=True)
    return int(payload["epoch"])


def _save_fusion(
    model: MotionRefiner, path: Path, identity: dict[str, Any], epoch: int
) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "schema": FUSION_CHECKPOINT,
            "identity": identity,
            "epoch": epoch,
            "fusion": model.fusion.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _validate_fps(clips: list[Clip], checkpoint_fps: float) -> None:
    for clip in clips:
        tolerance = FPS_TOLERANCE_HZ + math.ulp(max(clip.fps, checkpoint_fps))
        if abs(clip.fps - checkpoint_fps) > tolerance:
            raise ValueError(f"{clip.clip_id}: clip FPS differs from the checkpoint")


def _validation(
    model: MotionRefiner, clips: list[TrainingClip], device: torch.device
) -> tuple[float, float]:
    model.eval()
    decisions: list[Decision] = []
    losses = []
    with torch.inference_mode():
        for item in clips:
            logits = (
                item.frames.to(device, with_motion=True).forward(model).event_logits
            )
            losses.append(float(item.loss(logits)))
            estimates = logits[0].sigmoid().cpu().numpy()
            decisions.extend(score_clip(estimates, item.clip.labels, item.clip.fps))
    macro_f1 = summary(decisions)["macro_f1"]
    loss = float(np.mean(losses))
    if macro_f1 is None or not math.isfinite(loss):
        raise ValueError("Validation needs finite loss and at least one scored event")
    return float(macro_f1), loss


def _fit(
    model: MotionRefiner,
    train: list[TrainingClip],
    val: list[TrainingClip],
    settings: PostTrainingConfig,
    device: torch.device,
    output: Path,
    identity: dict[str, Any],
) -> int:
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    shuffle = np.random.default_rng(settings.seed)
    history = []
    best = (-math.inf, -math.inf)
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses = []
        for index in shuffle.permutation(len(train)):
            item = train[index]
            optimizer.zero_grad(set_to_none=True)
            logits = (
                item.frames.to(device, with_motion=True).forward(model).event_logits
            )
            loss = item.loss(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.fusion.parameters(),
                settings.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            train_losses.append(float(loss.detach()))
        val_f1, val_bce = _validation(model, val, device)
        score = (val_f1, -val_bce)
        if score > best:
            best = score
            _save_fusion(model, output / "best.pt", identity, epoch)
        row = {
            "epoch": epoch,
            "updates": epoch * len(train),
            "train_bce": float(np.mean(train_losses)),
            "val_bce": val_bce,
            "val_macro_f1": val_f1,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        write_json_atomic(
            output / "history.json",
            {
                "manifest_sha256": identity["manifest_sha256"],
                "epochs": history,
            },
        )
        print(row, flush=True)
    return _restore_fusion(model, output / "best.pt", identity)


def _measure(
    model: Model, inputs: FrameInputs, device: torch.device
) -> tuple[RefinerOutput, list[float], int | None]:
    inputs.forward(model)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms = []
    result: RefinerOutput
    for repetition in range(TIMING_REPEATS):
        start = time.perf_counter()
        result = inputs.forward(model)
        if cuda:
            torch.cuda.synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1_000)
        if repetition + 1 < TIMING_REPEATS:
            del result
    peak = torch.cuda.max_memory_allocated(device) if cuda else None
    return result, elapsed_ms, peak


def _test(
    model: Model,
    clips: list[Clip],
    device: torch.device,
    output: Path,
    manifest_sha256: str,
    prepared_sha256: str,
) -> dict[str, Any]:
    model.eval()
    predictions = output / "predictions"
    predictions.mkdir()
    primary: list[Decision] = []
    independent: list[Decision] = []
    matches, files = {}, {}
    measurements: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for clip in clips:
            motion = clip.motion
            inputs = FrameInputs.from_clip(clip).to(
                device, with_motion=isinstance(model, MotionRefiner)
            )
            result, times, peak = _measure(model, inputs, device)
            coordinates = result.coordinates[0].cpu().numpy() * COORDINATE_SCALE
            logits = result.event_logits[0].cpu().numpy()
            probability = result.event_probability[0].cpu().numpy()
            del result, inputs
            if not np.isfinite(coordinates).all() or not np.isfinite(logits).all():
                raise ValueError(f"Nonfinite predictions for {clip.clip_id}")
            path = predictions / f"{clip.clip_id.replace('/', '--')}.npz"
            np.savez(
                path,
                clip_id=clip.clip_id,
                fps=clip.fps,
                frame=np.arange(len(motion.ball_valid), dtype=np.int32),
                manifest_sha256=manifest_sha256,
                prepared_sha256=prepared_sha256,
                event_classes=EVENT_CLASSES,
                event_probability=probability,
                event_logits=logits,
                coordinates_m=coordinates,
                ball_m=motion.ball_m,
                ball_valid=motion.ball_valid,
                root_valid=motion.root_valid,
                smpl_valid=motion.smpl_valid,
            )
            files[clip.clip_id] = {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
            measured = {"elapsed_ms": times, "peak_cuda_allocated_bytes": peak}
            measurements[clip.clip_id] = measured
            rows = score_clip(probability, clip.labels, clip.fps, motion)
            primary.extend(rows)
            matches[clip.clip_id] = {"annotation": rows}
            if clip.reference is not None:
                rows = score_clip(probability, clip.reference, clip.fps, motion)
                independent.extend(rows)
                matches[clip.clip_id]["independent"] = rows
    write_json_atomic(
        output / "matches.json",
        {
            "manifest_sha256": manifest_sha256,
            "prepared_sha256": prepared_sha256,
            "predictions": files,
            "clips": matches,
        },
    )
    medians = [float(np.median(row["elapsed_ms"])) for row in measurements.values()]
    peaks = [
        int(row["peak_cuda_allocated_bytes"])
        for row in measurements.values()
        if row["peak_cuda_allocated_bytes"] is not None
    ]
    return {
        "annotation": report(primary),
        "independent": report(independent),
        "timing": {
            "median_clip_ms": float(np.median(medians)),
            "sum_clip_medians_ms": sum(medians),
            "max_cuda_allocated_bytes": max(peaks, default=None),
            "clips": measurements,
        },
    }


def run_post_training(
    dataset: PreparedDataset,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    settings: PostTrainingConfig,
    device: torch.device,
    code_commit: str,
) -> dict[str, Any]:
    """Train with train/val, then freeze the selected weights before scoring test."""
    verify_file(checkpoint, checkpoint_sha256)
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(True)
    base, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    if not isinstance(base, RegressionRefiner) or base.config.physics_heads:
        raise ValueError(
            "Require a regression ball-only checkpoint without physics heads"
        )
    model: Model = (
        base if settings.mode == "ball-only" else MotionRefiner(base, settings.mode)
    )
    model.to(device)
    sigma = float(metadata["event_sigma_frames"])
    checkpoint_fps = float(metadata["fps"])
    if not math.isfinite(checkpoint_fps) or checkpoint_fps <= 0:
        raise ValueError("Checkpoint FPS must be positive and finite")
    manifest_sha256 = dataset.inputs["manifest_sha256"]
    identity = {
        "manifest_sha256": manifest_sha256,
        "prepared_sha256": dataset.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "settings": asdict(settings),
        "player_features": PLAYER_FEATURES,
        "event_classes": list(EVENT_CLASSES),
    }
    output.mkdir(parents=True, exist_ok=False)
    run = {
        **identity,
        "code_commit": code_commit,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "event_sigma_frames": sigma,
        "checkpoint_fps": checkpoint_fps,
        "fps_tolerance_hz": FPS_TOLERANCE_HZ,
        "picking": asdict(DEFAULT_PICKING),
        "tolerance_frames": MATCH_TOLERANCE,
        "uncertain_radius_frames": UNCERTAIN_RADIUS,
        "timing_repeats": TIMING_REPEATS,
        "inputs": str(dataset.root / "inputs.json"),
        "checkpoint": str(checkpoint),
        "splits": dataset.inputs["splits"],
        "model_config": asdict(base.config),
        "state": "started",
    }
    write_json_atomic(output / "run.json", run)
    best_epoch = None
    if isinstance(model, MotionRefiner):
        train, val = dataset.split("train"), dataset.split("val")
        _validate_fps(train + val, checkpoint_fps)
        best_epoch = _fit(
            model,
            [TrainingClip.prepare(clip, sigma) for clip in train],
            [TrainingClip.prepare(clip, sigma) for clip in val],
            settings,
            device,
            output,
            identity,
        )
        model.zero_grad(set_to_none=True)
        del train, val
    test = dataset.split("test")
    _validate_fps(test, checkpoint_fps)
    base_parameters = sum(p.numel() for p in base.parameters())
    metrics = {
        **_test(model, test, device, output, manifest_sha256, dataset.sha256),
        "manifest_sha256": manifest_sha256,
        "prepared_sha256": dataset.sha256,
        "best_epoch": best_epoch,
        "base_parameters": base_parameters,
        "added_parameters": sum(p.numel() for p in model.parameters())
        - base_parameters,
        "fusion_checkpoint_sha256": None
        if best_epoch is None
        else sha256_file(output / "best.pt"),
    }
    write_json_atomic(output / "metrics.json", metrics)
    write_json_atomic(
        output / "run.json", {**run, "state": "complete", "best_epoch": best_epoch}
    )
    return metrics
