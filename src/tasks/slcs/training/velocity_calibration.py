"""One train-only minibatch calibrating velocity supervision at initialization."""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.data.dataset import SLCSWindowDataset, collate_slcs
from src.tasks.slcs.model_io.factory import create_slcs_model_io
from src.tasks.slcs.training.losses import (
    ball_position_loss_term,
    ball_position_nll_loss_term,
    build_slcs_loss_inputs,
    make_ball_velocity_term,
)
from src.utils.device import resolve_device
from src.utils.io import save_json


def calibrate_gradient_ratio(
    supervised: Tensor, velocity: Tensor, prediction: Tensor, *, ratio: float
) -> dict[str, float]:
    """Set the weighted velocity output-gradient norm to ratio * supervised norm."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("gradient_ratio must be finite and positive")
    if not prediction.requires_grad:
        raise ValueError("prediction must require gradients")
    for name, value in (("supervised", supervised), ("velocity", velocity)):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"{name} loss must be a finite scalar")
    supervised_grad = torch.autograd.grad(supervised, prediction, retain_graph=True)[0]
    velocity_grad = torch.autograd.grad(velocity, prediction)[0]
    supervised_norm = float(torch.linalg.vector_norm(supervised_grad.double()))
    velocity_norm = float(torch.linalg.vector_norm(velocity_grad.double()))
    for name, norm in (("supervised", supervised_norm), ("velocity", velocity_norm)):
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"{name} gradient norm must be finite and positive, got {norm}"
            )
    weight = ratio * supervised_norm / velocity_norm
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("calibrated velocity weight must be finite and positive")
    return {
        "supervised_loss": float(supervised.detach()),
        "velocity_loss": float(velocity.detach()),
        "supervised_gradient_norm": supervised_norm,
        "velocity_gradient_norm": velocity_norm,
        "gradient_ratio": ratio,
        "ball_velocity_weight": weight,
        "weighted_velocity_gradient_norm": weight * velocity_norm,
    }


def _fragment(root: Path, fragment: str, kind: str) -> Path:
    parts = fragment.split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["slcs", kind]
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
        or "\\" in fragment
    ):
        raise ValueError(f"fragment must be slcs/{kind}/<experiment>/<run-id>")
    result = (root / fragment).resolve()
    if not result.is_relative_to(root):
        raise ValueError("fragment must resolve within output_root")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calibrate_training_run(
    *,
    training_run: str,
    output_root: Path,
    output: str,
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
    velocity_scale_mps: float,
    gradient_ratio: float = 0.1,
) -> Path:
    """Fresh initialization, one train batch, no checkpoint or held-out labels."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(velocity_scale_mps) or velocity_scale_mps <= 0:
        raise ValueError("velocity_scale_mps must be finite and positive")
    if not math.isfinite(gradient_ratio) or gradient_ratio <= 0:
        raise ValueError("gradient_ratio must be finite and positive")
    parsed_device = torch.device(device)
    if parsed_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda[:index]")
    if parsed_device.type == "cuda" and not all(
        os.environ.get(name) for name in ("TENNIS_RUN_ID", "TENNIS_REPRO_DIR")
    ):
        raise ValueError(
            "CUDA calibration requires training queue (TENNIS_RUN_ID/TENNIS_REPRO_DIR)"
        )
    if not output_root.is_absolute():
        raise ValueError("output_root must be an explicit absolute path")
    root = output_root.resolve()
    source = _fragment(root, training_run, "train")
    destination = _fragment(root, output, "analyze")
    if destination.exists():
        raise FileExistsError(f"Refusing existing calibration directory: {destination}")
    config_path = (source / "config.yaml").resolve(strict=True)
    if not config_path.is_relative_to(source):
        raise ValueError("config.yaml must resolve within the training run")
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError("Training config must be a mapping")
    runtime = SLCSTrainingRuntimeConfig.from_config(config)
    if runtime.data.overfit or runtime.data.pipeline.on_incomplete != "error":
        raise ValueError("Calibration rejects overfit and skip-incomplete datasets")
    resolved_device = resolve_device(parsed_device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive reservation; failed runs are not overwritten.

    _seed(seed)
    model, adapter, binding = create_slcs_model_io(runtime.model, runtime.data)
    model.to(resolved_device)
    model.train()
    dataset = SLCSWindowDataset(
        dataset_root=runtime.data.dataset_root,
        split_file=runtime.data.split_file,
        split="train",
        config=runtime.data.pipeline,
        stride=runtime.data.pipeline.train_stride,
        augment=True,
    )
    if len(dataset) < batch_size:
        raise ValueError(
            f"Need {batch_size} distinct train windows, found {len(dataset)}"
        )
    selection_rng = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=selection_rng)[
        :batch_size
    ].tolist()
    _seed(seed)  # Augmentation stream is independent of model construction.
    batch = {
        key: value.to(resolved_device)
        for key, value in collate_slcs([dataset[i] for i in indices]).items()
    }
    _seed(seed)  # Dropout stream is independent of augmentation draw count.
    targets = adapter.build_training_targets(batch)
    decoded = binding.run(batch)
    inputs = build_slcs_loss_inputs(decoded, targets)
    supervised = runtime.loss.ball_position_weight * ball_position_loss_term(
        inputs
    ) + runtime.loss.ball_position_nll_weight * ball_position_nll_loss_term(inputs)
    velocity = make_ball_velocity_term(velocity_scale_mps)(inputs)
    measurements = calibrate_gradient_ratio(
        supervised, velocity, decoded.ball_position, ratio=gradient_ratio
    )
    valid = targets.ball_mask & ~targets.padding_mask
    pair = (
        valid[:, 1:]
        & valid[:, :-1]
        & (batch["frame_idx"][:, 1:] - batch["frame_idx"][:, :-1] == 1)
    )
    confidence = torch.minimum(targets.ball_weight[:, 1:], targets.ball_weight[:, :-1])
    report: dict[str, Any] = {
        **measurements,
        "velocity_scale_mps": velocity_scale_mps,
        "seed": seed,
        "rng_order": [
            "seed all; fresh CPU model init; transfer device",
            "independent CPU Generator(seed): randperm train windows without replacement",
            "reseed all; load selected windows with saved training augmentation",
            "reseed all; single training-mode forward (dropout enabled)",
        ],
        "checkpoint_loaded": False,
        "model_training": model.training,
        "split": "train",
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "batch_size": batch_size,
        "train_window_count": len(dataset),
        "selected_indices": indices,
        "selected_windows": [
            {
                "index": i,
                "scene_id": dataset.prediction_ids[i],
                **asdict(dataset.metas[i]),
            }
            for i in indices
        ],
        "valid_pair_count": int(pair.sum()),
        "positive_weight_pair_count": int((pair & (confidence > 0)).sum()),
        "pair_confidence_sum": float(confidence[pair].sum()),
        "training_run": str(source),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training_config": OmegaConf.to_container(config, resolve=True),
        "gradient_target": "pred_ball_position (normalized court XYZ), L2 over entire minibatch",
    }
    result = destination / "calibration.json"
    save_json(report, result)
    return result
