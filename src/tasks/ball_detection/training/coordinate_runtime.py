"""Explicit precision and CPU input-pipeline controls for coordinate training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.training.coordinate_compilation import (
    COMPILE_MODES,
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import (
    CoordinateModel,
    predict_coordinates,
)


def coordinate_worker_init(_: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


@dataclass(frozen=True)
class CoordinateRuntime:
    precision: str = "fp32"
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 1
    cpu_threads: int = 2
    compile_mode: str = "off"
    compile_recompile_limit: int = 8

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("Coordinate precision must be fp32 or bf16")
        if self.num_workers < 0 or min(self.prefetch_factor, self.cpu_threads) < 1:
            raise ValueError("Invalid input pipeline worker/thread/prefetch configuration")
        if self.compile_mode not in COMPILE_MODES or self.compile_recompile_limit < 1:
            raise ValueError("Invalid coordinate compilation settings")

    def configure(self, device: torch.device) -> None:
        if self.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 coordinate training requires a supported CUDA device; no fallback")
        if self.pin_memory and device.type != "cuda":
            raise ValueError("Pinned coordinate inputs require CUDA")
        if self.compile_mode != "off" and device.type != "cuda":
            raise ValueError("Coordinate compilation requires CUDA; no eager fallback")
        torch.set_num_threads(self.cpu_threads)
        cv2.setNumThreads(1)
        # Match the measured runtime. Avoid a shape-dependent autotuning phase.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

    def configure_model(self, model: CoordinateModel) -> None:
        compile_coordinate_model(model, mode=self.compile_mode, recompile_limit=self.compile_recompile_limit)

    def loader(self, dataset: CoordinateWindowDataset, *, batch_size: int,
               sampler: Sampler[int] | None = None, seed: int = 0) -> DataLoader[Any]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=collate_coordinate_windows, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, worker_init_fn=coordinate_worker_init,
                          generator=torch.Generator().manual_seed(seed),
                          prefetch_factor=self.prefetch_factor if self.num_workers else None,
                          persistent_workers=self.num_workers > 0)


def coordinate_train_step(model: CoordinateModel, optimizer: torch.optim.Optimizer,
                          batch: dict[str, Any], device: torch.device,
                          runtime: CoordinateRuntime) -> tuple[Tensor, Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with coordinate_compile_scope(model):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=runtime.precision == "bf16"):
            prediction = predict_coordinates(model, batch, device)
            loss = coordinate_loss(prediction, batch["uv"].to(device), batch["position_valid"].to(device))
        loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), grad.detach()
