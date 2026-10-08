"""Unique-frame evaluation per FPS, with full and identity-matched common scopes."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.ball_detection.model_io.mdd_pose import (
    MDDPoseAdapter,
    MDDPoseInput,
    prepare_mdd_pose_inputs,
)
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryAdapter, MDDQueryInput
from src.tasks.ball_detection.models.mdd_pose import MDDPoseDetector, MDDQueryDetector
from src.tasks.ball_detection.training.coordinate_compilation import (
    coordinate_compile_scope,
)

CoordinateModel: TypeAlias = MDDPoseDetector | MDDQueryDetector


def predict_coordinates(model: CoordinateModel, batch: dict[str, Any], device: torch.device) -> Tensor:
    if isinstance(model, MDDQueryDetector):
        inputs = MDDQueryInput(batch["rgb"], batch["timestamps"])
        adapter = MDDQueryAdapter(model.config)
        call = adapter.build_call(inputs)
        rgb, timestamps = call.args
        if rgb is None or timestamps is None:
            raise ValueError("Native query requires RGB and timestamps")
        return adapter.decode_output(model(rgb.to(device, non_blocking=True), timestamps.to(device, non_blocking=True)))
    pose_inputs = MDDPoseInput(*(batch[key] for key in ("rgb", "pose", "pose_valid", "timestamps")))
    # Validate the small metadata on CPU before H2D. No value-dependent Python
    # validation enters the compiled model; uint8 RGB needs no full-image scan.
    tensors = prepare_mdd_pose_inputs(model.config, pose_inputs)
    return MDDPoseAdapter(model.config).decode_output(model(*(tensor.to(device, non_blocking=True) for tensor in tensors)))


class CoordinateMetrics:
    def __init__(self, frame_steps: tuple[int, ...]) -> None:
        self.frame_steps = frame_steps
        self.frames: dict[tuple[int, str, int], tuple[tuple[float, int], float, str, bool]] = {}

    def add(self, batch: dict[str, Any], errors: Tensor) -> None:
        for row, clip_id in enumerate(batch["clip_id"]):
            step = int(batch["frame_step"][row])
            if step not in self.frame_steps:
                raise ValueError("Evaluation encountered an undeclared frame step")
            common, source = bool(batch["common_evaluation"][row]), str(batch["source"][row])
            start = int(batch["start"][row])
            for position, frame in enumerate(batch["frame_indices"][row].tolist()):
                if not bool(batch["position_valid"][row, position]):
                    continue
                key = step, clip_id, int(frame)
                owner = abs(position - 15.5), start
                value = float(errors[row, position])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Coordinate error must be finite and nonnegative")
                previous = self.frames.get(key)
                if previous is not None and previous[2:] != (source, common):
                    raise ValueError("Frame has inconsistent source/common evaluation membership")
                if previous is None or owner < previous[0]:
                    self.frames[key] = owner, value, source, common

    def _summary(self, common: bool, source: str | None = None) -> dict[str, Any]:
        rows = [(key, row) for key, row in self.frames.items()
                if (not common or row[3]) and (source is None or row[2] == source)]
        groups = {}
        means = []
        for step in self.frame_steps:
            values = np.asarray([row[1] for key, row in rows if key[0] == step], np.float64)
            group: dict[str, Any] = dict(observed_frames=len(values), available=bool(len(values)))
            for name, value in (("mean_error_px", np.mean(values) if len(values) else None),
                                ("median_error_px", np.median(values) if len(values) else None),
                                ("p95_error_px", np.quantile(values, .95) if len(values) else None)):
                group[name] = float(value) if value is not None else None
            groups[str(step)] = group
            if len(values):
                means.append(group["mean_error_px"])
        return dict(by_frame_step=groups, frame_fps_pairs=len(rows),
                    unique_observed_frames=len({key[1:] for key, _ in rows}),
                    macro_mean_error_px=float(np.mean(means)) if len(means) == len(self.frame_steps) else None)

    def report(self) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("Evaluation has no observed coordinates")
        sources = sorted({row[2] for row in self.frames.values()})
        result = {}
        for scope, common in (("full", False), ("common", True)):
            result[scope] = self._summary(common)
            result[scope]["by_source"] = {source: self._summary(common, source) for source in sources}
        return dict(schema="mdd_coordinate_evaluation.v1", scopes=result,
                    ownership="closest window centre, then earliest start; counted separately for each FPS",
                    aggregate="equal mean of per-FPS mean source-pixel errors; not a pooled duplicate-frame mean")


def selection_score(report: dict[str, Any], scope: str) -> float:
    if scope not in {"full", "common"}:
        raise ValueError("Unknown validation selection scope")
    value = report["scopes"][scope]["macro_mean_error_px"]
    if value is None:
        raise ValueError(f"Validation scope {scope} has an empty configured FPS; no fallback to another scope")
    return float(value)


def evaluate_coordinates(model: CoordinateModel, loader: DataLoader[Any], device: torch.device,
                         frame_steps: tuple[int, ...], *, precision: str = "fp32") -> dict[str, Any]:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Evaluation precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 evaluation requires a supported CUDA device; no fallback")
    model.eval()
    metrics = CoordinateMetrics(frame_steps)
    with torch.no_grad(), coordinate_compile_scope(model):
        for batch in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                uv = predict_coordinates(model, batch, device).cpu()
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            metrics.add(batch, errors)
    return dict(precision=precision, **metrics.report())
