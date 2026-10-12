"""Coordinate model calls on decoded inputs, independent of training and metrics."""
from __future__ import annotations

from typing import Any, TypeAlias

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.mdd_pose import (
    MDDPoseAdapter,
    MDDPoseInput,
    prepare_mdd_pose_inputs,
)
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryAdapter, MDDQueryInput
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDPoseDetector,
    MDDQueryDetector,
)

CoordinateModel: TypeAlias = MDDPoseDetector | MDDQueryDetector | DeepMDDQueryDetector


def predict_coordinates(model: CoordinateModel, batch: dict[str, Any], device: torch.device) -> Tensor:
    """Modality and architecture are explicit; coordinate outputs never fabricate heatmaps."""
    if isinstance(model, DeepMDDQueryDetector):
        rgb, times = batch["rgb"], batch["timestamps"]
        if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[1:3] != (32, 3):
            raise ValueError("Deep MDD coordinates require RGB uint8 B,32,3,H,W")
        if times.shape != rgb.shape[:2] or times.dtype != torch.float32:
            raise ValueError("Timestamps must match the RGB window")
        return model(rgb.to(device, non_blocking=True), times.to(device, non_blocking=True))
    if isinstance(model, MDDQueryDetector):
        adapter = MDDQueryAdapter(model.config)
        call = adapter.build_call(MDDQueryInput(batch["rgb"], batch["timestamps"]))
        rgb, timestamps = call.args
        if rgb is None or timestamps is None:
            raise ValueError("Native query requires RGB and timestamps")
        return adapter.decode_output(model(rgb.to(device, non_blocking=True), timestamps.to(device, non_blocking=True)))
    inputs = MDDPoseInput(*(batch[key] for key in ("rgb", "pose", "pose_valid", "timestamps")))
    tensors = prepare_mdd_pose_inputs(model.config, inputs)
    return MDDPoseAdapter(model.config).decode_output(model(*(tensor.to(device, non_blocking=True) for tensor in tensors)))
