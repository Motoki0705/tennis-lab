"""Validated coordinate-model boundary, distinct from ConvNeXt heatmap IO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd_pose.config import MDDPoseConfig
from src.tasks.ball_detection.models.mdd_pose.model import MDDPoseDetector
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io

from .mdd_coordinates import decode_coordinates, validate_mdd_timestamps


@dataclass(frozen=True)
class MDDPoseInput:
    mdd: Tensor
    coordinates: Tensor
    valid: Tensor
    timestamps: Tensor


def prepare_mdd_pose_inputs(config: MDDPoseConfig, inputs: MDDPoseInput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not config.requires_pose:
        raise ValueError("Pose input is not accepted by query_only; use MDDQueryInput")
    mdd, coordinates, valid, timestamps = inputs.mdd, inputs.coordinates, inputs.valid, inputs.timestamps
    validate_mdd_timestamps(config, mdd, timestamps)
    b, _, t, _, _ = mdd.shape
    if coordinates.ndim != 5 or coordinates.shape[:2] != (b, t) or coordinates.shape[-2:] != (17, 2):
        raise ValueError("Expected aligned 32-frame MDD and COCO17 pose windows")
    if valid.dtype != torch.bool or valid.shape != coordinates.shape[:-1] or timestamps.shape != (b, t):
        raise ValueError("Invalid pose mask or timestamp shape")
    if any(x.device != mdd.device for x in (coordinates, valid, timestamps)):
        raise ValueError("All coordinate-model inputs must share one device")
    if coordinates.dtype != torch.float32 or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Coordinate-model inputs require finite float32 values")
    return mdd, coordinates, valid, timestamps


class MDDPoseAdapter:
    def __init__(self, config: MDDPoseConfig) -> None:
        if not config.requires_pose:
            raise ValueError("MDDPoseAdapter requires pose conditioning")
        self.config = config

    @property
    def model_type(self) -> type[nn.Module]:
        return MDDPoseDetector

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, MDDPoseDetector) or model.config != self.config:
            raise ValueError("MDD+pose model/adapter configuration mismatch")

    def build_call(self, inputs: MDDPoseInput) -> ModelCall:
        return ModelCall(args=prepare_mdd_pose_inputs(self.config, inputs))

    def decode_output(self, output: Tensor) -> Tensor:
        return decode_coordinates(self.config, output)


def build_mdd_pose_detector(config: MDDPoseConfig) -> BoundModelIO[MDDPoseInput, Tensor, Tensor]:
    return bind_model_io(MDDPoseDetector(config), MDDPoseAdapter(config))
