"""Validated coordinate-model boundary, distinct from ConvNeXt heatmap IO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDPoseDetector
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io


@dataclass(frozen=True)
class MDDPoseInput:
    mdd: Tensor
    coordinates: Tensor
    valid: Tensor
    timestamps: Tensor


def prepare_mdd_pose_inputs(config: MDDPoseConfig, inputs: MDDPoseInput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mdd, coordinates, valid, timestamps = inputs.mdd, inputs.coordinates, inputs.valid, inputs.timestamps
    if mdd.ndim != 5 or mdd.shape[0] < 1 or mdd.shape[1] != 2:
        raise ValueError("MDD requires B,2,T,H,W")
    b, _, t, h, w = mdd.shape
    if min(h, w) < 8 or h % 8 or w % 8:
        raise ValueError("MDD spatial sizes must be divisible by eight")
    if t != config.frames or coordinates.ndim != 5 or coordinates.shape[:2] != (b, t) or coordinates.shape[-2:] != (17, 2):
        raise ValueError("Expected aligned 32-frame MDD and COCO17 pose windows")
    if valid.dtype != torch.bool or valid.shape != coordinates.shape[:-1] or timestamps.shape != (b, t):
        raise ValueError("Invalid pose mask or timestamp shape")
    if any(x.device != mdd.device for x in (coordinates, valid, timestamps)):
        raise ValueError("All coordinate-model inputs must share one device")
    if any(x.dtype != torch.float32 or not bool(torch.isfinite(x).all()) for x in (mdd, coordinates, timestamps)):
        raise ValueError("Coordinate-model inputs require finite float32 values")
    if bool(((mdd < 0) | (mdd > 1)).any()):
        raise ValueError("MDD sigmoid features must be in [0,1]")
    if bool((timestamps.diff(dim=1) <= 0).any()):
        raise ValueError("Real timestamps must increase strictly")
    return mdd, coordinates, valid, timestamps


class MDDPoseAdapter:
    def __init__(self, config: MDDPoseConfig) -> None:
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
        if output.ndim != 3 or output.shape[1:] != (self.config.frames, 2) or not bool(torch.isfinite(output).all()):
            raise ValueError("Invalid per-frame coordinate output")
        return output


def build_mdd_pose_detector(config: MDDPoseConfig) -> BoundModelIO[MDDPoseInput, Tensor, Tensor]:
    return bind_model_io(MDDPoseDetector(config), MDDPoseAdapter(config))
