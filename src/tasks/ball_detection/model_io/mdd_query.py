"""Typed RGB input to the native MDD-only model; pose is never required."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDQueryDetector
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io

from .mdd_coordinates import decode_coordinates, validate_rgb_timestamps


@dataclass(frozen=True)
class MDDQueryInput:
    rgb: Tensor
    timestamps: Tensor


class MDDQueryAdapter:
    def __init__(self, config: MDDPoseConfig) -> None:
        if config.requires_pose:
            raise ValueError("MDDQueryAdapter requires query_only configuration")
        self.config = config

    @property
    def model_type(self) -> type[nn.Module]:
        model_type: type[nn.Module] = MDDQueryDetector
        return model_type

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, MDDQueryDetector) or model.config != self.config:
            raise ValueError("MDD-only model/adapter configuration mismatch")

    def build_call(self, inputs: MDDQueryInput) -> ModelCall:
        validate_rgb_timestamps(self.config, inputs.rgb, inputs.timestamps)
        return ModelCall(args=(inputs.rgb, inputs.timestamps))

    def decode_output(self, output: Tensor) -> Tensor:
        return decode_coordinates(self.config, output)


def build_mdd_query_detector(config: MDDPoseConfig, *, mdd_a: float = .2, mdd_b: float = .15) -> BoundModelIO[MDDQueryInput, Tensor, Tensor]:
    return bind_model_io(MDDQueryDetector(config, mdd_a=mdd_a, mdd_b=mdd_b), MDDQueryAdapter(config))
