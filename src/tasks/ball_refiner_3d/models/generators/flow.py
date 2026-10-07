"""Conditional x0 prediction network for flow matching."""

import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.components.embeddings import (
    coordinate_features,
    sinusoidal,
    time_embedding,
)
from src.tasks.ball_refiner_3d.models.components.temporal_transformer import (
    TemporalTransformer,
)


class FlowRefiner(TemporalTransformer):
    def __init__(self, config: ModelConfig) -> None:
        if config.architecture != "flow":
            raise ValueError("FlowRefiner requires architecture=flow")
        super().__init__(config, input_channels=7)
        self.flow_time = time_embedding(config.width)

    def forward(
        self, coordinates: Tensor, missing: Tensor, state: Tensor, time: Tensor
    ) -> RefinerOutput:
        features = torch.cat((coordinate_features(coordinates, missing), state), dim=-1)
        return self.encode(
            features, self.flow_time(sinusoidal(time * 1000, self.config.width))
        )
