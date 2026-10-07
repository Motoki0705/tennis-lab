"""Direct, bidirectional all-frame trajectory regression."""

from torch import Tensor

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.components.embeddings import coordinate_features
from src.tasks.ball_refiner_3d.models.components.temporal_transformer import (
    TemporalTransformer,
)


class RegressionRefiner(TemporalTransformer):
    def __init__(self, config: ModelConfig) -> None:
        if config.architecture != "regression":
            raise ValueError("RegressionRefiner requires architecture=regression")
        super().__init__(config, input_channels=4)

    def forward(self, coordinates: Tensor, missing: Tensor) -> RefinerOutput:
        return self.encode(coordinate_features(coordinates, missing))
