"""Explicit model selection; a configured architecture has one implementation."""

from typing import TypeAlias

from torch import Tensor

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.adapters import RefinerAdapter
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.generators.flow import FlowRefiner
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner
from src.tasks.base.model_io import BoundModelIO, bind_model_io

RefinerModel: TypeAlias = RegressionRefiner | FlowRefiner
RefinerBinding: TypeAlias = BoundModelIO[
    dict[str, Tensor], RefinerOutput, RefinerOutput
]


def bind_refiner(model: RefinerModel) -> RefinerBinding:
    return bind_model_io(
        model,
        RefinerAdapter(
            type(model),
            window_length=model.config.window_length,
            flow=isinstance(model, FlowRefiner),
        ),
    )


def build_refiner(config: ModelConfig) -> RefinerModel:
    if config.architecture == "regression":
        return RegressionRefiner(config)
    if config.architecture == "flow":
        return FlowRefiner(config)
    raise ValueError(f"Unsupported refiner architecture: {config.architecture}")
