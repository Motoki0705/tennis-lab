"""Central registry for every synthetic-dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import cast

from src.synthetic_data_generation.dataset.pipeline import DatasetPipeline


@dataclass(frozen=True)
class DatasetPipelineDefinition:
    """One lazily imported dataset pipeline registered in one central place."""

    name: str
    factory_module: str
    factory_name: str
    description: str


_DEFINITIONS = MappingProxyType(
    {
        "blcs": DatasetPipelineDefinition(
            name="blcs",
            factory_module="src.synthetic_data_generation.dataset.blcs.pipeline",
            factory_name="create_pipeline",
            description="Single/multi-ball Gaussian scenes and physical labels.",
        ),
        "plcs": DatasetPipelineDefinition(
            name="plcs",
            factory_module="src.synthetic_data_generation.dataset.plcs.pipeline",
            factory_name="create_pipeline",
            description="Single/multi-person controllable Gaussian-avatar scenes.",
        ),
        "court": DatasetPipelineDefinition(
            name="court",
            factory_module="src.synthetic_data_generation.dataset.court.pipeline",
            factory_name="create_pipeline",
            description="Novel-view RGB and symmetric multi-court keypoint labels.",
        ),
    }
)


def available_dataset_pipelines() -> tuple[DatasetPipelineDefinition, ...]:
    """Return every built-in dataset pipeline in deterministic order."""
    return tuple(_DEFINITIONS[name] for name in sorted(_DEFINITIONS))


def get_dataset_pipeline(name: str) -> DatasetPipeline:
    """Construct one exact pipeline without a fallback or plugin side effects."""
    try:
        definition = _DEFINITIONS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_DEFINITIONS))
        raise ValueError(
            f"Unknown synthetic dataset {name!r}; available choices: {choices}."
        ) from error
    module = import_module(definition.factory_module)
    factory = getattr(module, definition.factory_name, None)
    if not callable(factory):
        raise RuntimeError(
            f"Dataset factory {definition.factory_module}:"
            f"{definition.factory_name} is not callable."
        )
    pipeline = cast(DatasetPipeline, factory())
    if pipeline.dataset_name != name:
        raise RuntimeError(
            f"Dataset factory {name!r} returned {pipeline.dataset_name!r}."
        )
    return pipeline
