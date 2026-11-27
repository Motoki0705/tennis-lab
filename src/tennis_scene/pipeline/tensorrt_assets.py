"""Bind declared TensorRT model assets to the scene cache and model runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tennis_scene.pipeline.model_assets import file_identity
from src.utils.device import resolve_device
from src.utils.models.tensorrt_artifacts import EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import engine_environment, tensorrt_module


@dataclass(frozen=True)
class TensorRTModel:
    model: str
    assets: Mapping[str, Path]
    settings: dict[str, Any]
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]
    batch_multiplier: int = 1
    fixed_batch: bool = False

    def bind(self, root: Path, *, device: str, batch_size: int) -> EngineBundle:
        def digest(path: Path) -> str:
            return str(file_identity(path)["sha256"])

        bundle = EngineBundle.read(root, digest=digest).for_model(
            model=self.model,
            settings=self.settings,
            sources={name: digest(path) for name, path in self.assets.items()},
            inputs=self.inputs,
            outputs=self.outputs,
        )
        for name, profile in bundle.engine.profiles.items():
            if not self.fixed_batch:
                profile.require((self.batch_multiplier, *profile.optimum[1:]), tensor=name)
            profile.require(
                (self.batch_multiplier * batch_size, *profile.optimum[1:]), tensor=name
            )
        bundle.require_environment(
            engine_environment(resolve_device(device), tensorrt_module())
        )
        return bundle


def tensorrt_identity(
    bundles: tuple[EngineBundle, ...], *, batch_size: int
) -> tuple[dict[str, Path], dict[str, Any]]:
    return (
        {
            f"engine_{index}/{path.name}": path
            for index, bundle in enumerate(bundles)
            for path in bundle.files
        },
        {
            "backend": "tensorrt",
            "precision": "fp16",
            "batch_size": batch_size,
            "engines": [bundle.engine.model_dump(mode="json") for bundle in bundles],
        },
    )
