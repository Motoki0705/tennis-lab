"""Invoke the production CLI with the experiment's PyTorch FP16 forwards."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
from weakref import WeakMethod

import numpy as np
import torch
from run import CONDITION

from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
    ClipReIDEncoder,
)
from src.tennis_scene.pipeline import definition
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.tennis_scene.scripts.run_video_batch import main as video_batch_main
from src.utils.checksum import dual_sha256


def autocast_forward(network: torch.nn.Module, *, fp32_outputs: bool = False) -> None:
    original = WeakMethod(network.forward)

    def infer(*args: Any, **kwargs: Any) -> Any:
        forward = original()
        if forward is None:
            raise RuntimeError("The PyTorch model was released before its forward")
        with torch.autocast("cuda", dtype=torch.float16):
            result = forward(*args, **kwargs)
        if fp32_outputs:
            return {
                name: value.float() if value.is_floating_point() else value
                for name, value in result.items()
            }
        return result

    network.forward = infer


@contextmanager
def pt16_forwards() -> Iterator[None]:
    """Bind the experimental compute dtype to the real pipeline ModelSpecs."""
    implementation = dual_sha256(Path(__file__))
    appearance_model = definition.appearance_model
    hmr_specs = HMR2FeaturesModule.model_specs
    gvhmr_specs = GVHMRModule.model_specs

    def specialize(
        spec: ModelSpec[Any], initialize_forward: Callable[[Any], None]
    ) -> ModelSpec[Any]:
        if spec.dtype != "float32" or not spec.device.startswith("cuda"):
            raise ValueError(
                f"Expected a CUDA PyTorch FP32 declaration for {spec.kind}"
            )

        def initialize(model: Any) -> None:
            spec.initialize(model)
            initialize_forward(model)

        return replace(
            spec,
            dtype="float16",
            settings={
                **spec.settings,
                "compute_precision": "cuda_autocast_fp16",
                "experiment_sha256": implementation,
            },
            initialize=initialize,
        )

    def clip_spec(name: str, *args: Any, **kwargs: Any) -> ModelSpec[Any]:
        spec = appearance_model(name, *args, **kwargs)
        if name != CLIP_REID_PROFILE:
            return spec

        def initialize(encoder: ClipReIDEncoder) -> None:
            autocast_forward(encoder.model)

        return specialize(spec, initialize)

    def hmr2_spec(component: HMR2FeaturesModule) -> dict[str, ModelSpec[Any]]:
        spec = hmr_specs(component)
        return {
            "features": specialize(
                spec["features"], lambda model: autocast_forward(model._model)
            )
        }

    def gvhmr_spec(component: GVHMRModule) -> dict[str, ModelSpec[Any]]:
        spec = gvhmr_specs(component)

        def initialize(model: Any) -> None:
            autocast_forward(model._model.pipeline.denoiser3d, fp32_outputs=True)

        return {"mesh": specialize(spec["mesh"], initialize)}

    with (
        patch.object(definition, "appearance_model", clip_spec),
        patch.object(HMR2FeaturesModule, "model_specs", hmr2_spec),
        patch.object(GVHMRModule, "model_specs", gvhmr_spec),
    ):
        yield


def main() -> None:
    backend = sys.argv.pop(1)
    if backend not in {"pt", "trt"}:
        raise ValueError(f"Unsupported experimental backend {backend!r}")
    seed = CONDITION["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if backend == "pt":
        with pt16_forwards():
            video_batch_main()
    else:
        video_batch_main()


if __name__ == "__main__":
    main()
