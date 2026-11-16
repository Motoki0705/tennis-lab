"""Shared model declarations used by more than one scene component."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.submodules.models import ViTPosePose2D
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig


def vitpose_model(config: PeopleModelConfig, *, factory: Callable[..., ViTPosePose2D] = ViTPosePose2D,
                  show_progress: bool = False) -> ModelSpec[ViTPosePose2D]:
    runtime = config.runtime
    return ModelSpec("vitpose", {"checkpoint": config.vitpose_checkpoint}, "vitpose_huge_coco17",
        runtime.device, "float32", {"runtime": runtime.vitpose, "show_progress": show_progress},
        lambda: factory(config.vitpose_checkpoint, device=runtime.device, flip_test=runtime.vitpose.flip_test,
            batch_size=runtime.vitpose.batch_size, head_config=runtime.vitpose.head,
            precision="float32", show_progress=show_progress),
        initialize=lambda model: model.load(), release=lambda model: model.unload())


def appearance_model(name: str, checkpoint: Path, device: str,
                     factory: Callable[[], AppearanceEncoder]) -> ModelSpec[AppearanceEncoder]:
    # Crop selection stays with each consumer; encoder input/preprocessing are
    # fixed by the named encoder profile, independent of clip and camera size.
    return ModelSpec("appearance", {"checkpoint": checkpoint}, name, device, "float32",
        {"profile": name, "input": "rgb_0_1", "embedding": "l2_normalized", "pose_prompt": False},
        factory)
