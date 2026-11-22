"""Shared model declarations used by more than one scene component."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.submodules.models import ViTPosePose2D
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.tennis_scene.pipeline.inference_runtime import (
    model_identities as model_identities,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.tensorrt_assets import TensorRTModel, tensorrt_identity
from src.utils.io import json_value
from src.utils.models.loading.dinov3 import dinov3_source_assets
from src.utils.models.tensorrt_artifacts import TensorSpec


def vitpose_tensorrt(config: PeopleModelConfig) -> TensorRTModel:
    pose = config.runtime.vitpose
    factor = 2 ** pose.head.num_deconv_layers
    return TensorRTModel(
        "vitpose_huge_coco17",
        {"checkpoint": config.vitpose_checkpoint},
        {"head": json_value(pose.head), "flip_test": pose.flip_test},
        {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, 256, 192))},
        {"heatmaps": TensorSpec(dtype="float16", shape=(-1, pose.head.out_channels, 16 * factor, 12 * factor))},
        batch_multiplier=2 if pose.flip_test else 1,
    )


def vitpose_model(config: PeopleModelConfig, *, factory: Callable[..., ViTPosePose2D] = ViTPosePose2D,
                  show_progress: bool = False) -> ModelSpec[ViTPosePose2D]:
    runtime = config.runtime
    pose = runtime.vitpose
    assets = {"checkpoint": config.vitpose_checkpoint}
    options = json_value(pose)
    del options["engines"]
    settings = {"runtime": options, "show_progress": show_progress}
    factory_kwargs = {}
    if pose.engines:
        model = vitpose_tensorrt(config)
        bundle = model.bind(pose.engines[0], device=runtime.device, batch_size=pose.batch_size)
        engine_assets, identity = tensorrt_identity((bundle,), batch_size=pose.batch_size)
        assets.update(engine_assets)
        settings.update(identity)
        factory_kwargs["engine"] = bundle
    return ModelSpec("vitpose", assets, "vitpose_huge_coco17",
        runtime.device, {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}[pose.precision], settings,
        lambda: factory(config.vitpose_checkpoint, device=runtime.device, flip_test=runtime.vitpose.flip_test,
            batch_size=runtime.vitpose.batch_size, head_config=runtime.vitpose.head,
            precision=runtime.vitpose.precision, show_progress=show_progress, **factory_kwargs),
        initialize=lambda model: model.load(), release=lambda model: model.unload())


def appearance_model(name: str, checkpoint: Path, device: str,
                     factory: Callable[[], AppearanceEncoder], *, repository: Path | None = None) -> ModelSpec[AppearanceEncoder]:
    # Crop selection stays with each consumer; encoder input/preprocessing are
    # fixed by the named encoder profile, independent of clip and camera size.
    if name.startswith("dinov3_") and repository is None:
        raise ValueError("DINOv3 appearance requires an explicit implementation repository")
    sources = {}
    if name.startswith("dinov3_"):
        assert repository is not None
        sources = dinov3_source_assets(repository)
    return ModelSpec("appearance", {"checkpoint": checkpoint, **sources}, name, device, "float32",
        {"profile": name, "input": "rgb_0_1", "embedding": "l2_normalized", "pose_prompt": False},
        factory)
