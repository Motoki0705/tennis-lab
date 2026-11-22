"""Shared model declarations used by more than one scene component."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from src.submodules.configuration import BatchedRuntimeConfig
from src.submodules.models import ViTPosePose2D
from src.submodules.models.dino.architecture import (
    DINO_NUM_CLASSES,
    DINO_NUM_QUERIES,
    dino_source_assets,
)
from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
    AppearanceEncoder,
    ClipReIDEncoder,
)
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


def hmr2_tensorrt(config: PeopleModelConfig) -> TensorRTModel:
    return TensorRTModel(
        "hmr2_vit_huge",
        {"checkpoint": config.hmr2_checkpoint, "mean_params": config.smpl_assets.hmr2_mean_params},
        {},
        {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, 256, 256))},
        {"features": TensorSpec(dtype="float16", shape=(-1, 1024))},
    )


def clip_tensorrt(checkpoint: Path) -> TensorRTModel:
    return TensorRTModel(
        CLIP_REID_PROFILE,
        {"checkpoint": checkpoint},
        {},
        {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, 256, 128))},
        {"features": TensorSpec(dtype="float16", shape=(-1, 1280))},
    )


def dino_tensorrt(config: PeopleModelConfig, height: int, width: int) -> TensorRTModel:
    runtime = config.runtime.dino_detector
    if not 0 < min(height, width) <= max(height, width) <= runtime.max_long_side:
        raise ValueError(f"Invalid resized DINO dimensions: {height}x{width}")
    return TensorRTModel(
        "dino_4scale_swin_l",
        {"checkpoint": config.dino_checkpoint, **dino_source_assets(config.dino_repository)},
        {"sampling": "grid_sample", "short_side": runtime.short_side, "max_long_side": runtime.max_long_side},
        {"pixels": TensorSpec(dtype="float32", shape=(-1, 3, height, width))},
        {
            "pred_logits": TensorSpec(dtype="float32", shape=(-1, DINO_NUM_QUERIES, DINO_NUM_CLASSES)),
            "pred_boxes": TensorSpec(dtype="float32", shape=(-1, DINO_NUM_QUERIES, 4)),
        },
    )


def gvhmr_tensorrt(config: PeopleModelConfig) -> TensorRTModel:
    return TensorRTModel(
        "gvhmr_siga24_denoiser",
        {"checkpoint": config.gvhmr_checkpoint},
        {"attention_window": 120, "max_rotary_frames": 4096},
        {
            "length": TensorSpec(dtype="int64", shape=(1,)),
            "obs": TensorSpec(dtype="float16", shape=(1, -1, 17, 3)),
            "f_cliffcam": TensorSpec(dtype="float16", shape=(1, -1, 3)),
            "f_cam_angvel": TensorSpec(dtype="float16", shape=(1, -1, 6)),
            "f_imgseq": TensorSpec(dtype="float16", shape=(1, -1, 1024)),
        },
        {
            "pred_x": TensorSpec(dtype="float16", shape=(1, -1, 151)),
            "pred_cam": TensorSpec(dtype="float16", shape=(1, -1, 3)),
            "static_conf_logits": TensorSpec(dtype="float16", shape=(1, -1, 6)),
            "mask_out": TensorSpec(dtype="bool", shape=(1, -1)),
            "length_out": TensorSpec(dtype="int64", shape=(1,)),
        },
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
                     factory: Callable[[], AppearanceEncoder], *, repository: Path | None = None,
                     clip: BatchedRuntimeConfig | None = None) -> ModelSpec[AppearanceEncoder]:
    # Crop selection stays with each consumer; encoder input/preprocessing are
    # fixed by the named encoder profile, independent of clip and camera size.
    if name.startswith("dinov3_") and repository is None:
        raise ValueError("DINOv3 appearance requires an explicit implementation repository")
    sources = {}
    if name.startswith("dinov3_"):
        assert repository is not None
        sources = dinov3_source_assets(repository)
    dtype = "float32"
    settings: dict[str, object] = {"profile": name, "input": "rgb_0_1", "embedding": "l2_normalized", "pose_prompt": False}
    if name == CLIP_REID_PROFILE and clip is not None:
        if clip.batch_size != 64:
            settings["batch_size"] = clip.batch_size
        if clip.backend.engines:
            engine = clip_tensorrt(checkpoint).bind(
                clip.backend.engines[0], device=device, batch_size=clip.batch_size)
            engine_assets, identity = tensorrt_identity((engine,), batch_size=clip.batch_size)
            sources.update(engine_assets)
            settings.update(identity)
            dtype = "float16"
            factory = partial(ClipReIDEncoder, name, checkpoint, device, engine=engine)
    return ModelSpec("appearance", {"checkpoint": checkpoint, **sources}, name, device, dtype, settings, factory)
