"""Accuracy adapter for the current ViTPose-H scene path."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from pose_contract import PoseBatch, PoseInput, PoseStages

from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models import Pose2DFrameSequenceRequest, ViTPosePose2D
from src.tasks.person_tracking.features import pose_boxes
from src.tennis_scene.pipeline.tensorrt_assets import TensorRTModel
from src.utils.inference_benchmark import (
    Backend,
    BenchmarkAdapter,
    ConditionUnavailable,
    Status,
)
from src.utils.io import json_value
from src.utils.models.tensorrt_artifacts import PLAN_FILE, TensorSpec

HEAD = ViTPoseHeadConfig(
    in_channels=1280,
    out_channels=17,
    num_deconv_layers=2,
    num_deconv_filters=(256, 256),
    num_deconv_kernels=(4, 4),
    final_conv_kernel=1,
    num_conv_layers=0,
    num_conv_kernels=(),
)
MODEL_IDS = ("vitpose-h", "vitpose-h-no-flip")


def _request(rows: tuple[PoseInput, ...]) -> Pose2DFrameSequenceRequest:
    images: list[tuple[int, NDArray[np.uint8]]] = []
    frame_indices = []
    for row in rows:
        # A consecutive run of people from one decoded image shares its resize.
        if not images or row.frame_bgr is not images[-1][1]:
            images.append((len(images), row.frame_bgr))
        frame_indices.append(images[-1][0])
    return Pose2DFrameSequenceRequest(
        images,
        pose_boxes(np.stack([row.bbox_xyxy for row in rows])),
        torch.tensor(frame_indices, dtype=torch.int64),
    )


def make_adapter(
    *,
    model_id: str,
    backend: Backend,
    checkpoint: Path,
    device: str,
    engine: Path | None = None,
    source_root: Path | None = None,
) -> BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]:
    if model_id not in MODEL_IDS or source_root is not None:
        raise ValueError(
            "ViTPose requires its local implementation and a supported model ID"
        )
    flip = model_id == "vitpose-h"
    if backend is not Backend.TRT16 and engine is not None:
        raise ValueError("An engine can only be supplied to TRT16")

    def load() -> Callable[[tuple[PoseInput, ...], PoseStages], PoseBatch]:
        bundle = None
        if backend is Backend.TRT16:
            if engine is None:
                raise ConditionUnavailable(
                    Status.UNAVAILABLE,
                    "Provide a ViTPose TensorRT bundle for this flip setting",
                )
            declaration = TensorRTModel(
                "vitpose_huge_coco17",
                {"checkpoint": checkpoint},
                {"head": json_value(HEAD), "flip_test": flip},
                {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, 256, 192))},
                {"heatmaps": TensorSpec(dtype="float16", shape=(-1, 17, 64, 48))},
                batch_multiplier=2 if flip else 1,
            )
            bundle = declaration.bind(engine, device=device, batch_size=1)
        model = ViTPosePose2D(
            checkpoint,
            device=device,
            flip_test=flip,
            batch_size=1,
            head_config=HEAD,
            precision="fp16" if backend is not Backend.PT32 else "fp32",
            show_progress=False,
            engine=bundle,
        )
        model.load()

        def infer(rows: tuple[PoseInput, ...], stages: PoseStages) -> PoseBatch:
            if bundle is not None:
                internal_shape = (len(rows) * (2 if flip else 1), 3, 256, 192)
                bundle.engine.profiles["pixels"].require(
                    internal_shape, tensor="pixels"
                )
            model.batch_size = len(rows)
            # Use production crop, flip and decode as one operation for accuracy.
            # The stage timings for this model come from #2496.
            result = stages.forward(backend, lambda: model.predict(_request(rows)))
            return PoseBatch(result.keypoints.detach().cpu().numpy())

        return infer

    return BenchmarkAdapter(
        backend,
        load,
        {"checkpoint": checkpoint},
        {} if engine is None else {"plan": engine / PLAN_FILE},
    )
