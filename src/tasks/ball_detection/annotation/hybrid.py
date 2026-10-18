"""Strict hybrid checkpoint restore and native heatmap/presence evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import canonical_bytes
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import HybridCheckpointSpec, HybridInferenceConfig


@dataclass(frozen=True)
class HybridWindow:
    coordinates: NDArray[np.float32]
    presence_logits: NDArray[np.float32]
    presence_scores: NDArray[np.float32]
    present: NDArray[np.bool_]
    heatmap_peaks: NDArray[np.float32]
    heatmap_size: tuple[int, int]


class HybridPredictor:
    def __init__(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        config: HybridInferenceConfig,
    ) -> None:
        import torch

        from src.tasks.ball_detection.checkpoints.hybrid_training import (
            restore_hybrid_model,
        )
        from src.tasks.ball_detection.model_io.mdd_hybrid import MDDHybridAdapter

        spec = read_document(layout, context.model.configuration, HybridCheckpointSpec)
        saved = torch.load(
            verify_reference(layout, context.model.inference_weights),
            map_location="cpu",
            weights_only=True,
        )
        if saved.get("schema") != spec.checkpoint_schema or canonical_bytes(
            saved.get("model_config")
        ) != canonical_bytes(spec.architecture):
            raise ValueError(
                "hybrid checkpoint architecture differs from the frozen model spec"
            )
        model, recipe = restore_hybrid_model(saved)
        if (
            canonical_bytes(model.mdd.input_contract())
            != canonical_bytes(spec.input_contract)
            or recipe["manifest_sha256"] != spec.training_manifest_digest
        ):
            raise ValueError(
                "hybrid checkpoint preprocessing/training provenance differs from the frozen spec"
            )
        if recipe["presence_threshold"] != config.presence_threshold:
            raise ValueError(
                "hybrid inference presence threshold differs from the adopted checkpoint"
            )
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        if config.precision == "bfloat16" and (
            config.device != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 inference requires a supported CUDA device")
        self.precision = config.precision
        self.model = model.to(self.device).eval()
        self.adapter = MDDHybridAdapter(
            model.config,
            presence_threshold=config.presence_threshold,
            subpixel=config.subpixel,
        )
        self.adapter.validate_model_pair(self.model)

    def __call__(
        self, rgb: NDArray[np.uint8], seconds: NDArray[np.float32]
    ) -> HybridWindow:
        import torch

        from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput

        inputs = MDDQueryInput(
            torch.from_numpy(rgb)[None].to(self.device),
            torch.from_numpy(seconds)[None].to(self.device),
        )
        call = self.adapter.build_call(inputs)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.precision == "bfloat16",
            ),
        ):
            output = self.model(*call.args, **call.kwargs)
        prediction = self.adapter.decode_output(output)
        # Keep the ungated heatmap peak even for predicted absence. It is evidence,
        # not a calibrated presence probability or a teacher confidence.
        return HybridWindow(
            coordinates=prediction.coords[0].numpy(),
            presence_logits=prediction.presence_logits[0].numpy(),
            presence_scores=prediction.presence_logits[0].sigmoid().numpy(),
            present=prediction.present[0].numpy(),
            heatmap_peaks=prediction.heatmaps[0].flatten(1).amax(1).numpy(),
            heatmap_size=(prediction.heatmaps.shape[-2], prediction.heatmaps.shape[-1]),
        )
