"""Synthetic portable hybrid weights and deterministic window evidence; CPU only."""

from dataclasses import asdict

import numpy as np

from src.tasks.ball_detection.annotation.contracts import (
    HybridCheckpointSpec,
    HybridInferenceConfig,
)
from src.tasks.ball_detection.annotation.hybrid import HybridWindow
from src.tasks.base.annotation.artifacts import annotation_reference, write_document
from src.tasks.base.annotation.contracts import ModelRef
from src.tasks.base.annotation.paths import AnnotationLayout


def inference(*, jpeg: bool = False) -> HybridInferenceConfig:
    return HybridInferenceConfig(
        stride=16,
        device="cpu",
        precision="float32",
        presence_threshold=0.5,
        subpixel=False,
        decode="opencv_jpeg_rgb24_native" if jpeg else "pyav_rgb24_native",
    )


def checkpoint(layout: AnnotationLayout, *, presence_logit: float = -2.0) -> ModelRef:
    import torch

    from src.tasks.ball_detection.models.mdd import (
        DeepMDDHybridDetector,
        MDDHybridConfig,
    )
    from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract

    config = MDDHybridConfig(
        stem_channels=(4, 4, 8, 8),
        mixed_channels=(8, 16),
        residual_blocks=(0,) * 6,
        decoder_channels=8,
        dim=16,
        heads=2,
        layers=1,
        ffn_dim=64,
        dropout=0.0,
        rope_base=10000.0,
        activation_checkpointing=False,
        encoder_variant="convnext_v2",
        temporal_mixing="factorized",
        input_channels=5,
    )
    model = DeepMDDHybridDetector(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.presence_head[1].bias.fill_(presence_logit)
        model.heatmap_head[-1].bias.fill_(2.0)
    architecture = asdict(config)
    contract = model.mdd.input_contract()
    image_decode = jpeg_decoder_contract("opencv")
    recipe = dict(
        model=architecture,
        input_contract=contract,
        manifest_sha256="f" * 64,
        image_decode=image_decode,
        runtime=dict(jpeg_decoder="opencv"),
        presence_threshold=0.5,
        loss=dict(
            heatmap_weight=1.0, presence_weight=1.0, sigma_ratio=0.01, focal_gamma=2.0
        ),
        parent=dict(input_contract=contract, image_decode=image_decode),
    )
    directory = layout.path("campaigns", "hybrid-input")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.pt"
    torch.save(
        dict(
            schema="mdd_hybrid_training.v1",
            stage="hybrid",
            model_kind="mdd_hybrid",
            model_config=architecture,
            state_dict=model.state_dict(),
            recipe=recipe,
        ),
        path,
    )
    weights = annotation_reference(layout, path)
    spec = write_document(
        layout,
        directory / "spec.json",
        HybridCheckpointSpec(
            architecture=architecture,
            input_contract=contract,
            training_manifest_digest="f" * 64,
        ),
    )
    return ModelRef(
        task="ball",
        version="synthetic-hybrid",
        training_weights=weights,
        inference_weights=weights,
        configuration=spec,
    )


class SpyHybrid:
    loads = 0
    calls: list[np.ndarray] = []

    def __init__(self, *_args: object) -> None:
        type(self).loads += 1

    def __call__(self, rgb: np.ndarray, seconds: np.ndarray) -> HybridWindow:
        type(self).calls.append(seconds.copy())
        assert rgb.shape[:2] == (32, 3) and rgb.dtype == np.uint8
        assert seconds[0] == 0 and np.all(np.diff(seconds) > 0)
        present = np.arange(32) % 2 == 0
        coords = np.stack(
            (np.arange(32, dtype=np.float32) / 31, np.ones(32, np.float32)), axis=1
        )
        coords[~present] = np.nan
        return HybridWindow(
            coordinates=coords,
            presence_logits=np.where(present, 2.0, -2.0).astype(np.float32),
            presence_scores=np.where(present, 0.8807971, 0.11920292).astype(np.float32),
            present=present,
            heatmap_peaks=np.full(32, 0.9, np.float32),
            heatmap_size=((rgb.shape[-2] + 3) // 4, (rgb.shape[-1] + 3) // 4),
        )
