"""Composed DINO patch-query Court model with pose and dense raw outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.court_detection.configuration import CourtQueryModelConfig
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.models.query_encoder.backbone import (
    CourtQueryDINOv3Backbone,
)
from src.tasks.court_detection.models.query_encoder.contracts import (
    CourtQueryRawOutput,
    PatchTokenBatch,
)
from src.tasks.court_detection.models.query_encoder.decoders import (
    build_query_dense_decoder,
)
from src.tasks.court_detection.models.query_encoder.heads import (
    CourtPose10DHead,
    CourtQueryDenseHeads,
)
from src.tasks.court_detection.models.query_encoder.task_encoder import (
    CourtQueryTaskEncoder,
)


def _metadata_hw(value: Tensor, *, name: str) -> tuple[int, int]:
    if value.shape != (2,) or value.dtype != torch.long:
        raise ValueError(f"{name} must be an int64 tensor with shape (2,).")
    first, second = (int(item) for item in value.tolist())
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive.")
    return first, second


class CourtQueryEncoderModel(nn.Module):
    """Consume an explicit patch boundary; never infer or strip special tokens."""

    def __init__(
        self,
        config: CourtQueryModelConfig,
        target_bundle: CourtTargetBundleSpec,
        *,
        backbone: CourtQueryDINOv3Backbone,
    ) -> None:
        super().__init__()
        if config.name != "court_query_encoder":
            raise ValueError("CourtQueryEncoderModel requires its discriminated config.")
        if config.heads.dense_targets != target_bundle.kinds:
            raise ValueError(
                "Query model head subset must exactly match the target bundle."
            )
        self.in_channels = config.in_channels
        self.target_bundle_spec = target_bundle
        self.backbone = backbone
        self.task_encoder = CourtQueryTaskEncoder(
            input_dim=self.backbone.embed_dim,
            config=config.task_encoder,
        )
        self.decoder = build_query_dense_decoder(
            hidden_dim=config.task_encoder.hidden_dim,
            config=config.decoder,
        )
        self.pose_head = CourtPose10DHead(
            input_dim=config.task_encoder.hidden_dim,
            hidden_dim=config.heads.pose_hidden_dim,
            depth=config.heads.pose_depth,
        )
        self.dense_heads = CourtQueryDenseHeads(
            input_dim=self.decoder.output_channels,
            config=config.heads,
            target_bundle=target_bundle,
        )

    @classmethod
    def from_config(
        cls,
        config: CourtQueryModelConfig,
        target_bundle: CourtTargetBundleSpec,
    ) -> CourtQueryEncoderModel:
        return cls(
            config,
            target_bundle,
            backbone=CourtQueryDINOv3Backbone.from_config(config.backbone),
        )

    def forward(
        self,
        images: Tensor,
        patch_tokens: Tensor,
        grid_hw: Tensor,
        padded_hw: Tensor,
    ) -> CourtQueryRawOutput:
        """Forward only already-extracted, patch-only DINO tokens."""
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError("Query model images must have shape (B,3,H,W).")
        grid = _metadata_hw(grid_hw, name="grid_hw")
        padded = _metadata_hw(padded_hw, name="padded_hw")
        original = (int(images.shape[-2]), int(images.shape[-1]))
        patch_batch = PatchTokenBatch(
            tokens=patch_tokens,
            original_hw=original,
            padded_hw=padded,
            padding_hw=(padded[0] - original[0], padded[1] - original[1]),
            grid_hw=grid,
            patch_size=self.backbone.patch_size,
        )
        if patch_batch.batch_size != images.shape[0]:
            raise ValueError("Patch-token batch size must match the image batch.")
        return self.forward_patch_batch(images, patch_batch)

    def forward_patch_batch(
        self,
        images: Tensor,
        patch_batch: PatchTokenBatch,
    ) -> CourtQueryRawOutput:
        if patch_batch.original_hw != tuple(images.shape[-2:]):
            raise ValueError("Patch boundary original_hw must match the input images.")
        encoded = self.task_encoder(patch_batch)
        dense_features = self.decoder(
            encoded.taps,
            output_hw=patch_batch.original_hw,
        )
        if dense_features.shape[-2:] != patch_batch.original_hw:
            raise RuntimeError("Query dense decoder did not preserve input image H/W.")
        return CourtQueryRawOutput(
            pose=self.pose_head(encoded.pose_query),
            dense_logits=self.dense_heads(dense_features),
        )


__all__ = ["CourtQueryEncoderModel"]
