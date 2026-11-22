"""CLIP visual tower adapter shared by the PyTorch and ONNX entry points."""

from pathlib import Path
from types import MethodType

import torch
from inputs import CHECKPOINT_SHA256, INPUT_SHAPE, MODEL, sha256_file
from torch import nn

from src.tasks.player_association.appearance.encoders import ClipReIDEncoder

BATCH_SHAPE = (1, *INPUT_SHAPE)
FEATURE_DIMENSION = 1280
INPUT_NAME = "pixels"
OUTPUT_NAME = "features"


def visual_forward_with_dynamic_batch(visual, pixels):
    """Keep the CLS expansion tied to tensor shape during ONNX tracing."""
    patches = visual.conv1(pixels).flatten(2).transpose(1, 2)
    token = visual.class_embedding.expand(patches.shape[0], 1, -1)
    x = torch.cat((token, patches), 1) + visual.positional_embedding
    x = visual.transformer(visual.ln_pre(x).transpose(0, 1)).transpose(0, 1)
    cls = visual.ln_post(x[:, 0])
    return cls, cls @ visual.proj


class VisualFeatures(nn.Module):
    def __init__(self, checkpoint: Path, *, dynamic_batch=False):
        super().__init__()
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError(f"CLIP-ReID checkpoint SHA-256 mismatch: {checkpoint}")
        self.visual = ClipReIDEncoder(MODEL, checkpoint).model
        if dynamic_batch:
            self.visual.forward = MethodType(
                visual_forward_with_dynamic_batch, self.visual
            )

    def forward(self, pixels):
        cls, projected = self.visual(pixels)
        return torch.cat((cls, projected), dim=-1)
