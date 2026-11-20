"""CLIP visual tower adapter shared by the PyTorch and ONNX entry points."""

from pathlib import Path

import torch
from inputs import CHECKPOINT_SHA256, INPUT_SHAPE, MODEL, sha256_file
from torch import nn

from src.tasks.player_association.appearance.encoders import ClipReIDEncoder

BATCH_SHAPE = (1, *INPUT_SHAPE)
FEATURE_DIMENSION = 1280
INPUT_NAME = "pixels"
OUTPUT_NAME = "features"


class VisualFeatures(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError(f"CLIP-ReID checkpoint SHA-256 mismatch: {checkpoint}")
        self.visual = ClipReIDEncoder(MODEL, checkpoint).model

    def forward(self, pixels):
        cls, projected = self.visual(pixels)
        return torch.cat((cls, projected), dim=-1)
