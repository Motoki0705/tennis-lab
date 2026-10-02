"""Strict inference adapter for the official SOLIDER-REID MSMT17 Swin-Base.

Recipe/provenance and MIT notice: solider_vendor/NOTICE.md. No MMCV,
Torchreid fork, checkpoint conversion or pretrained-weight fallback is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.player_association.appearance.solider_vendor.swin_transformer import (
    swin_base_patch4_window7_224,
)


def load_backbone(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Only the known unused classifier and pre-BN recipe's neck may be omitted."""
    unused = {'classifier.weight': (1041, 1024), 'bottleneck.weight': (1024,),
        'bottleneck.bias': (1024,), 'bottleneck.running_mean': (1024,),
        'bottleneck.running_var': (1024,), 'bottleneck.num_batches_tracked': ()}
    extra = {k: v for k, v in state.items() if not k.startswith('base.')}
    if set(extra) != set(unused) or any(tuple(extra[k].shape) != shape for k, shape in unused.items()):
        raise ValueError('SOLIDER checkpoint must be the explicit MSMT17 pre-BN recipe')
    model.load_state_dict({k.removeprefix('base.'): v for k, v in state.items() if k.startswith('base.')}, strict=True)


@dataclass
class SoliderEncoder:
    name: str
    checkpoint: Path
    device: str = 'cpu'
    input_size: tuple[int, int] = (384, 128)

    def __post_init__(self) -> None:
        if self.input_size != (384, 128):
            raise ValueError('SOLIDER MSMT17 inference requires 384x128 crops')
        self.model = swin_base_patch4_window7_224(img_size=self.input_size, semantic_weight=.2)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(state, Mapping) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise TypeError('SOLIDER requires a tensor state dict')
        load_backbone(self.model, state)
        self.model.eval().to(self.device)

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 4 or tuple(crops.shape[1:]) != (3, *self.input_size) \
                or not torch.isfinite(crops).all() or bool(((crops < 0) | (crops > 1)).any()):
            raise ValueError('SOLIDER crops must be finite RGB [0,1] at 384x128')
        with torch.inference_mode():
            features, _ = self.model((crops.to(self.device) - .5) / .5)
            features = features.float()
            if features.shape != (len(crops), 1024) or not torch.isfinite(features).all() \
                    or bool((torch.linalg.vector_norm(features, dim=-1) == 0).any()):
                raise ValueError('Invalid SOLIDER backbone output')
            result: torch.Tensor = F.normalize(features, dim=-1).cpu()
        return result
