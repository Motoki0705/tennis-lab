"""Typed DINO frame-to-token boundary for SLCS token precomputation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    ModelOutputContractError,
)
from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
from src.utils.data.augmentation import IMAGENET_MEAN, IMAGENET_STD

_PATCH_TOKEN_KEY = "x_norm_patchtokens"


@runtime_checkable
class FrameTokenBackbone(Protocol):
    """Minimal loaded-backbone surface consumed by the bound encoder."""

    embed_dim: int
    patch_size: int

    def forward_features(self, inputs: Tensor) -> Mapping[str, object]:
        """Execute the model-specific feature method."""
        ...


@dataclass(frozen=True, slots=True)
class FrameTokenCall:
    """Normalized model tensor and the validated external batch size."""

    normalized_frames: Tensor
    batch_size: int


class SLCSFrameTokenIOAdapter:
    """Validate uint8 NHWC frames, normalize them, and decode DINO patch tokens."""

    def __init__(self, spec: DinoTokenSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self._mean = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            IMAGENET_STD, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)

    def validate_model(self, model: object) -> FrameTokenBackbone:
        """Reject a loaded backbone inconsistent with the configured token spec."""
        if not isinstance(model, FrameTokenBackbone):
            raise ModelAdapterMismatchError(
                "SLCS frame-token adapter requires a backbone exposing integer "
                "patch_size/embed_dim and forward_features()."
            )
        if type(model.patch_size) is not int or type(model.embed_dim) is not int:
            raise ModelAdapterMismatchError(
                "SLCS frame-token backbone patch_size and embed_dim must be integers."
            )
        actual = (model.patch_size, model.embed_dim)
        expected = (self.spec.patch_size, self.spec.embed_dim)
        if actual != expected:
            raise ModelAdapterMismatchError(
                "SLCS frame-token adapter expects backbone "
                f"(patch_size, embed_dim)={expected}, got {actual}."
            )
        return model

    def build_call(self, frames: NDArray[np.uint8]) -> FrameTokenCall:
        """Validate one external RGB frame batch before model execution."""
        if not isinstance(frames, np.ndarray):
            raise ModelInputContractError(
                "SLCS precompute frames must be a numpy array."
            )
        if frames.dtype != np.uint8:
            raise ModelInputContractError(
                f"SLCS precompute frames must have dtype uint8, got {frames.dtype}."
            )
        expected_tail = (self.spec.image_height, self.spec.image_width, 3)
        if frames.ndim != 4 or frames.shape[1:] != expected_tail:
            raise ModelInputContractError(
                "SLCS precompute frames must have shape "
                f"(B, {expected_tail[0]}, {expected_tail[1]}, 3), got "
                f"{frames.shape}."
            )
        batch_size = int(frames.shape[0])
        if batch_size <= 0:
            raise ModelInputContractError(
                "SLCS precompute frames require a non-empty batch."
            )
        owned_frames = np.array(frames, dtype=np.uint8, order="C", copy=True)
        tensor = torch.from_numpy(owned_frames).to(
            device=self.device, dtype=torch.float32
        )
        tensor = tensor.permute(0, 3, 1, 2).div(255.0)
        normalized = (tensor - self._mean) / self._std
        return FrameTokenCall(normalized_frames=normalized, batch_size=batch_size)

    def decode_output(
        self,
        output: Mapping[str, object],
        *,
        batch_size: int,
    ) -> NDArray[np.float16]:
        """Decode and convert the required DINO patch-token tensor."""
        if not isinstance(output, Mapping):
            raise ModelOutputContractError(
                "DINO forward_features must return a mapping, got "
                f"{type(output).__name__}."
            )
        if _PATCH_TOKEN_KEY not in output:
            raise ModelOutputContractError(
                f"DINO output is missing required {_PATCH_TOKEN_KEY!r}."
            )
        tokens = output[_PATCH_TOKEN_KEY]
        if not isinstance(tokens, Tensor) or not torch.is_floating_point(tokens):
            raise ModelOutputContractError(
                f"DINO {_PATCH_TOKEN_KEY!r} must be a floating tensor."
            )
        expected = (batch_size, self.spec.num_tokens, self.spec.embed_dim)
        if tuple(tokens.shape) != expected:
            raise ModelOutputContractError(
                f"DINO {_PATCH_TOKEN_KEY!r} must have shape {expected}, got "
                f"{tuple(tokens.shape)}."
            )
        wrong_device_type = tokens.device.type != self.device.type
        wrong_device_index = (
            self.device.index is not None and tokens.device.index != self.device.index
        )
        if wrong_device_type or wrong_device_index:
            raise ModelOutputContractError(
                f"DINO patch tokens must remain on {self.device}, got {tokens.device}."
            )
        if not bool(torch.isfinite(tokens).all()):
            raise ModelOutputContractError("DINO patch tokens contain non-finite values.")
        return np.asarray(
            tokens.detach().to(device="cpu", dtype=torch.float16).numpy(),
            dtype=np.float16,
        )


@dataclass(frozen=True, slots=True)
class BoundSLCSFrameTokenEncoder:
    """Once-selected DINO backbone and its typed SLCS I/O adapter."""

    model: FrameTokenBackbone
    adapter: SLCSFrameTokenIOAdapter

    def __call__(self, frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        call = self.adapter.build_call(frames)
        with torch.inference_mode():
            output = self.model.forward_features(call.normalized_frames)
        return self.adapter.decode_output(output, batch_size=call.batch_size)


__all__ = [
    "BoundSLCSFrameTokenEncoder",
    "FrameTokenBackbone",
    "FrameTokenCall",
    "SLCSFrameTokenIOAdapter",
]
