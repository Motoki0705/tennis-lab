"""Canonical input construction and output decoding for ball models."""

from __future__ import annotations

import math
import weakref
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.configuration import validate_model
from src.tasks.ball_detection.model_io.contracts import (
    BallInputLayout,
    BallInputMode,
    BallModelCall,
    BallModelInputSpec,
    BallModelIOError,
    BallPrediction,
    BallTrainingCall,
)
from src.tasks.base.model_io import ModelCall
from src.utils.data.heatmaps import (
    heatmaps_to_argmax,
    refine_peaks_log_parabolic,
    resize_heatmap_sequence,
)
from src.utils.models.loading import require_dinov3_patch_tokens

_RGB_TO_LUMINANCE = (0.299, 0.587, 0.114)


class BallModelExecutionBoundary(Protocol):
    """Prepare variant-specific tensors before the detector forward starts."""

    def bind_model(self, model: nn.Module) -> None:
        """Bind the construction-validated model instance."""
        ...

    def prepare(self, call: BallModelCall) -> BallModelCall:
        """Return a call containing only validated model arguments."""
        ...


def _run_dinov3_backbone(model: nn.Module, frames: Tensor) -> object:
    backbone = cast(Any, model).backbone
    return cast(Any, backbone.module).forward_features(frames)


def _run_frozen_dinov3_backbone(model: nn.Module, frames: Tensor) -> object:
    with torch.no_grad():
        return _run_dinov3_backbone(model, frames)


class DINOv3BallExecutionBoundary:
    """Own DINO execution, raw response decode, and RoPE preparation."""

    def __init__(self, *, frozen_backbone: bool) -> None:
        self._model_ref: weakref.ReferenceType[nn.Module] | None = None
        self._backbone_executor = (
            _run_frozen_dinov3_backbone
            if frozen_backbone
            else _run_dinov3_backbone
        )

    def bind_model(self, model: nn.Module) -> None:
        self._model_ref = weakref.ref(model)

    def prepare(self, call: BallModelCall) -> BallModelCall:
        if self._model_ref is None or (model := self._model_ref()) is None:
            raise BallModelIOError(
                "DINOv3 ball execution boundary is not bound to its model."
            )
        from src.tasks.ball_detection.models.dinov3_rope import (
            build_spatiotemporal_positions,
        )

        frames = call.model_input
        height, width = frames.shape[-2:]
        patch_size = int(cast(Any, model).patch_size)
        patch_height = height // patch_size
        patch_width = width // patch_size
        expected_tokens = patch_height * patch_width
        flat_frames = frames.reshape(
            call.batch_size * call.frame_count,
            frames.shape[2],
            height,
            width,
        )
        raw_output = self._backbone_executor(model, flat_frames)
        try:
            patch_tokens = require_dinov3_patch_tokens(
                raw_output,
                expected_batch_size=call.batch_size * call.frame_count,
                expected_embed_dim=int(cast(Any, model).backbone.embed_dim),
                expected_num_tokens=expected_tokens,
                context="ball DINOv3 forward_features",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BallModelIOError(str(error)) from error
        _require_float_tensor(patch_tokens, name="DINOv3 patch tokens", rank=3)
        _require_finite(patch_tokens, name="DINOv3 patch tokens")
        if patch_tokens.device != frames.device:
            raise BallModelIOError(
                "DINOv3 patch tokens must remain on the ball input device."
            )

        positions = build_spatiotemporal_positions(
            num_frames=call.frame_count,
            patch_height=patch_height,
            patch_width=patch_width,
            device=frames.device,
        )
        freqs_cis = cast(Any, model).decoder_rope(positions)
        expected_freq_shape = (
            call.frame_count * expected_tokens,
            1,
            int(cast(Any, model).decoder_rope_dim) // 2,
        )
        if (
            not isinstance(freqs_cis, Tensor)
            or not freqs_cis.is_complex()
            or freqs_cis.shape != expected_freq_shape
            or freqs_cis.device != frames.device
        ):
            raise BallModelIOError(
                "Ball RoPE frequencies violate the prepared tensor contract: "
                f"expected complex {expected_freq_shape} on {frames.device}."
            )
        _require_finite(freqs_cis, name="Ball RoPE frequencies")
        sequence_length = call.frame_count * expected_tokens
        attn_mask = torch.ones(
            call.batch_size,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=frames.device,
        )
        return replace(
            call,
            model_args=(frames, patch_tokens, freqs_cis, attn_mask),
        )


def build_ball_model_input_spec(config: object) -> BallModelInputSpec:
    """Resolve and validate the static ball model input contract."""
    model_cfg = validate_model(config)
    model_name = str(model_cfg["name"])
    input_mode = str(model_cfg["input_mode"]).strip().lower()
    if input_mode not in {"rgb", "mdd"}:
        raise BallModelIOError(
            f"model.input_mode must be one of ['rgb', 'mdd'], got {input_mode!r}."
        )
    input_layout = str(model_cfg["input_layout"]).strip().lower()
    if input_layout not in {"bcthw", "btchw"}:
        raise BallModelIOError(
            "model.input_layout must be one of ['bcthw', 'btchw'], "
            f"got {input_layout!r}."
        )
    in_channels = int(model_cfg["in_channels"])
    expected_channels = 3 if input_mode == "rgb" else 2
    if in_channels != expected_channels:
        raise BallModelIOError(
            f"model.in_channels must be {expected_channels} for {input_mode!r}, "
            f"got {in_channels}."
        )
    num_classes = int(model_cfg["num_classes"])
    if num_classes != 1:
        raise BallModelIOError(
            f"Ball heatmap models require model.num_classes=1, got {num_classes}."
        )
    configured_frames = int(model_cfg["num_frames"])
    if configured_frames <= 0:
        raise BallModelIOError("model.num_frames must be positive.")

    if model_name == "dinov3_rope":
        if input_mode != "rgb" or input_layout != "btchw":
            raise BallModelIOError(
                "dinov3_rope requires input_mode='rgb' and input_layout='btchw'."
            )
        image_size = tuple(int(value) for value in model_cfg["image_size"])
        if len(image_size) != 2 or min(image_size) <= 0:
            raise BallModelIOError("model.image_size must contain two positive values.")
        image_size_hw: tuple[int, int] | None = (image_size[0], image_size[1])
    else:
        if input_layout != "bcthw":
            raise BallModelIOError(f"{model_name} requires input_layout='bcthw'.")
        image_size_hw = None
    minimum_spatial_size = (
        4 * 2 ** (len(tuple(model_cfg["dims"])) - 1)
        if model_name == "conv_next_unet"
        else None
    )

    if input_mode == "mdd":
        mdd_a = float(model_cfg["mdd_a"])
        mdd_b = float(model_cfg["mdd_b"])
    else:
        mdd_a = 0.0
        mdd_b = 0.0
    a_value = abs(math.tanh(mdd_a))
    return BallModelInputSpec(
        model_name=model_name,
        input_mode=cast(BallInputMode, input_mode),
        input_layout=cast(BallInputLayout, input_layout),
        in_channels=in_channels,
        num_classes=num_classes,
        configured_frames=configured_frames,
        image_size_hw=image_size_hw,
        minimum_spatial_size=minimum_spatial_size,
        mdd_gain=5.0 / (0.45 * a_value + 1.0e-6),
        mdd_offset=0.6 * math.tanh(mdd_b),
    )


class BallModelIOAdapter:
    """Validate RGB batches and adapt them to one selected ball model."""

    def __init__(
        self,
        spec: BallModelInputSpec,
        *,
        expected_model_type: type[nn.Module],
        minimum_frames: int,
        execution_boundary: BallModelExecutionBoundary | None = None,
    ) -> None:
        if minimum_frames <= 0:
            raise BallModelIOError("minimum_frames must be positive.")
        if spec.in_channels <= 0 or spec.num_classes <= 0:
            raise BallModelIOError("Ball model channels must be positive.")
        if spec.configured_frames <= 0:
            raise BallModelIOError("Ball configured_frames must be positive.")
        self.spec = spec
        self.expected_model_type = expected_model_type
        self.minimum_frames = minimum_frames
        self.minimum_spatial_size = spec.minimum_spatial_size
        self.execution_boundary = execution_boundary
        self._prepare_model_call = (
            self._prepare_direct_model_call
            if execution_boundary is None
            else execution_boundary.prepare
        )

    @property
    def model_type(self) -> type[nn.Module]:
        """Return the concrete model type selected by the factory."""
        return self.expected_model_type

    def build_call(self, batch: Tensor) -> ModelCall:
        """Implement the shared model-I/O lifecycle for inference images."""
        prepared = self.prepare_model_call(batch)
        return ModelCall(args=prepared.model_args)

    def decode_output(self, output: Tensor) -> Tensor:
        """Implement the shared lifecycle's canonical heatmap decode."""
        _require_float_tensor(output, name="logits", rank=5)
        if output.shape[1] != 1:
            raise BallModelIOError(
                f"Ball logits must have one output channel, got {output.shape[1]}."
            )
        return torch.sigmoid(output.squeeze(1))

    def validate_model_pair(self, model: nn.Module) -> None:
        """Reject a mismatched model/adapter pair at composition time."""
        if type(model) is not self.expected_model_type:
            raise BallModelIOError(
                f"Adapter for {self.spec.model_name!r} requires "
                f"{self.expected_model_type.__name__}, got {type(model).__name__}."
            )
        in_channels = getattr(model, "in_channels", None)
        num_classes = getattr(model, "num_classes", None)
        if in_channels != self.spec.in_channels or num_classes != self.spec.num_classes:
            raise BallModelIOError(
                "Ball model attributes do not match the selected adapter: "
                f"in_channels={in_channels!r}, num_classes={num_classes!r}."
            )
        if self.execution_boundary is not None:
            self.execution_boundary.bind_model(model)

    def prepare_model_call(self, images: Tensor) -> BallModelCall:
        """Build the complete validated argument list before model entry."""
        return self._prepare_model_call(self.prepare_images(images))

    @staticmethod
    def _prepare_direct_model_call(call: BallModelCall) -> BallModelCall:
        return call

    def prepare_images(self, images: Tensor) -> BallModelCall:
        """Validate raw ``(B,T,3,H,W)`` images before model execution."""
        _require_float_tensor(images, name="images", rank=5)
        if images.dtype != torch.float32:
            raise BallModelIOError(
                f"images must use torch.float32, got {images.dtype}."
            )
        _require_finite(images, name="images")
        if bool(torch.any((images < 0.0) | (images > 1.0))):
            raise BallModelIOError("images values must be in [0, 1].")
        batch_size, frame_count, channels, height, width = images.shape
        if batch_size <= 0:
            raise BallModelIOError("images must contain at least one sample.")
        if channels != 3:
            raise BallModelIOError(
                f"images must contain RGB channels at axis 2, got {channels}."
            )
        if frame_count < self.minimum_frames:
            raise BallModelIOError(
                f"{self.spec.model_name} requires at least {self.minimum_frames} "
                f"frame(s), got {frame_count}."
            )
        if height <= 0 or width <= 0:
            raise BallModelIOError("images height and width must be positive.")
        if (
            self.spec.image_size_hw is not None
            and (height, width) != self.spec.image_size_hw
        ):
            raise BallModelIOError(
                f"{self.spec.model_name} requires image size {self.spec.image_size_hw}, "
                f"got {(height, width)}."
            )
        if self.minimum_spatial_size is not None and (
            height < self.minimum_spatial_size or width < self.minimum_spatial_size
        ):
            raise BallModelIOError(
                f"{self.spec.model_name} requires H and W >= "
                f"{self.minimum_spatial_size}, got {(height, width)}."
            )
        model_input = self._to_model_input(images)
        return BallModelCall(
            images=images,
            model_input=model_input,
            model_args=(model_input,),
            batch_size=batch_size,
            frame_count=frame_count,
        )

    def prepare_training_batch(
        self,
        batch: Mapping[str, Any],
    ) -> BallTrainingCall:
        """Validate every tensor used by training before the forward pass."""
        images = _required_tensor(batch, "images")
        target_heatmaps = _required_tensor(batch, "heatmaps")
        coords = _required_tensor(batch, "coords")
        visibility = _required_tensor(batch, "visibility")
        original_size = _required_tensor(batch, "original_size")
        model_call = self.prepare_model_call(images)

        _require_float_tensor(target_heatmaps, name="heatmaps", rank=4)
        _require_float_tensor(coords, name="coords", rank=4)
        _require_finite(target_heatmaps, name="heatmaps")
        _require_finite(coords, name="coords")
        if bool(torch.any((target_heatmaps < 0.0) | (target_heatmaps > 1.0))):
            raise BallModelIOError("heatmaps values must be in [0, 1].")
        if visibility.ndim != 3 or (
            visibility.dtype != torch.bool and not visibility.is_floating_point()
        ):
            raise BallModelIOError(
                "visibility must be a rank-3 boolean or floating tensor."
            )
        if visibility.is_floating_point():
            _require_finite(visibility, name="visibility")
            if bool(torch.any((visibility < 0.0) | (visibility > 1.0))):
                raise BallModelIOError("visibility values must be in [0, 1].")
        if original_size.ndim != 2 or original_size.shape[-1] != 2:
            raise BallModelIOError("original_size must have shape (B, 2).")
        if original_size.dtype == torch.bool or original_size.is_complex():
            raise BallModelIOError("original_size must use a real numeric dtype.")
        _require_finite(original_size, name="original_size")
        if bool(torch.any(original_size <= 0)):
            raise BallModelIOError("original_size values must be positive.")

        batch_size = model_call.batch_size
        frame_count = model_call.frame_count
        if target_heatmaps.shape[:2] != (batch_size, frame_count):
            raise BallModelIOError("heatmaps batch/time dimensions must match images.")
        if coords.shape[:2] != (batch_size, frame_count) or coords.shape[-1] != 2:
            raise BallModelIOError("coords must have shape (B, T, K, 2).")
        if visibility.shape != coords.shape[:-1]:
            raise BallModelIOError("visibility must have shape (B, T, K).")
        if original_size.shape[0] != batch_size:
            raise BallModelIOError("original_size batch dimension must match images.")
        if coords.shape[2] <= 0:
            raise BallModelIOError("coords must reserve at least one instance slot.")
        return BallTrainingCall(
            model_call=model_call,
            target_heatmaps=target_heatmaps,
            coords=coords,
            visibility=visibility,
            original_size=original_size,
        )

    def validate_logits(self, logits: Tensor, call: BallModelCall) -> None:
        """Validate model output immediately at the model-I/O boundary."""
        _require_float_tensor(logits, name="logits", rank=5)
        _require_finite(logits, name="logits")
        if logits.shape[:3] != (call.batch_size, 1, call.frame_count):
            raise BallModelIOError(
                "Ball logits must have shape prefix (B, 1, T); got "
                f"{tuple(logits.shape)} for B={call.batch_size}, T={call.frame_count}."
            )

    def training_logits(self, logits: Tensor, call: BallTrainingCall) -> Tensor:
        """Decode model logits to the training heatmap resolution."""
        self.validate_logits(logits, call.model_call)
        squeezed = logits.squeeze(1)
        target_size = cast(tuple[int, int], tuple(call.target_heatmaps.shape[-2:]))
        return resize_heatmap_sequence(squeezed, target_size)

    def probability_heatmaps(
        self,
        logits: Tensor,
        call: BallModelCall,
        *,
        target_size_hw: tuple[int, int] | None = None,
    ) -> Tensor:
        """Decode validated logits into probability heatmaps."""
        self.validate_logits(logits, call)
        squeezed = logits.squeeze(1)
        if target_size_hw is not None:
            squeezed = resize_heatmap_sequence(squeezed, target_size_hw)
        return torch.sigmoid(squeezed)

    def resized_logits(
        self,
        logits: Tensor,
        call: BallModelCall,
        *,
        target_size_hw: tuple[int, int],
    ) -> Tensor:
        """Validate and resize logits without changing their numerical meaning."""
        self.validate_logits(logits, call)
        return resize_heatmap_sequence(logits.squeeze(1), target_size_hw)

    def prediction(
        self,
        logits: Tensor,
        call: BallModelCall,
        *,
        subpixel_refine: bool,
    ) -> BallPrediction:
        """Decode logits into the canonical typed inference result."""
        heatmaps = self.probability_heatmaps(logits, call)
        coords, confidence = heatmaps_to_argmax(heatmaps)
        if subpixel_refine:
            coords = refine_peaks_log_parabolic(heatmaps, coords)
        return BallPrediction(
            coords=coords.cpu(),
            confidence=confidence.cpu(),
            heatmaps=heatmaps.cpu(),
        )

    def mdd_features(self, images: Tensor) -> Tensor:
        """Build canonical ``(B,2,T,H,W)`` MDD features for visualization."""
        call = self.prepare_images(images)
        return self._rgb_frames_to_mdd(call.images)

    def _to_model_input(self, images: Tensor) -> Tensor:
        if self.spec.input_mode == "rgb":
            features = images
        else:
            features = self._rgb_frames_to_mdd(images).permute(0, 2, 1, 3, 4)
        if self.spec.input_layout == "btchw":
            return features.contiguous()
        return features.permute(0, 2, 1, 3, 4).contiguous()

    def _rgb_frames_to_mdd(self, images: Tensor) -> Tensor:
        weights = images.new_tensor(_RGB_TO_LUMINANCE).view(1, 1, 3, 1, 1)
        luminance = (images * weights).sum(dim=2)
        brighten = torch.zeros_like(luminance)
        darken = torch.zeros_like(luminance)
        frame_diffs = luminance[:, 1:] - luminance[:, :-1]
        brighten[:, 1:] = self._power_normalize(torch.clamp(frame_diffs, min=0.0))
        darken[:, 1:] = self._power_normalize(torch.clamp(-frame_diffs, min=0.0))
        return torch.stack([brighten, darken], dim=1)

    def _power_normalize(self, values: Tensor) -> Tensor:
        logits = torch.clamp(
            self.spec.mdd_gain * (values.abs() - self.spec.mdd_offset),
            min=-80.0,
            max=80.0,
        )
        return torch.sigmoid(logits)


def _required_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    if key not in batch:
        raise BallModelIOError(f"Ball batch is missing required field {key!r}.")
    value = batch[key]
    if not isinstance(value, Tensor):
        raise BallModelIOError(f"Ball batch field {key!r} must be a Tensor.")
    return value


def _require_float_tensor(tensor: Tensor, *, name: str, rank: int) -> None:
    if tensor.ndim != rank:
        raise BallModelIOError(
            f"{name} must be rank {rank}, got shape {tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        raise BallModelIOError(f"{name} must use a floating dtype, got {tensor.dtype}.")


def _require_finite(tensor: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise BallModelIOError(f"{name} must contain only finite values.")


__all__ = [
    "BallModelIOAdapter",
    "DINOv3BallExecutionBoundary",
    "build_ball_model_input_spec",
]
