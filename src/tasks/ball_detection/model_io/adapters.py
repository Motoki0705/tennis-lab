"""Canonical input construction and output decoding for ball models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.configuration import validate_model
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import (
    DEFAULT_CANDIDATE_CONFIG,
    BallCandidateConfig,
    BallModelCall,
    BallModelInputSpec,
    BallModelIOError,
    BallPrediction,
    BallTrainingCall,
)
from src.tasks.ball_detection.model_io.mdd import luminance_to_mdd, mdd_coefficients
from src.tasks.ball_detection.model_io.normalization import (
    IDENTITY_NORMALIZATION,
    BallImageNormalization,
)
from src.tasks.base.model_io import ModelCall
from src.utils.data.heatmaps import (
    heatmaps_to_argmax,
    refine_peaks_log_parabolic,
    resize_heatmap_sequence,
)

_RGB_TO_LUMINANCE = (0.299, 0.587, 0.114)


def build_ball_model_input_spec(config: object) -> BallModelInputSpec:
    """Resolve and validate the static ball model input contract."""
    model_cfg = validate_model(config)
    model_name = str(model_cfg["name"])
    # Saved ConvNeXt checkpoints retain these explicit, constant metadata fields.
    # There is no RGB/layout execution branch.
    in_channels = int(model_cfg["in_channels"])
    num_classes = int(model_cfg["num_classes"])
    configured_frames = int(model_cfg["num_frames"])
    minimum_spatial_size = 4 * 2 ** (len(tuple(model_cfg["dims"])) - 1)
    mdd_a = float(model_cfg["mdd_a"])
    mdd_b = float(model_cfg["mdd_b"])
    gain, offset = mdd_coefficients(mdd_a, mdd_b)
    return BallModelInputSpec(
        model_name=model_name,
        input_mode="mdd",
        input_layout="bcthw",
        in_channels=in_channels,
        num_classes=num_classes,
        configured_frames=configured_frames,
        image_size_hw=None,
        minimum_spatial_size=minimum_spatial_size,
        mdd_gain=gain,
        mdd_offset=offset,
    )


class BallModelIOAdapter:
    """Validate RGB batches and adapt them to one selected ball model."""

    def __init__(
        self,
        spec: BallModelInputSpec,
        *,
        expected_model_type: type[nn.Module],
        minimum_frames: int,
    ) -> None:
        if minimum_frames <= 0:
            raise BallModelIOError("minimum_frames must be positive.")
        if spec.input_mode != "mdd" or spec.input_layout != "bcthw" or spec.in_channels != 2:
            raise BallModelIOError("Ball models require MDD, bcthw layout and two channels")
        if spec.configured_frames <= 0:
            raise BallModelIOError("Ball configured_frames must be positive.")
        self.spec = spec
        self.expected_model_type = expected_model_type
        self.minimum_frames = minimum_frames
        self.minimum_spatial_size = spec.minimum_spatial_size

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

    def prepare_model_call(
        self, images: Tensor, *,
        image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION,
        preprocessed: bool = False,
    ) -> BallModelCall:
        """Build the complete validated argument list before model entry."""
        return self._prepare_direct_model_call(self.prepare_images(
            images, image_normalization=image_normalization, preprocessed=preprocessed,
        ))

    @staticmethod
    def _prepare_direct_model_call(call: BallModelCall) -> BallModelCall:
        return call

    def prepare_images(
        self, images: Tensor, *,
        image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION,
        preprocessed: bool = False,
    ) -> BallModelCall:
        """Validate RGB and apply its declared transform before layout/MDD conversion."""
        if type(preprocessed) is not bool:
            raise BallModelIOError("preprocessed must be a boolean.")
        _require_float_tensor(images, name="images", rank=5)
        if images.dtype != torch.float32:
            raise BallModelIOError(
                f"images must use torch.float32, got {images.dtype}."
            )
        _require_finite(images, name="images")
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
        if preprocessed:
            image_normalization.validate_preprocessed_range(images)
        else:
            IDENTITY_NORMALIZATION.validate_preprocessed_range(images)
            images = image_normalization.apply(images)
        _require_finite(images, name="preprocessed images")
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
        *, image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION,
    ) -> BallTrainingCall:
        """Validate every tensor used by training before the forward pass."""
        images = _required_tensor(batch, "images")
        target_heatmaps = _required_tensor(batch, "heatmaps")
        coords = _required_tensor(batch, "coords")
        visibility = _required_tensor(batch, "visibility")
        supervised = _required_tensor(batch, "supervised")
        original_size = _required_tensor(batch, "original_size")
        model_call = self.prepare_model_call(
            images, image_normalization=image_normalization, preprocessed=True,
        )

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
        if supervised.dtype != torch.bool or supervised.shape != (batch_size, frame_count):
            raise BallModelIOError("supervised must be a boolean (B, T) frame mask.")
        if bool(torch.any(~supervised[..., None] & (visibility > 0))):
            raise BallModelIOError("An unsupervised frame must not carry a visible target.")
        return BallTrainingCall(
            model_call=model_call,
            target_heatmaps=target_heatmaps,
            coords=coords,
            visibility=visibility,
            supervised=supervised,
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
        candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG,
    ) -> BallPrediction:
        """Decode logits into the canonical typed inference result."""
        heatmaps = self.probability_heatmaps(logits, call).float()
        coords, confidence = heatmaps_to_argmax(heatmaps)
        if subpixel_refine:
            coords = refine_peaks_log_parabolic(heatmaps, coords)
        return BallPrediction(
            coords=coords.cpu(),
            confidence=confidence.cpu(),
            heatmaps=heatmaps.cpu(),
            candidates=decode_candidates(
                heatmaps, config=candidate_config, subpixel_refine=subpixel_refine,
            ),
        )

    def mdd_features(
        self, images: Tensor, *,
        image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION,
        preprocessed: bool = False,
    ) -> Tensor:
        """Build MDD from raw RGB or explicitly declared dataset-preprocessed RGB."""
        call = self.prepare_images(
            images, image_normalization=image_normalization, preprocessed=preprocessed,
        )
        return self._rgb_frames_to_mdd(call.images)

    def _to_model_input(self, images: Tensor) -> Tensor:
        return self._rgb_frames_to_mdd(images).contiguous()

    def _rgb_frames_to_mdd(self, images: Tensor) -> Tensor:
        weights = images.new_tensor(_RGB_TO_LUMINANCE).view(1, 1, 3, 1, 1)
        luminance = (images * weights).sum(dim=2)
        return luminance_to_mdd(luminance, gain=self.spec.mdd_gain, offset=self.spec.mdd_offset)


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
    "build_ball_model_input_spec",
]
