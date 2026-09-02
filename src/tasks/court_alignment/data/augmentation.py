"""Typed image-only augmentations for ground-court procedural samples.

Every registered transform consumes one shared ``numpy.random.Generator`` and
returns a new sample without changing its geometry payload. The registry and
ordered compose contract make B00 domain randomization explicit in Hydra while
keeping validation/test inputs clean.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_image(image: Tensor) -> None:
    """Validate the strict line-evidence image contract."""

    if not isinstance(image, Tensor):
        raise TypeError("Augmentable image must be a torch.Tensor.")
    if (
        image.ndim != 3
        or image.shape[0] != 1
        or image.shape[1] <= 0
        or image.shape[2] <= 0
    ):
        raise ValueError("Augmentable image must have non-empty shape [1,H,W].")
    if not image.is_floating_point():
        raise TypeError("Augmentable image must have a floating-point dtype.")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("Augmentable image values must be finite.")
    minimum = float(image.amin().item())
    maximum = float(image.amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("Augmentable image values must lie in [0,1].")
    if maximum <= 0.0:
        raise ValueError("Augmentable image must contain positive line evidence.")
    if minimum >= 1.0:
        raise ValueError("Augmentable image must contain background evidence.")


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("augmentation rng must be numpy.random.Generator.")


@dataclass(frozen=True, slots=True)
class AugmentableGroundCourtSample:
    """Image and geometry payload passed through an augmentation."""

    image: Tensor  # [1,H,W] float line evidence in [0,1]
    keypoints: Tensor  # [M,14,2], output pixels
    visibility: Tensor  # [M,14], bool
    centers: Tensor  # [M,2], output pixels
    instance_ids: Tensor  # [M], int64

    def __post_init__(self) -> None:
        _validate_image(self.image)
        geometry = {
            "keypoints": self.keypoints,
            "visibility": self.visibility,
            "centers": self.centers,
            "instance_ids": self.instance_ids,
        }
        for name, value in geometry.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Augmentable {name} must be a torch.Tensor.")
        if self.keypoints.ndim != 3 or self.keypoints.shape[-2:] != (14, 2):
            raise ValueError("Augmentable keypoints must have shape [M,14,2].")
        count = self.keypoints.shape[0]
        if count <= 0:
            raise ValueError("Augmentable geometry must contain at least one instance.")
        if not self.keypoints.is_floating_point():
            raise TypeError("Augmentable keypoints must have a floating-point dtype.")
        if self.visibility.shape != (count, 14) or self.visibility.dtype != torch.bool:
            raise ValueError(
                "Augmentable visibility must have shape [M,14] and bool dtype."
            )
        if self.centers.shape != (count, 2):
            raise ValueError("Augmentable centers must have shape [M,2].")
        if not self.centers.is_floating_point():
            raise TypeError("Augmentable centers must have a floating-point dtype.")
        if self.instance_ids.shape != (count,) or self.instance_ids.dtype != torch.long:
            raise ValueError("Augmentable instance_ids must be int64 [M].")
        if any(value.device != self.image.device for value in geometry.values()):
            raise ValueError(
                "Augmentable image and geometry must be on the same device."
            )
        if not bool(torch.isfinite(self.keypoints).all().item()):
            raise ValueError("Augmentable keypoints must be finite.")
        if not bool(torch.isfinite(self.centers).all().item()):
            raise ValueError("Augmentable centers must be finite.")
        if bool((self.instance_ids < 0).any().item()):
            raise ValueError("Augmentable instance_ids must be non-negative.")
        if torch.unique(self.instance_ids).numel() != count:
            raise ValueError("Augmentable instance_ids must be unique.")


class GroundCourtAugmentation(Protocol):
    """Callable augmentation contract used by the procedural dataset."""

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample: ...


@dataclass(frozen=True, slots=True)
class GroundCourtAugmentationConfig:
    """Name and typed parameter mapping for one registered augmentation."""

    name: str = "identity"
    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("augmentation name must be a non-empty string.")
        if not isinstance(self.params, Mapping):
            raise TypeError("augmentation params must be a mapping.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundCourtAugmentationConfig:
        """Parse ``{'type': ..., 'params': {...}}`` with no unknown fields."""

        if not isinstance(value, Mapping):
            raise TypeError("augmentation config must be a mapping.")
        unknown = set(value) - {"name", "type", "params"}
        if unknown:
            raise ValueError(f"Unknown augmentation config fields: {sorted(unknown)}")
        if "name" in value and "type" in value:
            raise ValueError("augmentation config cannot define both name and type.")
        name = value.get("name", value.get("type", "identity"))
        params = value.get("params", {})
        if not isinstance(name, str):
            raise TypeError("augmentation config name must be a string.")
        if not isinstance(params, Mapping):
            raise TypeError("augmentation config params must be a mapping.")
        return cls(name=name, params=params)


def _replace_image(
    sample: AugmentableGroundCourtSample, image: Tensor
) -> AugmentableGroundCourtSample:
    """Replace only image data; geometry tensor identity is deliberately retained."""

    return AugmentableGroundCourtSample(
        image=image,
        keypoints=sample.keypoints,
        visibility=sample.visibility,
        centers=sample.centers,
        instance_ids=sample.instance_ids,
    )


def _prepare(sample: AugmentableGroundCourtSample, rng: np.random.Generator) -> None:
    if not isinstance(sample, AugmentableGroundCourtSample):
        raise TypeError("augmentation sample must be AugmentableGroundCourtSample.")
    _validate_image(sample.image)
    _validate_rng(rng)


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1].")
    return result


def _non_negative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _int_range(value: object, *, name: str, minimum: int = 0) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an integer pair.")
    items = tuple(value)
    if len(items) != 2 or any(type(item) is not int for item in items):
        raise TypeError(f"{name} must be an integer pair.")
    low, high = items
    if low < minimum or high < low:
        raise ValueError(f"{name} must be ordered with minimum {minimum}.")
    return low, high


def _float_range(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a numeric pair.")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"{name} must contain two values.")
    parsed = tuple(_non_negative_float(item, name=name) for item in items)
    low, high = parsed
    if low < minimum or high < low or (maximum is not None and high > maximum):
        upper = "" if maximum is None else f" and maximum {maximum}"
        raise ValueError(f"{name} must be ordered with minimum {minimum}{upper}.")
    return low, high


def _validate_params(params: Mapping[str, object], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown augmentation parameters: {sorted(unknown)}")


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return low
    return float(rng.uniform(low, high))


def _integer(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    low, high = bounds
    if low == high:
        return low
    return int(rng.integers(low, high + 1))


@dataclass(frozen=True, slots=True)
class IdentityAugmentation:
    """Baseline no-op augmentation."""

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        return sample


@dataclass(frozen=True, slots=True)
class RandomLineMorphology:
    """Random grey-scale dilation or erosion without moving annotations.

    An erosion proposal that would delete every line pixel is explicitly
    rejected, since an empty heatmap has no valid alignment target.
    """

    probability: float = 0.7
    dilate_probability: float = 0.8
    kernel_size_choices: tuple[int, ...] = (3,)
    iterations: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="probability")
        )
        object.__setattr__(
            self,
            "dilate_probability",
            _probability(self.dilate_probability, name="dilate_probability"),
        )
        if (
            not isinstance(self.kernel_size_choices, tuple)
            or not self.kernel_size_choices
        ):
            raise TypeError("kernel_size_choices must be a non-empty integer tuple.")
        if any(type(item) is not int for item in self.kernel_size_choices):
            raise TypeError("kernel_size_choices must contain integers.")
        if any(item <= 0 or item % 2 == 0 for item in self.kernel_size_choices):
            raise ValueError("kernel_size_choices must contain positive odd values.")
        _positive_int(self.iterations, name="iterations")

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        if rng.random() >= self.probability:
            return sample
        kernel = int(rng.choice(self.kernel_size_choices))
        dilate = rng.random() < self.dilate_probability
        result = sample.image
        for _ in range(self.iterations):
            padded = F.pad(
                result.unsqueeze(0),
                (kernel // 2,) * 4,
                mode="replicate",
            )
            if dilate:
                result = F.max_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0)
            else:
                result = -F.max_pool2d(-padded, kernel_size=kernel, stride=1).squeeze(0)
        if float(result.amax().item()) <= 0.0:
            result = sample.image
        return _replace_image(sample, result.clamp(0.0, 1.0))


@dataclass(frozen=True, slots=True)
class RandomHeatmapBlur:
    """Gaussian blur that retains continuous line probabilities."""

    probability: float = 0.7
    sigma_range: tuple[float, float] = (0.0, 1.5)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="probability")
        )
        object.__setattr__(
            self,
            "sigma_range",
            _float_range(self.sigma_range, name="sigma_range"),
        )

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        if rng.random() >= self.probability:
            return sample
        sigma = _uniform(rng, self.sigma_range)
        if sigma <= 1.0e-6:
            return sample
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(
            -radius,
            radius + 1,
            dtype=torch.float32,
            device=sample.image.device,
        )
        kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        working = sample.image.to(dtype=torch.float32).unsqueeze(0)
        horizontal = F.conv2d(
            F.pad(working, (radius, radius, 0, 0), mode="replicate"),
            kernel.view(1, 1, 1, -1),
        )
        blurred = F.conv2d(
            F.pad(horizontal, (0, 0, radius, radius), mode="replicate"),
            kernel.view(1, 1, -1, 1),
        ).squeeze(0)
        return _replace_image(
            sample, blurred.to(dtype=sample.image.dtype).clamp(0.0, 1.0)
        )


def _segment_mask(
    image: Tensor,
    *,
    center_x: float,
    center_y: float,
    angle_rad: float,
    length_px: float,
    width_px: float,
) -> Tensor:
    height, width = image.shape[-2:]
    ys = torch.arange(height, dtype=torch.float32, device=image.device).view(-1, 1)
    xs = torch.arange(width, dtype=torch.float32, device=image.device).view(1, -1)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    along = (xs - center_x) * cosine + (ys - center_y) * sine
    across = -(xs - center_x) * sine + (ys - center_y) * cosine
    return (along.abs() <= length_px / 2.0) & (across.abs() <= width_px / 2.0)


@dataclass(frozen=True, slots=True)
class RandomLineDropout:
    """Erase bounded oriented gaps centered on existing line evidence."""

    probability: float = 0.3
    gap_count_range: tuple[int, int] = (1, 4)
    gap_length_px_range: tuple[float, float] = (2.0, 12.0)
    gap_width_px_range: tuple[float, float] = (1.0, 4.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="probability")
        )
        object.__setattr__(
            self,
            "gap_count_range",
            _int_range(self.gap_count_range, name="gap_count_range", minimum=1),
        )
        object.__setattr__(
            self,
            "gap_length_px_range",
            _float_range(
                self.gap_length_px_range, name="gap_length_px_range", minimum=1.0
            ),
        )
        object.__setattr__(
            self,
            "gap_width_px_range",
            _float_range(
                self.gap_width_px_range, name="gap_width_px_range", minimum=1.0
            ),
        )

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        if rng.random() >= self.probability:
            return sample
        result = sample.image.clone()
        count = _integer(rng, self.gap_count_range)
        for _ in range(count):
            foreground = torch.nonzero(result[0] > 0.0, as_tuple=False)
            if foreground.shape[0] <= 1:
                break
            coordinate = foreground[int(rng.integers(0, foreground.shape[0]))]
            center_y = float(coordinate[0].item())
            center_x = float(coordinate[1].item())
            mask = _segment_mask(
                result,
                center_x=center_x,
                center_y=center_y,
                angle_rad=float(rng.uniform(0.0, math.pi)),
                length_px=_uniform(rng, self.gap_length_px_range),
                width_px=_uniform(rng, self.gap_width_px_range),
            )
            proposal = result.masked_fill(mask.unsqueeze(0), 0.0)
            if float(proposal.amax().item()) > 0.0:
                result = proposal
        return _replace_image(sample, result)


def _shift_without_wrap(image: Tensor, *, dx: int, dy: int) -> Tensor:
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0.0
    elif dy < 0:
        shifted[..., dy:, :] = 0.0
    if dx > 0:
        shifted[..., :, :dx] = 0.0
    elif dx < 0:
        shifted[..., :, dx:] = 0.0
    return shifted


@dataclass(frozen=True, slots=True)
class RandomGhostLines:
    """Add parallel shifted copies and optional unrelated long line clutter."""

    probability: float = 0.5
    copy_count_range: tuple[int, int] = (1, 3)
    offset_px_range: tuple[int, int] = (3, 15)
    amplitude_range: tuple[float, float] = (0.55, 1.0)
    long_line_count_range: tuple[int, int] = (0, 3)
    long_line_length_px_range: tuple[float, float] = (48.0, 220.0)
    long_line_width_px_range: tuple[float, float] = (1.0, 4.0)
    long_line_amplitude_range: tuple[float, float] = (0.55, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="probability")
        )
        object.__setattr__(
            self,
            "copy_count_range",
            _int_range(self.copy_count_range, name="copy_count_range", minimum=1),
        )
        object.__setattr__(
            self,
            "offset_px_range",
            _int_range(self.offset_px_range, name="offset_px_range", minimum=1),
        )
        object.__setattr__(
            self,
            "amplitude_range",
            _float_range(self.amplitude_range, name="amplitude_range", maximum=1.0),
        )
        object.__setattr__(
            self,
            "long_line_count_range",
            _int_range(self.long_line_count_range, name="long_line_count_range"),
        )
        object.__setattr__(
            self,
            "long_line_length_px_range",
            _float_range(
                self.long_line_length_px_range,
                name="long_line_length_px_range",
                minimum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "long_line_width_px_range",
            _float_range(
                self.long_line_width_px_range,
                name="long_line_width_px_range",
                minimum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "long_line_amplitude_range",
            _float_range(
                self.long_line_amplitude_range,
                name="long_line_amplitude_range",
                maximum=1.0,
            ),
        )

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        if rng.random() >= self.probability:
            return sample
        result = sample.image.clone()
        copy_count = _integer(rng, self.copy_count_range)
        for _ in range(copy_count):
            magnitude = _integer(rng, self.offset_px_range)
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            dx = int(round(magnitude * math.cos(angle)))
            dy = int(round(magnitude * math.sin(angle)))
            if dx == 0 and dy == 0:
                dx = magnitude
            copy = _shift_without_wrap(sample.image, dx=dx, dy=dy)
            result = torch.maximum(result, copy * _uniform(rng, self.amplitude_range))
        height, width = sample.image.shape[-2:]
        line_count = _integer(rng, self.long_line_count_range)
        for _ in range(line_count):
            mask = _segment_mask(
                result,
                center_x=float(rng.uniform(0.0, max(width - 1, 1))),
                center_y=float(rng.uniform(0.0, max(height - 1, 1))),
                angle_rad=float(rng.uniform(0.0, math.pi)),
                length_px=_uniform(rng, self.long_line_length_px_range),
                width_px=_uniform(rng, self.long_line_width_px_range),
            )
            amplitude = _uniform(rng, self.long_line_amplitude_range)
            clutter = mask.to(dtype=result.dtype).unsqueeze(0) * amplitude
            result = torch.maximum(result, clutter)
        return _replace_image(sample, result.clamp(0.0, 1.0))


@dataclass(frozen=True, slots=True)
class RandomProbabilityNoise:
    """Randomize confidence calibration and add isolated detector speckles."""

    probability: float = 0.8
    foreground_amplitude_range: tuple[float, float] = (0.55, 1.0)
    gamma_range: tuple[float, float] = (0.7, 1.5)
    speckle_fraction_range: tuple[float, float] = (0.001, 0.008)
    speckle_amplitude_range: tuple[float, float] = (0.2, 0.9)
    additive_std: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _probability(self.probability, name="probability")
        )
        object.__setattr__(
            self,
            "foreground_amplitude_range",
            _float_range(
                self.foreground_amplitude_range,
                name="foreground_amplitude_range",
                maximum=1.0,
            ),
        )
        gamma_range = _float_range(self.gamma_range, name="gamma_range")
        if gamma_range[0] <= 0.0:
            raise ValueError("gamma_range must be positive.")
        object.__setattr__(self, "gamma_range", gamma_range)
        object.__setattr__(
            self,
            "speckle_fraction_range",
            _float_range(
                self.speckle_fraction_range,
                name="speckle_fraction_range",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "speckle_amplitude_range",
            _float_range(
                self.speckle_amplitude_range,
                name="speckle_amplitude_range",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "additive_std",
            _non_negative_float(self.additive_std, name="additive_std"),
        )

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        if rng.random() >= self.probability:
            return sample
        amplitude = _uniform(rng, self.foreground_amplitude_range)
        gamma = _uniform(rng, self.gamma_range)
        result = (sample.image * amplitude).pow(gamma)
        height, width = result.shape[-2:]
        pixel_count = height * width
        fraction = _uniform(rng, self.speckle_fraction_range)
        speckle_count = min(pixel_count, int(round(pixel_count * fraction)))
        if speckle_count > 0:
            flat_indices = rng.choice(pixel_count, size=speckle_count, replace=False)
            amplitudes = rng.uniform(
                self.speckle_amplitude_range[0],
                self.speckle_amplitude_range[1],
                size=speckle_count,
            )
            index = torch.as_tensor(
                flat_indices, dtype=torch.long, device=result.device
            )
            values = torch.as_tensor(
                amplitudes, dtype=result.dtype, device=result.device
            )
            flat = result.view(-1)
            flat[index] = torch.maximum(flat[index], values)
        if self.additive_std > 0.0:
            noise = rng.normal(0.0, self.additive_std, size=(height, width))
            result = result + torch.as_tensor(
                noise, dtype=result.dtype, device=result.device
            ).unsqueeze(0)
        return _replace_image(sample, result.clamp(0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ComposeAugmentation:
    """Apply an ordered tuple of typed augmentations."""

    augmentations: tuple[GroundCourtAugmentation, ...]

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _prepare(sample, rng)
        result = sample
        for augmentation in self.augmentations:
            result = augmentation(result, rng)
            if not isinstance(result, AugmentableGroundCourtSample):
                raise TypeError(
                    "Ground-court augmentation must return "
                    "AugmentableGroundCourtSample."
                )
        return result


_AugmentationFactory = Callable[[Mapping[str, object]], GroundCourtAugmentation]
_AUGMENTATION_FACTORIES: dict[str, _AugmentationFactory] = {}


def register_augmentation(
    name: str,
) -> Callable[[_AugmentationFactory], _AugmentationFactory]:
    """Register a named factory; duplicate names fail during import/configuration."""

    normalized = name.strip() if isinstance(name, str) else ""
    if not normalized:
        raise ValueError("augmentation registry names must be non-empty strings.")

    def decorator(factory: _AugmentationFactory) -> _AugmentationFactory:
        if normalized in _AUGMENTATION_FACTORIES:
            raise ValueError(f"Augmentation already registered: {normalized!r}.")
        _AUGMENTATION_FACTORIES[normalized] = factory
        return factory

    return decorator


@register_augmentation("identity")
def _build_identity(params: Mapping[str, object]) -> GroundCourtAugmentation:
    if params:
        raise ValueError("identity augmentation does not accept parameters.")
    return IdentityAugmentation()


@register_augmentation("random_line_morphology")
def _build_random_line_morphology(
    params: Mapping[str, object],
) -> GroundCourtAugmentation:
    allowed = {"probability", "dilate_probability", "kernel_size_choices", "iterations"}
    _validate_params(params, allowed)
    choices = params.get("kernel_size_choices", (3,))
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise TypeError("kernel_size_choices must be a non-empty integer sequence.")
    return RandomLineMorphology(
        probability=_probability(params.get("probability", 0.7), name="probability"),
        dilate_probability=_probability(
            params.get("dilate_probability", 0.8), name="dilate_probability"
        ),
        kernel_size_choices=tuple(choices),
        iterations=_positive_int(params.get("iterations", 1), name="iterations"),
    )


@register_augmentation("random_heatmap_blur")
def _build_random_heatmap_blur(params: Mapping[str, object]) -> GroundCourtAugmentation:
    _validate_params(params, {"probability", "sigma_range"})
    return RandomHeatmapBlur(
        probability=_probability(params.get("probability", 0.7), name="probability"),
        sigma_range=_float_range(
            params.get("sigma_range", (0.0, 1.5)), name="sigma_range"
        ),
    )


@register_augmentation("random_line_dropout")
def _build_random_line_dropout(params: Mapping[str, object]) -> GroundCourtAugmentation:
    allowed = {
        "probability",
        "gap_count_range",
        "gap_length_px_range",
        "gap_width_px_range",
    }
    _validate_params(params, allowed)
    return RandomLineDropout(
        probability=_probability(params.get("probability", 0.3), name="probability"),
        gap_count_range=_int_range(
            params.get("gap_count_range", (1, 4)),
            name="gap_count_range",
            minimum=1,
        ),
        gap_length_px_range=_float_range(
            params.get("gap_length_px_range", (2.0, 12.0)),
            name="gap_length_px_range",
            minimum=1.0,
        ),
        gap_width_px_range=_float_range(
            params.get("gap_width_px_range", (1.0, 4.0)),
            name="gap_width_px_range",
            minimum=1.0,
        ),
    )


@register_augmentation("random_ghost_lines")
def _build_random_ghost_lines(params: Mapping[str, object]) -> GroundCourtAugmentation:
    allowed = {
        "probability",
        "copy_count_range",
        "offset_px_range",
        "amplitude_range",
        "long_line_count_range",
        "long_line_length_px_range",
        "long_line_width_px_range",
        "long_line_amplitude_range",
    }
    _validate_params(params, allowed)
    return RandomGhostLines(
        probability=_probability(params.get("probability", 0.5), name="probability"),
        copy_count_range=_int_range(
            params.get("copy_count_range", (1, 3)),
            name="copy_count_range",
            minimum=1,
        ),
        offset_px_range=_int_range(
            params.get("offset_px_range", (3, 15)),
            name="offset_px_range",
            minimum=1,
        ),
        amplitude_range=_float_range(
            params.get("amplitude_range", (0.55, 1.0)),
            name="amplitude_range",
            maximum=1.0,
        ),
        long_line_count_range=_int_range(
            params.get("long_line_count_range", (0, 3)),
            name="long_line_count_range",
        ),
        long_line_length_px_range=_float_range(
            params.get("long_line_length_px_range", (48.0, 220.0)),
            name="long_line_length_px_range",
            minimum=1.0,
        ),
        long_line_width_px_range=_float_range(
            params.get("long_line_width_px_range", (1.0, 4.0)),
            name="long_line_width_px_range",
            minimum=1.0,
        ),
        long_line_amplitude_range=_float_range(
            params.get("long_line_amplitude_range", (0.55, 1.0)),
            name="long_line_amplitude_range",
            maximum=1.0,
        ),
    )


@register_augmentation("random_probability_noise")
def _build_random_probability_noise(
    params: Mapping[str, object],
) -> GroundCourtAugmentation:
    allowed = {
        "probability",
        "foreground_amplitude_range",
        "gamma_range",
        "speckle_fraction_range",
        "speckle_amplitude_range",
        "additive_std",
    }
    _validate_params(params, allowed)
    return RandomProbabilityNoise(
        probability=_probability(params.get("probability", 0.8), name="probability"),
        foreground_amplitude_range=_float_range(
            params.get("foreground_amplitude_range", (0.55, 1.0)),
            name="foreground_amplitude_range",
            maximum=1.0,
        ),
        gamma_range=_float_range(
            params.get("gamma_range", (0.7, 1.5)), name="gamma_range"
        ),
        speckle_fraction_range=_float_range(
            params.get("speckle_fraction_range", (0.001, 0.008)),
            name="speckle_fraction_range",
            maximum=1.0,
        ),
        speckle_amplitude_range=_float_range(
            params.get("speckle_amplitude_range", (0.2, 0.9)),
            name="speckle_amplitude_range",
            maximum=1.0,
        ),
        additive_std=_non_negative_float(
            params.get("additive_std", 0.0), name="additive_std"
        ),
    )


def build_augmentation(
    config: GroundCourtAugmentationConfig | Mapping[str, object] | str | None = None,
) -> GroundCourtAugmentation:
    """Build a registered augmentation and reject unknown names immediately."""

    if config is None:
        resolved = GroundCourtAugmentationConfig()
    elif isinstance(config, GroundCourtAugmentationConfig):
        resolved = config
    elif isinstance(config, str):
        resolved = GroundCourtAugmentationConfig(name=config)
    elif isinstance(config, Mapping):
        resolved = GroundCourtAugmentationConfig.from_mapping(config)
    else:
        raise TypeError("Unsupported ground-court augmentation config.")
    try:
        factory = _AUGMENTATION_FACTORIES[resolved.name]
    except KeyError as error:
        available = ", ".join(sorted(_AUGMENTATION_FACTORIES))
        raise ValueError(
            f"Unknown ground-court augmentation {resolved.name!r}; available: {available}."
        ) from error
    return factory(resolved.params)


def build_augmentations(
    configs: Sequence[GroundCourtAugmentationConfig | Mapping[str, object] | str] = (),
) -> GroundCourtAugmentation:
    """Build an ordered augmentation pipeline, defaulting to identity."""

    built = tuple(build_augmentation(config) for config in configs)
    if not built:
        return IdentityAugmentation()
    return ComposeAugmentation(built)


__all__ = [
    "AugmentableGroundCourtSample",
    "ComposeAugmentation",
    "GroundCourtAugmentation",
    "GroundCourtAugmentationConfig",
    "IdentityAugmentation",
    "RandomGhostLines",
    "RandomHeatmapBlur",
    "RandomLineDropout",
    "RandomLineMorphology",
    "RandomProbabilityNoise",
    "build_augmentation",
    "build_augmentations",
    "register_augmentation",
]
