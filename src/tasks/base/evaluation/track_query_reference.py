"""Reference-frame metrics shared by BLCS and PLCS training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class PairedReferenceEvaluationError(ValueError):
    """Raised when paired inputs cannot produce an unambiguous metric."""


@dataclass(frozen=True, slots=True)
class AxisWisePositionError:
    """Mean absolute position error in target-frame X/Y/Z axes."""

    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        """Return stable metric field names for result bundles."""
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True, slots=True)
class PairedReferencePositionMetrics:
    """Core per-run position and local-reference-index metrics."""

    y_sign_accuracy: float
    axis_wise_position_error: AxisWisePositionError
    local_reference_index_error: dict[int, float]


def _require_pair(
    prediction: Tensor,
    target: Tensor,
    *,
    trailing_width: int,
    quantity: str,
) -> tuple[Tensor, Tensor]:
    """Validate one metric pair and normalize only its numeric precision.

    AMP predictions may be float16/bfloat16 while dataset targets remain
    float32.  Metrics use float32 for every non-float64 pair and preserve
    float64 when either operand requests it; shape and device identity stay
    strict.
    """
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError(f"{quantity} prediction and target must be tensors.")
    if prediction.shape != target.shape:
        raise PairedReferenceEvaluationError(
            f"{quantity} prediction/target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}."
        )
    if prediction.ndim < 1 or prediction.shape[-1] != trailing_width:
        raise PairedReferenceEvaluationError(
            f"{quantity} values must have trailing width {trailing_width}; got "
            f"{tuple(prediction.shape)}."
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise PairedReferenceEvaluationError(
            f"{quantity} prediction and target must use floating dtypes."
        )
    if prediction.device != target.device:
        raise PairedReferenceEvaluationError(
            f"{quantity} prediction and target must share a device."
        )
    metric_dtype = (
        torch.float64
        if torch.float64 in (prediction.dtype, target.dtype)
        else torch.float32
    )
    return prediction.to(dtype=metric_dtype), target.to(dtype=metric_dtype)


def _selected_rows(
    value: Tensor,
    *,
    valid_mask: Tensor | None,
    quantity: str,
) -> Tensor:
    leading_shape = value.shape[:-1]
    if valid_mask is None:
        selected = value.reshape(-1, value.shape[-1])
    else:
        if not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a torch.Tensor.")
        if valid_mask.dtype != torch.bool:
            raise PairedReferenceEvaluationError("valid_mask must have bool dtype.")
        if valid_mask.shape != leading_shape:
            raise PairedReferenceEvaluationError(
                f"valid_mask must have shape {tuple(leading_shape)}, got "
                f"{tuple(valid_mask.shape)}."
            )
        if valid_mask.device != value.device:
            raise PairedReferenceEvaluationError(
                "valid_mask must share the metric tensor device."
            )
        selected = value[valid_mask]
    if selected.numel() == 0:
        raise PairedReferenceEvaluationError(
            f"{quantity} metric has no selected observations."
        )
    if not torch.isfinite(selected).all().item():
        raise PairedReferenceEvaluationError(
            f"{quantity} metric inputs must be finite on selected observations."
        )
    return selected


def compute_y_sign_accuracy(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    zero_tolerance: float = 0.0,
) -> float:
    """Return Y-sign accuracy, explicitly excluding near-mid-plane targets."""
    prediction, target = _require_pair(
        prediction,
        target,
        trailing_width=3,
        quantity="position",
    )
    if not isinstance(zero_tolerance, (int, float)) or zero_tolerance < 0:
        raise PairedReferenceEvaluationError(
            f"zero_tolerance must be non-negative, got {zero_tolerance!r}."
        )
    predicted_rows = _selected_rows(
        prediction,
        valid_mask=valid_mask,
        quantity="Y-sign",
    )
    target_rows = _selected_rows(
        target,
        valid_mask=valid_mask,
        quantity="Y-sign",
    )
    non_mid_plane = target_rows[:, 1].abs() > float(zero_tolerance)
    if not non_mid_plane.any().item():
        raise PairedReferenceEvaluationError(
            "Y-sign metric has no targets outside the explicit mid-plane tolerance."
        )
    matches = torch.sign(predicted_rows[non_mid_plane, 1]).eq(
        torch.sign(target_rows[non_mid_plane, 1])
    )
    return float(matches.float().mean().item())


def compute_axis_wise_position_error(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> AxisWisePositionError:
    """Return selected-observation mean absolute X/Y/Z error."""
    prediction, target = _require_pair(
        prediction,
        target,
        trailing_width=3,
        quantity="position",
    )
    difference = (prediction - target).abs()
    selected = _selected_rows(
        difference,
        valid_mask=valid_mask,
        quantity="axis-wise position",
    )
    means = selected.mean(dim=0)
    return AxisWisePositionError(
        x=float(means[0].item()),
        y=float(means[1].item()),
        z=float(means[2].item()),
    )


def compute_heading_error_radians(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> float:
    """Return mean unsigned angle between selected 2D PLCS headings."""
    prediction, target = _require_pair(
        prediction,
        target,
        trailing_width=2,
        quantity="heading",
    )
    predicted_rows = _selected_rows(
        prediction,
        valid_mask=valid_mask,
        quantity="heading",
    )
    target_rows = _selected_rows(
        target,
        valid_mask=valid_mask,
        quantity="heading",
    )
    predicted_norm = torch.linalg.vector_norm(predicted_rows, dim=-1)
    target_norm = torch.linalg.vector_norm(target_rows, dim=-1)
    if (predicted_norm <= 0).any().item() or (target_norm <= 0).any().item():
        raise PairedReferenceEvaluationError(
            "Heading metric requires non-zero selected heading vectors."
        )
    cosine = (
        (predicted_rows * target_rows).sum(dim=-1) / (predicted_norm * target_norm)
    ).clamp(-1.0, 1.0)
    return float(torch.acos(cosine).mean().item())


def stratify_metric_by_reference_view_index(
    metric_values: Tensor,
    reference_view_index: Tensor,
) -> dict[int, float]:
    """Mean arbitrary per-sample metric values by batch-local reference index."""
    if not isinstance(metric_values, Tensor) or not isinstance(
        reference_view_index, Tensor
    ):
        raise TypeError("metric_values and reference_view_index must be tensors.")
    if metric_values.ndim < 1:
        raise PairedReferenceEvaluationError(
            "metric_values must have a leading batch axis."
        )
    batch_size = metric_values.shape[0]
    if reference_view_index.shape != (batch_size,):
        raise PairedReferenceEvaluationError(
            f"reference_view_index must have shape ({batch_size},)."
        )
    if reference_view_index.dtype != torch.int64:
        raise PairedReferenceEvaluationError(
            "reference_view_index must have dtype torch.int64."
        )
    if metric_values.device != reference_view_index.device:
        raise PairedReferenceEvaluationError(
            "metric_values and reference_view_index must share a device."
        )
    if not metric_values.is_floating_point():
        raise PairedReferenceEvaluationError("metric_values must be floating.")
    if not torch.isfinite(metric_values).all().item():
        raise PairedReferenceEvaluationError("metric_values must be finite.")
    if (reference_view_index < 0).any().item():
        raise PairedReferenceEvaluationError(
            "reference_view_index cannot contain padding or negative values."
        )
    metric_dtype = (
        torch.float64 if metric_values.dtype == torch.float64 else torch.float32
    )
    stable_values = metric_values.to(dtype=metric_dtype)
    per_sample = stable_values.reshape(batch_size, -1).mean(dim=1)
    result: dict[int, float] = {}
    for index in sorted(int(value) for value in torch.unique(reference_view_index)):
        selected = per_sample[reference_view_index.eq(index)]
        result[index] = float(selected.mean().item())
    return result


def compute_paired_reference_position_metrics(
    prediction: Tensor,
    target: Tensor,
    reference_view_index: Tensor,
    *,
    valid_mask: Tensor | None = None,
    zero_tolerance: float = 0.0,
) -> PairedReferencePositionMetrics:
    """Compute the common position report including local-index stratification."""
    axis_error = compute_axis_wise_position_error(
        prediction,
        target,
        valid_mask=valid_mask,
    )
    prediction, target = _require_pair(
        prediction,
        target,
        trailing_width=3,
        quantity="position",
    )
    sample_error = torch.linalg.vector_norm(prediction - target, dim=-1)
    if valid_mask is not None:
        if valid_mask.shape != sample_error.shape or valid_mask.dtype != torch.bool:
            raise PairedReferenceEvaluationError(
                "valid_mask must be bool and match the position leading axes."
            )
        counts = valid_mask.reshape(valid_mask.shape[0], -1).sum(dim=1)
        if (counts == 0).any().item():
            raise PairedReferenceEvaluationError(
                "Every sample needs at least one valid local-index metric value."
            )
        sample_error = (
            sample_error.masked_fill(~valid_mask, 0.0)
            .reshape(valid_mask.shape[0], -1)
            .sum(dim=1)
            / counts
        )
    return PairedReferencePositionMetrics(
        y_sign_accuracy=compute_y_sign_accuracy(
            prediction,
            target,
            valid_mask=valid_mask,
            zero_tolerance=zero_tolerance,
        ),
        axis_wise_position_error=axis_error,
        local_reference_index_error=stratify_metric_by_reference_view_index(
            sample_error,
            reference_view_index,
        ),
    )


__all__ = [
    "AxisWisePositionError",
    "PairedReferenceEvaluationError",
    "PairedReferencePositionMetrics",
    "compute_axis_wise_position_error",
    "compute_heading_error_radians",
    "compute_paired_reference_position_metrics",
    "compute_y_sign_accuracy",
    "stratify_metric_by_reference_view_index",
]
