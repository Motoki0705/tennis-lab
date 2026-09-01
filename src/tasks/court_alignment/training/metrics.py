"""One-to-one, symmetry-aware court-alignment evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.court_alignment.geometry.court import canonical_court_keypoints
from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    CourtInstances,
    CourtPeakDetections,
    decode_court_instances,
    decode_keypoint_peaks,
    fit_similarity_2d,
    group_peak_votes,
)
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX

NUM_KEYPOINTS = 14


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    mean_error_px: Tensor
    errors_px: Tensor
    prediction_indices: Tensor
    target_indices: Tensor
    visible_prediction_indices: Tensor
    visible_target_indices: Tensor


@dataclass(slots=True)
class _MetricTotals:
    sample_count: int = 0
    exact_count: int = 0
    count_absolute_error: int = 0
    predicted_count: int = 0
    target_count: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    matched_instance_count: int = 0
    center_error_sum: float = 0.0
    visible_keypoint_gt_count: int = 0
    visible_keypoint_matched_count: int = 0
    keypoint_error_sum: float = 0.0
    keypoint_pck_2_count: int = 0
    keypoint_pck_4_count: int = 0
    coverage_gate_evaluated_count: int = 0
    coverage_gate_pass_count: int = 0
    insufficient_coverage_count: int = 0
    sim2_pair_count: int = 0
    sim2_evaluation_count: int = 0
    sim2_translation_error_sum: float = 0.0
    sim2_rotation_error_sum: float = 0.0
    sim2_scale_error_sum: float = 0.0
    sim2_unavailable_count: int = 0
    no_match_penalty_count: int = 0
    no_match_penalty_sum: float = 0.0

    def merge(self, other: _MetricTotals) -> None:
        for field in fields(self):
            setattr(
                self,
                field.name,
                getattr(self, field.name) + getattr(other, field.name),
            )

    def compute(self) -> dict[str, float]:
        sample_denominator = max(self.sample_count, 1)
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = (
            self.true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            self.true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        fallback_error = (
            self.no_match_penalty_sum / self.no_match_penalty_count
            if self.no_match_penalty_count > 0
            else 0.0
        )
        center_error = (
            self.center_error_sum / self.matched_instance_count
            if self.matched_instance_count > 0
            else fallback_error
        )
        keypoint_error = (
            self.keypoint_error_sum / self.visible_keypoint_gt_count
            if self.visible_keypoint_gt_count > 0
            else fallback_error
        )
        pck_2 = (
            self.keypoint_pck_2_count / self.visible_keypoint_gt_count
            if self.visible_keypoint_gt_count > 0
            else 0.0
        )
        pck_4 = (
            self.keypoint_pck_4_count / self.visible_keypoint_gt_count
            if self.visible_keypoint_gt_count > 0
            else 0.0
        )
        visible_coverage = (
            self.visible_keypoint_matched_count / self.visible_keypoint_gt_count
            if self.visible_keypoint_gt_count > 0
            else 0.0
        )
        coverage_gate_pass_rate = (
            self.coverage_gate_pass_count / self.coverage_gate_evaluated_count
            if self.coverage_gate_evaluated_count > 0
            else 0.0
        )
        if self.sim2_evaluation_count > 0:
            sim2_translation = (
                self.sim2_translation_error_sum / self.sim2_evaluation_count
            )
            sim2_rotation = (
                self.sim2_rotation_error_sum / self.sim2_evaluation_count
            )
            sim2_scale = self.sim2_scale_error_sum / self.sim2_evaluation_count
        elif self.no_match_penalty_count > 0:
            sim2_translation = fallback_error
            sim2_rotation = 180.0
            sim2_scale = 1.0
        else:
            sim2_translation = 0.0
            sim2_rotation = 0.0
            sim2_scale = 0.0

        result = {
            "instance_tp": float(self.true_positive),
            "instance_fp": float(self.false_positive),
            "instance_fn": float(self.false_negative),
            "instance_precision": float(precision),
            "instance_recall": float(recall),
            "instance_f1": float(f1),
            "false_positive_count": float(self.false_positive),
            "instance_count_accuracy": self.exact_count / sample_denominator,
            "instance_count_mae": self.count_absolute_error / sample_denominator,
            "predicted_instance_count": self.predicted_count / sample_denominator,
            "target_instance_count": self.target_count / sample_denominator,
            "matched_instance_count": float(self.matched_instance_count),
            "matched_center_mean_error_px": float(center_error),
            "instance_kp_mean_error_px": float(keypoint_error),
            "instance_kp_pck_2px": float(pck_2),
            "instance_kp_pck_4px": float(pck_4),
            "visible_kp_gt": float(self.visible_keypoint_gt_count),
            "visible_kp_matched": float(self.visible_keypoint_matched_count),
            "visible_kp_coverage": float(visible_coverage),
            "coverage_gate_evaluated_count": float(
                self.coverage_gate_evaluated_count
            ),
            "coverage_gate_pass_count": float(self.coverage_gate_pass_count),
            "coverage_gate_pass_rate": float(coverage_gate_pass_rate),
            "insufficient_coverage_count": float(
                self.insufficient_coverage_count
            ),
            "sim2_pair_count": float(self.sim2_pair_count),
            "sim2_evaluation_count": float(self.sim2_evaluation_count),
            "sim2_unavailable_count": float(self.sim2_unavailable_count),
            "sim2_translation_error_px": float(sim2_translation),
            "sim2_rotation_error_deg": float(sim2_rotation),
            "sim2_scale_relative_error": float(sim2_scale),
        }
        # Stable aliases preserve existing ablation dashboards while making it
        # explicit that these values now come from matched court instances.
        result["peak_mean_error_px"] = result["instance_kp_mean_error_px"]
        result["recall_at_2px"] = result["instance_kp_pck_2px"]
        result["recall_at_4px"] = result["instance_kp_pck_4px"]
        result["kp_mean_distance_px"] = result["instance_kp_mean_error_px"]
        result["recall_2px"] = result["instance_kp_pck_2px"]
        result["recall_4px"] = result["instance_kp_pck_4px"]
        result["instance_center_mean_error_px"] = result[
            "matched_center_mean_error_px"
        ]
        result["visible_kp_gt_count"] = result["visible_kp_gt"]
        result["visible_kp_matched_count"] = result["visible_kp_matched"]
        result["coverage_gate_insufficient_count"] = result[
            "insufficient_coverage_count"
        ]
        return result


def _target_layout(keypoints: Tensor, *, batch_size: int) -> Tensor:
    """Return target instances as ``(B,N,14,2)``."""
    if (
        keypoints.ndim != 4
        or keypoints.shape[0] != batch_size
        or keypoints.shape[-1] != 2
    ):
        raise ValueError("keypoints must have shape (B,N,14,2) or (B,14,N,2).")
    if keypoints.shape[2] == NUM_KEYPOINTS:
        result = keypoints
    elif keypoints.shape[1] == NUM_KEYPOINTS:
        result = keypoints.permute(0, 2, 1, 3)
    else:
        raise ValueError("keypoints must have one axis of length fourteen.")
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError("keypoints must be finite floating point values.")
    return result


def _visibility_layout(
    visibility: Tensor | None,
    *,
    target_shape: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    if visibility is None:
        return torch.ones(target_shape, dtype=torch.bool, device=device)
    if visibility.shape == target_shape:
        result = visibility
    elif (
        visibility.ndim == 3
        and visibility.shape[0] == target_shape[0]
        and visibility.shape[1] == target_shape[2]
        and visibility.shape[2] == target_shape[1]
    ):
        result = visibility.permute(0, 2, 1)
    else:
        raise ValueError("visibility shape must match keypoints' instance/KP axes.")
    if result.dtype != torch.bool:
        raise TypeError("visibility must have boolean dtype.")
    if result.device != device:
        raise ValueError("keypoints and visibility must share a device.")
    return result


def _image_hw(
    image_size: Tensor | tuple[int, int] | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if image_size is None:
        raise ValueError(
            "image_size is required to define finite penalties for unmatched courts."
        )
    if isinstance(image_size, Tensor):
        if image_size.shape != (batch_size, 2) or image_size.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("image_size must have shape (B,2) and integer dtype.")
        if image_size.device != device:
            raise ValueError("image_size must share the target device.")
        result = image_size.to(dtype=dtype)
    else:
        if (
            len(image_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in image_size
            )
        ):
            raise ValueError("image_size must be an integer (height,width) pair.")
        result = torch.tensor(image_size, device=device, dtype=dtype).expand(
            batch_size, -1
        )
    if bool(torch.any(result <= 0)):
        raise ValueError("image_size values must be positive.")
    return result


def _num_courts(
    value: Tensor | None,
    *,
    batch_size: int,
    max_courts: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.full(
            (batch_size,), max_courts, dtype=torch.long, device=device
        )
    if value.shape != (batch_size,) or value.dtype not in {torch.int32, torch.int64}:
        raise ValueError("num_courts must have shape (B,) and integer dtype.")
    if value.device != device:
        raise ValueError("num_courts must share the target device.")
    if bool(torch.any((value < 0) | (value > max_courts))):
        raise ValueError("num_courts must fit the target instance axis.")
    return value.long()


def _pair_evaluation(
    prediction: CourtInstanceBatch,
    prediction_index: int,
    target_keypoints: Tensor,
    target_visibility: Tensor,
    *,
    minimum_common_keypoints: int,
    minimum_visible_keypoints: int,
    minimum_visible_fraction: float,
) -> _PairEvaluation | None:
    device = prediction.keypoints_px.device
    half_turn = torch.as_tensor(
        CAMERA_VIEW_HALF_TURN_INDEX, dtype=torch.long, device=device
    )
    prediction_valid = prediction.valid[prediction_index]
    visible_target_indices = torch.nonzero(
        target_visibility, as_tuple=False
    ).flatten()
    visible_count = int(visible_target_indices.numel())
    if visible_count < minimum_visible_keypoints:
        return None
    candidates: list[_PairEvaluation] = []
    visible_prediction_options = (
        visible_target_indices,
        half_turn[visible_target_indices],
    )
    for visible_prediction_indices in visible_prediction_options:
        present = prediction_valid[visible_prediction_indices]
        prediction_indices = visible_prediction_indices[present]
        selected_target_indices = visible_target_indices[present]
        common_count = int(prediction_indices.numel())
        coverage = common_count / visible_count
        if (
            common_count < minimum_common_keypoints
            or coverage < minimum_visible_fraction
        ):
            continue
        errors = torch.linalg.vector_norm(
            prediction.keypoints_px[prediction_index, prediction_indices]
            - target_keypoints[selected_target_indices],
            dim=-1,
        )
        candidates.append(
            _PairEvaluation(
                mean_error_px=errors.mean(),
                errors_px=errors,
                prediction_indices=prediction_indices,
                target_indices=selected_target_indices,
                visible_prediction_indices=visible_prediction_indices,
                visible_target_indices=visible_target_indices,
            )
        )
    if not candidates:
        return None
    # Direct correspondence wins an exact tie, providing a stable convention
    # for geometrically symmetric examples.
    return min(
        enumerate(candidates),
        key=lambda item: (float(item[1].mean_error_px), item[0]),
    )[1]


def _validate_metric_options(
    *,
    match_max_error_px: float,
    minimum_common_keypoints: int,
    minimum_visible_keypoints: int,
    minimum_visible_fraction: float,
    minimum_sim2_keypoints: int,
) -> None:
    if not math.isfinite(match_max_error_px) or match_max_error_px <= 0.0:
        raise ValueError("match_max_error_px must be finite and positive.")
    integer_options = (
        ("minimum_common_keypoints", minimum_common_keypoints, 1),
        ("minimum_visible_keypoints", minimum_visible_keypoints, 1),
        ("minimum_sim2_keypoints", minimum_sim2_keypoints, 2),
    )
    for name, value, lower_bound in integer_options:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower_bound <= value <= NUM_KEYPOINTS
        ):
            raise ValueError(
                f"{name} must be an integer in [{lower_bound},{NUM_KEYPOINTS}]."
            )
    if (
        type(minimum_visible_fraction) is not float
        or not math.isfinite(minimum_visible_fraction)
        or not 0.0 < minimum_visible_fraction <= 1.0
    ):
        raise ValueError("minimum_visible_fraction must be a float in (0,1].")


def _gated_assignment(costs: Tensor, *, max_error_px: float) -> list[tuple[int, int]]:
    """Return deterministic maximum-cardinality Hungarian assignments."""
    num_targets, num_predictions = costs.shape
    if num_targets == 0 or num_predictions == 0:
        return []
    values = costs.detach().to(device="cpu", dtype=torch.float64).numpy()
    gate = float(torch.as_tensor(max_error_px, dtype=costs.dtype).item())
    accepted = np.isfinite(values) & (values <= gate)
    unmatched_penalty = float(num_targets + 1) * gate
    if not math.isfinite(unmatched_penalty):
        raise OverflowError("Hungarian unmatched penalty is not finite.")
    matrix = np.full(
        (num_targets, num_predictions + num_targets), np.inf, dtype=np.float64
    )
    matrix[:, :num_predictions][accepted] = values[accepted]
    for target_index in range(num_targets):
        matrix[target_index, num_predictions + target_index] = unmatched_penalty
    target_indices, columns = linear_sum_assignment(matrix)
    return [
        (int(target_index), int(column))
        for target_index, column in zip(
            target_indices.tolist(), columns.tolist(), strict=True
        )
        if column < num_predictions
    ]


def _rotation_error_deg(first: Tensor, second: Tensor) -> float:
    difference = torch.atan2(torch.sin(first - second), torch.cos(first - second))
    return math.degrees(abs(float(difference)))


def _add_sim2_penalty(totals: _MetricTotals, *, diagonal: float) -> None:
    totals.sim2_evaluation_count += 1
    totals.sim2_unavailable_count += 1
    totals.sim2_translation_error_sum += diagonal
    totals.sim2_rotation_error_sum += 180.0
    totals.sim2_scale_error_sum += 1.0


def _evaluate_instances(
    predictions: CourtInstances | Sequence[CourtInstanceBatch],
    keypoints: Tensor,
    visibility: Tensor | None,
    *,
    centers: Tensor | None,
    num_courts: Tensor | None,
    image_size: Tensor | tuple[int, int] | None,
    target_normalized: bool,
    match_max_error_px: float,
    minimum_common_keypoints: int,
    minimum_visible_keypoints: int,
    minimum_visible_fraction: float,
    minimum_sim2_keypoints: int,
) -> _MetricTotals:
    samples = (
        predictions.samples
        if isinstance(predictions, CourtInstances)
        else tuple(predictions)
    )
    targets = _target_layout(keypoints, batch_size=len(samples))
    visible = _visibility_layout(
        visibility,
        target_shape=(targets.shape[0], targets.shape[1], targets.shape[2]),
        device=targets.device,
    )
    if any(sample.keypoints_px.device != targets.device for sample in samples):
        raise ValueError("Predicted instances and targets must share a device.")
    counts = _num_courts(
        num_courts,
        batch_size=len(samples),
        max_courts=targets.shape[1],
        device=targets.device,
    )
    sizes = _image_hw(
        image_size,
        batch_size=len(samples),
        device=targets.device,
        dtype=targets.dtype,
    )
    xy_scales = sizes[:, [1, 0]] - 1.0
    if target_normalized:
        targets = targets * xy_scales[:, None, None]
    if centers is None:
        target_centers = targets.new_zeros((len(samples), targets.shape[1], 2))
        for batch_index in range(len(samples)):
            for target_index in range(int(counts[batch_index])):
                target_valid = visible[batch_index, target_index]
                if not bool(target_valid.any()):
                    raise ValueError(
                        "A target court without visible keypoints requires explicit centers."
                    )
                target_centers[batch_index, target_index] = targets[
                    batch_index, target_index, target_valid
                ].mean(dim=0)
    else:
        if centers.shape != targets.shape[:2] + (2,):
            raise ValueError("centers must have shape (B,N,2) matching keypoints.")
        if (
            not centers.is_floating_point()
            or not bool(torch.isfinite(centers).all())
            or centers.device != targets.device
        ):
            raise ValueError(
                "centers must be finite floating point on the target device."
            )
        target_centers = (
            centers * xy_scales[:, None] if target_normalized else centers
        )

    totals = _MetricTotals()
    canonical = canonical_court_keypoints(
        dtype=targets.dtype, device=targets.device
    )
    for batch_index, sample in enumerate(samples):
        target_count = int(counts[batch_index])
        prediction_count = sample.num_instances
        totals.sample_count += 1
        totals.predicted_count += prediction_count
        totals.target_count += target_count
        totals.count_absolute_error += abs(prediction_count - target_count)
        totals.exact_count += int(prediction_count == target_count)
        totals.visible_keypoint_gt_count += int(
            visible[batch_index, :target_count].sum()
        )
        diagonal = max(
            float(torch.linalg.vector_norm(xy_scales[batch_index])), 1.0
        )

        evaluations: dict[tuple[int, int], _PairEvaluation] = {}
        costs = targets.new_full((target_count, prediction_count), float("inf"))
        for target_index in range(target_count):
            for prediction_index in range(prediction_count):
                evaluation = _pair_evaluation(
                    sample,
                    prediction_index,
                    targets[batch_index, target_index],
                    visible[batch_index, target_index],
                    minimum_common_keypoints=minimum_common_keypoints,
                    minimum_visible_keypoints=minimum_visible_keypoints,
                    minimum_visible_fraction=minimum_visible_fraction,
                )
                totals.coverage_gate_evaluated_count += 1
                if evaluation is not None:
                    totals.coverage_gate_pass_count += 1
                    evaluations[target_index, prediction_index] = evaluation
                    costs[target_index, prediction_index] = evaluation.mean_error_px
                else:
                    totals.insufficient_coverage_count += 1
        matches = _gated_assignment(costs, max_error_px=match_max_error_px)
        true_positive = len(matches)
        totals.true_positive += true_positive
        totals.false_positive += prediction_count - true_positive
        totals.false_negative += target_count - true_positive
        totals.matched_instance_count += true_positive
        if true_positive == 0 and (prediction_count > 0 or target_count > 0):
            totals.no_match_penalty_count += 1
            totals.no_match_penalty_sum += diagonal

        matched_target_indices = {target_index for target_index, _ in matches}
        for target_index, prediction_index in matches:
            evaluation = evaluations[target_index, prediction_index]
            totals.center_error_sum += float(
                torch.linalg.vector_norm(
                    sample.centers_px[prediction_index]
                    - target_centers[batch_index, target_index]
                )
            )
            present = sample.valid[
                prediction_index, evaluation.visible_prediction_indices
            ]
            visible_errors = targets.new_full(
                (evaluation.visible_target_indices.numel(),), diagonal
            )
            visible_errors[present] = evaluation.errors_px
            totals.visible_keypoint_matched_count += int(present.sum())
            totals.keypoint_error_sum += float(visible_errors.sum())
            totals.keypoint_pck_2_count += int((visible_errors <= 2.0).sum())
            totals.keypoint_pck_4_count += int((visible_errors <= 4.0).sum())

            # The selected target semantic indices are also the canonical
            # source indices for both fits.  This removes the pi ambiguity
            # when the half-turn correspondence was selected.
            if evaluation.prediction_indices.numel() < minimum_sim2_keypoints:
                _add_sim2_penalty(totals, diagonal=diagonal)
                continue
            source = canonical[evaluation.target_indices]
            try:
                predicted_fit = fit_similarity_2d(
                    source,
                    sample.keypoints_px[
                        prediction_index, evaluation.prediction_indices
                    ],
                )
                target_fit = fit_similarity_2d(
                    source,
                    targets[
                        batch_index, target_index, evaluation.target_indices
                    ],
                )
            except ValueError:
                _add_sim2_penalty(totals, diagonal=diagonal)
                continue
            totals.sim2_pair_count += 1
            totals.sim2_evaluation_count += 1
            totals.sim2_translation_error_sum += float(
                torch.linalg.vector_norm(
                    predicted_fit.translation_px - target_fit.translation_px
                )
            )
            totals.sim2_rotation_error_sum += _rotation_error_deg(
                predicted_fit.rotation_rad, target_fit.rotation_rad
            )
            totals.sim2_scale_error_sum += abs(
                float(predicted_fit.scale_px_per_metre)
                / float(target_fit.scale_px_per_metre)
                - 1.0
            )
        for target_index in range(target_count):
            if target_index in matched_target_indices:
                continue
            visible_count = int(visible[batch_index, target_index].sum())
            totals.keypoint_error_sum += diagonal * visible_count
            _add_sim2_penalty(totals, diagonal=diagonal)
        if target_count == 0:
            for _ in range(prediction_count):
                _add_sim2_penalty(totals, diagonal=diagonal)
    return totals


def instance_alignment_metrics(
    predictions: CourtInstances | Sequence[CourtInstanceBatch],
    keypoints: Tensor,
    visibility: Tensor | None = None,
    *,
    centers: Tensor | None = None,
    num_courts: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None,
    target_normalized: bool = False,
    match_max_error_px: float = 8.0,
    minimum_common_keypoints: int = 4,
    minimum_visible_keypoints: int = 4,
    minimum_visible_fraction: float = 0.5,
    minimum_sim2_keypoints: int = 4,
) -> dict[str, float]:
    """Evaluate decoded courts with gated one-to-one KP correspondence.

    A pair must meet the common-KP, visible-KP, and visible-fraction gates
    before Hungarian matching. Every visible GT keypoint remains in the PCK
    denominator; missing predictions receive the image-diagonal error. Sim(2)
    fits that are unavailable receive image-diagonal translation, 180-degree
    rotation, and unit relative-scale penalties in the reported mean.
    """
    _validate_metric_options(
        match_max_error_px=match_max_error_px,
        minimum_common_keypoints=minimum_common_keypoints,
        minimum_visible_keypoints=minimum_visible_keypoints,
        minimum_visible_fraction=minimum_visible_fraction,
        minimum_sim2_keypoints=minimum_sim2_keypoints,
    )
    return _evaluate_instances(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=num_courts,
        image_size=image_size,
        target_normalized=target_normalized,
        match_max_error_px=match_max_error_px,
        minimum_common_keypoints=minimum_common_keypoints,
        minimum_visible_keypoints=minimum_visible_keypoints,
        minimum_visible_fraction=minimum_visible_fraction,
        minimum_sim2_keypoints=minimum_sim2_keypoints,
    ).compute()


def _as_instances(
    predictions: CourtInstances | CourtPeakDetections | Tensor,
    center_votes: Tensor | None,
    *,
    threshold: float,
    nms_kernel: int,
    max_peaks: int,
    subpixel_refine: bool,
    cluster_distance_px: float,
    max_instances: int | None,
) -> CourtInstances:
    if isinstance(predictions, CourtInstances):
        return predictions
    if isinstance(predictions, Tensor):
        if center_votes is None:
            raise ValueError("center_votes are required when decoding logits.")
        return decode_court_instances(
            predictions,
            center_votes,
            threshold=threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
            subpixel_refine=subpixel_refine,
            cluster_distance_px=cluster_distance_px,
            max_instances=max_instances,
        )
    grouped = group_peak_votes(
        predictions.keypoints_px,
        predictions.center_votes_px,
        predictions.valid,
        predictions.scores,
        cluster_distance_px=cluster_distance_px,
        max_instances=max_instances,
    )
    return (
        CourtInstances((grouped,))
        if isinstance(grouped, CourtInstanceBatch)
        else grouped
    )


def compute_alignment_metrics(
    predictions: CourtInstances | CourtPeakDetections | Tensor,
    keypoints: Tensor,
    visibility: Tensor | None = None,
    *,
    center_votes: Tensor | None = None,
    centers: Tensor | None = None,
    num_courts: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None,
    target_normalized: bool = False,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
    subpixel_refine: bool = True,
    cluster_distance_px: float = 12.0,
    max_instances: int | None = None,
    match_max_error_px: float = 8.0,
    minimum_common_keypoints: int = 4,
    minimum_visible_keypoints: int = 4,
    minimum_visible_fraction: float = 0.5,
    minimum_sim2_keypoints: int = 4,
) -> dict[str, float]:
    """Decode if needed, then compute the unified instance metric set."""
    instances = _as_instances(
        predictions,
        center_votes,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
        subpixel_refine=subpixel_refine,
        cluster_distance_px=cluster_distance_px,
        max_instances=max_instances,
    )
    return instance_alignment_metrics(
        instances,
        keypoints,
        visibility,
        centers=centers,
        num_courts=num_courts,
        image_size=image_size,
        target_normalized=target_normalized,
        match_max_error_px=match_max_error_px,
        minimum_common_keypoints=minimum_common_keypoints,
        minimum_visible_keypoints=minimum_visible_keypoints,
        minimum_visible_fraction=minimum_visible_fraction,
        minimum_sim2_keypoints=minimum_sim2_keypoints,
    )


def peak_metrics(
    predictions: CourtPeakDetections | Tensor,
    keypoints: Tensor,
    visibility: Tensor | None = None,
    *,
    center_votes: Tensor | None = None,
    image_size: Tensor | tuple[int, int] | None = None,
    target_normalized: bool = False,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
) -> dict[str, float]:
    """Legacy channel-only peak diagnostic, separate from instance metrics."""
    if isinstance(predictions, Tensor):
        if predictions.ndim != 4:
            raise ValueError("prediction logits must have shape (B,14,H,W).")
        if center_votes is None:
            center_votes = torch.zeros(
                (predictions.shape[0], 2, *predictions.shape[-2:]),
                dtype=predictions.dtype,
                device=predictions.device,
            )
        detections = decode_keypoint_peaks(
            predictions,
            center_votes,
            threshold=threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
            subpixel_refine=True,
        )
        batch_size = predictions.shape[0]
    else:
        detections = predictions
        batch_size = predictions.keypoints_px.shape[0]
    target_instances = _target_layout(keypoints, batch_size=batch_size)
    visible_instances = _visibility_layout(
        visibility,
        target_shape=(
            target_instances.shape[0],
            target_instances.shape[1],
            target_instances.shape[2],
        ),
        device=keypoints.device,
    )
    targets = target_instances.permute(0, 2, 1, 3)
    visible = visible_instances.permute(0, 2, 1)
    if target_normalized:
        sizes = _image_hw(
            image_size,
            batch_size=batch_size,
            device=targets.device,
            dtype=targets.dtype,
        )
        targets = targets * (sizes[:, [1, 0]] - 1.0)[:, None, None]
    if image_size is not None:
        sizes = _image_hw(
            image_size,
            batch_size=batch_size,
            device=targets.device,
            dtype=targets.dtype,
        )
        penalties = torch.linalg.vector_norm(
            sizes[:, [1, 0]] - 1.0, dim=-1
        ).clamp_min(1.0)
    else:
        penalties = targets.new_ones((batch_size,))
    error_groups: list[Tensor] = []
    for batch_index in range(batch_size):
        for channel in range(NUM_KEYPOINTS):
            expected = targets[batch_index, channel][visible[batch_index, channel]]
            accepted = detections.keypoints_px[batch_index, channel][
                detections.valid[batch_index, channel]
            ]
            if expected.numel() == 0:
                continue
            if accepted.numel() == 0:
                error_groups.append(
                    penalties[batch_index].expand(expected.shape[0])
                )
            else:
                error_groups.append(torch.cdist(expected, accepted).amin(dim=1))
    if error_groups:
        errors = torch.cat(error_groups)
        mean_error = float(errors.mean())
        pck_2 = float((errors <= 2.0).float().mean())
        pck_4 = float((errors <= 4.0).float().mean())
        count = float(errors.numel())
    else:
        mean_error = pck_2 = pck_4 = count = 0.0
    return {
        "peak_mean_error_px": mean_error,
        "recall_at_2px": pck_2,
        "recall_at_4px": pck_4,
        "peak_count": count,
        "kp_mean_distance_px": mean_error,
        "recall_2px": pck_2,
        "recall_4px": pck_4,
    }


# Compatibility name retained for callers of the prototype API. The
# implementation now evaluates complete court instances and Hungarian matches.
instance_grouping_metrics = instance_alignment_metrics


class CourtAlignmentMetrics:
    """Accumulate unified court-instance statistics over an epoch."""

    def __init__(
        self,
        threshold: float = 0.25,
        nms_kernel: int = 3,
        max_peaks: int = 8,
        subpixel_refine: bool = True,
        cluster_distance_px: float = 12.0,
        max_instances: int | None = None,
        match_max_error_px: float = 8.0,
        minimum_common_keypoints: int = 4,
        minimum_visible_keypoints: int = 4,
        minimum_visible_fraction: float = 0.5,
        minimum_sim2_keypoints: int = 4,
    ) -> None:
        _validate_metric_options(
            match_max_error_px=match_max_error_px,
            minimum_common_keypoints=minimum_common_keypoints,
            minimum_visible_keypoints=minimum_visible_keypoints,
            minimum_visible_fraction=minimum_visible_fraction,
            minimum_sim2_keypoints=minimum_sim2_keypoints,
        )
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.max_peaks = max_peaks
        self.subpixel_refine = subpixel_refine
        self.cluster_distance_px = cluster_distance_px
        self.max_instances = max_instances
        self.match_max_error_px = match_max_error_px
        self.minimum_common_keypoints = minimum_common_keypoints
        self.minimum_visible_keypoints = minimum_visible_keypoints
        self.minimum_visible_fraction = minimum_visible_fraction
        self.minimum_sim2_keypoints = minimum_sim2_keypoints
        self._totals = _MetricTotals()

    def reset(self) -> None:
        self._totals = _MetricTotals()

    def update(
        self,
        predictions: CourtInstances | CourtPeakDetections | Tensor,
        keypoints: Tensor,
        visibility: Tensor | None = None,
        *,
        center_votes: Tensor | None = None,
        centers: Tensor | None = None,
        num_courts: Tensor | None = None,
        image_size: Tensor | tuple[int, int] | None,
        target_normalized: bool = False,
    ) -> None:
        instances = _as_instances(
            predictions,
            center_votes,
            threshold=self.threshold,
            nms_kernel=self.nms_kernel,
            max_peaks=self.max_peaks,
            subpixel_refine=self.subpixel_refine,
            cluster_distance_px=self.cluster_distance_px,
            max_instances=self.max_instances,
        )
        self._totals.merge(
            _evaluate_instances(
                instances,
                keypoints,
                visibility,
                centers=centers,
                num_courts=num_courts,
                image_size=image_size,
                target_normalized=target_normalized,
                match_max_error_px=self.match_max_error_px,
                minimum_common_keypoints=self.minimum_common_keypoints,
                minimum_visible_keypoints=self.minimum_visible_keypoints,
                minimum_visible_fraction=self.minimum_visible_fraction,
                minimum_sim2_keypoints=self.minimum_sim2_keypoints,
            )
        )

    def compute(self) -> dict[str, float]:
        return self._totals.compute()


__all__ = [
    "CourtAlignmentMetrics",
    "compute_alignment_metrics",
    "instance_alignment_metrics",
    "instance_grouping_metrics",
    "peak_metrics",
]
