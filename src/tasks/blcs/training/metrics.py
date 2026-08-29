"""Metrics for BLCS evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.base.evaluation import (
    compute_axis_wise_position_error,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    court_vectors_target_to_physical,
    resolve_court_keypoint_contract,
)
from src.tasks.base.training.metric_logging import format_metric_threshold
from src.utils.schema.court_normalization import denormalize_court_position

_PHYSICAL_COURT_KEYPOINT_CONTRACT = resolve_court_keypoint_contract(
    PHYSICAL_V1_SELECTOR
)


class BLCSMetrics:
    """Metrics tracker for BLCS evaluation.

    Tracks position errors and accuracy metrics over batches.
    """

    def __init__(
        self,
        *,
        position_threshold_m: float,
        endpoint_threshold_m: float,
        court_keypoint_contract: CourtKeypointContract = (
            _PHYSICAL_COURT_KEYPOINT_CONTRACT
        ),
    ) -> None:
        """Initialize metrics tracker.

        Args:
            position_threshold_m: Threshold for base position accuracy (meters), default 0.3m.
            endpoint_threshold_m: Threshold for base endpoint accuracy (meters), default 0.5m.

        """
        self.position_threshold_m = position_threshold_m
        self.endpoint_threshold_m = endpoint_threshold_m
        canonical_contract = resolve_court_keypoint_contract(
            court_keypoint_contract.selector
        )
        if court_keypoint_contract != canonical_contract:
            raise CourtKeypointContractMismatchError(
                "BLCS metric CourtKP20 contract must be canonical."
            )
        self.court_keypoint_contract = canonical_contract
        self.position_thresholds_m = tuple(
            dict.fromkeys(
                (
                    0.3,
                    self.position_threshold_m,
                    2.0 * self.position_threshold_m,
                    4.0 * self.position_threshold_m,
                )
            )
        )
        self.endpoint_thresholds_m = (
            self.endpoint_threshold_m,
            2.0 * self.endpoint_threshold_m,
        )
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.total_position_error = 0.0
        self.total_x_error = 0.0
        self.total_y_error = 0.0
        self.total_z_error = 0.0
        self.total_endpoint_error = 0.0
        self.num_frames: float = 0.0
        self.num_endpoint_sequences = 0
        self.num_correct_frames = [0.0 for _ in self.position_thresholds_m]
        self.num_correct_endpoints = [0.0 for _ in self.endpoint_thresholds_m]
        self.num_y_sign_correct = 0.0
        self.num_y_sign_targets = 0.0
        self.reference_index_errors: dict[int, list[float]] = {}

    def update(
        self,
        pred_position: Tensor,
        target_position: Tensor,
        mask: Tensor | None = None,
        court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...]
        | None = None,
        reference_view_index: Tensor | None = None,
    ) -> dict[str, float]:
        """Update metrics with a batch of predictions.

        Args:
            pred_position: Predicted positions (normalized), shape (B, T, 3).
            target_position: Target positions (normalized), shape (B, T, 3).
            mask: Visibility mask, shape (B, T).

        Returns:
            dict: Current batch metrics.

        """
        batch_size, seq_len, _ = pred_position.shape

        pred_m = denormalize_court_position(pred_position)
        target_m = denormalize_court_position(target_position)
        target_frame_pred_m = pred_m
        target_frame_target_m = target_m

        if court_reference_provenance is None:
            if self.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
                raise MissingCourtKeypointMetadataError(
                    "BLCS camera_view_v2 metrics require explicit Court reference "
                    "provenance."
                )
        else:
            if len(court_reference_provenance) != batch_size:
                raise ValueError(
                    "BLCS metric provenance must contain one record per batch item."
                )
            pred_rows: list[Tensor] = []
            target_rows: list[Tensor] = []
            for index, provenance in enumerate(court_reference_provenance):
                if not isinstance(provenance, CourtReferenceFrameProvenance):
                    raise TypeError(
                        "BLCS metric provenance entries must be validated records."
                    )
                if provenance.contract != self.court_keypoint_contract:
                    raise CourtKeypointContractMismatchError(
                        f"BLCS metric provenance[{index}] contract "
                        f"{provenance.contract_id!r} does not match runtime "
                        f"{self.court_keypoint_contract.contract_id!r}."
                    )
                pred_row = court_vectors_target_to_physical(pred_m[index], provenance)
                target_row = court_vectors_target_to_physical(
                    target_m[index], provenance
                )
                if not isinstance(pred_row, Tensor) or not isinstance(
                    target_row, Tensor
                ):
                    raise TypeError(
                        "BLCS metric frame conversion returned non-tensors."
                    )
                pred_rows.append(pred_row)
                target_rows.append(target_row)
            pred_m = torch.stack(pred_rows)
            target_m = torch.stack(target_rows)

        # Position distance is rotation invariant. Axis/sign diagnostics use the
        # authoritative reference target frame when v2 metadata is available.
        error = pred_m - target_m
        error_norm = torch.sqrt((error**2).sum(dim=-1) + 1e-8)  # (B, T)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=pred_position.device)
        valid_mask = mask.bool()

        reference_metrics: dict[str, float] = {}
        metric_error = error
        if reference_view_index is not None:
            if reference_view_index.shape != (batch_size,):
                raise ValueError(
                    "reference_view_index must match the BLCS metric batch axis."
                )
            if reference_view_index.dtype != torch.int64:
                raise ValueError("reference_view_index must have dtype torch.int64.")
            if reference_view_index.device != pred_position.device:
                raise ValueError(
                    "reference_view_index must share the BLCS metric tensor device."
                )
            metric_error = target_frame_pred_m - target_frame_target_m
            if valid_mask.any().item():
                axis_error = compute_axis_wise_position_error(
                    target_frame_pred_m,
                    target_frame_target_m,
                    valid_mask=valid_mask,
                )
                reference_metrics.update(
                    {
                        "x_error_m": axis_error.x,
                        "y_error_m": axis_error.y,
                        "z_error_m": axis_error.z,
                    }
                )
                selected_targets = target_frame_target_m[valid_mask]
                non_mid_plane = selected_targets[:, 1].ne(0)
                if non_mid_plane.any().item():
                    y_sign_accuracy = compute_y_sign_accuracy(
                        target_frame_pred_m,
                        target_frame_target_m,
                        valid_mask=valid_mask,
                    )
                    sign_count = float(non_mid_plane.sum().item())
                    self.num_y_sign_correct += y_sign_accuracy * sign_count
                    self.num_y_sign_targets += sign_count
                    reference_metrics["y_sign_accuracy"] = y_sign_accuracy
                sample_indices: list[int] = []
                sample_errors: list[Tensor] = []
                for sample_index in range(batch_size):
                    if valid_mask[sample_index].any().item():
                        sample_indices.append(sample_index)
                        sample_errors.append(
                            torch.linalg.vector_norm(
                                metric_error[sample_index, valid_mask[sample_index]],
                                dim=-1,
                            ).mean()
                        )
                sample_index_tensor = torch.tensor(
                    sample_indices,
                    dtype=torch.int64,
                    device=reference_view_index.device,
                )
                strata = stratify_metric_by_reference_view_index(
                    torch.stack(sample_errors),
                    reference_view_index[sample_index_tensor],
                )
                for index, value in strata.items():
                    reference_metrics[f"reference_index_{index}_position_error_m"] = (
                        value
                    )
                for sample_index, sample_error in zip(
                    sample_indices,
                    sample_errors,
                    strict=True,
                ):
                    reference_index = int(reference_view_index[sample_index].item())
                    self.reference_index_errors.setdefault(
                        reference_index,
                        [],
                    ).append(float(sample_error.item()))

        # Aggregate only actual observations. Padding-only sequences do not
        # contribute either frames or endpoints.
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return {}
        self.num_frames += num_valid
        self.num_endpoint_sequences += int(valid_mask.any(dim=1).sum().item())

        # Position errors
        masked_error = (error_norm * valid_mask).sum().item()
        self.total_position_error += masked_error

        # Per-axis errors
        self.total_x_error += (metric_error[:, :, 0].abs() * valid_mask).sum().item()
        self.total_y_error += (metric_error[:, :, 1].abs() * valid_mask).sum().item()
        self.total_z_error += (metric_error[:, :, 2].abs() * valid_mask).sum().item()

        # Accuracy (within thresholds)
        for i, threshold in enumerate(self.position_thresholds_m):
            within = (error_norm < threshold).float()
            self.num_correct_frames[i] += (within * valid_mask).sum().item()

        # Endpoint error (last valid frame per sequence)
        for b in range(batch_size):
            valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1]
                endpoint_error = error_norm[b, last_idx].item()
                self.total_endpoint_error += endpoint_error
                for i, threshold in enumerate(self.endpoint_thresholds_m):
                    if endpoint_error < threshold:
                        self.num_correct_endpoints[i] += 1

        # Return current batch metrics
        batch_metrics = {
            "position_error_m": masked_error / num_valid,
            "x_error_m": (metric_error[:, :, 0].abs() * valid_mask).sum().item()
            / num_valid,
            "y_error_m": (metric_error[:, :, 1].abs() * valid_mask).sum().item()
            / num_valid,
            "z_error_m": (metric_error[:, :, 2].abs() * valid_mask).sum().item()
            / num_valid,
        }
        batch_metrics.update(reference_metrics)
        return batch_metrics

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics.

        Returns:
            dict: Aggregated metrics.

        """
        if self.num_frames == 0:
            raise RuntimeError(
                "BLCSMetrics.compute() requires at least one valid frame; "
                "the epoch contained no metric observations."
            )

        metrics: dict[str, float] = {
            "position_error_m": self.total_position_error / self.num_frames,
            "x_error_m": self.total_x_error / self.num_frames,
            "y_error_m": self.total_y_error / self.num_frames,
            "z_error_m": self.total_z_error / self.num_frames,
            "endpoint_error_m": self.total_endpoint_error / self.num_endpoint_sequences,
        }

        for i, threshold in enumerate(self.position_thresholds_m):
            key = f"position_accuracy_{format_metric_threshold(threshold)}m"
            metrics[key] = self.num_correct_frames[i] / self.num_frames

        for i, threshold in enumerate(self.endpoint_thresholds_m):
            key = f"endpoint_accuracy_{format_metric_threshold(threshold)}m"
            metrics[key] = self.num_correct_endpoints[i] / self.num_endpoint_sequences

        if self.num_y_sign_targets:
            metrics["y_sign_accuracy"] = (
                self.num_y_sign_correct / self.num_y_sign_targets
            )
        metrics.update(
            {
                f"reference_index_{index}_position_error_m": sum(values) / len(values)
                for index, values in self.reference_index_errors.items()
            }
        )

        return metrics
