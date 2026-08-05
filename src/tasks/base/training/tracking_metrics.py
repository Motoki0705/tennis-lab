"""Shared lifecycle-aware localization, presence, and query-reuse metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from src.tasks.base.configuration import exact_config_mapping, require_config_value
from src.utils.configuration import ConfigurationTypeError, SemanticConfigurationError

Assignment = tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class TrackingMetricConfig:
    """Single composed-config authority for lifecycle metric thresholds."""

    presence_threshold: float
    duplicate_distance: float

    def __post_init__(self) -> None:
        for name in ("presence_threshold", "duplicate_distance"):
            value = getattr(self, name)
            if type(value) is not float:
                raise ConfigurationTypeError(
                    f"tracking_metrics.{name}: expected float, "
                    f"got {type(value).__name__}."
                )
        if not 0.0 < self.presence_threshold < 1.0:
            raise SemanticConfigurationError(
                "tracking_metrics.presence_threshold must be within (0, 1)."
            )
        if self.duplicate_distance <= 0.0:
            raise SemanticConfigurationError(
                "tracking_metrics.duplicate_distance must be > 0."
            )

    @classmethod
    def from_mapping(cls, value: object) -> TrackingMetricConfig:
        """Parse the exact two-key tracking-metric mapping with no defaults."""
        mapping = exact_config_mapping(
            value,
            path="tracking_metrics",
            required_keys=frozenset({"presence_threshold", "duplicate_distance"}),
        )
        return cls(
            presence_threshold=cast(
                "float",
                require_config_value(
                    mapping, "presence_threshold", float, path="tracking_metrics"
                ),
            ),
            duplicate_distance=cast(
                "float",
                require_config_value(
                    mapping, "duplicate_distance", float, path="tracking_metrics"
                ),
            ),
        )


def _boolean_segments(mask: Tensor) -> list[tuple[int, int]]:
    values = mask.bool().tolist()
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate([*values, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append((start, index))
            start = None
    return segments


def _instance_segments(
    instance_ids: Tensor,
    valid_frames: Tensor,
) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start: int | None = None
    current_id = -1
    values = instance_ids.tolist()
    valid = valid_frames.bool().tolist()
    for index in range(len(values) + 1):
        instance_id = int(values[index]) if index < len(values) and valid[index] else -1
        if instance_id != current_id:
            if current_id >= 0 and start is not None:
                segments.append((start, index, current_id))
            start = index if instance_id >= 0 else None
            current_id = instance_id
    return segments


def _match_predicted_segments(
    target_segments: Sequence[tuple[int, int, int]],
    predicted_active: Tensor,
    *,
    clip_length: int,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    predicted_segments = _boolean_segments(predicted_active)
    unused = set(range(len(predicted_segments)))
    birth_errors: list[Tensor] = []
    death_errors: list[Tensor] = []
    selected: list[int | None] = []
    for target_birth, target_death, _ in target_segments:
        if not unused:
            birth_errors.append(
                torch.tensor(
                    float(clip_length),
                    dtype=torch.float32,
                    device=predicted_active.device,
                )
            )
            death_errors.append(
                torch.tensor(
                    float(clip_length),
                    dtype=torch.float32,
                    device=predicted_active.device,
                )
            )
            selected.append(None)
            continue
        best = min(
            unused,
            key=lambda index: (
                -max(
                    0,
                    min(target_death, predicted_segments[index][1])
                    - max(target_birth, predicted_segments[index][0]),
                ),
                abs(target_birth - predicted_segments[index][0])
                + abs(target_death - predicted_segments[index][1]),
            ),
        )
        unused.remove(best)
        pred_birth, pred_death = predicted_segments[best]
        birth_errors.append(
            torch.tensor(
                float(abs(pred_birth - target_birth)),
                dtype=torch.float32,
                device=predicted_active.device,
            )
        )
        death_errors.append(
            torch.tensor(
                float(abs(pred_death - target_death)),
                dtype=torch.float32,
                device=predicted_active.device,
            )
        )
        selected.append(best)

    reuse = predicted_active.new_zeros((), dtype=torch.float32)
    for previous, current in zip(selected, selected[1:], strict=False):
        if previous is None or current is None or previous == current:
            continue
        previous_segment = predicted_segments[previous]
        current_segment = predicted_segments[current]
        if previous_segment[1] <= current_segment[0]:
            reuse += 1.0
    return birth_errors, death_errors, reuse


def common_lifecycle_tracking_metrics(
    prediction: dict[str, Tensor],
    batch: dict[str, Tensor],
    assignments: Sequence[Assignment],
    *,
    config: TrackingMetricConfig,
) -> dict[str, Tensor]:
    """Compute common metrics without connecting IDs across inactive gaps."""
    pred_position = prediction["position"]
    pred_active = prediction["presence_logits"].sigmoid() >= config.presence_threshold
    aligned_presence = torch.zeros_like(pred_active)
    position_errors: list[Tensor] = []
    birth_errors: list[Tensor] = []
    death_errors: list[Tensor] = []
    missed = pred_position.new_zeros(())
    segment_switches = pred_position.new_zeros(())
    query_reuse = pred_position.new_zeros(())
    matched_queries: list[set[int]] = []

    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        matched_queries.append(set(query_indices.tolist()))
        valid_frames = batch["frame_mask"][batch_index]
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index] & valid_frames
            )
            aligned_presence[batch_index, :, query_index] = active
            if active.any():
                position_errors.append(
                    torch.linalg.vector_norm(
                        pred_position[batch_index, active, query_index]
                        - batch["target_position"][batch_index, active, target_index],
                        dim=-1,
                    ).mean()
                )
                missed += (~pred_active[batch_index, active, query_index]).sum()

            target_segments = _instance_segments(
                batch["target_instance_id"][batch_index, :, target_index],
                valid_frames,
            )
            segment_birth, segment_death, reuse = _match_predicted_segments(
                target_segments,
                pred_active[batch_index, :, query_index] & valid_frames,
                clip_length=int(valid_frames.sum()),
            )
            birth_errors.extend(segment_birth)
            death_errors.extend(segment_death)
            query_reuse += reuse

            for birth, death, _ in target_segments:
                segment_range = torch.arange(birth, death, device=pred_position.device)
                distances = torch.linalg.vector_norm(
                    pred_position[batch_index, segment_range]
                    - batch["target_position"][
                        batch_index, segment_range, target_index, None
                    ],
                    dim=-1,
                ).masked_fill(~pred_active[batch_index, segment_range], float("inf"))
                has_prediction = pred_active[batch_index, segment_range].any(-1)
                nearest = distances.argmin(-1)
                consecutive = has_prediction[:-1] & has_prediction[1:]
                if consecutive.any():
                    segment_switches += (
                        (nearest[1:] != nearest[:-1]) & consecutive
                    ).sum()

    valid = batch["frame_mask"].unsqueeze(-1)
    true_positive = (pred_active & aligned_presence & valid).sum()
    false_positive = (pred_active & ~aligned_presence & valid).sum()
    false_negative = (~pred_active & aligned_presence & valid).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

    duplicate = pred_position.new_zeros(())
    inactive_false_positive = pred_position.new_zeros(())
    for batch_index in range(pred_active.shape[0]):
        unmatched = torch.ones(
            pred_active.shape[-1], dtype=torch.bool, device=pred_active.device
        )
        if matched_queries[batch_index]:
            unmatched[list(matched_queries[batch_index])] = False
        inactive_false_positive += (
            pred_active[batch_index]
            & unmatched[None]
            & batch["frame_mask"][batch_index, :, None]
        ).sum()
        for frame_index in (
            torch.nonzero(batch["frame_mask"][batch_index], as_tuple=False)
            .flatten()
            .tolist()
        ):
            active_queries = torch.nonzero(
                pred_active[batch_index, frame_index], as_tuple=False
            ).flatten()
            if active_queries.numel() < 2:
                continue
            distances = torch.cdist(
                pred_position[batch_index, frame_index, active_queries],
                pred_position[batch_index, frame_index, active_queries],
            )
            duplicate += torch.triu(
                distances < config.duplicate_distance, diagonal=1
            ).sum()

    illegal_overlap = pred_position.new_zeros(())
    target_ids = batch["target_instance_id"]
    for batch_index in range(target_ids.shape[0]):
        for frame_index in (
            torch.nonzero(batch["frame_mask"][batch_index], as_tuple=False)
            .flatten()
            .tolist()
        ):
            active_ids = target_ids[batch_index, frame_index]
            active_ids = active_ids[active_ids >= 0]
            if active_ids.numel() > 1:
                illegal_overlap += active_ids.numel() - active_ids.unique().numel()

    zero = pred_position.new_zeros(())
    return {
        "position_error": (
            torch.stack(position_errors).mean() if position_errors else zero
        ),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "lifecycle_presence_f1": f1,
        "birth_frame_error": (
            torch.stack(birth_errors).mean() if birth_errors else zero
        ),
        "death_frame_error": (
            torch.stack(death_errors).mean() if death_errors else zero
        ),
        "query_reuse_count": query_reuse,
        "illegal_overlap_count": illegal_overlap,
        "segment_id_switches": segment_switches,
        "id_switches": segment_switches,
        "duplicate_active_tracks": duplicate,
        "missed_gt_frames": missed,
        "inactive_query_false_positives": inactive_false_positive,
    }


__all__ = [
    "Assignment",
    "TrackingMetricConfig",
    "common_lifecycle_tracking_metrics",
]
