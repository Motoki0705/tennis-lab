"""Deterministic camera-local tracking from normalized 2-D observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor

from src.tasks.base.configuration import exact_config_mapping, require_config_value
from src.utils.configuration import ConfigurationTypeError, SemanticConfigurationError

CostReduction = Literal["mean", "median"]
OverflowPolicy = Literal["error"]

_ASSOCIATION_KEYS = frozenset(
    {
        "max_distance",
        "max_missed_frames",
        "min_reuse_gap_frames",
        "use_velocity_prediction",
        "min_common_keypoints",
        "cost_reduction",
        "overflow_policy",
    }
)
_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


@dataclass(frozen=True, slots=True)
class ObservationTrackingConfig:
    """Strict association and lifecycle settings for observation tracking."""

    max_distance: float
    max_missed_frames: int
    min_reuse_gap_frames: int
    use_velocity_prediction: bool
    min_common_keypoints: int
    cost_reduction: CostReduction
    overflow_policy: OverflowPolicy

    def __post_init__(self) -> None:
        typed_fields: tuple[tuple[str, object, type[object]], ...] = (
            ("max_distance", self.max_distance, float),
            ("max_missed_frames", self.max_missed_frames, int),
            ("min_reuse_gap_frames", self.min_reuse_gap_frames, int),
            ("use_velocity_prediction", self.use_velocity_prediction, bool),
            ("min_common_keypoints", self.min_common_keypoints, int),
            ("cost_reduction", self.cost_reduction, str),
            ("overflow_policy", self.overflow_policy, str),
        )
        for name, value, expected_type in typed_fields:
            if type(value) is not expected_type:
                raise ConfigurationTypeError(
                    f"data.association.{name}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}."
                )
        if not math.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise SemanticConfigurationError(
                "data.association.max_distance must be finite and > 0."
            )
        if self.max_missed_frames < 0:
            raise SemanticConfigurationError(
                "data.association.max_missed_frames must be non-negative."
            )
        if self.min_reuse_gap_frames < 0:
            raise SemanticConfigurationError(
                "data.association.min_reuse_gap_frames must be non-negative."
            )
        if self.min_common_keypoints <= 0:
            raise SemanticConfigurationError(
                "data.association.min_common_keypoints must be positive."
            )
        if self.cost_reduction not in ("mean", "median"):
            raise SemanticConfigurationError(
                "data.association.cost_reduction must be 'mean' or 'median'."
            )
        if self.overflow_policy != "error":
            raise SemanticConfigurationError(
                "data.association.overflow_policy must be the literal 'error'."
            )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        path: str = "data.association",
    ) -> ObservationTrackingConfig:
        """Parse an exact mapping without defaults or legacy fallbacks."""
        mapping = exact_config_mapping(
            value,
            path=path,
            required_keys=_ASSOCIATION_KEYS,
        )
        return cls(
            max_distance=cast(
                "float",
                require_config_value(mapping, "max_distance", float, path=path),
            ),
            max_missed_frames=cast(
                "int",
                require_config_value(mapping, "max_missed_frames", int, path=path),
            ),
            min_reuse_gap_frames=cast(
                "int",
                require_config_value(
                    mapping, "min_reuse_gap_frames", int, path=path
                ),
            ),
            use_velocity_prediction=cast(
                "bool",
                require_config_value(
                    mapping, "use_velocity_prediction", bool, path=path
                ),
            ),
            min_common_keypoints=cast(
                "int",
                require_config_value(
                    mapping, "min_common_keypoints", int, path=path
                ),
            ),
            cost_reduction=cast(
                "CostReduction",
                require_config_value(mapping, "cost_reduction", str, path=path),
            ),
            overflow_policy=cast(
                "OverflowPolicy",
                require_config_value(mapping, "overflow_policy", str, path=path),
            ),
        )


class TrackingCapacityError(RuntimeError):
    """Raised when one frame has more visible detections than fixed ``Q``."""

    def __init__(
        self,
        *,
        camera_index: int,
        frame_index: int,
        num_slots: int,
        free_slots: Sequence[int],
        unmatched_detection_ranks: Sequence[int],
    ) -> None:
        self.camera_index = camera_index
        self.frame_index = frame_index
        self.num_slots = num_slots
        self.free_slots = tuple(int(slot) for slot in free_slots)
        self.unmatched_detection_ranks = tuple(
            int(rank) for rank in unmatched_detection_ranks
        )
        super().__init__(
            "Observation tracking capacity exceeded: "
            f"camera={camera_index}, frame={frame_index}, num_slots={num_slots}, "
            f"free_slots={self.free_slots}, "
            f"unmatched_detection_ranks={self.unmatched_detection_ranks}."
        )


@dataclass(frozen=True, slots=True)
class TrackedObservations:
    """Fixed-Q tracked values plus association-only debug metadata.

    Camera-local tensors have shapes ``(T,Q,K,2)``, ``(T,Q,K)``, and
    ``(T,Q)``. Multiview tensors carry one leading view dimension. Carrier
    indices and optional provenance are debug outputs and are never tracking
    features.
    """

    values: Tensor
    visibility: Tensor
    detection_indices: Tensor
    debug_provenance: Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.ndim not in (4, 5) or self.values.shape[-1] != 2:
            raise ValueError(
                "Tracked values must have shape (T,Q,K,2) or (V,T,Q,K,2)."
            )
        if self.visibility.shape != self.values.shape[:-1]:
            raise ValueError("Tracked visibility must match values without UV axis.")
        if self.visibility.dtype != torch.bool:
            raise TypeError("Tracked visibility must have dtype torch.bool.")
        if self.detection_indices.shape != self.values.shape[:-2]:
            raise ValueError("Tracked detection indices must have shape (T,Q) or (V,T,Q).")
        if self.detection_indices.dtype != torch.long:
            raise TypeError("Tracked detection indices must have dtype torch.long.")
        if (
            self.debug_provenance is not None
            and self.debug_provenance.shape != self.detection_indices.shape
        ):
            raise ValueError("Tracked debug provenance must match detection indices.")


@dataclass(slots=True)
class _TrackState:
    last_values: Tensor
    last_visibility: Tensor
    last_frame: int
    previous_values: Tensor | None = None
    previous_visibility: Tensor | None = None
    previous_frame: int | None = None
    missed_frames: int = 0

    def prediction(self, frame_index: int, *, use_velocity: bool) -> tuple[Tensor, Tensor]:
        predicted = self.last_values.clone()
        if (
            use_velocity
            and self.previous_values is not None
            and self.previous_visibility is not None
            and self.previous_frame is not None
        ):
            elapsed = self.last_frame - self.previous_frame
            if elapsed <= 0:
                raise RuntimeError("Matched observation frames must be strictly ordered.")
            horizon = frame_index - self.last_frame
            common = self.last_visibility & self.previous_visibility
            velocity = (self.last_values[common] - self.previous_values[common]) / elapsed
            predicted[common] = self.last_values[common] + velocity * horizon
        return predicted, self.last_visibility

    def update(self, values: Tensor, visibility: Tensor, frame_index: int) -> None:
        self.previous_values = self.last_values
        self.previous_visibility = self.last_visibility
        self.previous_frame = self.last_frame
        self.last_values = values
        self.last_visibility = visibility
        self.last_frame = frame_index
        self.missed_frames = 0


def _validate_num_slots(num_slots: int) -> None:
    if type(num_slots) is not int:
        raise TypeError(f"num_slots must be int, got {type(num_slots).__name__}.")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")


def _validate_camera_index(camera_index: int) -> None:
    if type(camera_index) is not int:
        raise TypeError(
            f"camera_index must be int, got {type(camera_index).__name__}."
        )
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative.")


def _validate_camera_observations(
    values: Tensor,
    visibility: Tensor,
    *,
    config: ObservationTrackingConfig,
) -> None:
    if not isinstance(values, Tensor) or not isinstance(visibility, Tensor):
        raise TypeError("values and visibility must be torch.Tensor instances.")
    if values.ndim != 4 or values.shape[-1] != 2:
        raise ValueError("values must have shape (T,D,K,2).")
    if visibility.shape != values.shape[:-1]:
        raise ValueError("visibility must have shape (T,D,K) matching values.")
    if not torch.is_floating_point(values):
        raise TypeError("values must have a floating-point dtype.")
    if visibility.dtype != torch.bool:
        raise TypeError("visibility must have dtype torch.bool.")
    if values.device != visibility.device:
        raise ValueError("values and visibility must be on the same device.")
    if values.shape[0] <= 0:
        raise ValueError("values must contain at least one frame.")
    if values.shape[2] <= 0:
        raise ValueError("values must contain at least one keypoint.")
    if config.min_common_keypoints > values.shape[2]:
        raise ValueError(
            "data.association.min_common_keypoints exceeds the observation "
            f"keypoint count ({config.min_common_keypoints} > {values.shape[2]})."
        )
    visible_values = values[visibility]
    if visible_values.numel() > 0:
        if not bool(torch.isfinite(visible_values).all()):
            raise ValueError("Visible normalized UV coordinates must be finite.")
        if not bool(((visible_values >= 0.0) & (visible_values <= 1.0)).all()):
            raise ValueError("Visible normalized UV coordinates must be within [0, 1].")


def _normalized_detection(
    frame_values: Tensor,
    frame_visibility: Tensor,
    detection_index: int,
) -> tuple[Tensor, Tensor]:
    visibility = frame_visibility[detection_index].clone()
    values = torch.where(
        visibility.unsqueeze(-1),
        frame_values[detection_index],
        torch.zeros_like(frame_values[detection_index]),
    )
    return values, visibility


def _canonical_detection_indices(
    frame_values: Tensor,
    frame_visibility: Tensor,
) -> list[int]:
    visible_detections = torch.nonzero(
        frame_visibility.any(dim=-1), as_tuple=False
    ).flatten().tolist()

    def sort_key(detection_index: int) -> tuple[tuple[int, ...], tuple[float, ...], int]:
        detection_visibility = frame_visibility[detection_index]
        visibility_key = tuple(int(value) for value in detection_visibility.tolist())
        coordinate_key = tuple(
            float(coordinate)
            for keypoint_index in range(frame_values.shape[1])
            if bool(detection_visibility[keypoint_index])
            for coordinate in frame_values[detection_index, keypoint_index].tolist()
        )
        # The carrier index distinguishes exact model-visible duplicates only.
        # Such duplicates yield identical tracked values regardless of this key.
        return visibility_key, coordinate_key, detection_index

    return sorted(visible_detections, key=sort_key)


def _association_cost(
    predicted_values: Tensor,
    predicted_visibility: Tensor,
    detection_values: Tensor,
    detection_visibility: Tensor,
    *,
    config: ObservationTrackingConfig,
) -> float | None:
    common = predicted_visibility & detection_visibility
    if int(common.sum()) < config.min_common_keypoints:
        return None
    distances = torch.linalg.vector_norm(
        predicted_values[common] - detection_values[common], dim=-1
    )
    reduced = distances.mean() if config.cost_reduction == "mean" else distances.median()
    cost = float(reduced)
    return cost if cost <= config.max_distance else None


def _exact_deterministic_assignment(
    active_slots: Sequence[int],
    costs: Sequence[Sequence[float | None]],
) -> tuple[tuple[int, int], ...]:
    """Solve cardinality/cost/lex objectives exactly with bitmask dynamic programming."""
    if not active_slots or not costs or not costs[0]:
        return ()
    num_detections = len(costs[0])
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
    for row_index, slot_index in enumerate(active_slots):
        next_states = dict(states)
        for used_mask, (total_cost, pairs) in states.items():
            for detection_rank in range(num_detections):
                pair_cost = costs[row_index][detection_rank]
                detection_bit = 1 << detection_rank
                if pair_cost is None or used_mask & detection_bit:
                    continue
                next_mask = used_mask | detection_bit
                candidate = (
                    math.fsum((total_cost, pair_cost)),
                    (*pairs, (slot_index, detection_rank)),
                )
                current = next_states.get(next_mask)
                if current is None or candidate < current:
                    next_states[next_mask] = candidate
        states = next_states
    _, (_, best_pairs) = min(
        states.items(),
        key=lambda item: (
            -item[0].bit_count(),
            item[1][0],
            item[1][1],
        ),
    )
    return best_pairs


def limit_synthetic_false_positive_carriers(
    values: Tensor,
    visibility: Tensor,
    visibility_before_false_positive: Tensor,
    *,
    num_slots: int,
) -> tuple[Tensor, Tensor]:
    """Limit only false-positive-only carriers to the remaining fixed-Q capacity.

    Inputs use the shared ``(...,D,K,2)``/``(...,D,K)`` carrier convention.
    A currently visible carrier with any pre-false-positive visibility is
    genuine and is never removed.  Visible carriers with no such visibility
    are synthetic-only; their canonical model-visible order determines which
    fit after the genuine carriers.  Genuine overflow is intentionally left
    intact so the tracker raises :class:`TrackingCapacityError`.
    """
    _validate_num_slots(num_slots)
    if not isinstance(values, Tensor) or not isinstance(visibility, Tensor):
        raise TypeError("values and visibility must be torch.Tensor instances.")
    if not isinstance(visibility_before_false_positive, Tensor):
        raise TypeError("visibility_before_false_positive must be a torch.Tensor.")
    if values.ndim < 3 or values.shape[-1] != 2:
        raise ValueError("values must have shape (...,D,K,2).")
    if visibility.shape != values.shape[:-1]:
        raise ValueError("visibility must have shape (...,D,K) matching values.")
    if visibility_before_false_positive.shape != visibility.shape:
        raise ValueError(
            "visibility_before_false_positive must match visibility exactly."
        )
    if not torch.is_floating_point(values):
        raise TypeError("values must have a floating-point dtype.")
    if visibility.dtype != torch.bool:
        raise TypeError("visibility must have dtype torch.bool.")
    if visibility_before_false_positive.dtype != torch.bool:
        raise TypeError("visibility_before_false_positive must have dtype torch.bool.")
    if (
        values.device != visibility.device
        or values.device != visibility_before_false_positive.device
    ):
        raise ValueError("All false-positive capacity tensors must share a device.")
    if values.shape[-3] <= 0:
        raise ValueError("values must contain at least one carrier.")
    if values.shape[-2] <= 0:
        raise ValueError("values must contain at least one keypoint.")

    visible_values = values[visibility]
    if visible_values.numel() > 0:
        if not bool(torch.isfinite(visible_values).all()):
            raise ValueError("Visible normalized UV coordinates must be finite.")
        if not bool(((visible_values >= 0.0) & (visible_values <= 1.0)).all()):
            raise ValueError("Visible normalized UV coordinates must be within [0, 1].")

    limited_values = values.clone(memory_format=torch.contiguous_format)
    limited_visibility = visibility.clone(memory_format=torch.contiguous_format)
    num_carriers = values.shape[-3]
    num_keypoints = values.shape[-2]
    flat_values = limited_values.reshape(-1, num_carriers, num_keypoints, 2)
    flat_visibility = limited_visibility.reshape(-1, num_carriers, num_keypoints)
    flat_pre_false_positive = visibility_before_false_positive.reshape(
        -1, num_carriers, num_keypoints
    )
    for leading_index in range(flat_values.shape[0]):
        carrier_visible = flat_visibility[leading_index].any(dim=-1)
        genuine_carrier = flat_pre_false_positive[leading_index].any(dim=-1)
        genuine_visible_count = int((carrier_visible & genuine_carrier).sum())
        allowed_synthetic_count = max(num_slots - genuine_visible_count, 0)
        canonical_indices = _canonical_detection_indices(
            flat_values[leading_index], flat_visibility[leading_index]
        )
        synthetic_indices = [
            carrier_index
            for carrier_index in canonical_indices
            if not bool(genuine_carrier[carrier_index])
        ]
        rejected_indices = synthetic_indices[allowed_synthetic_count:]
        if rejected_indices:
            flat_values[leading_index, rejected_indices] = 0
            flat_visibility[leading_index, rejected_indices] = False
    return limited_values, limited_visibility


def gather_tracked_debug_provenance(
    debug_provenance: Tensor,
    detection_indices: Tensor,
) -> Tensor:
    """Gather scalar carrier metadata after association, filling padding with ``-1``."""
    if not isinstance(debug_provenance, Tensor) or not isinstance(
        detection_indices, Tensor
    ):
        raise TypeError("debug_provenance and detection_indices must be tensors.")
    if detection_indices.ndim not in (2, 3):
        raise ValueError("detection_indices must have shape (T,Q) or (V,T,Q).")
    if debug_provenance.ndim != detection_indices.ndim:
        raise ValueError(
            "debug_provenance must have shape (T,D) or (V,T,D) matching indices."
        )
    if debug_provenance.shape[:-1] != detection_indices.shape[:-1]:
        raise ValueError("Debug provenance and tracked indices prefixes must match.")
    if detection_indices.dtype != torch.long:
        raise TypeError("detection_indices must have dtype torch.long.")
    if debug_provenance.dtype not in _INTEGER_DTYPES:
        raise TypeError("debug_provenance must have an integer dtype.")
    if debug_provenance.device != detection_indices.device:
        raise ValueError("Debug provenance and tracked indices must share a device.")
    num_detections = debug_provenance.shape[-1]
    if bool((debug_provenance < -1).any()):
        raise ValueError("debug_provenance values must be -1 or non-negative.")
    if bool((detection_indices < -1).any()) or bool(
        (detection_indices >= num_detections).any()
    ):
        raise ValueError("detection_indices contain an out-of-range carrier index.")
    output = torch.full(
        detection_indices.shape,
        -1,
        dtype=debug_provenance.dtype,
        device=debug_provenance.device,
    )
    if num_detections == 0:
        return output
    flat_indices = detection_indices.reshape(-1, detection_indices.shape[-1])
    flat_provenance = debug_provenance.reshape(-1, num_detections)
    safe_indices = flat_indices.clamp_min(0)
    gathered = torch.gather(flat_provenance, 1, safe_indices)
    gathered[flat_indices < 0] = -1
    return gathered.reshape_as(output)


def _track_camera_core(
    values: Tensor,
    visibility: Tensor,
    *,
    num_slots: int,
    config: ObservationTrackingConfig,
    camera_index: int,
) -> TrackedObservations:
    _validate_num_slots(num_slots)
    _validate_camera_index(camera_index)
    _validate_camera_observations(values, visibility, config=config)

    num_frames, _, num_keypoints, _ = values.shape
    tracked_values = torch.zeros(
        (num_frames, num_slots, num_keypoints, 2),
        dtype=values.dtype,
        device=values.device,
    )
    tracked_visibility = torch.zeros(
        (num_frames, num_slots, num_keypoints),
        dtype=torch.bool,
        device=values.device,
    )
    detection_indices = torch.full(
        (num_frames, num_slots),
        -1,
        dtype=torch.long,
        device=values.device,
    )
    states: list[_TrackState | None] = [None] * num_slots
    reusable_after_frame = [0] * num_slots

    for frame_index in range(num_frames):
        canonical_indices = _canonical_detection_indices(
            values[frame_index], visibility[frame_index]
        )
        free_before_matching = tuple(
            slot_index
            for slot_index, state in enumerate(states)
            if state is None and frame_index >= reusable_after_frame[slot_index]
        )
        if len(canonical_indices) > num_slots:
            raise TrackingCapacityError(
                camera_index=camera_index,
                frame_index=frame_index,
                num_slots=num_slots,
                free_slots=free_before_matching,
                unmatched_detection_ranks=tuple(range(len(canonical_indices))),
            )

        detections = [
            _normalized_detection(
                values[frame_index], visibility[frame_index], detection_index
            )
            for detection_index in canonical_indices
        ]
        active_slots = [
            slot_index for slot_index, state in enumerate(states) if state is not None
        ]
        costs: list[list[float | None]] = []
        for slot_index in active_slots:
            state = states[slot_index]
            if state is None:
                raise RuntimeError("Active observation track unexpectedly has no state.")
            predicted_values, predicted_visibility = state.prediction(
                frame_index,
                use_velocity=config.use_velocity_prediction,
            )
            costs.append(
                [
                    _association_cost(
                        predicted_values,
                        predicted_visibility,
                        detection_values,
                        detection_visibility,
                        config=config,
                    )
                    for detection_values, detection_visibility in detections
                ]
            )
        matches = _exact_deterministic_assignment(active_slots, costs)
        matched_slots = {slot_index for slot_index, _ in matches}
        matched_detection_ranks = {detection_rank for _, detection_rank in matches}

        for slot_index, detection_rank in matches:
            detection_values, detection_visibility = detections[detection_rank]
            state = states[slot_index]
            if state is None:
                raise RuntimeError("Matched observation track unexpectedly has no state.")
            state.update(detection_values, detection_visibility, frame_index)
            tracked_values[frame_index, slot_index] = detection_values
            tracked_visibility[frame_index, slot_index] = detection_visibility
            detection_indices[frame_index, slot_index] = canonical_indices[detection_rank]

        for slot_index in active_slots:
            if slot_index in matched_slots:
                continue
            state = states[slot_index]
            if state is None:
                raise RuntimeError("Unmatched observation track unexpectedly has no state.")
            state.missed_frames += 1
            if state.missed_frames > config.max_missed_frames:
                states[slot_index] = None
                reusable_after_frame[slot_index] = (
                    frame_index + config.min_reuse_gap_frames
                )

        birth_ranks = [
            detection_rank
            for detection_rank in range(len(detections))
            if detection_rank not in matched_detection_ranks
        ]
        free_slots = [
            slot_index
            for slot_index, state in enumerate(states)
            if state is None and frame_index >= reusable_after_frame[slot_index]
        ]
        if len(birth_ranks) > len(free_slots):
            slot_deficit = len(birth_ranks) - len(free_slots)
            cooldown_slots = sorted(
                (
                    slot_index
                    for slot_index, state in enumerate(states)
                    if state is None
                    and frame_index < reusable_after_frame[slot_index]
                ),
                key=lambda slot_index: (
                    reusable_after_frame[slot_index],
                    slot_index,
                ),
            )
            pressure_slots = cooldown_slots[:slot_deficit]

            def retained_sort_key(slot_index: int) -> tuple[int, int, int]:
                state = states[slot_index]
                if state is None:
                    raise RuntimeError(
                        "Retained pressure candidate unexpectedly has no state."
                    )
                return (-state.missed_frames, state.last_frame, slot_index)

            retained_slots = sorted(
                (
                    slot_index
                    for slot_index, state in enumerate(states)
                    if state is not None and slot_index not in matched_slots
                ),
                key=retained_sort_key,
            )
            remaining_deficit = slot_deficit - len(pressure_slots)
            if remaining_deficit > 0:
                pressure_slots.extend(retained_slots[:remaining_deficit])
            if len(pressure_slots) != slot_deficit:
                raise RuntimeError(
                    "Fixed-Q pressure recycling could not satisfy a non-overflow frame."
                )
            free_slots.extend(pressure_slots)
        available_slots = sorted(free_slots)
        for detection_rank, slot_index in zip(
            birth_ranks,
            available_slots[: len(birth_ranks)],
            strict=True,
        ):
            detection_values, detection_visibility = detections[detection_rank]
            states[slot_index] = _TrackState(
                last_values=detection_values,
                last_visibility=detection_visibility,
                last_frame=frame_index,
            )
            tracked_values[frame_index, slot_index] = detection_values
            tracked_visibility[frame_index, slot_index] = detection_visibility
            detection_indices[frame_index, slot_index] = canonical_indices[detection_rank]

    return TrackedObservations(
        values=tracked_values,
        visibility=tracked_visibility,
        detection_indices=detection_indices,
    )


def track_camera_observations(
    values: Tensor,
    visibility: Tensor,
    *,
    num_slots: int,
    config: ObservationTrackingConfig,
    camera_index: int = 0,
    debug_provenance: Tensor | None = None,
) -> TrackedObservations:
    """Track one camera's unordered ``(T,D,K,2)`` observation carriers."""
    if not isinstance(config, ObservationTrackingConfig):
        raise TypeError("config must be an ObservationTrackingConfig instance.")
    tracked = _track_camera_core(
        values,
        visibility,
        num_slots=num_slots,
        config=config,
        camera_index=camera_index,
    )
    gathered = (
        None
        if debug_provenance is None
        else gather_tracked_debug_provenance(
            debug_provenance, tracked.detection_indices
        )
    )
    return TrackedObservations(
        values=tracked.values,
        visibility=tracked.visibility,
        detection_indices=tracked.detection_indices,
        debug_provenance=gathered,
    )


def track_multiview_observations(
    values: Tensor,
    visibility: Tensor,
    *,
    num_slots: int,
    config: ObservationTrackingConfig,
    camera_indices: Sequence[int] | None = None,
    debug_provenance: Tensor | None = None,
) -> TrackedObservations:
    """Track ``(V,T,D,K,2)`` observations with independent per-view state."""
    if not isinstance(config, ObservationTrackingConfig):
        raise TypeError("config must be an ObservationTrackingConfig instance.")
    if not isinstance(values, Tensor) or not isinstance(visibility, Tensor):
        raise TypeError("values and visibility must be torch.Tensor instances.")
    if values.ndim != 5 or values.shape[-1] != 2:
        raise ValueError("values must have shape (V,T,D,K,2).")
    if visibility.shape != values.shape[:-1]:
        raise ValueError("visibility must have shape (V,T,D,K) matching values.")
    if values.shape[0] <= 0:
        raise ValueError("Multiview observations must contain at least one view.")
    resolved_camera_indices = (
        tuple(range(values.shape[0]))
        if camera_indices is None
        else tuple(camera_indices)
    )
    if len(resolved_camera_indices) != values.shape[0]:
        raise ValueError("camera_indices must contain one index per view.")
    for camera_index in resolved_camera_indices:
        _validate_camera_index(camera_index)

    camera_results = [
        _track_camera_core(
            values[view_index],
            visibility[view_index],
            num_slots=num_slots,
            config=config,
            camera_index=camera_index,
        )
        for view_index, camera_index in enumerate(resolved_camera_indices)
    ]
    tracked_values = torch.stack([result.values for result in camera_results])
    tracked_visibility = torch.stack(
        [result.visibility for result in camera_results]
    )
    detection_indices = torch.stack(
        [result.detection_indices for result in camera_results]
    )
    gathered = (
        None
        if debug_provenance is None
        else gather_tracked_debug_provenance(debug_provenance, detection_indices)
    )
    return TrackedObservations(
        values=tracked_values,
        visibility=tracked_visibility,
        detection_indices=detection_indices,
        debug_provenance=gathered,
    )


__all__ = [
    "ObservationTrackingConfig",
    "TrackedObservations",
    "TrackingCapacityError",
    "gather_tracked_debug_provenance",
    "limit_synthetic_false_positive_carriers",
    "track_camera_observations",
    "track_multiview_observations",
]
