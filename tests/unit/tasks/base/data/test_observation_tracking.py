from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from src.tasks.base.data import limit_synthetic_false_positive_carriers
from src.tasks.base.data.observation_tracking import (
    ObservationTrackingConfig,
    TrackingCapacityError,
    gather_tracked_debug_provenance,
    track_camera_observations,
    track_multiview_observations,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)


def _config(**overrides: object) -> ObservationTrackingConfig:
    values: dict[str, object] = {
        "max_distance": 0.25,
        "max_missed_frames": 2,
        "min_reuse_gap_frames": 0,
        "use_velocity_prediction": False,
        "min_common_keypoints": 1,
        "cost_reduction": "mean",
        "overflow_policy": "error",
    }
    values.update(overrides)
    return ObservationTrackingConfig.from_mapping(values)


def _point_observations(coordinates: list[list[tuple[float, float]]]) -> tuple[Tensor, Tensor]:
    values = torch.tensor(coordinates, dtype=torch.float32).unsqueeze(2)
    visibility = torch.ones(values.shape[:-1], dtype=torch.bool)
    return values, visibility


def _permute_carriers(
    values: Tensor,
    visibility: Tensor,
    permutations: list[list[int]],
) -> tuple[Tensor, Tensor]:
    return (
        torch.stack(
            [values[frame, permutation] for frame, permutation in enumerate(permutations)]
        ),
        torch.stack(
            [
                visibility[frame, permutation]
                for frame, permutation in enumerate(permutations)
            ]
        ),
    )


def test_tracking_is_replayable_and_independent_of_carrier_order() -> None:
    values, visibility = _point_observations(
        [
            [(0.15, 0.20), (0.85, 0.80)],
            [(0.20, 0.20), (0.80, 0.80)],
            [(0.25, 0.20), (0.75, 0.80)],
        ]
    )
    permuted_values, permuted_visibility = _permute_carriers(
        values, visibility, [[1, 0], [0, 1], [1, 0]]
    )

    first = track_camera_observations(
        values, visibility, num_slots=2, config=_config()
    )
    replay = track_camera_observations(
        values, visibility, num_slots=2, config=_config()
    )
    permuted = track_camera_observations(
        permuted_values,
        permuted_visibility,
        num_slots=2,
        config=_config(),
    )

    torch.testing.assert_close(first.values, replay.values)
    torch.testing.assert_close(first.visibility, replay.visibility)
    torch.testing.assert_close(first.values, permuted.values)
    torch.testing.assert_close(first.visibility, permuted.visibility)


def test_exact_assignment_tie_uses_lexicographically_smallest_pairs() -> None:
    values, visibility = _point_observations(
        [
            [(0.25, 0.50), (0.75, 0.50)],
            [(0.50, 0.25), (0.50, 0.75)],
        ]
    )
    permuted_values, permuted_visibility = _permute_carriers(
        values,
        visibility,
        [[1, 0], [1, 0]],
    )

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.50),
    )
    permuted = track_camera_observations(
        permuted_values,
        permuted_visibility,
        num_slots=2,
        config=_config(max_distance=0.50),
    )

    assert tracked.detection_indices[1].tolist() == [0, 1]
    torch.testing.assert_close(
        tracked.values[1, :, 0],
        torch.tensor([[0.50, 0.25], [0.50, 0.75]]),
    )
    torch.testing.assert_close(tracked.values, permuted.values)
    torch.testing.assert_close(tracked.visibility, permuted.visibility)


def test_pose_equal_cost_tie_is_independent_of_carrier_order() -> None:
    values = torch.tensor(
        [
            [
                [[0.25, 0.50]] * 4,
                [[0.75, 0.50]] * 4,
            ],
            [
                [[0.50, 0.25]] * 4,
                [[0.50, 0.75]] * 4,
            ],
        ],
        dtype=torch.float32,
    )
    visibility = torch.ones(values.shape[:-1], dtype=torch.bool)
    permuted_values, permuted_visibility = _permute_carriers(
        values,
        visibility,
        [[1, 0], [1, 0]],
    )
    config = _config(
        max_distance=0.50,
        min_common_keypoints=4,
        cost_reduction="median",
    )

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=config,
    )
    permuted = track_camera_observations(
        permuted_values,
        permuted_visibility,
        num_slots=2,
        config=config,
    )

    torch.testing.assert_close(tracked.values, permuted.values)
    torch.testing.assert_close(tracked.visibility, permuted.visibility)
    torch.testing.assert_close(
        tracked.values[1, :, :, 1],
        torch.tensor([[0.25] * 4, [0.75] * 4]),
    )


def test_assignment_maximizes_valid_cardinality_before_total_cost() -> None:
    values, visibility = _point_observations(
        [
            [(0.00, 0.50), (0.15, 0.50)],
            [(0.00, 0.50), (0.08, 0.50)],
        ]
    )

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.11),
    )

    # The zero-cost slot-0/rank-0 pair alone is cheaper, but rank 1 can match
    # slot 1 only if slot 0 takes rank 0, producing the required two matches.
    assert tracked.detection_indices[1].tolist() == [0, 1]
    assert tracked.visibility[1].all()


def test_velocity_prediction_preserves_identity_through_a_crossing() -> None:
    values, visibility = _point_observations(
        [
            [(0.20, 0.50), (0.80, 0.50)],
            [(0.40, 0.50), (0.60, 0.50)],
            [(0.40, 0.50), (0.60, 0.50)],
        ]
    )

    velocity = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.30, use_velocity_prediction=True),
    )
    last_position = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.30, use_velocity_prediction=False),
    )

    assert velocity.detection_indices[2].tolist() == [1, 0]
    assert last_position.detection_indices[2].tolist() == [0, 1]
    torch.testing.assert_close(
        velocity.values[2, :, 0, 0], torch.tensor([0.60, 0.40])
    )


def test_distance_gate_and_false_positive_provenance_are_explicit() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.10), (0.00, 0.00)],
            [(0.12, 0.10), (0.80, 0.80)],
        ]
    )
    visibility[0, 1] = False
    provenance = torch.tensor([[7, 99], [7, -1]])

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.05),
        debug_provenance=provenance,
    )

    assert tracked.debug_provenance is not None
    assert tracked.debug_provenance.tolist() == [[7, -1], [7, -1]]
    torch.testing.assert_close(tracked.values[1, 0, 0], torch.tensor([0.12, 0.10]))
    torch.testing.assert_close(tracked.values[1, 1, 0], torch.tensor([0.80, 0.80]))


def test_miss_reappearance_retirement_and_normal_reuse_boundaries_are_exact() -> None:
    values = torch.zeros((8, 1, 1, 2), dtype=torch.float32)
    visibility = torch.zeros((8, 1, 1), dtype=torch.bool)
    values[0, 0, 0] = torch.tensor([0.10, 0.10])
    values[3, 0, 0] = torch.tensor([0.12, 0.10])
    values[7, 0, 0] = torch.tensor([0.80, 0.80])
    visibility[[0, 3, 7], 0, 0] = True

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=1,
        config=_config(
            max_distance=0.05,
            max_missed_frames=2,
            min_reuse_gap_frames=1,
        ),
    )

    assert tracked.detection_indices[:, 0].tolist() == [0, -1, -1, 0, -1, -1, -1, 0]
    assert tracked.visibility[3, 0, 0]
    assert tracked.visibility[7, 0, 0]


def test_retained_pressure_recycles_the_stalest_unmatched_state() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.50), (0.50, 0.50), (0.90, 0.50)],
            [(0.50, 0.50), (0.90, 0.50), (0.00, 0.00)],
            [(0.30, 0.50), (0.90, 0.50), (0.00, 0.00)],
        ]
    )
    visibility[1:, 2] = False

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=3,
        config=_config(max_distance=0.05, max_missed_frames=4),
    )

    torch.testing.assert_close(
        tracked.values[2, :, 0, 0], torch.tensor([0.30, 0.00, 0.90])
    )
    assert tracked.visibility[2, :, 0].tolist() == [True, False, True]


def test_cooldown_only_pressure_bypasses_reuse_gap_for_current_birth() -> None:
    values = torch.zeros((3, 1, 1, 2), dtype=torch.float32)
    visibility = torch.zeros((3, 1, 1), dtype=torch.bool)
    values[0, 0, 0] = torch.tensor([0.10, 0.20])
    values[2, 0, 0] = torch.tensor([0.80, 0.70])
    visibility[[0, 2], 0, 0] = True

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=1,
        config=_config(
            max_distance=0.05,
            max_missed_frames=0,
            min_reuse_gap_frames=5,
        ),
    )

    assert tracked.detection_indices[:, 0].tolist() == [0, -1, 0]
    torch.testing.assert_close(tracked.values[2, 0, 0], values[2, 0, 0])


def test_cooldown_pressure_prefers_the_earliest_normal_reuse_frame() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.50), (0.50, 0.50), (0.90, 0.50)],
            [(0.50, 0.50), (0.90, 0.50), (0.00, 0.00)],
            [(0.90, 0.50), (0.00, 0.00), (0.00, 0.00)],
            [(0.30, 0.50), (0.90, 0.50), (0.00, 0.00)],
        ]
    )
    visibility[1, 2] = False
    visibility[2, 1:] = False
    visibility[3, 2] = False

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=3,
        config=_config(
            max_distance=0.05,
            max_missed_frames=0,
            min_reuse_gap_frames=5,
        ),
    )

    torch.testing.assert_close(tracked.values[3, 0, 0], torch.tensor([0.30, 0.50]))
    assert tracked.visibility[3, :, 0].tolist() == [True, False, True]


def test_pressure_prefers_cooldown_slot_before_retained_state() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.50), (0.50, 0.50), (0.90, 0.50)],
            [(0.10, 0.50), (0.50, 0.50), (0.00, 0.00)],
            [(0.10, 0.50), (0.00, 0.00), (0.00, 0.00)],
            [(0.10, 0.50), (0.30, 0.50), (0.00, 0.00)],
        ]
    )
    visibility[1, 2] = False
    visibility[2, 1:] = False
    visibility[3, 2] = False

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=3,
        config=_config(
            max_distance=0.05,
            max_missed_frames=2,
            min_reuse_gap_frames=5,
        ),
    )

    assert tracked.visibility[3, :, 0].tolist() == [True, False, True]
    torch.testing.assert_close(tracked.values[3, 2, 0], torch.tensor([0.30, 0.50]))


def test_pressure_never_preempts_match_and_gathers_current_provenance() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.20), (0.90, 0.80), (0.00, 0.00)],
            [(0.12, 0.20), (0.50, 0.50), (0.70, 0.50)],
        ]
    )
    visibility[0, 2] = False

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=3,
        config=_config(max_distance=0.05, max_missed_frames=3),
        debug_provenance=torch.tensor([[10, 20, 99], [11, -1, -1]]),
    )

    torch.testing.assert_close(
        tracked.values[1, :, 0],
        torch.tensor([[0.12, 0.20], [0.50, 0.50], [0.70, 0.50]]),
    )
    assert tracked.debug_provenance is not None
    assert tracked.debug_provenance[1].tolist() == [11, -1, -1]


def test_pressure_ties_and_carrier_permutations_choose_the_same_slot() -> None:
    values, visibility = _point_observations(
        [
            [(0.10, 0.50), (0.90, 0.50)],
            [(0.50, 0.50), (0.00, 0.00)],
        ]
    )
    visibility[1, 1] = False
    permuted_values, permuted_visibility = _permute_carriers(
        values, visibility, [[1, 0], [1, 0]]
    )

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(max_distance=0.05, max_missed_frames=3),
    )
    permuted = track_camera_observations(
        permuted_values,
        permuted_visibility,
        num_slots=2,
        config=_config(max_distance=0.05, max_missed_frames=3),
    )

    torch.testing.assert_close(tracked.values, permuted.values)
    torch.testing.assert_close(tracked.visibility, permuted.visibility)
    assert tracked.visibility[1, :, 0].tolist() == [True, False]


def test_spare_capacity_preserves_normal_miss_and_reuse_gap_behavior() -> None:
    values = torch.zeros((4, 2, 1, 2), dtype=torch.float32)
    visibility = torch.zeros((4, 2, 1), dtype=torch.bool)
    values[0, 0, 0] = torch.tensor([0.10, 0.50])
    values[2, 0, 0] = torch.tensor([0.80, 0.50])
    values[3, 0, 0] = torch.tensor([0.30, 0.50])
    values[3, 1, 0] = torch.tensor([0.80, 0.50])
    visibility[0, 0, 0] = True
    visibility[2, 0, 0] = True
    visibility[3, :, 0] = True

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(
            max_distance=0.05,
            max_missed_frames=0,
            min_reuse_gap_frames=2,
        ),
    )

    assert tracked.visibility[2, :, 0].tolist() == [False, True]
    torch.testing.assert_close(
        tracked.values[3, :, 0], torch.tensor([[0.30, 0.50], [0.80, 0.50]])
    )


def test_pose_requires_configured_common_visible_keypoints() -> None:
    values = torch.zeros((2, 1, 8, 2), dtype=torch.float32)
    visibility = torch.zeros((2, 1, 8), dtype=torch.bool)
    values[0, 0, :4] = 0.25
    visibility[0, 0, :4] = True
    values[1, 0, 4:] = 0.25
    visibility[1, 0, 4:] = True

    tracked = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(min_common_keypoints=4, cost_reduction="median"),
    )

    assert not tracked.visibility[1, 0].any()
    assert tracked.visibility[1, 1, 4:].all()


def test_pose_median_reduction_is_applied_to_common_joint_distances() -> None:
    values = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    visibility = torch.ones((2, 1, 4), dtype=torch.bool)
    values[0, 0] = torch.tensor([[0.2, 0.2]] * 4)
    values[1, 0] = torch.tensor(
        [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.6, 0.2]]
    )

    median = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(
            max_distance=0.05,
            min_common_keypoints=4,
            cost_reduction="median",
        ),
    )
    mean = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(
            max_distance=0.05,
            min_common_keypoints=4,
            cost_reduction="mean",
        ),
    )

    assert median.visibility[1, 0].all()
    assert not mean.visibility[1, 0].any()
    assert mean.visibility[1, 1].all()


def test_q_one_overflow_raises_typed_camera_frame_and_free_slot_evidence() -> None:
    values, visibility = _point_observations(
        [[(0.10, 0.10), (0.90, 0.90)]]
    )

    with pytest.raises(TrackingCapacityError, match=r"camera=3, frame=0") as error:
        track_camera_observations(
            values,
            visibility,
            num_slots=1,
            config=_config(),
            camera_index=3,
        )

    assert error.value.num_slots == 1
    assert error.value.free_slots == (0,)
    assert error.value.unmatched_detection_ranks == (0, 1)


def test_multiview_tracking_is_exactly_camera_local() -> None:
    values = torch.tensor(
        [
            [[[[0.10, 0.20]]], [[[0.15, 0.20]]]],
            [[[[0.90, 0.80]]], [[[0.85, 0.80]]]],
        ],
        dtype=torch.float32,
    )
    visibility = torch.ones(values.shape[:-1], dtype=torch.bool)

    multiview = track_multiview_observations(
        values,
        visibility,
        num_slots=1,
        config=_config(max_distance=0.10),
        camera_indices=(5, 9),
    )
    for view_index, camera_index in enumerate((5, 9)):
        camera = track_camera_observations(
            values[view_index],
            visibility[view_index],
            num_slots=1,
            config=_config(max_distance=0.10),
            camera_index=camera_index,
        )
        torch.testing.assert_close(multiview.values[view_index], camera.values)
        torch.testing.assert_close(
            multiview.visibility[view_index], camera.visibility
        )


def test_debug_provenance_never_changes_association_or_model_visible_values() -> None:
    values, visibility = _point_observations(
        [
            [(0.20, 0.20), (0.80, 0.80)],
            [(0.25, 0.20), (0.75, 0.80)],
        ]
    )
    first = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(),
        debug_provenance=torch.tensor([[10, -1], [10, -1]]),
    )
    second = track_camera_observations(
        values,
        visibility,
        num_slots=2,
        config=_config(),
        debug_provenance=torch.tensor([[999, 888], [777, 666]]),
    )

    torch.testing.assert_close(first.values, second.values)
    torch.testing.assert_close(first.visibility, second.visibility)
    torch.testing.assert_close(first.detection_indices, second.detection_indices)
    assert first.debug_provenance is not None
    assert second.debug_provenance is not None
    assert not torch.equal(first.debug_provenance, second.debug_provenance)


def test_provenance_gather_uses_minus_one_for_padding() -> None:
    gathered = gather_tracked_debug_provenance(
        torch.tensor([[[5, -1], [6, 7]]]),
        torch.tensor([[[1, -1, 0], [-1, 0, 1]]]),
    )
    assert gathered.tolist() == [[[-1, -1, 5], [-1, 6, 7]]]


def test_synthetic_false_positive_cap_uses_canonical_model_visible_order() -> None:
    values = torch.tensor(
        [[[[0.80, 0.50]], [[0.40, 0.50]], [[0.10, 0.50]],
          [[0.60, 0.50]], [[0.20, 0.50]]]],
        dtype=torch.float32,
    )
    visibility = torch.ones(values.shape[:-1], dtype=torch.bool)
    before_false_positive = torch.zeros_like(visibility)
    before_false_positive[..., 0, :] = True
    before_false_positive[..., 2, :] = True
    permutation = [3, 0, 4, 2, 1]

    limited_values, limited_visibility = limit_synthetic_false_positive_carriers(
        values,
        visibility,
        before_false_positive,
        num_slots=3,
    )
    permuted_values, permuted_visibility = limit_synthetic_false_positive_carriers(
        values[..., permutation, :, :],
        visibility[..., permutation, :],
        before_false_positive[..., permutation, :],
        num_slots=3,
    )

    assert limited_visibility.any(dim=-1).sum().item() == 3
    assert limited_visibility[..., 0, :].all()
    assert limited_visibility[..., 2, :].all()
    assert limited_visibility[..., 4, :].all()
    assert not limited_visibility[..., [1, 3], :].any()
    assert limited_values[..., [1, 3], :, :].eq(0).all()
    selected = limited_values[limited_visibility].reshape(-1, 2)[:, 0].sort().values
    permuted_selected = (
        permuted_values[permuted_visibility].reshape(-1, 2)[:, 0].sort().values
    )
    torch.testing.assert_close(selected, torch.tensor([0.10, 0.20, 0.80]))
    torch.testing.assert_close(selected, permuted_selected)
    assert visibility.all()


def test_synthetic_cap_handles_partial_pose_and_preserves_mixed_genuine_carrier() -> None:
    values = torch.zeros((3, 17, 2), dtype=torch.float32)
    visibility = torch.zeros((3, 17), dtype=torch.bool)
    before_false_positive = torch.zeros_like(visibility)
    values[0, :2] = 0.80
    visibility[0, :2] = True
    before_false_positive[0, 0] = True
    values[1, :4] = 0.60
    visibility[1, :4] = True
    values[2, 4:8] = 0.20
    visibility[2, 4:8] = True

    limited_values, limited_visibility = limit_synthetic_false_positive_carriers(
        values,
        visibility,
        before_false_positive,
        num_slots=2,
    )

    torch.testing.assert_close(limited_values[0], values[0])
    torch.testing.assert_close(limited_visibility[0], visibility[0])
    assert not limited_visibility[1].any()
    assert limited_values[1].eq(0).all()
    assert limited_visibility[2, 4:8].all()
    torch.testing.assert_close(limited_values[2], values[2])


def test_synthetic_cap_writes_back_from_strided_leading_layout() -> None:
    views, frames, carriers, keypoints = 2, 3, 4, 2
    x_coordinates = (
        torch.tensor([0.80, 0.40, 0.10, 0.60])
        .view(1, carriers, 1, 1, 1)
        .expand(views, carriers, frames, keypoints, 1)
    )
    carrier_major_values = torch.cat(
        [x_coordinates, torch.full_like(x_coordinates, 0.50)], dim=-1
    )
    carrier_major_visibility = torch.ones(
        (views, carriers, frames, keypoints), dtype=torch.bool
    )
    carrier_major_genuine = torch.zeros_like(carrier_major_visibility)
    carrier_major_genuine[:, 0] = True
    values = carrier_major_values.permute(0, 2, 1, 3, 4).requires_grad_()
    visibility = carrier_major_visibility.permute(0, 2, 1, 3)
    before_false_positive = carrier_major_genuine.permute(0, 2, 1, 3)
    original_values = values.detach().clone()
    original_visibility = visibility.clone()

    layout_preserving_values_clone = values.clone()
    layout_preserving_visibility_clone = visibility.clone()
    assert not layout_preserving_values_clone.is_contiguous()
    assert not layout_preserving_visibility_clone.is_contiguous()
    assert (
        layout_preserving_values_clone.reshape(
            -1, carriers, keypoints, 2
        ).untyped_storage().data_ptr()
        != layout_preserving_values_clone.untyped_storage().data_ptr()
    )
    assert (
        layout_preserving_visibility_clone.reshape(
            -1, carriers, keypoints
        ).untyped_storage().data_ptr()
        != layout_preserving_visibility_clone.untyped_storage().data_ptr()
    )

    limited_values, limited_visibility = limit_synthetic_false_positive_carriers(
        values,
        visibility,
        before_false_positive,
        num_slots=2,
    )

    assert limited_values.is_contiguous()
    assert limited_visibility.is_contiguous()
    assert limited_values.dtype == values.dtype
    assert limited_values.device == values.device
    assert limited_values.requires_grad
    assert limited_visibility[..., [0, 2], :].all()
    assert not limited_visibility[..., [1, 3], :].any()
    assert limited_values[..., [1, 3], :, :].eq(0).all()
    torch.testing.assert_close(values.detach(), original_values)
    torch.testing.assert_close(visibility, original_visibility)
    limited_values.sum().backward()
    assert values.grad is not None
    assert values.grad[..., [0, 2], :, :].eq(1).all()
    assert values.grad[..., [1, 3], :, :].eq(0).all()


def test_genuine_over_q_is_preserved_for_typed_tracker_rejection() -> None:
    values = torch.tensor(
        [[[[0.10, 0.20]], [[0.40, 0.50]], [[0.80, 0.90]]]],
        dtype=torch.float32,
    )
    visibility = torch.ones(values.shape[:-1], dtype=torch.bool)
    before_false_positive = torch.zeros_like(visibility)
    before_false_positive[:, :2] = True

    limited_values, limited_visibility = limit_synthetic_false_positive_carriers(
        values,
        visibility,
        before_false_positive,
        num_slots=1,
    )

    torch.testing.assert_close(limited_values[:, :2], values[:, :2])
    assert limited_visibility[:, :2].all()
    assert not limited_visibility[:, 2].any()
    assert limited_values[:, 2].eq(0).all()
    with pytest.raises(TrackingCapacityError, match=r"frame=0"):
        track_camera_observations(
            limited_values,
            limited_visibility,
            num_slots=1,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"max_distance": float("inf")}, SemanticConfigurationError),
        ({"max_distance": 0.0}, SemanticConfigurationError),
        ({"max_distance": 1}, ConfigurationTypeError),
        ({"max_missed_frames": -1}, SemanticConfigurationError),
        ({"max_missed_frames": True}, ConfigurationTypeError),
        ({"min_reuse_gap_frames": -1}, SemanticConfigurationError),
        ({"use_velocity_prediction": 1}, ConfigurationTypeError),
        ({"min_common_keypoints": 0}, SemanticConfigurationError),
        ({"cost_reduction": "sum"}, SemanticConfigurationError),
        ({"overflow_policy": "truncate"}, SemanticConfigurationError),
    ],
)
def test_config_rejects_invalid_values_without_fallback(
    overrides: Mapping[str, object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        _config(**overrides)


def test_config_rejects_missing_and_unknown_keys() -> None:
    complete: dict[str, object] = {
        "max_distance": 0.1,
        "max_missed_frames": 2,
        "min_reuse_gap_frames": 0,
        "use_velocity_prediction": True,
        "min_common_keypoints": 1,
        "cost_reduction": "mean",
        "overflow_policy": "error",
    }
    missing = dict(complete)
    missing.pop("max_distance")
    with pytest.raises(MissingConfigurationKeyError):
        ObservationTrackingConfig.from_mapping(missing)
    with pytest.raises(UnknownConfigurationKeyError):
        ObservationTrackingConfig.from_mapping({**complete, "legacy_mode": True})


@pytest.mark.parametrize(
    ("mutator", "error_type"),
    [
        (lambda values, visibility: values.__setitem__((0, 0, 0, 0), float("nan")), ValueError),
        (lambda values, visibility: values.__setitem__((0, 0, 0, 0), 1.1), ValueError),
        (lambda values, visibility: visibility.to(torch.float32), TypeError),
    ],
)
def test_visible_observation_contract_is_strict(
    mutator: object,
    error_type: type[Exception],
) -> None:
    values, visibility = _point_observations([[(0.20, 0.20)]])
    if callable(mutator):
        result = mutator(values, visibility)
        if isinstance(result, Tensor):
            visibility = result
    with pytest.raises(error_type):
        track_camera_observations(
            values, visibility, num_slots=1, config=_config()
        )


def test_invisible_nonfinite_coordinates_are_ignored_and_zero_filled() -> None:
    values, visibility = _point_observations([[(0.20, 0.20), (0.30, 0.30)]])
    visibility[0, 1] = False
    values[0, 1] = float("nan")

    tracked = track_camera_observations(
        values, visibility, num_slots=2, config=_config()
    )

    assert torch.isfinite(tracked.values).all()
    assert tracked.values[0, 1].eq(0).all()


@pytest.mark.parametrize("num_slots", [0, -1, True, 1.5])
def test_num_slots_must_be_an_exact_positive_integer(num_slots: object) -> None:
    values, visibility = _point_observations([[(0.20, 0.20)]])
    with pytest.raises((TypeError, ValueError)):
        track_camera_observations(
            values,
            visibility,
            num_slots=num_slots,  # type: ignore[arg-type]
            config=_config(),
        )
