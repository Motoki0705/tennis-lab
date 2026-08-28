from __future__ import annotations

import pytest
import torch

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    _gated_one_to_one_assignment,
    common_lifecycle_tracking_metrics,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)


def _metrics_from_x_positions(
    *,
    target_x: list[list[float]],
    prediction_x: list[list[float]],
    prediction_active: list[list[bool]],
    target_instance_id: list[list[int]],
    target_active: list[list[bool]] | None = None,
    id_switch_distance: float = 0.05,
) -> dict[str, torch.Tensor]:
    target_x_tensor = torch.tensor(target_x, dtype=torch.float32)
    prediction_x_tensor = torch.tensor(prediction_x, dtype=torch.float32)
    target_ids = torch.tensor(target_instance_id, dtype=torch.long)
    target_position = torch.zeros(*target_x_tensor.shape, 3)
    target_position[..., 0] = target_x_tensor
    prediction_position = torch.zeros(*prediction_x_tensor.shape, 3)
    prediction_position[..., 0] = prediction_x_tensor
    active = torch.tensor(prediction_active, dtype=torch.bool)
    target_presence = (
        torch.tensor(target_active, dtype=torch.bool)
        if target_active is not None
        else target_ids >= 0
    )
    num_targets = target_position.shape[-2]
    num_queries = prediction_position.shape[-2]
    num_assigned = min(num_targets, num_queries)
    return common_lifecycle_tracking_metrics(
        {
            "position": prediction_position.unsqueeze(0),
            "presence_logits": torch.where(active, 20.0, -20.0).unsqueeze(0),
        },
        {
            "target_position": target_position.unsqueeze(0),
            "target_presence": target_presence.unsqueeze(0),
            "target_instance_id": target_ids.unsqueeze(0),
            "frame_mask": torch.ones(1, target_ids.shape[0], dtype=torch.bool),
        },
        [
            (
                torch.arange(num_assigned, dtype=torch.long),
                torch.arange(num_assigned, dtype=torch.long),
            )
        ],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.01,
            id_switch_distance=id_switch_distance,
        ),
    )


def test_metrics_measure_two_segments_as_one_legal_query_reuse() -> None:
    target_presence = torch.zeros(1, 12, 1, dtype=torch.bool)
    target_presence[:, 1:4, 0] = True
    target_presence[:, 7:10, 0] = True
    target_instance_id = torch.full((1, 12, 1), -1, dtype=torch.long)
    target_instance_id[:, 1:4, 0] = 20
    target_instance_id[:, 7:10, 0] = 21
    target_position = torch.zeros(1, 12, 1, 3)
    target_position[:, 1:4, 0, 0] = 1.0
    target_position[:, 7:10, 0, 0] = 2.0
    prediction_position = torch.zeros(1, 12, 2, 3)
    prediction_position[:, :, 0] = target_position[:, :, 0]
    presence_logits = torch.full((1, 12, 2), -20.0)
    presence_logits[:, 1:4, 0] = 20.0
    presence_logits[:, 7:10, 0] = 20.0

    metrics = common_lifecycle_tracking_metrics(
        {"position": prediction_position, "presence_logits": presence_logits},
        {
            "target_position": target_position,
            "target_presence": target_presence,
            "target_instance_id": target_instance_id,
            "frame_mask": torch.ones(1, 12, dtype=torch.bool),
        },
        [(torch.tensor([0]), torch.tensor([0]))],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.1,
            id_switch_distance=0.05,
        ),
    )

    assert metrics["birth_frame_error"].item() == 0.0
    assert metrics["death_frame_error"].item() == 0.0
    assert metrics["lifecycle_presence_f1"].item() == 1.0
    assert metrics["query_reuse_count"].item() == 1.0
    assert metrics["segment_id_switches"].item() == 0.0
    assert metrics["id_switches"].item() == 0.0
    assert metrics["illegal_overlap_count"].item() == 0.0


def test_tracking_metric_config_requires_an_independent_finite_positive_gate() -> None:
    valid = {
        "presence_threshold": 0.5,
        "duplicate_distance": 0.05,
        "id_switch_distance": 0.05,
    }

    with pytest.raises(MissingConfigurationKeyError, match="id_switch_distance"):
        TrackingMetricConfig.from_mapping(
            {key: value for key, value in valid.items() if key != "id_switch_distance"}
        )
    with pytest.raises(UnknownConfigurationKeyError, match="id_switch_distance_m"):
        TrackingMetricConfig.from_mapping({**valid, "id_switch_distance_m": 0.05})
    for wrong_type_value in (1, True, "0.05", None):
        with pytest.raises(ConfigurationTypeError, match="id_switch_distance"):
            TrackingMetricConfig.from_mapping({**valid, "id_switch_distance": wrong_type_value})
    for invalid_numeric_value in (0.0, -0.01, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SemanticConfigurationError, match="id_switch_distance"):
            TrackingMetricConfig.from_mapping({**valid, "id_switch_distance": invalid_numeric_value})


@pytest.mark.parametrize(
    ("prediction_x", "prediction_active", "expected_switches"),
    [
        (
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[True, False], [True, False], [True, False]],
            0.0,
        ),
        (
            [[0.0, 1.0], [1.0, 0.0]],
            [[True, False], [False, True]],
            1.0,
        ),
        (
            [[0.0, 1.0], [1.0, 1.0]],
            [[True, False], [False, True]],
            0.0,
        ),
    ],
    ids=["stable", "replacement", "far-false-positive"],
)
def test_id_switch_assignment_handles_stability_replacement_and_gate(
    prediction_x: list[list[float]],
    prediction_active: list[list[bool]],
    expected_switches: float,
) -> None:
    frames = len(prediction_x)
    metrics = _metrics_from_x_positions(
        target_x=[[0.0]] * frames,
        prediction_x=prediction_x,
        prediction_active=prediction_active,
        target_instance_id=[[10]] * frames,
    )

    assert metrics["id_switches"].item() == expected_switches
    assert metrics["segment_id_switches"].item() == expected_switches
    assert metrics["query_reuse_count"].item() == 0.0


def test_crossing_targets_keep_previous_frame_correspondence() -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0, 0.04], [0.04, 0.0]],
        prediction_x=[[0.0, 0.04], [0.0, 0.04]],
        prediction_active=[[True, True], [True, True]],
        target_instance_id=[[10, 20], [10, 20]],
    )

    assert metrics["id_switches"].item() == 0.0


@pytest.mark.parametrize(
    ("continued_query_x", "expected_switches"),
    [(0.05, 0.0), (0.050001, 1.0)],
    ids=["equal-gate-retained", "outside-gate-reassigned"],
)
def test_previous_correspondence_has_priority_only_while_gated(
    continued_query_x: float,
    expected_switches: float,
) -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [continued_query_x, 0.0]],
        prediction_active=[[True, False], [True, True]],
        target_instance_id=[[10], [10]],
    )

    assert metrics["id_switches"].item() == expected_switches


def test_query_competition_remains_one_to_one_across_a_prediction_gap() -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0, 0.02], [0.0, 0.02], [0.0, 0.02]],
        prediction_x=[[0.0, 0.02], [0.0, 0.02], [0.0, 0.02]],
        prediction_active=[[True, True], [True, False], [True, True]],
        target_instance_id=[[10, 20], [10, 20], [10, 20]],
    )

    assert metrics["id_switches"].item() == 0.0


def test_gated_assignment_maximizes_cardinality_before_distance() -> None:
    distances = torch.tensor(
        [[0.0, 0.049], [0.01, 0.051]],
        dtype=torch.float64,
    )

    assert _gated_one_to_one_assignment(
        distances,
        max_distance=0.05,
    ) == [(0, 1), (1, 0)]


def test_gated_assignment_does_not_treat_a_lower_cost_as_an_equal_tie() -> None:
    distance = torch.tensor(0.02, dtype=torch.float64)
    representably_lower = torch.nextafter(distance, torch.zeros_like(distance))
    distances = torch.tensor(
        [
            [distance.item(), distance.item()],
            [representably_lower.item(), distance.item()],
        ],
        dtype=torch.float64,
    )

    assert _gated_one_to_one_assignment(
        distances,
        max_distance=0.05,
    ) == [(0, 1), (1, 0)]


def test_gated_assignment_preserves_lower_cost_when_normalization_rounds_to_tie() -> None:
    max_distance = 0.05
    half_gate = torch.tensor(max_distance / 2.0, dtype=torch.float64)
    lower = torch.nextafter(half_gate, torch.full_like(half_gate, float("inf")))
    higher = torch.nextafter(lower, torch.full_like(lower, float("inf")))
    assert lower.item() < higher.item()
    assert lower.item() / max_distance == higher.item() / max_distance
    distances = torch.tensor(
        [
            [lower.item(), lower.item()],
            [lower.item(), higher.item()],
        ],
        dtype=torch.float64,
    )

    assert _gated_one_to_one_assignment(
        distances,
        max_distance=max_distance,
    ) == [(0, 1), (1, 0)]


def test_gated_assignment_fails_closed_when_dummy_penalty_overflows() -> None:
    distances = torch.zeros((2, 1), dtype=torch.float64)

    with pytest.raises(OverflowError, match="unmatched penalty"):
        _gated_one_to_one_assignment(
            distances,
            max_distance=torch.finfo(torch.float64).max,
        )


def test_equal_distance_tie_uses_stable_target_then_query_index_order() -> None:
    distances = torch.full((2, 2), 0.02, dtype=torch.float64)
    expected_assignment = [(0, 0), (1, 1)]

    for _ in range(20):
        assert _gated_one_to_one_assignment(
            distances,
            max_distance=0.05,
        ) == expected_assignment

    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0]],
        prediction_x=[[-0.01, 0.01], [-0.01, 0.01]],
        prediction_active=[[True, True], [False, True]],
        target_instance_id=[[10], [10]],
    )

    assert metrics["id_switches"].item() == 1.0


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_prediction_is_an_unmatched_gap_not_a_correspondence(
    nonfinite: float,
) -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [1.0, nonfinite], [1.0, 0.0]],
        prediction_active=[[True, False], [False, True], [False, True]],
        target_instance_id=[[10], [10], [10]],
    )

    assert metrics["id_switches"].item() == 1.0


def test_last_valid_query_crosses_prediction_gap_but_not_lifecycle_boundary() -> None:
    within_lifecycle = _metrics_from_x_positions(
        target_x=[[0.0], [0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        prediction_active=[[True, False], [False, False], [False, True]],
        target_instance_id=[[10], [10], [10]],
    )
    across_lifecycles = _metrics_from_x_positions(
        target_x=[[0.0], [0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        prediction_active=[[True, False], [False, False], [False, True]],
        target_instance_id=[[10], [-1], [11]],
    )

    assert within_lifecycle["id_switches"].item() == 1.0
    assert across_lifecycles["id_switches"].item() == 0.0


def test_adjacent_lifecycle_change_resets_last_valid_identity() -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [1.0, 0.0]],
        prediction_active=[[True, False], [False, True]],
        target_instance_id=[[10], [11]],
    )

    assert metrics["id_switches"].item() == 0.0


def test_last_valid_query_crosses_an_unmatched_target_frame_in_same_lifecycle() -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0], [0.0]],
        prediction_x=[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        prediction_active=[[True, False], [False, False], [False, True]],
        target_instance_id=[[10], [10], [10]],
        target_active=[[True], [False], [True]],
    )

    assert metrics["id_switches"].item() == 1.0


@pytest.mark.parametrize(
    ("initial_distance", "expected_switches"),
    [(0.049, 1.0), (0.05, 1.0), (0.051, 0.0)],
    ids=["inside", "equal", "outside"],
)
def test_id_switch_gate_boundary_is_explicit(
    initial_distance: float,
    expected_switches: float,
) -> None:
    metrics = _metrics_from_x_positions(
        target_x=[[0.0], [0.0]],
        prediction_x=[[initial_distance, 1.0], [1.0, 0.0]],
        prediction_active=[[True, False], [False, True]],
        target_instance_id=[[10], [10]],
    )

    assert metrics["id_switches"].item() == expected_switches


def _count_rich_metrics(
    rich_sequences: list[bool],
) -> dict[str, torch.Tensor]:
    batch_size = len(rich_sequences)
    target_position = torch.zeros(1, 6, 2, 3)
    target_position[0, 0, 1, 0] = 0.02
    target_presence = torch.tensor(
        [[[True, True], [True, False], [False, False], [True, False], [True, False], [True, False]]]
    )
    target_ids = torch.tensor(
        [[[10, 10], [10, -1], [-1, -1], [11, -1], [11, -1], [11, -1]]]
    )
    prediction_position = torch.ones(1, 6, 3, 3)
    prediction_position[0, 0, 0] = 0.0
    prediction_position[0, 0, 1] = 0.0
    prediction_position[0, 0, 1, 0] = 0.02
    prediction_position[0, 3, 0] = 0.0
    prediction_position[0, 4:6, 1] = 0.0
    prediction_active = torch.tensor(
        [
            [
                [True, True, False],
                [False, False, False],
                [False, False, True],
                [True, False, False],
                [False, True, False],
                [False, True, False],
            ]
        ]
    )
    prediction = {
        "position": prediction_position.repeat(batch_size, 1, 1, 1),
        "presence_logits": torch.where(prediction_active, 20.0, -20.0).repeat(
            batch_size, 1, 1
        ),
    }
    batch = {
        "target_position": target_position.repeat(batch_size, 1, 1, 1),
        "target_presence": target_presence.repeat(batch_size, 1, 1),
        "target_instance_id": target_ids.repeat(batch_size, 1, 1),
        "frame_mask": torch.ones(batch_size, 6, dtype=torch.bool),
    }
    for batch_index, is_rich in enumerate(rich_sequences):
        if is_rich:
            continue
        prediction["presence_logits"][batch_index].fill_(-20.0)
        batch["target_presence"][batch_index].fill_(False)
        batch["target_instance_id"][batch_index].fill_(-1)
    assignments = [
        (torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(batch_size)
    ]
    return common_lifecycle_tracking_metrics(
        prediction,
        batch,
        assignments,
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
    )


def test_all_additive_counts_are_reported_as_per_sequence_means() -> None:
    single = _count_rich_metrics([True])
    repeated = _count_rich_metrics([True, True])
    expected = {
        "query_reuse_count": 1.0,
        "illegal_overlap_count": 1.0,
        "segment_id_switches": 1.0,
        "id_switches": 1.0,
        "duplicate_active_tracks": 1.0,
        "missed_gt_frames": 3.0,
        "inactive_query_false_positives": 1.0,
    }

    for name, expected_value in expected.items():
        assert single[name].item() == expected_value
        torch.testing.assert_close(repeated[name], single[name])
    assert repeated["id_switches"] is repeated["segment_id_switches"]


def test_actual_count_metrics_are_invariant_for_heterogeneous_short_batches() -> None:
    sequences = [True, False, True, True, False]
    count_keys = (
        "query_reuse_count",
        "illegal_overlap_count",
        "segment_id_switches",
        "id_switches",
        "duplicate_active_tracks",
        "missed_gt_frames",
        "inactive_query_false_positives",
    )
    expected = _count_rich_metrics(sequences)

    for partition in ((1, 1, 1, 1, 1), (2, 2, 1)):
        offset = 0
        weighted = {name: torch.tensor(0.0) for name in count_keys}
        for batch_size in partition:
            metrics = _count_rich_metrics(sequences[offset : offset + batch_size])
            for name in count_keys:
                weighted[name] += metrics[name] * batch_size
            offset += batch_size
        assert offset == len(sequences)
        for name in count_keys:
            torch.testing.assert_close(
                weighted[name] / len(sequences),
                expected[name],
            )


def test_common_metric_key_contract_matches_the_baseline_inventory() -> None:
    metrics = _count_rich_metrics([True])

    assert set(metrics) == {
        "position_error",
        "presence_precision",
        "presence_recall",
        "presence_f1",
        "lifecycle_presence_f1",
        "birth_frame_error",
        "death_frame_error",
        "query_reuse_count",
        "illegal_overlap_count",
        "segment_id_switches",
        "id_switches",
        "duplicate_active_tracks",
        "missed_gt_frames",
        "inactive_query_false_positives",
    }
