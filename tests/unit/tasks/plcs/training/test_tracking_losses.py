from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.plcs.training.tracking_losses import (
    PLCSTrackingLoss,
    _presence_pairwise_loss,
)
from src.utils.geometry.court_pose import canonical_pose_to_world_pose
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.schema.court_normalization import normalize_court_position


def _tracking_config(**overrides: float):
    return OmegaConf.merge(
        OmegaConf.load(Path("src/tasks/plcs/configs/loss/tracking.yaml")),
        overrides,
    )


def _criterion(**overrides: float) -> PLCSTrackingLoss:
    return PLCSTrackingLoss(_tracking_config(**overrides))


def _hybrid_config():
    return OmegaConf.merge(
        OmegaConf.load(Path("src/tasks/plcs/configs/loss/tracking.yaml")),
        OmegaConf.load(Path("src/tasks/plcs/configs/loss/_base.yaml")),
        {
            "position_weight": 1.0,
            "position_smooth_l1_beta": 0.1,
            "rotation_weight": 0.05,
            "angle_weight": 0.05,
            "canonical_pose_weight": 1.0,
            "reprojection_weight": 1.0,
            "reprojection_smooth_l1_beta": 0.01,
        },
    )


def _hybrid_criterion() -> PLCSTrackingLoss:
    return PLCSTrackingLoss(_hybrid_config())


def _fixture():
    torch.manual_seed(8)
    prediction = {
        "position": torch.rand(1, 5, 3, 3, requires_grad=True),
        "rotation": torch.nn.functional.normalize(
            torch.rand(1, 5, 3, 2), dim=-1
        ).requires_grad_(),
        "presence_logits": torch.randn(1, 5, 3, requires_grad=True),
    }
    batch = {
        "target_position": torch.rand(1, 5, 2, 3),
        "target_rotation": torch.nn.functional.normalize(
            torch.rand(1, 5, 2, 2), dim=-1
        ),
        "target_presence": torch.tensor(
            [[[1, 0], [1, 1], [1, 1], [0, 1], [0, 0]]], dtype=torch.bool
        ),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 5, dtype=torch.bool),
    }
    return prediction, batch


def _cardinality_fixture(
    presence_logits: torch.Tensor,
    target_presence: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch_size, frames, queries = presence_logits.shape
    target_slots = target_presence.shape[-1]
    if padding_mask is None:
        padding_mask = torch.zeros(
            batch_size,
            2,
            frames,
            dtype=torch.bool,
            device=presence_logits.device,
        )
    unit_rotation = torch.tensor(
        [1.0, 0.0],
        dtype=presence_logits.dtype,
        device=presence_logits.device,
    )
    prediction = {
        "position": torch.zeros(
            batch_size,
            frames,
            queries,
            3,
            dtype=presence_logits.dtype,
            device=presence_logits.device,
        ),
        "rotation": unit_rotation.expand(batch_size, frames, queries, 2).clone(),
        "presence_logits": presence_logits,
    }
    batch = {
        "target_position": torch.zeros(
            batch_size,
            frames,
            target_slots,
            3,
            dtype=presence_logits.dtype,
            device=presence_logits.device,
        ),
        "target_rotation": unit_rotation.expand(
            batch_size, frames, target_slots, 2
        ).clone(),
        "target_presence": target_presence,
        "target_slot_mask": target_presence.any(dim=1),
        "padding_mask": padding_mask,
    }
    return prediction, batch


def _brute_force_poisson_binomial_nll(
    logits: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    probabilities = logits.double().sigmoid()
    outcome_probabilities: list[torch.Tensor] = []
    for outcome in product((False, True), repeat=probabilities.numel()):
        if sum(outcome) != target_count:
            continue
        terms = [
            probability if present else 1.0 - probability
            for probability, present in zip(probabilities, outcome, strict=True)
        ]
        outcome_probabilities.append(torch.stack(terms).prod())
    return -torch.stack(outcome_probabilities).sum().log()


def _matching_sensitive_fixture():
    target_a = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    target_b = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    query_a = [
        -0.5,
        2.0,
        0.5,
        0.5,
        2.0,
        2.0,
        4.0,
        -2.0,
        4.0,
        -4.0,
        -0.5,
        -0.5,
    ]
    query_b = [
        -2.0,
        -4.0,
        -2.0,
        -0.5,
        0.5,
        2.0,
        2.0,
        4.0,
        0.5,
        2.0,
        4.0,
        2.0,
    ]
    frames = len(target_a)
    target_presence = torch.tensor(
        [[[target_a[frame], target_b[frame]] for frame in range(frames)]],
        dtype=torch.bool,
    )
    presence_logits = torch.tensor([[query_a, query_b]]).transpose(1, 2)
    rotation = torch.tensor([1.0, 0.0]).expand(1, frames, 2, 2).clone()
    position = torch.zeros(1, frames, 2, 3)
    prediction = {
        "position": position,
        "rotation": rotation,
        "presence_logits": presence_logits,
    }
    batch = {
        "target_position": position.clone(),
        "target_rotation": rotation.clone(),
        "target_presence": target_presence,
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 1, frames, dtype=torch.bool),
    }
    return prediction, batch


def _hybrid_fixture():
    frames = 4
    queries = 2
    joints = 17
    views = 2
    target_position = torch.tensor(
        [
            [
                [[0.05, -0.08, 0.02], [-0.20, 0.12, 0.04]],
                [[0.06, -0.07, 0.02], [-0.19, 0.11, 0.04]],
                [[0.07, -0.06, 0.02], [-0.18, 0.10, 0.04]],
                [[0.08, -0.05, 0.02], [-0.17, 0.09, 0.04]],
            ]
        ]
    )
    target_rotation = torch.nn.functional.normalize(
        torch.tensor(
            [
                [
                    [[1.0, 0.1], [0.6, 0.8]],
                    [[0.98, 0.2], [0.5, 0.86]],
                    [[0.95, 0.3], [0.4, 0.92]],
                    [[0.9, 0.4], [0.3, 0.95]],
                ]
            ]
        ),
        dim=-1,
    )
    canonical = torch.linspace(
        -0.7,
        0.9,
        frames * queries * joints * 3,
    ).reshape(1, frames, queries, joints, 3)
    canonical = canonical.clone()
    canonical[..., 2] += 1.2
    target_world = canonical_pose_to_world_pose(
        canonical,
        target_position,
        target_rotation,
    )
    camera_R = torch.eye(3).view(1, 1, 3, 3).expand(1, views, -1, -1)
    camera_C = torch.tensor([[[0.0, 0.0, -20.0], [1.0, -1.0, -18.0]]])
    camera_f = torch.tensor([[800.0, 900.0]])
    camera_cx = torch.full((1, views), 640.0)
    camera_cy = torch.full((1, views), 360.0)
    camera_w = torch.full((1, views), 1280.0)
    camera_h = torch.full((1, views), 720.0)
    target_uv, in_front = DifferentiablePinholeProjection()(
        target_world,
        camera_R,
        camera_C,
        camera_f,
        camera_cx,
        camera_cy,
        camera_w,
        camera_h,
    )
    assert in_front.all()
    target_presence = torch.tensor(
        [[[1, 1], [1, 1], [1, 0], [1, 1]]], dtype=torch.bool
    )
    target_vis = target_presence[:, None, :, :, None].expand(
        -1, views, -1, -1, joints
    ).clone()
    prediction = {
        "position": target_position.clone().requires_grad_(),
        "rotation": target_rotation.clone().requires_grad_(),
        "presence_logits": torch.where(
            target_presence,
            torch.full_like(target_presence, 12.0, dtype=torch.float32),
            torch.full_like(target_presence, -12.0, dtype=torch.float32),
        ).requires_grad_(),
        "canonical_pose": canonical.clone().requires_grad_(),
    }
    batch = {
        "target_position": target_position,
        "target_rotation": target_rotation,
        # Deliberately wrong: the loss must derive canonical pose from world joints.
        "target_canonical_pose_3d": torch.full_like(canonical, 999.0),
        "target_human_kp_3d": target_world,
        "target_presence": target_presence,
        "target_slot_mask": torch.ones(1, queries, dtype=torch.bool),
        "target_instance_id": torch.tensor(
            [[[10, 20], [10, 20], [10, -1], [10, 20]]]
        ),
        "padding_mask": torch.zeros(1, views, frames, dtype=torch.bool),
        "human_kp_target": target_uv.detach(),
        "human_vis_target": target_vis,
        "camera_R": camera_R,
        "camera_C": camera_C,
        "camera_f": camera_f,
        "camera_cx": camera_cx,
        "camera_cy": camera_cy,
        "camera_w": camera_w,
        "camera_h": camera_h,
    }
    return prediction, batch


def _compute(
    criterion: PLCSTrackingLoss,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
):
    inputs, assignments = criterion.prepare_inputs(prediction, batch)
    return criterion(inputs), assignments


def test_player_loss_is_invariant_to_gt_person_order_and_smoothness_is_off() -> None:
    prediction, batch = _fixture()
    original, _ = _compute(_criterion(), prediction, batch)
    permutation = torch.tensor([1, 0])
    permuted = dict(batch)
    for key in ("target_position", "target_rotation", "target_presence"):
        permuted[key] = batch[key][:, :, permutation]
    permuted["target_slot_mask"] = batch["target_slot_mask"][:, permutation]
    reordered, _ = _compute(_criterion(), prediction, permuted)
    torch.testing.assert_close(original["total"], reordered["total"])
    assert original["track_smoothness"].item() == 0.0


def test_all_persons_absent_does_not_produce_nan() -> None:
    prediction, batch = _fixture()
    batch["target_slot_mask"].zero_()
    batch["target_presence"].zero_()
    losses, assignments = _compute(_criterion(), prediction, batch)
    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 0
    losses["total"].backward()


def test_matching_accepts_bfloat16_predictions() -> None:
    prediction, batch = _fixture()
    prediction = {
        key: value.detach().to(torch.bfloat16).requires_grad_()
        for key, value in prediction.items()
    }
    batch["target_position"] = batch["target_position"].to(torch.bfloat16)
    batch["target_rotation"] = batch["target_rotation"].to(torch.bfloat16)

    losses, assignments = _compute(_criterion(), prediction, batch)

    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 2


def test_final_inactive_weight_does_not_change_fixed_matching_assignment() -> None:
    prediction, batch = _matching_sensitive_fixture()
    low_losses, low_assignments = _compute(
        _criterion(
            presence_inactive_weight=0.25,
            match_presence_inactive_weight=0.25,
        ),
        prediction,
        batch,
    )
    high_losses, high_assignments = _compute(
        _criterion(
            presence_inactive_weight=2.0,
            match_presence_inactive_weight=0.25,
        ),
        prediction,
        batch,
    )

    torch.testing.assert_close(low_assignments[0][0], high_assignments[0][0])
    torch.testing.assert_close(low_assignments[0][1], high_assignments[0][1])
    assert low_assignments[0][1].tolist() == [1, 0]
    assert not torch.isclose(low_losses["presence"], high_losses["presence"])


def test_matching_inactive_weight_can_change_assignment_independently() -> None:
    prediction, batch = _matching_sensitive_fixture()
    _, low_assignments = _compute(
        _criterion(
            presence_inactive_weight=0.25,
            match_presence_inactive_weight=0.25,
        ),
        prediction,
        batch,
    )
    _, high_assignments = _compute(
        _criterion(
            presence_inactive_weight=0.25,
            match_presence_inactive_weight=2.0,
        ),
        prediction,
        batch,
    )

    assert low_assignments[0][1].tolist() == [1, 0]
    assert high_assignments[0][1].tolist() == [0, 1]


def test_legacy_tracking_loss_config_couples_matching_inactive_weight() -> None:
    config = _tracking_config(presence_inactive_weight=2.0)
    del config["match_presence_inactive_weight"]

    criterion = PLCSTrackingLoss(config)
    _, assignments = _compute(criterion, *_matching_sensitive_fixture())

    assert criterion.presence_inactive_weight == 2.0
    assert criterion.match_presence_inactive_weight == 2.0
    assert assignments[0][1].tolist() == [0, 1]


def test_present_malformed_matching_inactive_weight_does_not_fallback() -> None:
    config = _tracking_config()
    config.match_presence_inactive_weight = None

    with pytest.raises(ValueError, match="finite non-negative number"):
        PLCSTrackingLoss(config)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_matching_inactive_weight_must_be_finite_and_nonnegative(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _criterion(match_presence_inactive_weight=value)


def test_tracking_position_loss_is_equal_for_same_physical_xyz_error() -> None:
    criterion = _criterion()
    values = []
    for axis in range(3):
        error_m = torch.zeros(1, 1, 1, 3)
        error_m[..., axis] = 0.25
        prediction = {
            "position": normalize_court_position(error_m),
            "rotation": torch.tensor([[[[1.0, 0.0]]]]),
            "presence_logits": torch.full((1, 1, 1), 20.0),
        }
        batch = {
            "target_position": torch.zeros_like(error_m),
            "target_rotation": torch.tensor([[[[1.0, 0.0]]]]),
            "target_presence": torch.ones(1, 1, 1, dtype=torch.bool),
            "target_slot_mask": torch.ones(1, 1, dtype=torch.bool),
            "padding_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
        }
        values.append(_compute(criterion, prediction, batch)[0]["position"])

    torch.testing.assert_close(torch.stack(values), values[0].expand(3))


def test_legacy_tracking_config_retains_original_loss_contract() -> None:
    losses, _assignments = _compute(_criterion(), *_fixture())

    assert set(losses) == {
        "total",
        "position",
        "rotation",
        "presence",
        "track_smoothness",
    }


def test_cardinality_loss_is_exactly_zero_when_soft_count_matches_gt_count() -> None:
    presence_logits = torch.zeros(1, 2, 4, requires_grad=True)
    target_presence = torch.tensor(
        [[[True, True, False, False], [False, True, True, False]]]
    )

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(presence_logits, target_presence),
    )

    torch.testing.assert_close(
        losses["cardinality"],
        torch.zeros_like(losses["cardinality"]),
    )


def test_cardinality_loss_uses_mean_beta_one_smooth_l1_of_frame_counts() -> None:
    # Four zero logits predict a soft count of 2. GT counts are 1 and 4, so
    # beta=1 Smooth-L1 terms are 0.5 and 1.5, respectively.
    presence_logits = torch.zeros(1, 2, 4, requires_grad=True)
    target_presence = torch.tensor(
        [[[True, False, False, False], [True, True, True, True]]]
    )

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(presence_logits, target_presence),
    )

    torch.testing.assert_close(losses["cardinality"], torch.tensor(1.0))


def test_cardinality_loss_is_invariant_to_prediction_query_permutation() -> None:
    presence_logits = torch.tensor(
        [[[-2.0, -0.5, 0.7, 3.0], [1.5, -3.0, 0.2, -0.8]]]
    )
    target_presence = torch.tensor(
        [[[True, False, False], [True, True, False]]]
    )
    original, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(presence_logits, target_presence),
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(
            presence_logits[:, :, permutation],
            target_presence,
        ),
    )

    torch.testing.assert_close(original["cardinality"], permuted["cardinality"])


def test_cardinality_overcount_gradient_suppresses_every_presence_logit() -> None:
    presence_logits = torch.zeros(1, 2, 4, requires_grad=True)
    target_presence = torch.tensor(
        [[[True, False], [False, True]]]
    )
    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(presence_logits, target_presence),
    )

    losses["cardinality"].backward()

    assert presence_logits.grad is not None
    assert torch.all(presence_logits.grad > 0.0)


def test_cardinality_excludes_only_frames_padded_in_every_view() -> None:
    presence_logits = torch.zeros(1, 3, 4, requires_grad=True)
    target_presence = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )
    padding_mask = torch.tensor(
        [[[False, False, True], [False, True, True]]]
    )

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(
            presence_logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    # Frames 0 and 1 each have |count error|=1 (loss 0.5); frame 2 is excluded.
    torch.testing.assert_close(losses["cardinality"], torch.tensor(0.5))


def test_cardinality_mean_uses_the_number_of_valid_frames_across_the_batch() -> None:
    presence_logits = torch.zeros(2, 3, 4, requires_grad=True)
    target_presence = torch.tensor(
        [
            [
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            [
                [True, True, False, False],
                [True, True, True, True],
                [False, False, False, False],
            ],
        ]
    )
    padding_mask = torch.tensor(
        [
            [[False, True, True], [False, True, True]],
            [[False, False, True], [False, True, True]],
        ]
    )

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(
            presence_logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    # The three valid-frame losses are 0.5, 0.0, and 1.5.
    torch.testing.assert_close(losses["cardinality"], torch.tensor(2.0 / 3.0))


def test_cardinality_boolean_selection_excludes_nan_in_padded_frame() -> None:
    presence_logits = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [float("nan")] * 4]],
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 2, 2, dtype=torch.bool)
    padding_mask = torch.tensor(
        [[[False, True], [False, True]]],
    )

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(
            presence_logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    torch.testing.assert_close(losses["cardinality"], torch.tensor(1.5))
    losses["cardinality"].backward()
    assert presence_logits.grad is not None
    assert torch.isfinite(presence_logits.grad).all()
    torch.testing.assert_close(
        presence_logits.grad[:, 1],
        torch.zeros_like(presence_logits.grad[:, 1]),
    )


def test_cardinality_is_finite_differentiable_zero_when_every_frame_is_padded() -> None:
    presence_logits = torch.full(
        (1, 3, 4),
        float("nan"),
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 3, 2, dtype=torch.bool)
    padding_mask = torch.ones(1, 2, 3, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(
            presence_logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    assert torch.isfinite(losses["cardinality"])
    torch.testing.assert_close(
        losses["cardinality"],
        torch.zeros_like(losses["cardinality"]),
    )
    losses["cardinality"].backward()
    assert presence_logits.grad is not None
    torch.testing.assert_close(
        presence_logits.grad,
        torch.zeros_like(presence_logits.grad),
    )


def test_cardinality_does_not_hide_nan_in_valid_frame() -> None:
    presence_logits = torch.full(
        (1, 1, 4),
        float("nan"),
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 1, 2, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(cardinality_weight=1.0),
        *_cardinality_fixture(presence_logits, target_presence),
    )

    assert torch.isnan(losses["cardinality"])


def test_cardinality_weight_zero_preserves_total_and_positive_weight_adds_term() -> None:
    prediction, batch = _fixture()
    legacy, _assignments = _compute(_criterion(), prediction, batch)
    zero, _assignments = _compute(
        _criterion(cardinality_weight=0.0), prediction, batch
    )
    positive, _assignments = _compute(
        _criterion(cardinality_weight=2.0), prediction, batch
    )

    assert "cardinality" not in legacy
    assert "cardinality" not in zero
    torch.testing.assert_close(zero["total"], legacy["total"])
    torch.testing.assert_close(
        positive["total"],
        legacy["total"] + 2.0 * positive["cardinality"],
    )


@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), 10**400, None],
    ids=["negative", "nan", "infinity", "overflowing-integer", "invalid-type"],
)
def test_cardinality_weight_must_be_finite_and_nonnegative(value: object) -> None:
    config_values = cast(
        "dict[str, object]",
        OmegaConf.to_container(_tracking_config()),
    )
    config_values["cardinality_weight"] = value

    with pytest.raises(ValueError, match="finite.*non-negative"):
        PLCSTrackingLoss(SimpleNamespace(**config_values))


@pytest.mark.parametrize("target_count", range(5))
def test_cardinality_nll_matches_all_sixteen_q4_outcomes(
    target_count: int,
) -> None:
    logits = torch.tensor([-2.0, -0.3, 0.8, 2.2]).reshape(1, 1, 4)
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    target_presence[..., :target_count] = True

    prediction, batch = _cardinality_fixture(logits, target_presence)
    batch["target_slot_mask"].zero_()
    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        prediction,
        batch,
    )

    expected = _brute_force_poisson_binomial_nll(logits[0, 0], target_count)
    torch.testing.assert_close(
        losses["cardinality_nll"].double(),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("target_count", [0, 4])
def test_cardinality_nll_boundary_counts_match_closed_form(
    target_count: int,
) -> None:
    logits = torch.tensor([-3.0, -0.5, 0.75, 4.0]).reshape(1, 1, 4)
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    target_presence[..., :target_count] = True

    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    expected = (
        torch.nn.functional.softplus(logits.double()).sum()
        if target_count == 0
        else torch.nn.functional.softplus(-logits.double()).sum()
    )
    torch.testing.assert_close(losses["cardinality_nll"], expected)


def test_cardinality_nll_loss_and_gradient_are_query_permutation_invariant() -> None:
    logits = torch.tensor(
        [[[-2.0, -0.3, 0.8, 2.2]]],
        requires_grad=True,
    )
    target_presence = torch.tensor([[[True, True, False, False]]])
    original, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )
    original["cardinality_nll"].backward()
    assert logits.grad is not None
    original_gradient = logits.grad.detach().clone()

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_logits = (
        logits.detach()[:, :, permutation].clone().requires_grad_(True)
    )
    permuted, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(permuted_logits, target_presence),
    )
    permuted["cardinality_nll"].backward()

    torch.testing.assert_close(
        original["cardinality_nll"],
        permuted["cardinality_nll"],
    )
    assert permuted_logits.grad is not None
    torch.testing.assert_close(
        permuted_logits.grad,
        original_gradient[:, :, permutation],
    )


def test_cardinality_nll_is_float64_and_backward_finite_for_bfloat16() -> None:
    logits = torch.tensor(
        [[[-30.0, -2.0, 3.0, 25.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target_presence = torch.tensor([[[True, True, False, False]]])

    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    assert losses["cardinality_nll"].dtype == torch.float64
    assert torch.isfinite(losses["cardinality_nll"])
    losses["cardinality_nll"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("logit_value", [3e38, -3e38], ids=["positive", "negative"])
@pytest.mark.parametrize("target_count", range(5))
def test_cardinality_nll_extreme_finite_float32_is_finite_forward_and_backward(
    logit_value: float,
    target_count: int,
) -> None:
    logits = torch.full(
        (1, 1, 4),
        logit_value,
        dtype=torch.float32,
        requires_grad=True,
    )
    assert torch.isfinite(logits).all()
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    target_presence[..., :target_count] = True

    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    assert losses["cardinality_nll"].dtype == torch.float64
    assert torch.isfinite(losses["cardinality_nll"])
    losses["cardinality_nll"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logit_value", "target_count"),
    [(float("inf"), 0), (float("-inf"), 4)],
    ids=["positive-infinity-impossible-zero", "negative-infinity-impossible-all"],
)
def test_cardinality_nll_keeps_impossible_nonfinite_events_nonfinite(
    logit_value: float,
    target_count: int,
) -> None:
    logits = torch.full((1, 1, 4), logit_value, requires_grad=True)
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    target_presence[..., :target_count] = True

    prediction, batch = _cardinality_fixture(logits, target_presence)
    batch["target_slot_mask"].zero_()
    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        prediction,
        batch,
    )

    assert torch.isinf(losses["cardinality_nll"])


def test_cardinality_nll_mean_uses_all_valid_frames_and_excludes_padded_nan() -> None:
    logits = torch.zeros(2, 3, 4)
    logits[0, 1:] = float("nan")
    logits[1, 2] = float("nan")
    logits.requires_grad_(True)
    target_presence = torch.tensor(
        [
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            [
                [True, True, False, False],
                [True, True, True, True],
                [False, False, False, False],
            ],
        ]
    )
    padding_mask = torch.tensor(
        [
            [[False, True, True], [False, True, True]],
            [[False, False, True], [False, True, True]],
        ]
    )

    prediction, batch = _cardinality_fixture(
        logits,
        target_presence,
        padding_mask=padding_mask,
    )
    batch["target_slot_mask"].zero_()
    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        prediction,
        batch,
    )

    expected = torch.stack(
        [
            _brute_force_poisson_binomial_nll(torch.zeros(4), count)
            for count in (0, 2, 4)
        ]
    ).mean()
    torch.testing.assert_close(losses["cardinality_nll"].double(), expected)
    losses["cardinality_nll"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(
        logits.grad[0, 1:],
        torch.zeros_like(logits.grad[0, 1:]),
    )
    torch.testing.assert_close(
        logits.grad[1, 2],
        torch.zeros_like(logits.grad[1, 2]),
    )


def test_cardinality_nll_all_padded_is_empty_selection_finite_zero() -> None:
    logits = torch.full(
        (1, 3, 4),
        float("nan"),
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 3, 4, dtype=torch.bool)
    padding_mask = torch.ones(1, 2, 3, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(
            logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    assert torch.isfinite(losses["cardinality_nll"])
    torch.testing.assert_close(
        losses["cardinality_nll"],
        torch.zeros_like(losses["cardinality_nll"]),
    )
    losses["cardinality_nll"].backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_cardinality_nll_does_not_hide_valid_frame_nan() -> None:
    logits = torch.full(
        (1, 1, 4),
        float("nan"),
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(cardinality_nll_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    assert torch.isnan(losses["cardinality_nll"])


def test_soft_cardinality_and_exact_count_nll_can_be_combined() -> None:
    prediction, batch = _fixture()
    baseline, _assignments = _compute(_criterion(), prediction, batch)
    combined, _assignments = _compute(
        _criterion(cardinality_weight=0.7, cardinality_nll_weight=1.3),
        prediction,
        batch,
    )

    assert {"cardinality", "cardinality_nll"}.issubset(combined)
    torch.testing.assert_close(
        combined["total"],
        baseline["total"]
        + 0.7 * combined["cardinality"]
        + 1.3 * combined["cardinality_nll"],
    )


def test_cardinality_nll_missing_and_zero_weights_are_exact_noops() -> None:
    prediction, batch = _fixture()
    missing, _assignments = _compute(_criterion(), prediction, batch)
    zero, _assignments = _compute(
        _criterion(cardinality_nll_weight=0.0),
        prediction,
        batch,
    )

    assert "cardinality_nll" not in missing
    assert "cardinality_nll" not in zero
    assert set(zero) == set(missing)
    torch.testing.assert_close(zero["total"], missing["total"])


@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), 10**400, None],
    ids=["negative", "nan", "infinity", "overflowing-integer", "invalid-type"],
)
def test_cardinality_nll_weight_must_be_finite_and_nonnegative(
    value: object,
) -> None:
    config_values = cast(
        "dict[str, object]",
        OmegaConf.to_container(_tracking_config()),
    )
    config_values["cardinality_nll_weight"] = value

    with pytest.raises(ValueError, match="finite.*non-negative"):
        PLCSTrackingLoss(SimpleNamespace(**config_values))


@pytest.mark.parametrize(
    ("invalid_field", "expected_message"),
    [
        ("presence_logits_rank", r"presence_logits.*\(B,T,Q\)"),
        ("target_presence_shape", r"target_presence.*\(B,T,P\)"),
        ("padding_mask_shape", r"padding_mask.*\(B,V,T\)"),
        ("target_presence_dtype", r"target_presence.*boolean"),
        ("unreachable_count", r"player count greater than.*predicted queries"),
    ],
)
def test_cardinality_nll_rejects_invalid_shape_and_count_contracts(
    invalid_field: str,
    expected_message: str,
) -> None:
    logits = torch.zeros(1, 2, 4)
    target_presence = torch.zeros(1, 2, 4, dtype=torch.bool)
    prediction, batch = _cardinality_fixture(logits, target_presence)
    if invalid_field == "presence_logits_rank":
        prediction["presence_logits"] = logits[:, 0]
    elif invalid_field == "target_presence_shape":
        batch["target_presence"] = target_presence[:, :1]
    elif invalid_field == "padding_mask_shape":
        batch["padding_mask"] = batch["padding_mask"][:, :, :1]
    elif invalid_field == "target_presence_dtype":
        batch["target_presence"] = target_presence.float()
    else:
        unreachable_target = torch.ones(1, 2, 5, dtype=torch.bool)
        prediction, batch = _cardinality_fixture(logits, unreachable_target)

    with pytest.raises(ValueError, match=expected_message):
        _compute(_criterion(cardinality_nll_weight=1.0), prediction, batch)


def test_presence_hard_negative_matches_stable_logits_formula() -> None:
    logits = torch.tensor([[[-2.0, -0.3, 0.8, 2.2]]], requires_grad=True)
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    gamma = 1.5

    losses, _assignments = _compute(
        _criterion(
            presence_hard_negative_weight=1.0,
            presence_hard_negative_gamma=gamma,
        ),
        *_cardinality_fixture(logits, target_presence),
    )

    logits64 = logits.double()
    expected = (
        torch.exp(gamma * torch.nn.functional.logsigmoid(logits64))
        * -torch.nn.functional.logsigmoid(-logits64)
    ).mean()
    torch.testing.assert_close(losses["presence_hard_negative"], expected)


def test_presence_hard_negative_gamma_zero_equals_negative_bce() -> None:
    logits = torch.tensor([[[-2.0, -0.3, 0.8, 2.2]]], requires_grad=True)
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(
            presence_hard_negative_weight=1.0,
            presence_hard_negative_gamma=0.0,
        ),
        *_cardinality_fixture(logits, target_presence),
    )

    expected = -torch.nn.functional.logsigmoid(-logits.double()).mean()
    torch.testing.assert_close(losses["presence_hard_negative"], expected)


@pytest.mark.parametrize("logit_value", [float("-inf"), float("inf")])
def test_presence_hard_negative_gamma_zero_matches_nonfinite_negative_bce(
    logit_value: float,
) -> None:
    logits = torch.tensor([[[logit_value]]], requires_grad=True)
    target_presence = torch.zeros(1, 1, 1, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(
            presence_hard_negative_weight=1.0,
            presence_hard_negative_gamma=0.0,
        ),
        *_cardinality_fixture(logits, target_presence),
    )

    expected = -torch.nn.functional.logsigmoid(-logits.double()).mean()
    if logit_value < 0.0:
        assert torch.isfinite(losses["presence_hard_negative"])
        torch.testing.assert_close(
            losses["presence_hard_negative"],
            torch.zeros_like(losses["presence_hard_negative"]),
        )
    else:
        assert torch.isinf(losses["presence_hard_negative"])
    torch.testing.assert_close(losses["presence_hard_negative"], expected)


def test_presence_hard_negative_gradients_exclude_active_and_transition_slots() -> None:
    logits = torch.zeros(1, 6, 1, requires_grad=True)
    target_presence = torch.tensor(
        [[[False], [False], [True], [True], [False], [False]]]
    )

    losses, _assignments = _compute(
        _criterion(
            presence_hard_negative_weight=1.0,
            transition_radius=0,
        ),
        *_cardinality_fixture(logits, target_presence),
    )
    losses["presence_hard_negative"].backward()

    assert logits.grad is not None
    assert torch.all(logits.grad[0, [0, 1, 5], 0] > 0.0)
    torch.testing.assert_close(
        logits.grad[0, [2, 3, 4], 0],
        torch.zeros(3),
    )


def test_presence_hard_negative_includes_matched_and_unmatched_negatives() -> None:
    logits = torch.zeros(1, 4, 2, requires_grad=True)
    target_presence = torch.tensor(
        [[[True], [True], [False], [False]]]
    )

    losses, assignments = _compute(
        _criterion(
            presence_hard_negative_weight=1.0,
            transition_radius=0,
        ),
        *_cardinality_fixture(logits, target_presence),
    )
    losses["presence_hard_negative"].backward()

    matched_query = assignments[0][0].item()
    unmatched_query = 1 - matched_query
    assert logits.grad is not None
    assert logits.grad[0, 3, matched_query] > 0.0
    torch.testing.assert_close(
        logits.grad[0, :3, matched_query],
        torch.zeros(3),
    )
    assert torch.all(logits.grad[0, :, unmatched_query] > 0.0)


def test_presence_hard_negative_padding_selection_excludes_nan() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0], [float("nan"), float("nan")]]],
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 2, 2, dtype=torch.bool)
    padding_mask = torch.tensor([[[False, True], [False, True]]])

    losses, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(
            logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    assert torch.isfinite(losses["presence_hard_negative"])
    losses["presence_hard_negative"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]))


def test_presence_hard_negative_all_padded_is_differentiable_empty_zero() -> None:
    logits = torch.full((1, 2, 2), float("nan"), requires_grad=True)
    target_presence = torch.zeros(1, 2, 2, dtype=torch.bool)
    padding_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(
            logits,
            target_presence,
            padding_mask=padding_mask,
        ),
    )

    torch.testing.assert_close(
        losses["presence_hard_negative"],
        torch.zeros_like(losses["presence_hard_negative"]),
    )
    losses["presence_hard_negative"].backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_presence_hard_negative_no_selected_negative_is_differentiable_zero() -> None:
    logits = torch.zeros(1, 2, 1, requires_grad=True)
    target_presence = torch.ones(1, 2, 1, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    torch.testing.assert_close(
        losses["presence_hard_negative"],
        torch.zeros_like(losses["presence_hard_negative"]),
    )
    losses["presence_hard_negative"].backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_presence_hard_negative_does_not_hide_selected_nan() -> None:
    logits = torch.full((1, 1, 2), float("nan"), requires_grad=True)
    target_presence = torch.zeros(1, 1, 2, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    assert torch.isnan(losses["presence_hard_negative"])


def test_presence_hard_negative_loss_and_gradient_are_query_permutation_invariant() -> None:
    logits = torch.tensor(
        [[[-2.0, -0.3, 0.8, 2.2]]],
        requires_grad=True,
    )
    target_presence = torch.zeros(1, 1, 4, dtype=torch.bool)
    original, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )
    original["presence_hard_negative"].backward()
    assert logits.grad is not None
    original_gradient = logits.grad.detach().clone()

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_logits = logits.detach()[:, :, permutation].clone().requires_grad_(True)
    permuted, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(permuted_logits, target_presence),
    )
    permuted["presence_hard_negative"].backward()

    torch.testing.assert_close(
        original["presence_hard_negative"],
        permuted["presence_hard_negative"],
    )
    assert permuted_logits.grad is not None
    torch.testing.assert_close(
        permuted_logits.grad,
        original_gradient[:, :, permutation],
    )


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.bfloat16, [-30.0, 30.0]),
        (torch.float32, [-3e38, 3e38]),
    ],
    ids=["bfloat16", "extreme-float32"],
)
def test_presence_hard_negative_is_finite_forward_and_backward(
    dtype: torch.dtype,
    values: list[float],
) -> None:
    logits = torch.tensor([[values]], dtype=dtype, requires_grad=True)
    target_presence = torch.zeros(1, 1, 2, dtype=torch.bool)

    losses, _assignments = _compute(
        _criterion(presence_hard_negative_weight=1.0),
        *_cardinality_fixture(logits, target_presence),
    )

    assert losses["presence_hard_negative"].dtype == torch.float64
    assert torch.isfinite(losses["presence_hard_negative"])
    losses["presence_hard_negative"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_presence_hard_negative_missing_and_zero_weights_are_exact_noops() -> None:
    prediction, batch = _fixture()
    missing, _assignments = _compute(_criterion(), prediction, batch)
    zero, _assignments = _compute(
        _criterion(presence_hard_negative_weight=0.0),
        prediction,
        batch,
    )

    assert "presence_hard_negative" not in missing
    assert "presence_hard_negative" not in zero
    assert set(zero) == set(missing)
    torch.testing.assert_close(zero["total"], missing["total"])


def test_presence_hard_negative_does_not_change_fixed_assignment() -> None:
    prediction, batch = _fixture()
    _, baseline_assignments = _compute(
        _criterion(
            match_presence_weight=0.0,
            presence_hard_negative_weight=0.0,
        ),
        prediction,
        batch,
    )
    _, auxiliary_assignments = _compute(
        _criterion(
            match_presence_weight=0.0,
            presence_hard_negative_weight=2.0,
        ),
        prediction,
        batch,
    )

    torch.testing.assert_close(
        baseline_assignments[0][0],
        auxiliary_assignments[0][0],
    )
    torch.testing.assert_close(
        baseline_assignments[0][1],
        auxiliary_assignments[0][1],
    )


@pytest.mark.parametrize("field", ["presence_hard_negative_weight", "presence_hard_negative_gamma"])
@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), 10**400, None],
    ids=["negative", "nan", "infinity", "overflowing-integer", "invalid-type"],
)
def test_presence_hard_negative_settings_must_be_finite_and_nonnegative(
    field: str,
    value: object,
) -> None:
    config_values = cast(
        "dict[str, object]",
        OmegaConf.to_container(_tracking_config()),
    )
    config_values[field] = value

    with pytest.raises(ValueError, match="finite.*non-negative"):
        PLCSTrackingLoss(SimpleNamespace(**config_values))


def test_presence_hard_negative_gamma_defaults_to_two() -> None:
    criterion = _criterion(presence_hard_negative_weight=1.0)

    assert criterion.presence_hard_negative_gamma == 2.0


def test_presence_pairwise_matches_hand_calculation_and_balances_frames() -> None:
    logits = torch.tensor(
        [[[0.0, 0.5, 0.5, 0.5], [0.0, 0.0, 2.5, 2.5]]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[[True, False, False, False], [True, True, False, False]]]
    )
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    # Frame 0 has three pairs with loss 1; frame 1 has two non-transition
    # pairs with loss 3. Frame balancing gives (1 + 3) / 2, not 9 / 5.
    torch.testing.assert_close(loss, torch.tensor(2.0, dtype=torch.float64))


@pytest.mark.parametrize("target_value", [False, True], ids=["no-active", "all-active"])
def test_presence_pairwise_no_positive_or_no_negative_is_differentiable_zero(
    target_value: bool,
) -> None:
    logits = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.full((1, 2, 3), target_value, dtype=torch.bool)
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_presence_pairwise_excludes_transition_only_pair_and_nan() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0], [float("nan"), 0.0]]],
        requires_grad=True,
    )
    target = torch.tensor([[[False, False], [True, False]]])
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_presence_pairwise_active_and_inactive_gradients_have_ranking_direction() -> None:
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    target = torch.tensor([[[True, False]]])
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0, 0] < 0.0
    assert logits.grad[0, 0, 1] > 0.0


def test_presence_pairwise_uses_matched_and_unmatched_negatives() -> None:
    logits = torch.zeros(1, 5, 3, requires_grad=True)
    target_presence = torch.tensor(
        [
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, True],
            ]
        ]
    )

    losses, assignments = _compute(
        _criterion(
            presence_pairwise_weight=1.0,
            transition_radius=0,
        ),
        *_cardinality_fixture(logits, target_presence),
    )
    losses["presence_pairwise"].backward()

    target_to_query = {
        target_index: query_index
        for query_index, target_index in zip(
            assignments[0][0].tolist(),
            assignments[0][1].tolist(),
            strict=True,
        )
    }
    positive_query = target_to_query[0]
    matched_negative_query = target_to_query[1]
    unmatched_query = ({0, 1, 2} - set(target_to_query.values())).pop()
    assert logits.grad is not None
    assert logits.grad[0, 0, positive_query] < 0.0
    assert logits.grad[0, 0, matched_negative_query] > 0.0
    assert logits.grad[0, 0, unmatched_query] > 0.0


def test_presence_pairwise_excludes_padded_nan_before_computation() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0], [float("nan"), float("nan")]]],
        requires_grad=True,
    )
    target = torch.tensor([[[True, False], [True, False]]])
    valid = torch.tensor([[[True, True], [False, False]]])

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]))


def test_presence_pairwise_all_pairless_nan_is_differentiable_zero() -> None:
    logits = torch.full((1, 2, 2), float("nan"), requires_grad=True)
    target = torch.zeros(1, 2, 2, dtype=torch.bool)
    valid = torch.zeros_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_presence_pairwise_does_not_hide_selected_nan() -> None:
    logits = torch.tensor([[[float("nan"), 0.0]]], requires_grad=True)
    target = torch.tensor([[[True, False]]])
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    assert torch.isnan(loss)


def test_presence_pairwise_loss_and_gradient_are_query_permutation_invariant() -> None:
    logits = torch.tensor([[[-1.0, 0.2, 1.5, -0.4]]], requires_grad=True)
    target = torch.tensor([[[True, False, True, False]]])
    valid = torch.ones_like(target)
    original = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )
    original.backward()
    assert logits.grad is not None
    original_gradient = logits.grad.detach().clone()

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_logits = logits.detach()[:, :, permutation].clone().requires_grad_(True)
    permuted = _presence_pairwise_loss(
        permuted_logits,
        target[:, :, permutation],
        valid[:, :, permutation],
        margin=0.5,
        transition_radius=0,
    )
    permuted.backward()

    torch.testing.assert_close(original, permuted)
    assert permuted_logits.grad is not None
    torch.testing.assert_close(
        permuted_logits.grad,
        original_gradient[:, :, permutation],
    )


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.bfloat16, [-30.0, 30.0]),
        (torch.float32, [-3e38, 3e38]),
    ],
    ids=["bfloat16", "extreme-float32"],
)
def test_presence_pairwise_is_finite_forward_and_backward(
    dtype: torch.dtype,
    values: list[float],
) -> None:
    logits = torch.tensor([[values]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[[True, False]]])
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.5,
        transition_radius=0,
    )

    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_presence_pairwise_margin_zero_is_unshifted_ranking_hinge() -> None:
    logits = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], requires_grad=True)
    target = torch.tensor([[[True, False], [True, False]]])
    valid = torch.ones_like(target)

    loss = _presence_pairwise_loss(
        logits,
        target,
        valid,
        margin=0.0,
        transition_radius=0,
    )

    # Frame losses are relu(1 - 2)=0 and relu(2 - 1)=1.
    torch.testing.assert_close(loss, torch.tensor(0.5, dtype=torch.float64))


def test_presence_pairwise_missing_and_zero_weights_are_exact_noops() -> None:
    prediction, batch = _fixture()
    missing, _assignments = _compute(_criterion(), prediction, batch)
    zero, _assignments = _compute(
        _criterion(presence_pairwise_weight=0.0),
        prediction,
        batch,
    )

    assert "presence_pairwise" not in missing
    assert "presence_pairwise" not in zero
    assert set(zero) == set(missing)
    torch.testing.assert_close(zero["total"], missing["total"])


def test_presence_pairwise_does_not_change_fixed_assignment() -> None:
    prediction, batch = _fixture()
    _, baseline_assignments = _compute(
        _criterion(match_presence_weight=0.0, presence_pairwise_weight=0.0),
        prediction,
        batch,
    )
    _, auxiliary_assignments = _compute(
        _criterion(match_presence_weight=0.0, presence_pairwise_weight=2.0),
        prediction,
        batch,
    )

    torch.testing.assert_close(
        baseline_assignments[0][0],
        auxiliary_assignments[0][0],
    )
    torch.testing.assert_close(
        baseline_assignments[0][1],
        auxiliary_assignments[0][1],
    )


@pytest.mark.parametrize("field", ["presence_pairwise_weight", "presence_pairwise_margin"])
@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), 10**400, None],
    ids=["negative", "nan", "infinity", "overflowing-integer", "invalid-type"],
)
def test_presence_pairwise_settings_must_be_finite_and_nonnegative(
    field: str,
    value: object,
) -> None:
    config_values = cast(
        "dict[str, object]",
        OmegaConf.to_container(_tracking_config()),
    )
    config_values[field] = value

    with pytest.raises(ValueError, match="finite.*non-negative"):
        PLCSTrackingLoss(SimpleNamespace(**config_values))


def test_presence_pairwise_margin_defaults_to_half() -> None:
    criterion = _criterion(presence_pairwise_weight=1.0)

    assert criterion.presence_pairwise_margin == 0.5


def test_hybrid_loss_is_zero_for_exact_matched_pose_and_ignores_raw_canonical() -> None:
    losses, _assignments = _compute(_hybrid_criterion(), *_hybrid_fixture())

    for name in ("position", "rotation", "angle", "canonical_pose", "reprojection"):
        torch.testing.assert_close(
            losses[name],
            torch.zeros_like(losses[name]),
            atol=1e-6,
            rtol=0.0,
        )


def test_hybrid_loss_is_invariant_to_prediction_query_permutation() -> None:
    prediction, batch = _hybrid_fixture()
    original, _ = _compute(_hybrid_criterion(), prediction, batch)
    permutation = torch.tensor([1, 0])
    permuted_prediction = {
        key: value.detach()[:, :, permutation].clone()
        for key, value in prediction.items()
    }

    permuted, _ = _compute(_hybrid_criterion(), permuted_prediction, batch)

    for name in original:
        torch.testing.assert_close(original[name], permuted[name])


def test_reprojection_excludes_inactive_padded_and_invisible_targets() -> None:
    prediction, batch = _hybrid_fixture()
    batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    # Invisible joint in a valid view/frame.
    batch["human_vis_target"][0, 0, 0, 0, 0] = False
    batch["human_kp_target"][0, 0, 0, 0, 0] += 100.0
    # One padded view does not supervise either target at this frame.
    batch["padding_mask"][0, 1, 1] = True
    batch["human_kp_target"][0, 1, 1] += 100.0
    # Target slot 1 is inactive at frame 2 even if malformed visibility says true.
    batch["human_vis_target"][0, :, 2, 1] = True
    batch["human_kp_target"][0, :, 2, 1] += 100.0
    # A completely padded frame is excluded from both canonical and reprojection.
    batch["padding_mask"][0, :, 3] = True
    batch["human_kp_target"][0, :, 3] += 100.0
    batch["target_human_kp_3d"][0, 3] += 100.0

    losses, _assignments = _compute(_hybrid_criterion(), prediction, batch)

    torch.testing.assert_close(
        losses["canonical_pose"],
        torch.zeros_like(losses["canonical_pose"]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        losses["reprojection"],
        torch.zeros_like(losses["reprojection"]),
        atol=1e-6,
        rtol=0.0,
    )


def test_reprojection_keeps_visible_predictions_behind_camera_supervised() -> None:
    exact, batch = _hybrid_fixture()
    behind_camera_pose = exact["canonical_pose"].detach().clone()
    behind_camera_pose[..., 2] = -30.0
    prediction = dict(exact)
    prediction["canonical_pose"] = behind_camera_pose.requires_grad_()

    losses, _assignments = _compute(_hybrid_criterion(), prediction, batch)

    assert losses["reprojection"] > 0.0
    losses["reprojection"].backward()
    gradient = prediction["canonical_pose"].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_reprojection_rejects_misaligned_clean_visibility_shape() -> None:
    prediction, batch = _hybrid_fixture()
    batch = dict(batch)
    batch["human_vis_target"] = batch["human_vis_target"][..., :-1]

    with pytest.raises(
        ValueError,
        match=r"human_vis_target: expected .*17.* got",
    ):
        _compute(_hybrid_criterion(), prediction, batch)


def test_hybrid_loss_backpropagates_to_position_rotation_and_canonical() -> None:
    exact, batch = _hybrid_fixture()
    prediction = {
        "position": (exact["position"].detach() + 0.01).requires_grad_(),
        "rotation": (
            exact["rotation"].detach() + torch.tensor([0.02, -0.01])
        ).requires_grad_(),
        "presence_logits": exact["presence_logits"].detach().requires_grad_(),
        "canonical_pose": (
            exact["canonical_pose"].detach() + 0.015
        ).requires_grad_(),
    }

    losses, _assignments = _compute(_hybrid_criterion(), prediction, batch)
    losses["total"].backward()

    for name in ("position", "rotation", "canonical_pose"):
        gradient = prediction[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.0


def test_nonzero_unsupported_regularizer_is_rejected() -> None:
    config = OmegaConf.merge(_hybrid_config(), {"joint_angle_weight": 0.1})

    with pytest.raises(ValueError, match=r"nonzero loss\.joint_angle_weight"):
        PLCSTrackingLoss(config)


def test_object_config_enables_declared_pose_loss_outputs() -> None:
    config_values = cast(
        "dict[str, object]",
        OmegaConf.to_container(_hybrid_config()),
    )
    config = SimpleNamespace(**config_values)

    losses, _assignments = _compute(
        PLCSTrackingLoss(config),
        *_hybrid_fixture(),
    )

    assert {"angle", "canonical_pose", "reprojection"}.issubset(losses)
