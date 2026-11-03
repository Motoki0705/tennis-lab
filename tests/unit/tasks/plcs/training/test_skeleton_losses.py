"""Missing-feature contracts for PLCS static and temporal skeleton losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from src.tasks.plcs.training.losses import (
    DEFAULT_LOSS_TERMS,
    PLCSLoss,
    PLCSLossConfig,
    PLCSLossInputs,
    bone_length_loss,
    joint_angle_loss,
)
from tests.support.geometry.skeleton import coco_pose_m

# Recorded before the domain change, using only geometrically valid poses.
BASELINES = {
    "joint_angle": 0.0003059997976398596,
    "torsion_angle": 0.0006813444224289174,
    "torso_twist": 3.259967653948529e-05,
    "bone_length": 9.981159105728281e-05,
    "joint_angle_velocity": 0.0004109714745601445,
    "torsion_angle_velocity": 0.001247279455934956,
    "torso_twist_velocity": 4.889951480922794e-05,
}


def _poses() -> tuple[Tensor, Tensor]:
    target = coco_pose_m().repeat(1, 3, 1, 1)
    pred = target.clone()
    # Float32 offsets retained for the unchanged pre-change baseline.
    pred[0, 0, 7] += torch.tensor([0.12, -0.18, 0.09])
    pred[0, 1, 9] += torch.tensor([-0.21, 0.15, 0.11])
    pred[0, 2, 5] += torch.tensor([0.08, 0.04, -0.16])
    return pred, target


def _inputs(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> PLCSLossInputs:
    frame_shape = pred.shape[:-2]
    position = pred.new_zeros(*frame_shape, 3)
    rotation = pred.new_tensor([1.0, 0.0]).expand(*frame_shape, 2)
    return PLCSLossInputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position,
        target_rotation=rotation,
        frame_mask=mask,
        pred_canonical_pose=pred,
        target_canonical_pose=target,
    )


def _loss_config(**enabled: float) -> PLCSLossConfig:
    weights: dict[str, object] = {name + "_weight": 0.0 for name in DEFAULT_LOSS_TERMS}
    weights.update(
        position_smooth_l1_beta=1.0,
        canonical_pose_smooth_l1_beta=1.0,
        reprojection_smooth_l1_beta=0.01,
        joint_angle_velocity_angle_weights=None,
        torsion_angle_velocity_angle_weights=None,
    )
    weights.update(enabled)
    return PLCSLossConfig.from_dict(weights)


@pytest.mark.parametrize("name", BASELINES)
def test_all_valid_values_match_previous_loss_and_padding_is_excluded(
    name: str,
) -> None:
    pred, target = _poses()
    pred.requires_grad_()
    term = DEFAULT_LOSS_TERMS[name]
    expected = term(_inputs(pred, target))
    assert float(expected) == pytest.approx(BASELINES[name], rel=1e-12, abs=1e-15)
    expected.backward()
    # Nonfinite padding verifies exclusion before API validation/computation.
    mixed_pred = torch.cat(
        [pred.detach(), pred.new_full((1, 1, 17, 3), float("nan"))], dim=1
    ).requires_grad_()
    mixed_target = torch.cat([target, torch.zeros_like(target[:, :1])], dim=1)
    actual = term(
        _inputs(mixed_pred, mixed_target, torch.tensor([[True, True, True, False]]))
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert mixed_pred.grad is not None and pred.grad is not None
    torch.testing.assert_close(mixed_pred.grad[:, :3], pred.grad)
    torch.testing.assert_close(
        mixed_pred.grad[:, 3:], torch.zeros_like(mixed_pred.grad[:, 3:])
    )


@pytest.mark.parametrize("name", BASELINES)
def test_pred_degeneracy_is_an_error_even_when_target_is_missing(name: str) -> None:
    pred, target = _poses()
    pred[:, 1] = 0
    target[:, 1] = 0
    with pytest.raises(ValueError, match="pred.*(invalid|degenerate)"):
        DEFAULT_LOSS_TERMS[name](_inputs(pred, target))


@pytest.mark.parametrize("name", BASELINES)
def test_all_missing_target_features_raise(name: str) -> None:
    pred, target = _poses()
    target.zero_()
    with pytest.raises(ValueError, match="(No valid|no valid|all.invalid)"):
        DEFAULT_LOSS_TERMS[name](_inputs(pred, target))


def test_partial_bone_missingness_uses_frame_means_and_zero_missing_gradient() -> None:
    # Two disjoint bones and two frames. Relative errors: [1/2, missing],
    # then [1/4, 3/4]. Frame means 1/8 and 5/32, hence overall 9/64.
    target = torch.tensor(
        [
            [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [4, 0, 0], [0, 1, 0], [0, 5, 0]],
        ],
        dtype=torch.float64,
    )
    pred = torch.tensor(
        [
            [[0, 0, 0], [3, 0, 0], [0, 1, 0], [0, 6, 0]],
            [[0, 0, 0], [5, 0, 0], [0, 1, 0], [0, 8, 0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = ((0, 1), (2, 3))
    loss = bone_length_loss(pred, target, edges)
    torch.testing.assert_close(loss, pred.new_tensor(9 / 64))
    loss.backward()
    assert pred.grad is not None
    torch.testing.assert_close(pred.grad[0, 2:], torch.zeros_like(pred.grad[0, 2:]))
    # A fully missing frame does not dilute the loss.
    padded_pred = torch.cat([pred.detach(), pred.detach()[:1]]).requires_grad_()
    padded_target = torch.cat([target, torch.zeros_like(target[:1])])
    torch.testing.assert_close(
        bone_length_loss(padded_pred, padded_target, edges), loss
    )


def test_partial_joint_missingness_does_not_dilute_other_features() -> None:
    # Disjoint triples: first angle changes from 90 to 0 degrees, the second
    # is unobserved. Smooth-L1(pi/2) = pi/2 - 1/2.
    target = torch.tensor(
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=torch.float64,
    )
    pred = torch.tensor(
        [[1, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = joint_angle_loss(pred, target, ((0, 1, 2), (3, 4, 5)))
    torch.testing.assert_close(actual, pred.new_tensor(torch.pi / 2 - 0.5))
    actual.backward()
    assert pred.grad is not None
    torch.testing.assert_close(pred.grad[3:], torch.zeros_like(pred.grad[3:]))


@pytest.mark.parametrize(
    "name", ["joint_angle_velocity", "torsion_angle_velocity", "torso_twist_velocity"]
)
def test_velocity_uses_both_endpoints_without_bridging_missing_frame(name: str) -> None:
    pred, target = _poses()
    # First transition remains observed, second has a missing target endpoint.
    target[:, 2] = 0
    term = DEFAULT_LOSS_TERMS[name]
    actual = term(_inputs(pred, target))
    expected = term(_inputs(pred[:, :2], target[:, :2]))
    torch.testing.assert_close(actual, expected)
    # Valid observations separated by a gap never form a transition.
    target[:, 1] = 0
    target[:, 2] = coco_pose_m()
    with pytest.raises(ValueError, match="(No valid|no valid|all.invalid)"):
        term(_inputs(pred, target))


@pytest.mark.parametrize("name", ["joint_angle_velocity", "torsion_angle_velocity"])
def test_velocity_feature_mask_and_denominator_are_per_angle(name: str) -> None:
    pred, target = _poses()
    target[0, 2, 9] = target[0, 2, 7]  # Last left forearm is unobserved.
    # Left elbow/left-arm motion in the last transition is excluded. Change
    # only the last predicted wrist: other observed features cannot change.
    term = DEFAULT_LOSS_TERMS[name]
    before = term(_inputs(pred, target))
    pred[0, 2, 9] += pred.new_tensor([1.0, 2.0, 3.0])
    after = term(_inputs(pred, target))
    torch.testing.assert_close(after, before)


@pytest.mark.parametrize("name", BASELINES)
def test_nonfinite_nonpadding_prediction_and_target_are_rejected(name: str) -> None:
    pred, target = _poses()
    pred[0, 1, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        DEFAULT_LOSS_TERMS[name](_inputs(pred, target))
    pred, target = _poses()
    target[0, 1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DEFAULT_LOSS_TERMS[name](_inputs(pred, target))


def test_disabled_terms_are_not_evaluated_and_not_reported_as_raw_measurements() -> (
    None
):
    config = _loss_config(position_weight=1.0)
    disabled_calls: list[str] = []

    def disabled(_inputs: PLCSLossInputs) -> Tensor:
        disabled_calls.append("called")
        raise AssertionError("Disabled term must not execute")

    terms: dict[str, Callable[[PLCSLossInputs], Tensor]] = {
        name: disabled for name in DEFAULT_LOSS_TERMS if name != "position"
    }
    terms["position"] = DEFAULT_LOSS_TERMS["position"]
    loss_fn = PLCSLoss(config, loss_terms=terms)
    pred, target = _poses()
    # Finite degenerate prediction remains usable for position-only training.
    pred.zero_()
    inputs = _inputs(pred, target)
    prepared = loss_fn.prepare_inputs(
        inputs.pred_position,
        inputs.pred_rotation,
        inputs.target_position,
        inputs.target_rotation,
        pred_canonical_pose=pred,
        target_human_kp_3d=target,
        padding_mask=None,
    )
    losses = loss_fn(prepared)
    assert set(losses) == {"position", "total"}
    assert not disabled_calls
    # Enabling the geometric term must immediately reject the same pose.
    loss_fn = PLCSLoss(replace(config, bone_length_weight=1.0))
    with pytest.raises(ValueError, match="pred.*(invalid|degenerate)"):
        loss_fn.prepare_inputs(
            inputs.pred_position,
            inputs.pred_rotation,
            inputs.target_position,
            inputs.target_rotation,
            pred_canonical_pose=pred,
            target_human_kp_3d=target,
            padding_mask=None,
        )


def test_combined_loss_excludes_multiview_padding_before_canonicalization() -> None:
    pred, target = _poses()
    pred = torch.cat([pred, torch.full_like(pred[:, :1], torch.nan)], dim=1)
    target = torch.cat([target, torch.full_like(target[:, :1], torch.nan)], dim=1)
    pred.requires_grad_()
    inputs = _inputs(pred, target)
    target_position = inputs.target_position.clone()
    target_rotation = inputs.target_rotation.clone()
    target_position[:, -1] = torch.nan
    target_rotation[:, -1] = torch.nan
    padding = torch.tensor([[[False, False, False, True]]]).expand(1, 2, 4)
    loss_fn = PLCSLoss(_loss_config(**{name + "_weight": 1.0 for name in BASELINES}))
    losses = loss_fn(
        loss_fn.prepare_inputs(
            inputs.pred_position,
            inputs.pred_rotation,
            target_position,
            target_rotation,
            pred_canonical_pose=pred,
            target_human_kp_3d=target,
            padding_mask=padding,
        )
    )
    assert set(losses) == {*BASELINES, "total"}
    for name, baseline in BASELINES.items():
        assert float(losses[name].detach()) == pytest.approx(baseline, abs=1e-15)
    losses["total"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad[:, -1], torch.zeros_like(pred.grad[:, -1]))


@pytest.mark.parametrize("name", BASELINES)
def test_missing_target_frame_preserves_observed_loss_and_gradient(name: str) -> None:
    pred, target = _poses()
    target[:, 2] = 0
    single_pred = pred[:, :2].clone().requires_grad_()
    expected = DEFAULT_LOSS_TERMS[name](_inputs(single_pred, target[:, :2]))
    expected.backward()
    pred.requires_grad_()
    actual = DEFAULT_LOSS_TERMS[name](_inputs(pred, target))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert pred.grad is not None and single_pred.grad is not None
    torch.testing.assert_close(pred.grad[:, :2], single_pred.grad)
    torch.testing.assert_close(pred.grad[:, 2], torch.zeros_like(pred.grad[:, 2]))


@pytest.mark.parametrize("weighted", [False, True])
def test_velocity_averages_only_observed_features_within_each_transition(
    weighted: bool,
) -> None:
    import math
    from functools import partial

    from src.tasks.plcs.training.losses import _angle_velocity_loss_term
    from src.utils.geometry.skeleton import compute_joint_angles

    # Two disjoint joint angles. GT remains pi/2; the second angle is missing
    # only at the final endpoint. Predicted velocities are [.2,.6], [.4,.2].
    def make_pose(first: float, second: float) -> list[list[float]]:
        points: list[list[float]] = []
        for angle in (first, second):
            points += [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [math.cos(angle), math.sin(angle), 0.0],
            ]
        return points

    pred = torch.tensor(
        [
            [
                make_pose(math.pi / 2 + a, math.pi / 2 + b)
                for a, b in ((0.0, 0.0), (0.2, 0.6), (0.6, 0.8))
            ]
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[make_pose(math.pi / 2, math.pi / 2)] * 3], dtype=torch.float64
    )
    target[0, 2, 3:] = 0
    weights = pred.new_tensor([2.0, 0.5]) if weighted else None
    result = _angle_velocity_loss_term(
        _inputs(pred, target),
        partial(compute_joint_angles, triplets=((0, 1, 2), (3, 4, 5))),
        wrap=False,
        angle_weights=weights,
    )
    # Unweighted means: (.02+.18)/2=.1 then .08 -> .09 overall.
    # Weighted means: (.04+.09)/2=.065 then .16 -> .1125 overall.
    torch.testing.assert_close(result, pred.new_tensor(0.1125 if weighted else 0.09))
