"""CPU tests for DINO court matching and losses."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from src.tasks.court_alignment.geometry.court import (
    GroundCourtInstance,
    court_keypoints_for_instance,
)
from src.tasks.court_alignment.geometry.oriented_box import build_detr_court_targets
from src.tasks.court_alignment.training.detr_losses import (
    CourtDetrCriterion,
    CourtDetrHungarianMatcher,
)


def _targets(instances: list[GroundCourtInstance]) -> list[dict[str, torch.Tensor]]:
    if instances:
        keypoints = torch.stack(
            [court_keypoints_for_instance(instance) for instance in instances]
        )[None]
        visibility = torch.ones((1, len(instances), 14), dtype=torch.bool)
    else:
        keypoints = torch.empty((1, 0, 14, 2))
        visibility = torch.empty((1, 0, 14), dtype=torch.bool)
    return cast(
        list[dict[str, torch.Tensor]],
        build_detr_court_targets(keypoints, visibility, image_size=(800, 800)),
    )


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1.0e-5, 1.0 - 1.0e-5))


def _raw_court(target_court_boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            _inverse_sigmoid(target_court_boxes[:, 2:3]),
            target_court_boxes[:, 3:],
        ),
        dim=-1,
    )


def _main_outputs(
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, object]:
    target = targets[0]
    if target["boxes"].shape[0] == 0:
        return {
            "pred_logits": torch.full((1, 2, 1), -2.0),
            "pred_boxes": torch.full((1, 2, 4), 0.5),
            "pred_court_boxes": torch.tensor([[[0.0, 1.0, 0.0]]]).repeat(1, 2, 1),
        }
    return {
        "pred_logits": torch.full((1, target["boxes"].shape[0], 1), 3.0),
        "pred_boxes": target["boxes"][None].clone(),
        "pred_court_boxes": _raw_court(target["court_boxes"])[None],
    }


def test_hungarian_matching_uses_geometry_for_swapped_queries() -> None:
    targets = _targets(
        [
            GroundCourtInstance(0, (100.0, 120.0), 0.2, 4.0),
            GroundCourtInstance(1, (270.0, 250.0), 1.2, 5.0),
        ]
    )
    outputs = {
        "pred_logits": torch.full((1, 2, 1), 5.0),
        "pred_boxes": targets[0]["boxes"].flip(0)[None],
        "pred_court_boxes": _raw_court(targets[0]["court_boxes"].flip(0))[None],
    }

    matcher = CourtDetrHungarianMatcher()
    matcher.validate_inputs(outputs, targets)
    assignments = matcher(outputs, targets)

    assert assignments[0][0].tolist() == [0, 1]
    assert assignments[0][1].tolist() == [1, 0]


def test_empty_targets_produce_finite_differentiable_losses() -> None:
    targets = _targets([])
    logits = torch.randn((1, 4, 1), requires_grad=True)
    boxes = torch.full((1, 4, 4), 0.5, requires_grad=True)
    court = torch.randn((1, 4, 3), requires_grad=True)

    losses = CourtDetrCriterion()(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_court_boxes": court},
        targets,
    )

    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["loss_bbox"] == 0.0
    assert losses["loss_giou"] == 0.0
    assert losses["loss_scale"] == 0.0
    assert losses["loss_axis"] == 0.0
    losses["loss_total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()
    assert court.grad is not None and torch.isfinite(court.grad).all()


def test_all_heads_receive_gradients_for_matched_targets() -> None:
    targets = _targets([GroundCourtInstance(0, (190.0, 170.0), 0.7, 4.5)])
    logits = torch.tensor([[[0.5], [-1.0]]], requires_grad=True)
    boxes = torch.tensor(
        [[[0.5, 0.45, 0.4, 0.5], [0.15, 0.15, 0.1, 0.1]]],
        requires_grad=True,
    )
    court = torch.tensor(
        [[[0.0, math.cos(1.4 + math.pi), math.sin(1.4 + math.pi)], [0.0, 1.0, 0.0]]],
        requires_grad=True,
    )

    losses = CourtDetrCriterion()(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_court_boxes": court},
        targets,
    )
    losses["loss_total"].backward()

    for value in (logits.grad, boxes.grad, court.grad):
        assert value is not None
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0


def test_auxiliary_outputs_receive_same_loss_family_and_gradients() -> None:
    targets = _targets([GroundCourtInstance(0, (180.0, 190.0), 0.4, 4.0)])
    target = targets[0]
    main_logits = torch.full((1, 1, 1), 4.0, requires_grad=True)
    main_boxes = target["boxes"][None].detach().clone().requires_grad_()
    main_court = (
        _raw_court(target["court_boxes"])[None].detach().clone().requires_grad_()
    )
    aux_logits = torch.full((1, 1, 1), 2.0, requires_grad=True)
    aux_boxes = torch.full((1, 1, 4), 0.4, requires_grad=True)
    aux_court = torch.tensor([[[0.1, 0.5, 0.5]]], requires_grad=True)
    outputs = {
        "pred_logits": main_logits,
        "pred_boxes": main_boxes,
        "pred_court_boxes": main_court,
        "aux_outputs": [
            {
                "pred_logits": aux_logits,
                "pred_boxes": aux_boxes,
                "pred_court_boxes": aux_court,
            }
        ],
    }

    losses = CourtDetrCriterion()(outputs, targets)

    for name in ("class", "bbox", "giou", "scale", "axis"):
        assert f"loss_{name}_aux_0" in losses
    losses["loss_total"].backward()
    assert aux_logits.grad is not None
    assert aux_boxes.grad is not None
    assert aux_court.grad is not None


def test_intermediate_and_encoder_outputs_receive_detection_losses() -> None:
    targets = _targets([GroundCourtInstance(0, (240.0, 260.0), 0.3, 5.0)])
    outputs: dict[str, object] = _main_outputs(targets)
    intermediate_logits = torch.tensor([[[1.0], [-1.0]]], requires_grad=True)
    intermediate_boxes = torch.tensor(
        [[[0.31, 0.34, 0.2, 0.25], [0.8, 0.8, 0.1, 0.1]]],
        requires_grad=True,
    )
    encoder_logits = torch.tensor([[[0.5], [-0.5]]], requires_grad=True)
    encoder_boxes = torch.tensor(
        [[[0.29, 0.31, 0.18, 0.22], [0.7, 0.7, 0.15, 0.15]]],
        requires_grad=True,
    )
    outputs["interm_outputs"] = {
        "pred_logits": intermediate_logits,
        "pred_boxes": intermediate_boxes,
    }
    outputs["enc_outputs"] = [
        {"pred_logits": encoder_logits, "pred_boxes": encoder_boxes}
    ]

    losses = CourtDetrCriterion()(outputs, targets)

    for suffix in ("_interm", "_enc_0"):
        assert f"loss_class{suffix}" in losses
        assert f"loss_bbox{suffix}" in losses
        assert f"loss_giou{suffix}" in losses
        assert f"loss_scale{suffix}" not in losses
        assert f"loss_axis{suffix}" not in losses
    losses["loss_total"].backward()
    for gradient in (
        intermediate_logits.grad,
        intermediate_boxes.grad,
        encoder_logits.grad,
        encoder_boxes.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_dn_main_and_aux_use_fixed_positive_slots_and_negative_classification() -> None:
    targets = _targets(
        [
            GroundCourtInstance(0, (180.0, 200.0), 0.2, 4.0),
            GroundCourtInstance(1, (520.0, 540.0), 1.0, 5.0),
        ]
    )
    outputs: dict[str, object] = _main_outputs(targets)
    positive_slots = torch.tensor([0, 1, 4, 5])
    negative_slots = torch.tensor([2, 3, 6, 7])
    repeated_targets = targets[0]["boxes"].repeat(2, 1)
    known_boxes = torch.full((8, 4), 0.9)
    known_boxes[positive_slots] = repeated_targets
    known_boxes[positive_slots, 0] += 0.01
    known_boxes = known_boxes[None].requires_grad_()
    known_logits = torch.full((1, 8, 1), -4.0)
    known_logits[:, positive_slots] = 4.0
    known_logits = known_logits.requires_grad_()
    aux_boxes = torch.full((1, 8, 4), 0.4, requires_grad=True)
    aux_logits = torch.zeros((1, 8, 1), requires_grad=True)
    outputs["dn_meta"] = {
        "pad_size": 8,
        "num_dn_group": 2,
        "output_known_lbs_bboxes": {
            "pred_logits": known_logits,
            "pred_boxes": known_boxes,
            "aux_outputs": [{"pred_logits": aux_logits, "pred_boxes": aux_boxes}],
        },
    }

    losses = CourtDetrCriterion()(outputs, targets)

    assert float(losses["loss_bbox_dn"]) == pytest.approx(0.01, abs=1.0e-6)
    for suffix in ("_dn", "_dn_aux_0"):
        assert f"loss_class{suffix}" in losses
        assert f"loss_bbox{suffix}" in losses
        assert f"loss_giou{suffix}" in losses
        assert f"loss_scale{suffix}" not in losses
        assert f"loss_axis{suffix}" not in losses
    losses["loss_total"].backward()
    assert known_logits.grad is not None
    assert torch.count_nonzero(known_logits.grad[:, negative_slots]) > 0
    assert known_boxes.grad is not None
    assert torch.count_nonzero(known_boxes.grad[:, positive_slots]) > 0
    assert torch.count_nonzero(known_boxes.grad[:, negative_slots]) == 0
    assert aux_logits.grad is not None
    assert aux_boxes.grad is not None


def test_dn_with_empty_targets_is_finite_and_supervises_all_queries_as_negative() -> (
    None
):
    targets = _targets([])
    outputs: dict[str, object] = _main_outputs(targets)
    known_logits = torch.zeros((1, 4, 1), requires_grad=True)
    known_boxes = torch.full((1, 4, 4), 0.5, requires_grad=True)
    outputs["dn_meta"] = {
        "pad_size": 4,
        "num_dn_group": 1,
        "output_known_lbs_bboxes": {
            "pred_logits": known_logits,
            "pred_boxes": known_boxes,
        },
    }

    losses = CourtDetrCriterion()(outputs, targets)

    assert torch.isfinite(losses["loss_class_dn"])
    assert losses["loss_class_dn"] > 0.0
    assert losses["loss_bbox_dn"] == 0.0
    assert losses["loss_giou_dn"] == 0.0
    losses["loss_total"].backward()
    assert known_logits.grad is not None and torch.count_nonzero(known_logits.grad) > 0
    assert known_boxes.grad is not None and torch.count_nonzero(known_boxes.grad) == 0


def test_empty_targets_keep_every_supervision_branch_finite_and_differentiable() -> (
    None
):
    targets = _targets([])

    def prediction(*, with_court: bool, queries: int = 3) -> dict[str, torch.Tensor]:
        result = {
            "pred_logits": torch.randn((1, queries, 1), requires_grad=True),
            "pred_boxes": torch.full(
                (1, queries, 4), 0.5, requires_grad=True
            ),
        }
        if with_court:
            result["pred_court_boxes"] = torch.randn(
                (1, queries, 3), requires_grad=True
            )
        return result

    main = prediction(with_court=True)
    auxiliary = prediction(with_court=True)
    intermediate = prediction(with_court=False)
    encoder = prediction(with_court=False)
    known: dict[str, object] = prediction(with_court=False, queries=4)
    known_auxiliary = prediction(with_court=False, queries=4)
    known["aux_outputs"] = [known_auxiliary]
    outputs: dict[str, object] = {
        **main,
        "aux_outputs": [auxiliary],
        "interm_outputs": intermediate,
        "enc_outputs": [encoder],
        "dn_meta": {
            "pad_size": 4,
            "num_dn_group": 1,
            "output_known_lbs_bboxes": known,
        },
    }
    criterion = CourtDetrCriterion()
    criterion.validate_inputs(outputs, targets)

    losses = criterion(outputs, targets)

    expected_suffixes = ("", "_aux_0", "_interm", "_enc_0", "_dn", "_dn_aux_0")
    for suffix in expected_suffixes:
        assert f"loss_class{suffix}" in losses
        assert f"loss_bbox{suffix}" in losses
        assert f"loss_giou{suffix}" in losses
    assert all(torch.isfinite(value) for value in losses.values())

    losses["loss_total"].backward()
    tensors = [
        tensor
        for branch in (
            main,
            auxiliary,
            intermediate,
            encoder,
            known,
            known_auxiliary,
        )
        for tensor in branch.values()
        if isinstance(tensor, torch.Tensor)
    ]
    assert all(tensor.grad is not None for tensor in tensors)
    assert all(torch.isfinite(tensor.grad).all() for tensor in tensors if tensor.grad is not None)


def test_eval_none_and_official_zero_pad_dn_metadata_are_normal() -> None:
    targets = _targets([])
    for dn_meta in (None, {"pad_size": 0, "num_dn_group": 1}):
        outputs: dict[str, object] = _main_outputs(targets)
        outputs["dn_meta"] = dn_meta
        losses = CourtDetrCriterion()(outputs, targets)
        assert all("_dn" not in name for name in losses)
        assert torch.isfinite(losses["loss_total"])


@pytest.mark.parametrize(
    "dn_meta, message",
    [
        ({"pad_size": 4, "num_dn_group": 1}, "output_known"),
        (
            {
                "pad_size": 5,
                "num_dn_group": 2,
                "output_known_lbs_bboxes": {},
            },
            "divisible",
        ),
        (
            {
                "pad_size": 4,
                "num_dn_group": 1,
                "output_known_lbs_bboxes": {
                    "pred_logits": torch.zeros((1, 3, 1)),
                    "pred_boxes": torch.full((1, 3, 4), 0.5),
                },
            },
            "query count",
        ),
    ],
)
def test_malformed_dn_metadata_fails_explicitly(
    dn_meta: dict[str, object], message: str
) -> None:
    targets = _targets([GroundCourtInstance(0, (200.0, 200.0), 0.2, 4.0)])
    outputs: dict[str, object] = _main_outputs(targets)
    outputs["dn_meta"] = dn_meta
    criterion = CourtDetrCriterion()

    with pytest.raises((TypeError, ValueError), match=message):
        criterion.validate_inputs(outputs, targets)
