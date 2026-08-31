"""PLCS physical-detection corruption boundary tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.data.augmentation import (
    PLCSObservationAugmentation,
    PLCSObservationTrackingResult,
)
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/plcs/configs/data/_augmentation.yaml"
)


def _augmentation_config(
    *,
    enabled: bool,
    gaussian_noise: bool = False,
    visibility_dropout: bool = False,
    false_positive: bool = False,
) -> DictConfig:
    config = OmegaConf.load(_AUGMENTATION_CONFIG).augmentation
    if not isinstance(config, DictConfig):
        raise AssertionError("PLCS augmentation config must be a mapping.")
    config.enabled = enabled
    for block_name in (
        "uv_scale",
        "gaussian_noise",
        "visibility_dropout",
        "temporal_jitter",
        "burst_dropout",
        "false_positive",
        "edge_degradation",
        "speed_conditioned",
    ):
        config[block_name].enabled = False
    config.gaussian_noise.enabled = gaussian_noise
    config.gaussian_noise.prob = 1.0
    config.gaussian_noise.human_std = 0.001
    config.gaussian_noise.court_std = 0.001
    config.visibility_dropout.enabled = visibility_dropout
    config.visibility_dropout.prob = 1.0
    config.visibility_dropout.human_drop_prob = 1.0
    config.visibility_dropout.court_drop_prob = 0.0
    config.false_positive.enabled = false_positive
    config.false_positive.prob = 1.0
    config.false_positive.human_prob_absent = 1.0
    config.false_positive.human_prob_after_dropout = 1.0
    config.false_positive.human_after_dropout_window = 0
    config.false_positive.court_prob_absent = 0.0
    config.false_positive.court_prob_after_dropout = 0.0
    return config


def _sample(
    *,
    visible: bool = True,
    num_detections: int = 2,
    views: int = 1,
    frames: int = 2,
) -> dict[str, torch.Tensor]:
    human_kp = torch.zeros(views, frames, num_detections, 17, 2)
    for detection_index in range(num_detections):
        human_kp[:, :, detection_index] = (detection_index + 1) / (
            num_detections + 1
        )
    human_vis = torch.full(
        (views, frames, num_detections, 17), visible, dtype=torch.bool
    )
    if not visible:
        human_kp.zero_()
    provenance = (
        torch.arange(num_detections, dtype=torch.long)
        .add(7)
        .view(1, 1, num_detections)
        .expand(views, frames, -1)
        .clone()
    )
    if not visible:
        provenance.fill_(-1)
    return {
        "human_kp": human_kp,
        "human_vis": human_vis,
        "detection_gt_index": provenance,
        "court_kp": torch.rand(views, frames, 14, 2),
        "court_vis": torch.ones(views, frames, 14, dtype=torch.bool),
        "target_position": torch.rand(frames, num_detections, 3),
        "clean_human_kp": human_kp.clone(),
    }


def _observation_sample(visibility: torch.Tensor) -> dict[str, torch.Tensor]:
    keypoints = torch.linspace(
        0.1,
        0.9,
        visibility.numel() * 2,
        dtype=torch.float32,
    ).reshape(*visibility.shape, 2)
    keypoints = torch.where(visibility.unsqueeze(-1), keypoints, 0.0)
    views, frames = visibility.shape[:2]
    return {
        "human_kp": keypoints,
        "human_vis": visibility,
        "court_kp": torch.linspace(
            0.0,
            1.0,
            views * frames * 14 * 2,
        ).reshape(views, frames, 14, 2),
        "court_vis": torch.ones(views, frames, 14, dtype=torch.bool),
    }


def _assert_augmented_samples_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key])


@pytest.mark.parametrize(
    ("num_slots", "error_type"),
    [(True, TypeError), (0, ValueError)],
)
def test_tracking_augmentation_rejects_invalid_query_capacity(
    num_slots: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="num_slots"):
        PLCSTrackingDetectionAugmentation(
            _augmentation_config(enabled=False),
            num_slots=num_slots,
        )


def test_ordinary_and_tracking_only_augmentation_are_fixed_seed_equivalent() -> None:
    sample = _observation_sample(
        torch.tensor(
            [
                [
                    [True, False, True, False],
                    [False, True, False, True],
                ]
            ],
            dtype=torch.bool,
        )
    )
    augmentation = PLCSObservationAugmentation(
        _augmentation_config(
            enabled=True,
            gaussian_noise=True,
            false_positive=True,
        )
    )

    torch.manual_seed(832)
    ordinary = augmentation.forward(sample)
    ordinary_rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(832)
    tracking = augmentation.forward_with_tracking_provenance(sample)
    tracking_rng_state = torch.random.get_rng_state().clone()

    _assert_augmented_samples_equal(ordinary, tracking.sample)
    torch.testing.assert_close(ordinary_rng_state, tracking_rng_state)
    assert ordinary is not sample
    assert tracking.sample is not sample


def test_tracking_only_capture_handles_disabled_no_activation_dropout_and_active_fp() -> None:
    visibility = torch.tensor(
        [[[True, False, True], [False, True, False]]],
        dtype=torch.bool,
    )
    sample = _observation_sample(visibility)
    disabled_augmentation = PLCSObservationAugmentation(
        _augmentation_config(enabled=False)
    )
    disabled = disabled_augmentation.forward_with_tracking_provenance(sample)

    no_activation_config = _augmentation_config(
        enabled=True,
        false_positive=True,
    )
    no_activation_config.false_positive.prob = 0.0
    no_activation = PLCSObservationAugmentation(
        no_activation_config
    ).forward_with_tracking_provenance(sample)

    dropped_then_replaced = PLCSObservationAugmentation(
        _augmentation_config(
            enabled=True,
            visibility_dropout=True,
            false_positive=True,
        )
    ).forward_with_tracking_provenance(sample)

    absent_sample = _observation_sample(torch.zeros_like(visibility))
    torch.manual_seed(833)
    active = PLCSObservationAugmentation(
        _augmentation_config(enabled=True, false_positive=True)
    ).forward_with_tracking_provenance(absent_sample)

    assert disabled.sample is sample
    assert disabled_augmentation.forward(sample) is sample
    torch.testing.assert_close(
        disabled.human_visibility_before_false_positive,
        visibility,
    )
    assert no_activation.sample is not sample
    torch.testing.assert_close(
        no_activation.human_visibility_before_false_positive,
        visibility,
    )
    torch.testing.assert_close(no_activation.sample["human_vis"], visibility)
    assert not (
        dropped_then_replaced.human_visibility_before_false_positive.any()
    )
    assert dropped_then_replaced.sample["human_vis"].all()
    assert not active.human_visibility_before_false_positive.any()
    assert active.sample["human_vis"].all()


def test_tracking_provenance_is_local_for_repeated_interleaved_and_reentrant_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_visibility = torch.tensor(
        [[[True, False, False], [False, False, False]]]
    )
    second_visibility = torch.tensor(
        [[[False, True, False], [True, True, False]]]
    )
    first_sample = _observation_sample(first_visibility)
    second_sample = _observation_sample(second_visibility)
    first_augmentation = PLCSObservationAugmentation(
        _augmentation_config(enabled=True)
    )
    second_augmentation = PLCSObservationAugmentation(
        _augmentation_config(enabled=True)
    )

    first_result = first_augmentation.forward_with_tracking_provenance(first_sample)
    interleaved_result = second_augmentation.forward_with_tracking_provenance(
        second_sample
    )
    repeated_result = first_augmentation.forward_with_tracking_provenance(second_sample)

    torch.testing.assert_close(
        first_result.human_visibility_before_false_positive,
        first_visibility,
    )
    torch.testing.assert_close(
        interleaved_result.human_visibility_before_false_positive,
        second_visibility,
    )
    torch.testing.assert_close(
        repeated_result.human_visibility_before_false_positive,
        second_visibility,
    )

    original_apply = first_augmentation._apply_false_positive
    nested_visibility: list[torch.Tensor] = []
    reentering = False

    def reentrant_apply(
        keypoints: torch.Tensor,
        visibility: torch.Tensor,
        *,
        entity: Literal["human", "court"],
        dropped_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal reentering
        if entity == "human" and not reentering:
            reentering = True
            nested_result = first_augmentation.forward_with_tracking_provenance(
                second_sample
            )
            nested_visibility.append(
                nested_result.human_visibility_before_false_positive
            )
            reentering = False
        return original_apply(
            keypoints,
            visibility,
            entity=entity,
            dropped_mask=dropped_mask,
        )

    monkeypatch.setattr(first_augmentation, "_apply_false_positive", reentrant_apply)
    outer_result = first_augmentation.forward_with_tracking_provenance(first_sample)

    torch.testing.assert_close(
        outer_result.human_visibility_before_false_positive,
        first_visibility,
    )
    torch.testing.assert_close(nested_visibility[0], second_visibility)
    assert not hasattr(
        first_augmentation,
        "human_visibility_before_false_positive",
    )
    assert not hasattr(
        second_augmentation,
        "human_visibility_before_false_positive",
    )


@pytest.mark.parametrize(
    ("malformation", "error_type", "message"),
    [
        ("human_rank", ValueError, "human_kp must have shape"),
        ("human_uv_axis", ValueError, "human_kp must have shape"),
        ("joint_count", ValueError, "exactly 17 COCO keypoints"),
        ("visibility_shape", ValueError, "human_vis must match human_kp"),
        ("visibility_dtype", TypeError, "human_vis must have dtype torch.bool"),
        (
            "provenance_shape",
            ValueError,
            "detection_gt_index must match the physical detection carrier axis",
        ),
        (
            "provenance_dtype",
            TypeError,
            "detection_gt_index must have dtype torch.long",
        ),
    ],
)
def test_callable_preserves_malformed_physical_input_failures(
    malformation: str,
    error_type: type[Exception],
    message: str,
) -> None:
    sample = _sample()
    if malformation == "human_rank":
        sample["human_kp"] = sample["human_kp"][0]
    elif malformation == "human_uv_axis":
        sample["human_kp"] = sample["human_kp"][..., :1]
    elif malformation == "joint_count":
        sample["human_kp"] = sample["human_kp"][..., :16, :]
    elif malformation == "visibility_shape":
        sample["human_vis"] = sample["human_vis"][..., :16]
    elif malformation == "visibility_dtype":
        sample["human_vis"] = sample["human_vis"].float()
    elif malformation == "provenance_shape":
        sample["detection_gt_index"] = sample["detection_gt_index"][..., :1]
    elif malformation == "provenance_dtype":
        sample["detection_gt_index"] = sample["detection_gt_index"].to(torch.int32)
    else:
        raise AssertionError(f"Unknown malformed-input case: {malformation}")

    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=False),
        num_slots=2,
    )
    with pytest.raises(error_type, match=message):
        augmentation(sample)


def test_tracking_augmentation_preserves_physical_width_and_non_inputs() -> None:
    sample = _sample()
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=False),
        num_slots=1,
    )

    result = augmentation(sample)

    assert result["human_kp"].shape == (1, 2, 2, 17, 2)
    torch.testing.assert_close(result["target_position"], sample["target_position"])
    torch.testing.assert_close(result["clean_human_kp"], sample["clean_human_kp"])
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])
    torch.testing.assert_close(
        result["detection_gt_index"], sample["detection_gt_index"]
    )
    torch.testing.assert_close(result["human_kp"], sample["human_kp"])
    torch.testing.assert_close(result["human_vis"], sample["human_vis"])
    assert result["human_vis"].any(-1).sum(-1).eq(2).all()


def test_tracking_noise_changes_physical_keypoints_without_carrier_reordering() -> None:
    torch.manual_seed(24)
    sample = _sample()
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, gaussian_noise=True),
        num_slots=2,
    )

    result = augmentation(sample)

    assert not torch.equal(result["human_kp"], sample["human_kp"])
    assert bool(((result["human_kp"] - sample["human_kp"]).abs() < 0.01).all())
    torch.testing.assert_close(
        result["detection_gt_index"], sample["detection_gt_index"]
    )
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])


def test_false_positive_only_carriers_are_capped_at_q_with_negative_provenance() -> None:
    torch.manual_seed(25)
    sample = _sample(visible=False)
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, false_positive=True),
        num_slots=1,
    )

    result = augmentation(sample)

    carrier_visible = result["human_vis"].any(-1)
    assert carrier_visible.sum(-1).eq(1).all()
    assert result["human_kp"][~result["human_vis"]].eq(0).all()
    assert result["detection_gt_index"].eq(-1).all()
    torch.testing.assert_close(result["court_kp"], sample["court_kp"])
    torch.testing.assert_close(result["court_vis"], sample["court_vis"])


def test_false_positive_after_complete_dropout_does_not_restore_gt_provenance() -> None:
    torch.manual_seed(26)
    sample = _sample()
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(
            enabled=True,
            visibility_dropout=True,
            false_positive=True,
        ),
        num_slots=2,
    )

    result = augmentation(sample)

    assert bool(result["human_vis"].all())
    assert result["detection_gt_index"].eq(-1).all()


def _run_controlled_false_positive_capacity(
    sample: dict[str, torch.Tensor],
    augmented_keypoints: torch.Tensor,
    augmented_visibility: torch.Tensor,
    visibility_before_false_positive: torch.Tensor,
    *,
    num_slots: int,
    strided_output: bool = False,
) -> dict[str, torch.Tensor]:
    augmentation = PLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=False),
        num_slots=num_slots,
    )
    views, frames, detections, joints = augmented_visibility.shape

    def controlled_forward_with_tracking_provenance(
        adapted: dict[str, torch.Tensor],
    ) -> PLCSObservationTrackingResult:
        flattened_keypoints = augmented_keypoints.reshape(
            views, frames, detections * joints, 2
        )
        flattened_visibility = augmented_visibility.reshape(
            views, frames, detections * joints
        )
        flattened_genuine = visibility_before_false_positive.reshape(
            views, frames, detections * joints
        )
        if strided_output:
            flattened_keypoints = (
                augmented_keypoints.permute(0, 2, 3, 1, 4)
                .contiguous()
                .reshape(views, detections * joints, frames, 2)
                .permute(0, 2, 1, 3)
            )
            flattened_visibility = (
                augmented_visibility.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(views, detections * joints, frames)
                .permute(0, 2, 1)
            )
            flattened_genuine = (
                visibility_before_false_positive.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(views, detections * joints, frames)
                .permute(0, 2, 1)
            )
        return PLCSObservationTrackingResult(
            sample={
                **adapted,
                "human_kp": flattened_keypoints,
                "human_vis": flattened_visibility,
            },
            human_visibility_before_false_positive=flattened_genuine,
        )

    observation = cast("Any", augmentation.observation)
    observation.forward_with_tracking_provenance = (
        controlled_forward_with_tracking_provenance
    )
    return augmentation(sample)


def test_capacity_is_permutation_invariant_and_preserves_mixed_partial_pose() -> None:
    sample = _sample(visible=False, num_detections=4)
    augmented_keypoints = torch.zeros((1, 2, 4, 17, 2))
    augmented_visibility = torch.zeros((1, 2, 4, 17), dtype=torch.bool)
    visibility_before_false_positive = torch.zeros_like(augmented_visibility)

    augmented_keypoints[:, :, 0] = 0.90
    augmented_visibility[:, :, 0] = True
    visibility_before_false_positive[:, :, 0, 0] = True
    sample["human_kp"][:, :, 0, 0] = 0.90
    sample["human_vis"][:, :, 0, 0] = True
    sample["detection_gt_index"][:, :, 0] = 7

    augmented_keypoints[:, :, 1, :4] = 0.40
    augmented_visibility[:, :, 1, :4] = True
    augmented_keypoints[:, :, 2, 4:8] = 0.20
    augmented_visibility[:, :, 2, 4:8] = True
    augmented_keypoints[:, :, 3] = 0.60
    augmented_visibility[:, :, 3] = True

    result = _run_controlled_false_positive_capacity(
        sample,
        augmented_keypoints,
        augmented_visibility,
        visibility_before_false_positive,
        num_slots=2,
    )

    permutation = torch.tensor([2, 0, 3, 1])
    inverse_permutation = torch.argsort(permutation)
    permuted_sample = {
        key: value.clone() for key, value in sample.items()
    }
    permuted_sample["human_kp"] = sample["human_kp"][:, :, permutation]
    permuted_sample["human_vis"] = sample["human_vis"][:, :, permutation]
    permuted_sample["detection_gt_index"] = sample["detection_gt_index"][
        :, :, permutation
    ]
    permuted_result = _run_controlled_false_positive_capacity(
        permuted_sample,
        augmented_keypoints[:, :, permutation],
        augmented_visibility[:, :, permutation],
        visibility_before_false_positive[:, :, permutation],
        num_slots=2,
    )

    torch.testing.assert_close(
        result["human_kp"],
        permuted_result["human_kp"][:, :, inverse_permutation],
    )
    torch.testing.assert_close(
        result["human_vis"],
        permuted_result["human_vis"][:, :, inverse_permutation],
    )
    torch.testing.assert_close(
        result["detection_gt_index"],
        permuted_result["detection_gt_index"][:, :, inverse_permutation],
    )
    assert result["human_vis"].any(-1).sum(-1).eq(2).all()
    assert result["human_vis"][:, :, 0].all()
    torch.testing.assert_close(result["human_kp"][:, :, 0], augmented_keypoints[:, :, 0])
    assert result["detection_gt_index"][:, :, 0].eq(7).all()
    assert result["detection_gt_index"][:, :, 1:].eq(-1).all()


def test_capacity_limit_handles_non_contiguous_pose_layout() -> None:
    views, frames, detections, joints = 2, 3, 4, 17
    sample = _sample(
        visible=False,
        num_detections=detections,
        views=views,
        frames=frames,
    )
    augmented_keypoints = torch.zeros((views, frames, detections, joints, 2))
    augmented_visibility = torch.ones(
        (views, frames, detections, joints), dtype=torch.bool
    )
    visibility_before_false_positive = torch.zeros_like(augmented_visibility)
    for detection_index, coordinate in enumerate((0.90, 0.40, 0.20, 0.60)):
        augmented_keypoints[:, :, detection_index] = coordinate
    visibility_before_false_positive[:, :, 0, 0] = True
    sample["human_kp"][:, :, 0, 0] = 0.90
    sample["human_vis"][:, :, 0, 0] = True
    sample["detection_gt_index"][:, :, 0] = 7

    strided_values = (
        augmented_keypoints.permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(views, detections * joints, frames, 2)
        .permute(0, 2, 1, 3)
        .reshape(views, frames, detections, joints, 2)
    )
    strided_visibility = (
        augmented_visibility.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(views, detections * joints, frames)
        .permute(0, 2, 1)
        .reshape(views, frames, detections, joints)
    )
    values_clone = strided_values.clone()
    visibility_clone = strided_visibility.clone()
    assert not values_clone.is_contiguous()
    assert not visibility_clone.is_contiguous()
    assert (
        values_clone.reshape(-1, detections, joints, 2)
        .untyped_storage()
        .data_ptr()
        != values_clone.untyped_storage().data_ptr()
    )
    assert (
        visibility_clone.reshape(-1, detections, joints)
        .untyped_storage()
        .data_ptr()
        != visibility_clone.untyped_storage().data_ptr()
    )

    result = _run_controlled_false_positive_capacity(
        sample,
        augmented_keypoints,
        augmented_visibility,
        visibility_before_false_positive,
        num_slots=2,
        strided_output=True,
    )

    carrier_visible = result["human_vis"].any(-1)
    assert carrier_visible[..., [0, 2]].all()
    assert not carrier_visible[..., [1, 3]].any()
    assert result["human_kp"][..., [1, 3], :, :].eq(0).all()
    assert result["detection_gt_index"][..., 0].eq(7).all()
    assert result["detection_gt_index"][..., 1:].eq(-1).all()
