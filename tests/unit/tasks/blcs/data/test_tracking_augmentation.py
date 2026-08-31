from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.blcs.data.augmentation import (
    BLCSBallObservationAugmentation,
    BLCSBallObservationTrackingResult,
)
from src.tasks.blcs.data.observation_candidates import (
    PhysicalObservationCandidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingDetectionAugmentation,
)
from src.tasks.blcs.data.types import BLCSMultiViewSample

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/blcs/configs/data/_augmentation.yaml"
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
        raise AssertionError("BLCS augmentation config must be a mapping.")
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
    config.gaussian_noise.ball_std = 0.001
    config.gaussian_noise.court_std = 0.001
    config.visibility_dropout.enabled = visibility_dropout
    config.visibility_dropout.prob = 1.0
    config.visibility_dropout.drop_prob = 1.0
    config.false_positive.enabled = false_positive
    config.false_positive.prob = 1.0
    config.false_positive.prob_absent = 1.0
    config.false_positive.prob_after_dropout = 1.0
    config.false_positive.after_dropout_window = 1
    return config


def _detections(*, visible: bool = True) -> PhysicalObservationCandidates:
    uv = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
    vis = torch.full((1, 2, 2), visible, dtype=torch.bool)
    ids = torch.arange(2).view(1, 1, 2).expand_as(vis)
    return PhysicalObservationCandidates(
        uv=torch.where(vis.unsqueeze(-1), uv, torch.zeros_like(uv)),
        vis=vis,
        gt_index=torch.where(vis, ids, -1),
    )


def _apply(
    augmentation: BLCSTrackingDetectionAugmentation,
    detections: PhysicalObservationCandidates,
) -> PhysicalObservationCandidates:
    views, frames = detections.vis.shape[:2]
    return augmentation(
        detections,
        court_kp=torch.rand(views, frames, 14, 2),
        court_vis=torch.ones(views, frames, 14, dtype=torch.bool),
    )


def _single_frame_detections(
    x_coordinates: list[float],
    *,
    visible: bool,
) -> PhysicalObservationCandidates:
    uv = torch.tensor([[[[x, 0.5] for x in x_coordinates]]], dtype=torch.float32)
    vis = torch.full((1, 1, len(x_coordinates)), visible, dtype=torch.bool)
    ids = torch.arange(len(x_coordinates)).view(1, 1, -1)
    return PhysicalObservationCandidates(
        uv=torch.where(vis.unsqueeze(-1), uv, torch.zeros_like(uv)),
        vis=vis,
        gt_index=torch.where(vis, ids, -1),
    )


def _observation_sample(visibility: torch.Tensor) -> BLCSMultiViewSample:
    ball_uv = torch.linspace(
        0.1,
        0.9,
        visibility.numel() * 2,
        dtype=torch.float32,
    ).reshape(*visibility.shape, 2)
    ball_uv = torch.where(visibility.unsqueeze(-1), ball_uv, 0.0)
    views, frames = visibility.shape
    return cast(
        "BLCSMultiViewSample",
        {
            "ball_uv": ball_uv,
            "ball_vis": visibility,
            "court_kp": torch.linspace(
                0.0,
                1.0,
                views * frames * 14 * 2,
            ).reshape(views, frames, 14, 2),
            "court_vis": torch.ones(views, frames, 14, dtype=torch.bool),
        },
    )


def _assert_augmented_samples_equal(
    left: BLCSMultiViewSample,
    right: BLCSMultiViewSample,
) -> None:
    assert left.keys() == right.keys()
    torch.testing.assert_close(left["ball_uv"], right["ball_uv"])
    torch.testing.assert_close(left["ball_vis"], right["ball_vis"])
    torch.testing.assert_close(left["court_kp"], right["court_kp"])
    torch.testing.assert_close(left["court_vis"], right["court_vis"])
    if "ball_uv_target" in left and "ball_uv_target" in right:
        torch.testing.assert_close(left["ball_uv_target"], right["ball_uv_target"])
    if "ball_vis_target" in left and "ball_vis_target" in right:
        torch.testing.assert_close(left["ball_vis_target"], right["ball_vis_target"])


def test_num_slots_is_required_and_must_be_a_positive_integer() -> None:
    config = _augmentation_config(enabled=False)
    parameter = signature(BLCSTrackingDetectionAugmentation).parameters["num_slots"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is Parameter.empty

    for invalid_num_slots in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError), match="num_slots"):
            BLCSTrackingDetectionAugmentation(
                config,
                num_slots=invalid_num_slots,  # type: ignore[arg-type]
            )


def test_ordinary_and_tracking_only_augmentation_are_fixed_seed_equivalent() -> None:
    sample = _observation_sample(
        torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        )
    )
    augmentation = BLCSBallObservationAugmentation(
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
    visibility = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    sample = _observation_sample(visibility)
    disabled_augmentation = BLCSBallObservationAugmentation(
        _augmentation_config(enabled=False)
    )
    disabled = disabled_augmentation.forward_with_tracking_provenance(sample)

    no_activation_config = _augmentation_config(
        enabled=True,
        false_positive=True,
    )
    no_activation_config.false_positive.prob = 0.0
    no_activation = BLCSBallObservationAugmentation(
        no_activation_config
    ).forward_with_tracking_provenance(sample)

    dropped_then_replaced = BLCSBallObservationAugmentation(
        _augmentation_config(
            enabled=True,
            visibility_dropout=True,
            false_positive=True,
        )
    ).forward_with_tracking_provenance(sample)

    absent_sample = _observation_sample(torch.zeros_like(visibility))
    torch.manual_seed(833)
    active = BLCSBallObservationAugmentation(
        _augmentation_config(enabled=True, false_positive=True)
    ).forward_with_tracking_provenance(absent_sample)

    assert disabled.sample is sample
    assert disabled_augmentation.forward(sample) is sample
    torch.testing.assert_close(disabled.visibility_before_false_positive, visibility)
    assert no_activation.sample is not sample
    torch.testing.assert_close(
        no_activation.visibility_before_false_positive,
        visibility,
    )
    torch.testing.assert_close(no_activation.sample["ball_vis"], visibility)
    assert not dropped_then_replaced.visibility_before_false_positive.any()
    assert dropped_then_replaced.sample["ball_vis"].all()
    assert not active.visibility_before_false_positive.any()
    assert active.sample["ball_vis"].all()


def test_tracking_provenance_is_local_for_repeated_interleaved_and_reentrant_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_visibility = torch.tensor([[True, False], [False, False]])
    second_visibility = torch.tensor([[False, True], [True, True]])
    first_sample = _observation_sample(first_visibility)
    second_sample = _observation_sample(second_visibility)
    first_augmentation = BLCSBallObservationAugmentation(
        _augmentation_config(enabled=True)
    )
    second_augmentation = BLCSBallObservationAugmentation(
        _augmentation_config(enabled=True)
    )

    first_result = first_augmentation.forward_with_tracking_provenance(first_sample)
    interleaved_result = second_augmentation.forward_with_tracking_provenance(
        second_sample
    )
    repeated_result = first_augmentation.forward_with_tracking_provenance(second_sample)

    torch.testing.assert_close(
        first_result.visibility_before_false_positive,
        first_visibility,
    )
    torch.testing.assert_close(
        interleaved_result.visibility_before_false_positive,
        second_visibility,
    )
    torch.testing.assert_close(
        repeated_result.visibility_before_false_positive,
        second_visibility,
    )

    original_apply = first_augmentation._apply_false_positive
    nested_visibility: list[torch.Tensor] = []
    reentering = False

    def reentrant_apply(
        ball_uv: torch.Tensor,
        ball_vis: torch.Tensor,
        *,
        dropped_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal reentering
        if not reentering:
            reentering = True
            nested_result = first_augmentation.forward_with_tracking_provenance(
                second_sample
            )
            nested_visibility.append(
                nested_result.visibility_before_false_positive
            )
            reentering = False
        return original_apply(ball_uv, ball_vis, dropped_mask=dropped_mask)

    monkeypatch.setattr(first_augmentation, "_apply_false_positive", reentrant_apply)
    outer_result = first_augmentation.forward_with_tracking_provenance(first_sample)

    torch.testing.assert_close(
        outer_result.visibility_before_false_positive,
        first_visibility,
    )
    torch.testing.assert_close(nested_visibility[0], second_visibility)
    assert not hasattr(first_augmentation, "visibility_before_false_positive")
    assert not hasattr(second_augmentation, "visibility_before_false_positive")


def test_disabled_tracking_augmentation_is_identity_on_physical_width() -> None:
    detections = _detections()
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=False),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert result.uv.shape == (1, 2, 2, 2)
    torch.testing.assert_close(result.uv, detections.uv)
    torch.testing.assert_close(result.vis, detections.vis)
    torch.testing.assert_close(result.gt_index, detections.gt_index)


def test_tracking_noise_changes_coordinates_before_any_q_axis_exists() -> None:
    torch.manual_seed(23)
    detections = _detections()
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, gaussian_noise=True),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert not torch.equal(result.uv, detections.uv)
    assert bool(((result.uv - detections.uv).abs() < 0.01).all())
    torch.testing.assert_close(result.gt_index, detections.gt_index)


def test_dropout_clears_visibility_coordinates_and_provenance() -> None:
    detections = _detections()
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, visibility_dropout=True),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert not result.vis.any()
    assert not result.uv.any()
    assert (result.gt_index == -1).all()


def test_false_positives_have_negative_provenance() -> None:
    torch.manual_seed(31)
    detections = _detections(visible=False)
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, false_positive=True),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert result.vis.all()
    assert bool(((result.uv >= 0.0) & (result.uv <= 1.0)).all())
    assert (result.gt_index == -1).all()


def test_false_positive_after_dropout_does_not_recover_gt_provenance() -> None:
    torch.manual_seed(37)
    detections = _detections()
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(
            enabled=True,
            visibility_dropout=True,
            false_positive=True,
        ),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert result.vis.all()
    assert (result.gt_index == -1).all()


def test_synthetic_false_positives_are_bounded_to_q_with_fixed_carrier_shapes() -> None:
    torch.manual_seed(41)
    detections = PhysicalObservationCandidates(
        uv=torch.zeros(1, 2, 4, 2),
        vis=torch.zeros(1, 2, 4, dtype=torch.bool),
        gt_index=torch.full((1, 2, 4), -1, dtype=torch.long),
    )
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, false_positive=True),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    assert result.uv.shape == detections.uv.shape
    assert result.vis.shape == detections.vis.shape
    assert result.vis.sum(dim=-1).tolist() == [[2, 2]]
    assert result.uv[~result.vis].eq(0).all()
    assert result.gt_index.eq(-1).all()


def test_false_positive_capacity_selection_is_deterministic_and_permutation_invariant() -> None:
    class _FixedFalsePositiveObservation:
        def __init__(self, post_false_positive_uv: torch.Tensor) -> None:
            self.post_false_positive_uv = post_false_positive_uv

        def forward_with_tracking_provenance(
            self, sample: dict[str, torch.Tensor]
        ) -> BLCSBallObservationTrackingResult:
            return BLCSBallObservationTrackingResult(
                sample=cast(
                    "BLCSMultiViewSample",
                    {
                        **sample,
                        "ball_uv": self.post_false_positive_uv.clone(),
                        "ball_vis": torch.ones_like(
                            sample["ball_vis"], dtype=torch.bool
                        ),
                    },
                ),
                visibility_before_false_positive=sample["ball_vis"].clone(),
            )

    def run(post_false_positive_uv: torch.Tensor) -> PhysicalObservationCandidates:
        views, frames, carriers, _ = post_false_positive_uv.shape
        detections = PhysicalObservationCandidates(
            uv=torch.zeros_like(post_false_positive_uv),
            vis=torch.zeros(post_false_positive_uv.shape[:-1], dtype=torch.bool),
            gt_index=torch.full(
                post_false_positive_uv.shape[:-1], -1, dtype=torch.long
            ),
        )
        augmentation = BLCSTrackingDetectionAugmentation(
            _augmentation_config(enabled=True),
            num_slots=2,
        )
        flattened_uv = post_false_positive_uv.permute(0, 2, 1, 3).reshape(
            -1, frames, 2
        )
        production_uv = flattened_uv.reshape(views, carriers, frames, 2).permute(
            0, 2, 1, 3
        )
        production_vis = (
            torch.ones((views * carriers, frames), dtype=torch.bool)
            .reshape(views, carriers, frames)
            .permute(0, 2, 1)
        )
        layout_preserving_uv_clone = production_uv.unsqueeze(-2).clone()
        layout_preserving_vis_clone = production_vis.unsqueeze(-1).clone()
        assert not layout_preserving_uv_clone.is_contiguous()
        assert not layout_preserving_vis_clone.is_contiguous()
        assert (
            layout_preserving_uv_clone.reshape(
                -1, carriers, 1, 2
            ).untyped_storage().data_ptr()
            != layout_preserving_uv_clone.untyped_storage().data_ptr()
        )
        assert (
            layout_preserving_vis_clone.reshape(
                -1, carriers, 1
            ).untyped_storage().data_ptr()
            != layout_preserving_vis_clone.untyped_storage().data_ptr()
        )
        augmentation.observation = _FixedFalsePositiveObservation(flattened_uv)  # type: ignore[assignment]
        return _apply(augmentation, detections)

    x_coordinates = (
        torch.tensor([0.8, 0.3, 0.6, 0.1], dtype=torch.float32)
        .view(1, 1, 4, 1)
        .expand(2, 3, -1, -1)
    )
    values = torch.cat(
        [x_coordinates, torch.full_like(x_coordinates, 0.5)], dim=-1
    )
    first = run(values)
    replay = run(values)
    permuted = run(values[:, :, [2, 0, 3, 1]])

    torch.testing.assert_close(first.uv, replay.uv)
    torch.testing.assert_close(first.vis, replay.vis)
    first_selected = first.uv[..., 0].masked_select(first.vis).reshape(2, 3, 2)
    first_selected = first_selected.sort(dim=-1).values
    permuted_selected = permuted.uv[..., 0].masked_select(permuted.vis)
    permuted_selected = permuted_selected.reshape(2, 3, 2).sort(dim=-1).values
    torch.testing.assert_close(
        first_selected, torch.tensor([0.1, 0.3]).expand(2, 3, 2)
    )
    torch.testing.assert_close(first_selected, permuted_selected)
    assert first.uv[~first.vis].eq(0).all()
    assert first.gt_index.eq(-1).all()
    assert permuted.gt_index.eq(-1).all()


def test_genuine_overflow_is_preserved_for_tracker_rejection() -> None:
    detections = _single_frame_detections([0.1, 0.4, 0.8], visible=True)
    augmentation = BLCSTrackingDetectionAugmentation(
        _augmentation_config(enabled=True, false_positive=True),
        num_slots=2,
    )

    result = _apply(augmentation, detections)

    torch.testing.assert_close(result.uv, detections.uv)
    torch.testing.assert_close(result.vis, detections.vis)
    torch.testing.assert_close(result.gt_index, detections.gt_index)
