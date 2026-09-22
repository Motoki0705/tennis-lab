"""Preserve observable ordering, capacity, gradients, and tracking after batching work."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.tasks.base.data import observation_tracking as tracking


def _scalar_order(values: Tensor, visibility: Tensor) -> list[int]:
    """Pre-optimization scalar oracle, including the exact-duplicate tie break."""
    candidates = visibility.any(-1).nonzero().flatten().tolist()

    def key(index: int) -> tuple[tuple[int, ...], tuple[float, ...], int]:
        return (
            tuple(int(x) for x in visibility[index].tolist()),
            tuple(
                float(x)
                for joint in range(values.shape[-2])
                if bool(visibility[index, joint])
                for x in values[index, joint].tolist()
            ),
            index,
        )

    return sorted(candidates, key=key)


def _scalar_cap(
    values: Tensor, visibility: Tensor, before_fp: Tensor, *, slots: int
) -> tuple[Tensor, Tensor]:
    output = values.clone(memory_format=torch.contiguous_format)
    masks = visibility.clone(memory_format=torch.contiguous_format)
    carriers, joints = values.shape[-3:-1]
    flat = output.reshape(-1, carriers, joints, 2)
    flat_masks = masks.reshape(-1, carriers, joints)
    flat_before = before_fp.reshape(-1, carriers, joints)
    for frame in range(flat.shape[0]):
        present = flat_masks[frame].any(-1)
        if int(present.sum()) <= slots:
            continue
        genuine = flat_before[frame].any(-1)
        available = max(slots - int((genuine & present).sum()), 0)
        ordered = _scalar_order(flat[frame], flat_masks[frame])
        synthetic = [index for index in ordered if not bool(genuine[index])]
        for rejected in synthetic[available:]:
            flat[frame, rejected] = 0
            flat_masks[frame, rejected] = False
    return output, masks


def _scalar_prediction(
    state: tracking._TrackState, frame_index: int, *, use_velocity: bool
) -> tuple[Tensor, Tensor]:
    result = state.last_values.clone()
    if (
        use_velocity
        and state.previous_values is not None
        and state.previous_visibility is not None
        and state.previous_frame is not None
    ):
        elapsed = state.last_frame - state.previous_frame
        common = state.last_visibility & state.previous_visibility
        velocity = (state.last_values[common] - state.previous_values[common]) / elapsed
        result[common] = state.last_values[common] + velocity * (
            frame_index - state.last_frame
        )
    return result, state.last_visibility


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("elapsed", [1, 3, 7])
def test_batched_motion_prediction_preserves_values_and_gradients(
    dtype: torch.dtype, elapsed: int
) -> None:
    rng = torch.Generator().manual_seed(317)
    previous = torch.rand(17, 2, generator=rng, dtype=dtype).requires_grad_()
    last = torch.rand(17, 2, generator=rng, dtype=dtype).requires_grad_()
    state = tracking._TrackState(
        last_values=last,
        last_visibility=torch.rand(17, generator=rng) > 0.2,
        last_frame=elapsed,
        previous_values=previous,
        previous_visibility=torch.rand(17, generator=rng) > 0.2,
        previous_frame=0,
    )
    actual, _ = state.prediction(elapsed + 3, use_velocity=True)
    expected, _ = _scalar_prediction(state, elapsed + 3, use_velocity=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (last, previous))
    expected_grads = torch.autograd.grad(expected.sum(), (last, previous))
    for result, baseline in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("joints", [1, 17])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_bulk_order_preserves_scalar_keys_with_duplicates_and_hidden_nans(
    joints: int, dtype: torch.dtype
) -> None:
    rng = torch.Generator().manual_seed(713)
    for _ in range(8):
        values = torch.rand(8, joints, 2, generator=rng).to(dtype)
        visibility = torch.rand(8, joints, generator=rng) > 0.35
        values.masked_fill_(~visibility[..., None], float("nan"))
        values[3], visibility[3] = values[1].clone(), visibility[1].clone()
        visibility[5] = False
        assert tracking._canonical_detection_indices(
            values, visibility
        ) == _scalar_order(values, visibility)
    assert tracking._canonical_detection_indices(values, visibility & False) == []


@pytest.mark.parametrize("joints", [1, 17])
@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_capacity_matches_scalar_policy_including_genuine_overflow_and_gradients(
    joints: int, slots: int
) -> None:
    rng = torch.Generator().manual_seed(921)
    # Deliberately use a noncontiguous view/frame layout and partial visibility.
    values = torch.rand(2, 7, 9, joints, 2, generator=rng).permute(0, 2, 1, 3, 4)
    visibility = (torch.rand(2, 7, 9, joints, generator=rng) > 0.2).permute(0, 2, 1, 3)
    before = visibility & (torch.rand(visibility.shape, generator=rng) > 0.8)
    before[:, 0] = False  # Every candidate is synthetic.
    before[:, 1] = visibility[:, 1]  # Genuine overflow must not be truncated.
    visibility[:, 2] = False  # Empty frame.
    visibility[:, 3, 3:], before[:, 3, 3:] = False, False  # Below capacity.
    source = values.detach().requires_grad_()
    reference_source = values.detach().clone().requires_grad_()
    actual = tracking.limit_synthetic_false_positive_carriers(
        source, visibility, before, num_slots=slots
    )
    expected = _scalar_cap(reference_source, visibility, before, slots=slots)
    for result, baseline in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)
        assert result.is_contiguous()
    weights = torch.rand(values.shape, generator=rng)
    (actual[0] * weights).sum().backward()
    (expected[0] * weights).sum().backward()
    torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)


@pytest.mark.parametrize("joints", [1, 17])
@pytest.mark.parametrize("seed", [0, 17, 1024])
def test_complete_tracking_matches_scalar_order_capacity_and_motion_prediction(
    joints: int, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = torch.Generator().manual_seed(seed)
    values = torch.rand(24, 7, joints, 2, generator=rng)
    visibility = torch.rand(24, 7, joints, generator=rng) > 0.3
    visibility[6:9, :2] = False
    visibility[18] = False
    values[:, 2], visibility[:, 2] = values[:, 1].clone(), visibility[:, 1].clone()
    before = visibility.clone()
    before[:, 3:] = False
    provenance = torch.arange(7).expand(24, -1).clone()
    provenance[:, 3:] = -1
    cfg = tracking.ObservationTrackingConfig(
        max_distance=0.4,
        max_missed_frames=2,
        min_reuse_gap_frames=3,
        use_velocity_prediction=True,
        min_common_keypoints=1 if joints == 1 else 4,
        cost_reduction="mean" if joints == 1 else "median",
        overflow_policy="error",
    )
    capped = tracking.limit_synthetic_false_positive_carriers(
        values, visibility, before, num_slots=4
    )
    actual = tracking.track_camera_observations(
        *capped, num_slots=4, config=cfg, debug_provenance=provenance
    )
    with monkeypatch.context() as patch:
        patch.setattr(tracking, "_canonical_detection_indices", _scalar_order)
        patch.setattr(tracking._TrackState, "prediction", _scalar_prediction)
        baseline = tracking.track_camera_observations(
            *_scalar_cap(values, visibility, before, slots=4),
            num_slots=4,
            config=cfg,
            debug_provenance=provenance,
        )
    for name in ("values", "visibility", "detection_indices", "debug_provenance"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(baseline, name), rtol=0, atol=0
        )
