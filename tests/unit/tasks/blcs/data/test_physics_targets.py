"""BLCS boundary rejects missing/corrupt physics and retains clean observations."""

import pytest
import torch

from src.tasks.blcs.physics.rollout import reconstruct_segments, target_parameters
from src.tasks.blcs.physics.targets import PhysicsTargets
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
)
from tests.support.tasks.blcs_physics import physics_sample, prepared_samples


def test_physics_is_teacher_only_and_clean_target_is_independent():
    sample = physics_sample()
    clean = sample["ball_uv_target"].clone()
    sample["ball_uv"].fill_(0.8)
    batch, target = prepared_samples([sample])
    assert torch.equal(target.target_uv[0], clean)
    assert set(target.call.kwargs) == {
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    }
    del batch["physics_clock"]
    with pytest.raises(KeyError, match="physics_clock"):
        PhysicsTargets.from_batch(batch, target.loss_mask)


def test_invalid_clock_and_segment_padding_are_rejected():
    batch, target = prepared_samples(
        [physics_sample(), physics_sample(3, name="short")]
    )
    batch["physics_clock"][0, 3] += 1
    with pytest.raises(ValueError, match="clock"):
        PhysicsTargets.from_batch(batch, target.loss_mask)
    batch["physics_clock"][0, 3] -= 1
    batch["physics_segment"][1, 3] = 0
    with pytest.raises(ValueError, match="Padded"):
        PhysicsTargets.from_batch(batch, target.loss_mask)


def test_head_reconstruction_uses_only_its_parameters_reset_events_and_clock():
    _, target = prepared_samples([physics_sample(12, hits=(20,))])
    resets = torch.zeros_like(target.loss_mask)
    resets[:, 1:] = target.physics.segment[:, 1:] != target.physics.segment[:, :-1]
    p, v = reconstruct_segments(
        denormalize_court_position(target.position),
        denormalize_court_velocity(target.velocity),
        target_parameters(target.physics),
        resets,
        target.loss_mask,
        target.physics.clock,
        gravity=9.8,
    )
    assert torch.allclose(p, denormalize_court_position(target.position), atol=2e-6)
    assert torch.allclose(v, denormalize_court_velocity(target.velocity), atol=3e-6)
    # An incorrect inferred segmentation is a real measurable failure, not an oracle fallback.
    wrong, _ = reconstruct_segments(
        denormalize_court_position(target.position),
        denormalize_court_velocity(target.velocity),
        target_parameters(target.physics),
        torch.zeros_like(resets),
        target.loss_mask,
        target.physics.clock,
        gravity=9.8,
    )
    assert (wrong - p).abs().max() > 0.01
