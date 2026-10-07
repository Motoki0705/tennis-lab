"""Physics heads: units, segmentation, integrated reconstruction and objectives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.configuration.training import PhysicsLossConfig
from src.tasks.ball_refiner_3d.inference.clip import predict_normalized
from src.tasks.ball_refiner_3d.inference.segmentation import predicted_segments
from src.tasks.ball_refiner_3d.model_io.checkpoint import (
    checkpoint_flight_clock,
    upgrade_model_config,
)
from src.tasks.ball_refiner_3d.model_io.factory import bind_refiner, build_refiner
from src.tasks.ball_refiner_3d.models.components.embeddings import coordinate_features
from src.tasks.ball_refiner_3d.models.components.temporal_transformer import MAX_FRAMES
from src.tasks.ball_refiner_3d.physics.reconstruction import (
    integrate_segments,
    segment_layout,
)
from src.tasks.ball_refiner_3d.physics.targets import (
    FlightClock,
    local_segments,
    physics_targets,
    segments_from_events,
)
from src.tasks.ball_refiner_3d.physics.units import (
    Field,
    State,
    decode_field,
    decode_state,
    encode_field,
    encode_state,
)
from src.tasks.ball_refiner_3d.training.losses.physics import (
    physics_losses,
    weighted_physics,
)
from tests.support.physics.ball_record import simulated_record


def _model(physics: bool = True) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_refiner(
        ModelConfig(
            3, "regression", 16, 1, 2, 0.0, 2, 64, 8, 10000.0, "swiglu", physics
        )
    )


def test_units_roundtrip_and_reject_vertical_wind() -> None:
    field = Field(
        torch.tensor([[1.5, -2.0, 0.0]]), torch.tensor([0.013]), torch.tensor([0.0017])
    )
    decoded = decode_field(encode_field(field))
    for before, after in zip(field, decoded, strict=True):
        torch.testing.assert_close(before, after)
    state = State(torch.randn(4, 3), torch.randn(4, 3) * 20, torch.randn(4, 3) * 200)
    for before, after in zip(state, decode_state(encode_state(state)), strict=True):
        torch.testing.assert_close(before, after)
    with pytest.raises(ValueError, match="horizontal"):
        encode_field(field._replace(wind=torch.tensor([[0.0, 0.0, 1.0]])))


def test_segments_open_at_frame_zero_and_each_event() -> None:
    assert segments_from_events(np.array([3, 7]), 10).tolist() == [
        0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    ]  # fmt: skip
    assert segments_from_events(np.array([0]), 3).tolist() == [0, 0, 0]
    assert local_segments(np.array([4, 4, 5, 5, 5, 7])).tolist() == [0, 0, 1, 1, 1, 2]
    probability = np.zeros(12)
    probability[[5, 6, 9]] = [0.9, 0.7, 0.8]
    # 6 is not a peak, so flights open at 0, 5 and 9.
    assert predicted_segments(probability).tolist() == [0] * 5 + [1] * 4 + [2] * 3
    with pytest.raises(ValueError):
        segment_layout(torch.tensor([[0, 1, 0]]))


def test_ground_truth_parameters_reconstruct_every_window() -> None:
    positions, record = simulated_record(48, output_fps=60, sim_fps=240, hits=(50, 132))
    targets = physics_targets(record, positions)
    clock = FlightClock.of([record])
    for start, length in ((0, 48), (5, 30), (14, 20)):
        labels = local_segments(targets.segment[start : start + length])
        firsts = np.flatnonzero(np.diff(labels, prepend=-1))
        integrated = integrate_segments(
            torch.from_numpy(targets.state[start + firsts])[None],
            torch.from_numpy(targets.field)[None],
            torch.from_numpy(labels)[None],
            clock,
        )[0]
        np.testing.assert_allclose(
            integrated.numpy(), positions[start : start + length], atol=1e-5
        )


def test_padded_batches_integrate_each_row_independently() -> None:
    positions, record = simulated_record(30, output_fps=60, sim_fps=240, hits=(40,))
    targets = physics_targets(record, positions)
    clock = FlightClock.of([record])
    labels = torch.from_numpy(targets.segment)
    firsts = np.flatnonzero(np.diff(targets.segment, prepend=-1))
    states = torch.from_numpy(targets.state[firsts])
    padded_labels = torch.full((2, 30), -1)
    padded_labels[0] = labels
    first_length = int(firsts[1])  # the second flight opens at the hit frame
    padded_labels[1, :first_length] = 0
    padded_states = torch.zeros(2, len(firsts), 9)
    padded_states[0] = states
    padded_states[1, 0] = states[0]
    field = torch.from_numpy(targets.field).expand(2, -1)
    result = integrate_segments(padded_states, field, padded_labels, clock)
    np.testing.assert_allclose(result[0].numpy(), positions, atol=1e-5)
    np.testing.assert_allclose(
        result[1, :first_length].numpy(), positions[:first_length], atol=1e-5
    )
    assert (result[1, first_length:] == 0).all()


def test_segment_state_depends_only_on_its_own_frames() -> None:
    model = _model().eval()
    coordinates = torch.randn(1, 20, 3)
    missing = torch.zeros(1, 20, dtype=torch.bool)
    segment = torch.tensor([[0] * 8 + [1] * 12])
    tokens = model.trunk(coordinate_features(coordinates, missing), missing)
    first = model.heads(tokens, missing, segment).physics
    changed = tokens.clone()
    changed[:, 8:] += 1.0
    second = model.heads(changed, missing, segment).physics
    torch.testing.assert_close(first.segment_states[:, 0], second.segment_states[:, 0])
    assert not torch.allclose(first.segment_states[:, 1], second.segment_states[:, 1])
    assert not torch.allclose(first.field, second.field)


def test_physics_inputs_are_validated_before_computation() -> None:
    binding = bind_refiner(_model())
    batch = {
        "coordinates": torch.zeros(1, 6, 3),
        "missing": torch.zeros(1, 6, dtype=torch.bool),
        "padding": torch.zeros(1, 6, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="-1 exactly on padded"):
        binding.run({**batch, "segment": torch.tensor([[0, 0, 0, -1, 1, 1]])})
    padding = torch.tensor([[False] * 4 + [True] * 2])
    with pytest.raises(ValueError, match="Padded frames must be missing"):
        binding.run({**batch, "padding": padding})
    with pytest.raises(ValueError, match="requires physics heads"):
        bind_refiner(_model(physics=False)).run(
            {**batch, "segment": torch.zeros(1, 6, dtype=torch.long)}
        )


@pytest.mark.parametrize(
    "gradient,direct_moves,integrated_moves",
    [("direct", True, False), ("integrated", False, True), ("both", True, True)],
)
def test_consistency_gradient_reaches_only_the_configured_side(
    gradient: str, direct_moves: bool, integrated_moves: bool
) -> None:
    positions, record = simulated_record(24, output_fps=60, sim_fps=240, hits=(40,))
    targets = physics_targets(record, positions)
    model = _model()
    missing = torch.zeros(1, 24, dtype=torch.bool)
    segment = torch.from_numpy(targets.segment)[None]
    output = model(
        torch.from_numpy(positions / [10, 20, 5]).float()[None],
        missing,
        missing,
        segment,
    )
    output.coordinates.retain_grad()
    output.physics.segment_states.retain_grad()
    firsts = np.flatnonzero(np.diff(targets.segment, prepend=-1))
    batch = {
        "padding": missing,
        "segment": segment,
        "target": torch.from_numpy(positions / [10, 20, 5]).float()[None],
        "segment_target": torch.from_numpy(targets.state[firsts])[None],
        "field_target": torch.from_numpy(targets.field)[None],
        "surface_target": torch.tensor([targets.surface]),
    }
    config = PhysicsLossConfig(0.0, 0.0, 0.0, 1.0, gradient)
    losses = physics_losses(output, batch, FlightClock.of([record]), config)
    assert set(losses) == {"field", "segment", "reconstruction", "consistency"}
    weighted_physics(losses, config).backward()
    direct = output.coordinates.grad
    integrated = output.physics.segment_states.grad
    assert (direct is not None and direct.abs().sum() > 0) == direct_moves
    assert (integrated is not None and integrated.abs().sum() > 0) == integrated_moves


def test_whole_clip_physics_prediction_uses_given_or_predicted_segments() -> None:
    positions, record = simulated_record(40, output_fps=60, sim_fps=240, hits=(60,))
    clock = FlightClock.of([record])
    model = _model().eval()
    coordinates = torch.from_numpy(positions / [10, 20, 5]).float()[None]
    missing = torch.zeros(1, 40, dtype=torch.bool)
    with pytest.raises(ValueError, match="flight clock"):
        predict_normalized(
            model, coordinates, missing, batch_size=1, seed=0, clock=None
        )
    truth = torch.from_numpy(record.frame_segment())[None]
    given = predict_normalized(
        model, coordinates, missing, batch_size=1, seed=0, clock=clock, segment=truth
    )
    assert torch.equal(given.physics.segment, truth)
    expected = integrate_segments(
        given.physics.segment_states, given.physics.field, truth, clock
    ) / torch.tensor([10.0, 20.0, 5.0])
    torch.testing.assert_close(given.physics.integrated, expected)
    predicted = predict_normalized(
        model, coordinates, missing, batch_size=1, seed=0, clock=clock
    )
    labels = predicted_segments(predicted.event_probability[0].numpy())
    np.testing.assert_array_equal(predicted.physics.segment[0].numpy(), labels)
    torch.testing.assert_close(
        predicted.physics.surface_probability.sum(dim=-1), torch.ones(1)
    )


def test_legacy_model_configs_upgrade_explicitly_and_clock_is_required() -> None:
    legacy = {
        "dimensions": 3,
        "architecture": "regression",
        "width": 16,
        "layers": 1,
        "heads": 2,
        "dropout": 0.0,
        "window_length": 128,
        "flow_steps": 2,
        "ffn_dim": 64,
        "rope_dim": 8,
        "rope_theta": 10000.0,
        "ffn_type": "swiglu",
    }
    upgraded = upgrade_model_config(legacy)
    assert "window_length" not in upgraded and upgraded["physics_heads"] is False
    assert upgrade_model_config(upgraded) is upgraded
    with pytest.raises(ValueError, match="Unrecognized"):
        upgrade_model_config({k: v for k, v in legacy.items() if k != "window_length"})
    with pytest.raises(ValueError, match="no flight clock"):
        checkpoint_flight_clock({"schema": "ball_refiner_3d.events.v1"})


def test_sequences_beyond_the_rope_table_are_rejected() -> None:
    model = _model(physics=False).eval()
    frames = MAX_FRAMES + 1
    missing = torch.zeros(1, frames, dtype=torch.bool)
    with pytest.raises(ValueError, match="limited to"):
        model(torch.zeros(1, frames, 3), missing, missing)
