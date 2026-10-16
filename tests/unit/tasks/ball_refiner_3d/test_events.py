"""Binary probabilities, Gaussian supervision, whole-clip inference and default objectives."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.ball_refiner_3d.configuration.data import WindowConfig
from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.configuration.training import (
    ReconstructionConfig,
    training_config,
)
from src.tasks.ball_refiner_3d.data.sampling import sample_batch
from src.tasks.ball_refiner_3d.data.schema import (
    CorruptedTrajectory,
    PreparedRally,
    Rally,
)
from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.inference.clip import predict_normalized
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.model_io.factory import build_refiner
from src.tasks.ball_refiner_3d.physics.targets import PhysicsTargets
from src.tasks.ball_refiner_3d.training.losses.events import event_loss
from src.tasks.ball_refiner_3d.training.schedules import reconstruction_weight_at
from src.utils.paths import PROJECT_ROOT


def test_gaussian_peaks_overlap_and_outside_window_tails():
    events: np.ndarray = np.zeros(16, np.uint8)
    events[[0, 5, 7, 15]] = [1, 2, 3, 1]
    target = gaussian_event_target(events, 2.0)
    assert target[[0, 5, 7, 15]].tolist() == [1.0] * 4
    assert target[6] == pytest.approx(np.exp(-0.5 * (0.5**2)))  # max, not sum
    assert 0 < target[3] < 1
    # An event outside [8:12] still contributes a tail inside that training crop.
    assert np.all(target[8:12] > 0)
    assert not gaussian_event_target(np.zeros(5, np.uint8), 2.0).any()
    with pytest.raises(ValueError):
        gaussian_event_target(events, 0)


def test_sampling_crops_full_rally_event_targets_without_rebuilding():
    # This test exercises sampling itself with a target tail produced before cropping.
    events: np.ndarray = np.zeros(14, np.uint8)
    events[0] = 1
    target = gaussian_event_target(events, 4.0)[None]
    rally = _prepared(14, target)
    batch = sample_batch(
        [rally],
        30,
        WindowConfig(5, 0.0, 5),
        np.random.default_rng(1),
        torch.device("cpu"),
    )
    assert batch["coordinates"].shape == batch["target"].shape == (30, 5, 3)
    assert batch["missing"].shape == batch["event_target"].shape == (30, 5)
    assert not batch["padding"].any()
    labels = batch["event_target"]
    assert torch.all(labels > 0)
    assert torch.any(labels[:, 0] < 1)  # no event in these windows, but tails remain


def _prepared(
    frames: int,
    event_target: np.ndarray,
    segment: np.ndarray | None = None,
) -> PreparedRally:
    labels = np.zeros(frames, np.int64) if segment is None else segment
    state: np.ndarray = np.arange(frames * 9, dtype=np.float32).reshape(frames, 9)
    return PreparedRally(
        Mock(spec=Rally),
        Mock(spec=CorruptedTrajectory),
        np.ones((1, frames, 3), np.float32),
        np.zeros((1, frames), bool),
        np.zeros((1, frames, 3), np.float32),
        event_target,
        PhysicsTargets(labels, state, np.arange(4, dtype=np.float32), 2),
    )


def test_long_windows_are_padded_missing_and_segmented_per_window():
    frames = 40
    segment: np.ndarray = np.repeat([0, 1, 2], [10, 12, 18])
    rally = _prepared(frames, np.zeros((1, frames), np.float32), segment)
    batch = sample_batch(
        [rally],
        64,
        WindowConfig(8, 0.5, 30),
        np.random.default_rng(3),
        torch.device("cpu"),
    )
    lengths = (~batch["padding"]).sum(dim=1)
    assert lengths.min() == 8 and 8 < lengths.max() <= 30
    assert batch["coordinates"].shape[1] == lengths.max()
    assert torch.equal(batch["padding"], batch["segment"] < 0)
    assert batch["missing"][batch["padding"]].all()
    assert (batch["coordinates"][batch["padding"]] == 0).all()
    for row in range(64):
        labels = batch["segment"][row, : lengths[row]].numpy()
        assert labels[0] == 0 and np.all(np.diff(labels) >= 0)
        # Each window segment is supervised by the state at its first frame;
        # the synthetic state encodes its frame index (state[t, 0] == 9 t).
        firsts = np.flatnonzero(np.diff(labels, prepend=-1))
        values = batch["segment_target"][row, : len(firsts)].numpy()
        start = int(values[0, 0]) // 9
        np.testing.assert_array_equal(
            labels,
            np.cumsum(np.diff(segment[start : start + len(labels)], prepend=-1) != 0)
            - 1,
        )
        np.testing.assert_array_equal(values, rally.physics.state[start + firsts])
        assert not batch["segment_target"][row, len(firsts) :].any()
    assert (batch["surface_target"] == 2).all()


def test_default_is_equal_position_event_weights_no_gan_or_loss_schedule():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs"),
    ):
        cfg = compose(config_name="train")
    raw = training_config(cfg).raw
    training = raw["training"]
    assert not training["gan"]["enabled"] and not training["gan"]["schedule_enabled"]
    assert not raw["loss"]["reconstruction"]["enabled"]
    position = ReconstructionConfig(**raw["loss"]["reconstruction"])
    assert raw["loss"]["event"]["weight"] == 1
    assert [reconstruction_weight_at(i, position) for i in (0, 2000, 3999)] == [
        1.0,
        1.0,
        1.0,
    ]
    cfg.model.dimensions = 2
    with pytest.raises(ValueError, match="dimensions=3"):
        training_config(cfg)


def test_event_head_softmax_is_per_frame_and_soft_ce_has_correct_gradient():
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 2, 64, 8, 10000.0, "swiglu", False)
    )
    unpadded = torch.zeros(2, 16, dtype=torch.bool)
    output = model(torch.randn(2, 16, 3), unpadded, unpadded)
    assert output.coordinates.shape == (2, 16, 3) and output.event_logits.shape == (
        2,
        16,
        2,
    )
    torch.testing.assert_close(
        output.event_logits.softmax(-1).sum(-1), torch.ones(2, 16)
    )
    target = torch.full((2, 16), 0.3)
    event_loss(output.event_logits, target, ~unpadded).backward()
    assert model.event_head[-1].weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0
    logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    event_loss(
        logits, torch.tensor([[0.8]]), torch.ones(1, 1, dtype=torch.bool)
    ).backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[0.3, -0.3]]]))


def test_each_whole_clip_is_predicted_in_one_forward(monkeypatch):
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 2, 64, 8, 10000.0, "swiglu", False)
    )
    coordinates = torch.randn(3, 300, 3)
    calls = []

    def predict(_model, coords, missing, *, generator, segment=None):
        calls.append(tuple(coords.shape))
        logits = torch.zeros(*coords.shape[:2], 2)
        return RefinerOutput(coords * 2, logits)

    monkeypatch.setattr(
        "src.tasks.ball_refiner_3d.inference.clip.predict_sequences", predict
    )
    result = predict_normalized(
        model,
        coordinates,
        torch.zeros(3, 300, dtype=torch.bool),
        batch_size=2,
        seed=42,
        clock=None,
    )
    # No temporal windows: two forwards over complete 300-frame sequences.
    assert calls == [(2, 300, 3), (1, 300, 3)]
    torch.testing.assert_close(result.coordinates, coordinates * 2)
    torch.testing.assert_close(result.event_probability, torch.full((3, 300), 0.5))


def test_long_clips_do_not_change_short_predictions():
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 2, 64, 8, 10000.0, "swiglu", False)
    ).eval()
    torch.manual_seed(0)
    long = torch.randn(1, 900, 3)
    missing = torch.zeros(1, 900, dtype=torch.bool)
    short = model(long[:, :50], missing[:, :50], missing[:, :50]).coordinates
    full = model(long, missing, missing)
    assert torch.isfinite(full.coordinates).all()
    again = model(long[:, :50], missing[:, :50], missing[:, :50]).coordinates
    torch.testing.assert_close(short, again, rtol=0, atol=0)


def test_padding_does_not_change_unpadded_predictions():
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 2, 64, 8, 10000.0, "swiglu", False)
    ).eval()
    torch.manual_seed(1)
    values = torch.randn(1, 20, 3)
    missing = torch.zeros(1, 20, dtype=torch.bool)
    alone = model(values, missing, missing).coordinates
    padded_values = torch.cat((values, torch.zeros(1, 7, 3)), dim=1)
    padding = torch.zeros(1, 27, dtype=torch.bool)
    padding[:, 20:] = True
    padded = model(padded_values, padding, padding).coordinates[:, :20]
    torch.testing.assert_close(alone, padded, rtol=1e-5, atol=1e-6)


def test_legacy_checkpoint_does_not_silently_gain_untrained_event_head(tmp_path):
    path = tmp_path / "old.ckpt"
    torch.save({"schema": "ball_refiner.coordinates.v3"}, path)
    with pytest.raises(ValueError, match="event-head checkpoint"):
        load_checkpoint(path, torch.device("cpu"))
