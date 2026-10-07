"""Binary probabilities, Gaussian supervision, window ownership and default objectives."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

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
from src.tasks.ball_refiner_3d.inference.windowing import (
    predict_normalized,
    window_owners,
    window_starts,
)
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.model_io.factory import build_refiner
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
    rally = PreparedRally(
        Mock(spec=Rally),
        Mock(spec=CorruptedTrajectory),
        np.zeros((1, 14, 3), np.float32),
        np.zeros((1, 14), bool),
        np.zeros((1, 14, 3), np.float32),
        target,
    )
    coords, mask, xyz, labels = sample_batch(
        [rally], 30, 5, np.random.default_rng(1), torch.device("cpu")
    )
    assert coords.shape == xyz.shape == (30, 5, 3) and mask.shape == labels.shape == (
        30,
        5,
    )
    assert torch.all(labels > 0)
    assert torch.any(labels[:, 0] < 1)  # no event in these windows, but tails remain


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
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 16, 2, 64, 8, 10000.0, "swiglu")
    )
    output = model(torch.randn(2, 16, 3), torch.zeros(2, 16, dtype=torch.bool))
    assert output.coordinates.shape == (2, 16, 3) and output.event_logits.shape == (
        2,
        16,
        2,
    )
    torch.testing.assert_close(
        output.event_logits.softmax(-1).sum(-1), torch.ones(2, 16)
    )
    target = torch.full((2, 16), 0.3)
    event_loss(output.event_logits, target).backward()
    assert model.event_head[-1].weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0
    logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    event_loss(logits, torch.tensor([[0.8]])).backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[0.3, -0.3]]]))


def test_coordinates_and_events_use_identical_window_owners(monkeypatch):
    model = build_refiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.0, 8, 2, 64, 8, 10000.0, "swiglu")
    )
    coordinates = torch.arange(13.0).view(1, 13, 1).expand(1, 13, 3).clone()

    def predict(_model, coords, missing, *, generator):
        start = coords[:, :1, :1].expand_as(coords)
        logits = torch.stack((torch.zeros_like(start[..., 0]), start[..., 0]), -1)
        return RefinerOutput(start, logits)

    monkeypatch.setattr(
        "src.tasks.ball_refiner_3d.inference.windowing.predict_window", predict
    )
    result = predict_normalized(
        model, coordinates, torch.zeros(1, 13, dtype=torch.bool), batch_size=2, seed=42
    )
    starts = window_starts(13, 8, 4)
    expected = torch.tensor(
        np.array(starts)[window_owners(13, starts, 8)], dtype=torch.float32
    )
    torch.testing.assert_close(result.coordinates[0, :, 0], expected)
    torch.testing.assert_close(result.event_probability[0], expected.sigmoid())


def test_legacy_checkpoint_does_not_silently_gain_untrained_event_head(tmp_path):
    path = tmp_path / "old.ckpt"
    torch.save({"schema": "ball_refiner.coordinates.v3"}, path)
    with pytest.raises(ValueError, match="event-head checkpoint"):
        load_checkpoint(path, torch.device("cpu"))
