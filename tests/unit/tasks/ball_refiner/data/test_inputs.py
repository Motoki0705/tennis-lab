"""Inference inputs never require amodal targets or fabricate context."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.data.inputs import (
    collate_inputs,
    detector_only_input,
    slice_input,
)
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.evaluation import predict_clip


def test_inputs_from_evidence_need_no_targets_and_preserve_source_time(window_data):
    clip, config = window_data
    inputs = detector_only_input(clip.evidence, 0, clip.record.frame_count, config)
    sliced = slice_input(inputs, 3, 8)
    torch.testing.assert_close(sliced.timestamps_seconds[0], torch.from_numpy(clip.evidence.timestamps_seconds[3:8]))
    assert sliced.court_uv.shape == (1, config.court_keypoints, 2)
    assert sliced.pose_uv.shape == (1, 5, 0, 4, 2)
    sliced.candidates.coords.zero_()
    assert inputs.candidates.coords.any() and clip.evidence.candidates.coords.any()
    with pytest.raises(ValueError, match="Detector-only"):
        detector_only_input(clip.evidence, 0, 5, replace(config, use_court=True))


def test_camera_and_time_axes_cannot_be_sliced_implicitly(window_data):
    clip, config = window_data
    inputs = detector_only_input(clip.evidence, 0, 5, config)
    with pytest.raises(ValueError, match="one camera"):
        slice_input(collate_inputs([inputs, inputs]), 0, 3)
    for start, stop in ((-1, 3), (0, 6), (2, 2)):
        with pytest.raises(ValueError, match="real frame"):
            slice_input(inputs, start, stop)
    with pytest.raises(ValueError, match="empty"):
        collate_inputs([])


def test_validation_prediction_never_materializes_target_windows(window_data, monkeypatch):
    clip, model_config = window_data
    pair = build_ball_refiner_2d(model_config)
    config = SimpleNamespace(model=model_config, window_length=5, stride=3, training=SimpleNamespace(batch_size=2))

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference requested a target window")

    monkeypatch.setattr(type(clip.targets), "target", forbidden)
    result = predict_clip(pair, clip, config, device=torch.device("cpu"), gap=np.zeros(clip.record.frame_count, bool))
    assert result.means.shape[:2] == (1, clip.record.frame_count)
