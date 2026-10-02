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


def test_window_omits_only_all_missing_people_and_preserves_gmm_and_gradients(window_data):
    clip, config = window_data
    base = detector_only_input(clip.evidence, 0, 11, config)
    valid = torch.zeros(1, 11, 80, 4, dtype=torch.bool)
    valid[:, 0, 0] = True  # A track outside the selected window.
    valid[:, 3, 19, 0] = True  # Even a single valid elbow keeps this person.
    valid[:, 5:7, 79] = True
    inputs = replace(base, pose_uv=torch.rand(1, 11, 80, 4, 2),
                     pose_confidence=torch.rand(1, 11, 80, 4), pose_valid=valid)
    compact = slice_input(inputs, 3, 8)
    assert compact.pose_uv.shape == (1, 5, 2, 4, 2)
    torch.testing.assert_close(compact.pose_uv, inputs.pose_uv[:, 3:8, [19, 79]])
    uncompact = replace(compact, pose_uv=inputs.pose_uv[:, 3:8],
                        pose_confidence=inputs.pose_confidence[:, 3:8], pose_valid=valid[:, 3:8])
    pair = build_ball_refiner_2d(replace(config, use_pose=True, dropout=0.0, pose_dropout=0.0))
    pair.model.eval()
    with torch.no_grad():
        pair.model.pose_context.gate.fill_(.7)  # The zero-init gate must not hide mistakes.
    predictions = []
    gradients = []
    for value in (uncompact, compact):
        pair.model.zero_grad(set_to_none=True)
        prediction = pair.run(value)
        predictions.append(prediction)
        loss = (prediction.means.square().sum() + prediction.scale_tril.square().sum()
                + prediction.mixture_logits.square().sum() + prediction.presence_logits.square().sum())
        loss.backward()
        gradients.append({name: parameter.grad.clone() for name, parameter in pair.model.named_parameters()
                          if parameter.grad is not None})
    for field in ("means", "scale_tril", "mixture_logits", "presence_logits"):
        torch.testing.assert_close(getattr(predictions[0], field), getattr(predictions[1], field), atol=2e-6, rtol=2e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-4)
    compact.pose_uv.zero_()
    assert inputs.pose_uv[:, 3:8, [19, 79]].any()


def test_empty_and_differently_populated_windows_collate_without_track_truncation(window_data):
    clip, config = window_data
    base = detector_only_input(clip.evidence, 0, 11, config)
    valid = torch.zeros(1, 11, 70, 4, dtype=torch.bool)
    valid[:, :3] = True
    inputs = replace(base, pose_uv=torch.rand(1, 11, 70, 4, 2),
                     pose_confidence=torch.ones(1, 11, 70, 4), pose_valid=valid)
    populated, empty = slice_input(inputs, 0, 3), slice_input(inputs, 5, 8)
    assert populated.pose_uv.shape[2] == 70 and empty.pose_uv.shape[2] == 0
    batch = collate_inputs([populated, empty])
    assert batch.pose_uv.shape == (2, 3, 70, 4, 2)
    assert batch.pose_valid[0].all() and not batch.pose_valid[1].any()
    pair = build_ball_refiner_2d(replace(config, use_pose=True, dropout=0.0))
    pair.model.eval()
    with torch.no_grad():
        pair.model.pose_context.gate.fill_(.7)
        torch.testing.assert_close(pair.run(empty).means[0], pair.run(batch).means[1])
