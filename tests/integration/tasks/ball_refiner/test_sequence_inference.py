"""Annotation-free all-frame inference, atomic window ownership and context flow."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.gaps import mask_detector_evidence
from src.tasks.ball_refiner.data.inputs import (
    CANDIDATE_FIELDS,
    collate_inputs,
    slice_input,
)
from src.tasks.ball_refiner.inference import predict_sequence
from src.tasks.ball_refiner.refiner_2d import Refiner2DInput, build_ball_refiner_2d
from tests.integration.tasks.ball_refiner.test_refiner_2d import config


@pytest.fixture
def sequence():
    torch.manual_seed(730)
    frames, people = 12, 2
    candidate_config = BallCandidateConfig(max_candidates=3, patch_size=3, nms_kernel=3)
    candidates = decode_candidates(torch.rand(1, frames, 7, 9), config=candidate_config, subpixel_refine=True)
    inputs = Refiner2DInput(
        candidates=candidates,
        timestamps_seconds=torch.arange(frames, dtype=torch.float32)[None].square() / 120,
        pose_uv=torch.rand(1, frames, people, 4, 2), pose_confidence=torch.rand(1, frames, people, 4),
        pose_valid=torch.ones(1, frames, people, 4, dtype=torch.bool),
        court_uv=torch.rand(1, 5, 2), court_confidence=torch.rand(1, 5),
        court_valid=torch.ones(1, 5, dtype=torch.bool),
    )
    pair = build_ball_refiner_2d(config())
    with torch.no_grad():
        pair.model.pose_context.gate.fill_(.7)
        pair.model.court_context.gate.fill_(.4)
    return pair, inputs


def test_every_gmm_field_comes_from_the_same_real_window_without_labels(sequence):
    pair, inputs = sequence
    result = predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    # Explicit expected owners include centre ties and the real backfilled tail.
    expected_start = np.array([0, 0, 0, 0, 3, 3, 3, 6, 6, 7, 7, 7])
    np.testing.assert_array_equal(result.window_start, expected_start)
    np.testing.assert_array_equal(result.time_index, np.arange(12) - expected_start)
    assert pair.model.training  # caller mode is restored
    pair.model.eval()
    for start in (0, 3, 6, 7):
        with torch.no_grad():
            reference = pair.run(slice_input(inputs, start, start + 5))
        selected = np.flatnonzero(expected_start == start)
        for field in fields(reference):
            actual = getattr(result.distribution, field.name)
            assert actual.device.type == "cpu" and not actual.requires_grad
            torch.testing.assert_close(actual[:, selected], getattr(reference, field.name)[:, selected - start],
                                       rtol=2e-5, atol=2e-6)


def test_batch_partition_preserves_distribution_and_does_not_mutate_input(sequence):
    pair, inputs = sequence
    before = {key: getattr(inputs.candidates, key).clone() for key in CANDIDATE_FIELDS}
    one = predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=1, device=torch.device("cpu"))
    many = predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=9, device=torch.device("cpu"))
    for field in fields(one.distribution):
        torch.testing.assert_close(getattr(one.distribution, field.name), getattr(many.distribution, field.name),
                                   rtol=2e-5, atol=2e-6)
    for key in CANDIDATE_FIELDS:
        torch.testing.assert_close(before[key], getattr(inputs.candidates, key), rtol=0, atol=0)
    # Mutating exported tensors must not mutate input evidence either.
    one.distribution.means.zero_()
    torch.testing.assert_close(before["coords"], inputs.candidates.coords, rtol=0, atol=0)


def test_global_gap_has_identical_mask_in_all_overlapping_windows(sequence, monkeypatch):
    pair, inputs = sequence
    gap = torch.zeros(1, 12, dtype=torch.bool)
    gap[:, 3:8] = True
    missing = mask_detector_evidence(inputs, gap)
    calls = []
    original = pair.adapter.build_call

    def capture(batch):
        calls.append(batch)
        return original(batch)

    monkeypatch.setattr(pair.adapter, "build_call", capture)
    result = predict_sequence(pair, missing, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    assert result.distribution.presence_probability.shape == (1, 12)
    for batch in calls:
        for row in range(batch.timestamps_seconds.shape[0]):
            source_frames = torch.searchsorted(inputs.timestamps_seconds[0], batch.timestamps_seconds[row])
            masked = gap[0, source_frames]
            for key in CANDIDATE_FIELDS:
                assert not getattr(batch.candidates, key)[row, masked].any()
            torch.testing.assert_close(batch.pose_uv[row], inputs.pose_uv[0, source_frames])
            torch.testing.assert_close(batch.court_uv[row], inputs.court_uv[0])


def test_context_is_used_and_all_missing_frames_are_still_predicted(sequence):
    pair, inputs = sequence
    before = predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    missing = mask_detector_evidence(inputs, torch.ones(1, 12, dtype=torch.bool))
    missing = replace(missing, pose_valid=torch.zeros_like(missing.pose_valid), court_valid=torch.zeros_like(missing.court_valid))
    after = predict_sequence(pair, missing, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    assert after.distribution.means.shape == (1, 12, 4, 2)
    assert not torch.equal(before.distribution.means, after.distribution.means)
    assert torch.linalg.eigvalsh(after.distribution.covariance).min() > 0
    changed_context = replace(inputs, pose_uv=inputs.pose_uv + 1, court_uv=inputs.court_uv + 1)
    context = predict_sequence(pair, changed_context, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    assert not torch.equal(before.distribution.means, context.distribution.means)


@pytest.mark.parametrize("window,stride,batch_size", [(13, 3, 2), (5, 6, 2), (5, 3, 0), (True, 1, 2), (5, 3, 2.5)])
def test_invalid_or_padded_window_policy_is_rejected(sequence, window, stride, batch_size):
    pair, inputs = sequence
    with pytest.raises(ValueError):
        predict_sequence(pair, inputs, window_length=window, stride=stride, batch_size=batch_size, device=torch.device("cpu"))


def test_global_timeline_discontinuity_is_not_hidden_between_disjoint_windows(sequence):
    pair, inputs = sequence
    times = inputs.timestamps_seconds.clone()
    times[:, 6:] = times[:, 6:] - times[:, 6:7]
    with pytest.raises(ValueError, match="strictly increasing"):
        predict_sequence(pair, replace(inputs, timestamps_seconds=times), window_length=6, stride=6,
                         batch_size=2, device=torch.device("cpu"))


def test_failed_prediction_restores_caller_model_mode(sequence, monkeypatch):
    pair, inputs = sequence

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate inference failure")

    monkeypatch.setattr(pair.model, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    assert pair.model.training


def test_forged_window_provenance_is_rejected(sequence):
    pair, inputs = sequence
    result = predict_sequence(pair, inputs, window_length=5, stride=3, batch_size=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="provenance"):
        replace(result, time_index=result.time_index + 1)
    with pytest.raises(ValueError, match="provenance"):
        replace(result, window_start=result.window_start.astype(np.int32))


def test_multiple_cameras_are_not_flattened_into_one_timeline(sequence):
    pair, inputs = sequence
    with pytest.raises(ValueError, match="one camera"):
        predict_sequence(pair, collate_inputs([inputs, inputs]), window_length=5, stride=3,
                         batch_size=2, device=torch.device("cpu"))
