"""Candidate precision, missing evidence, gradients and legacy config compatibility."""

from dataclasses import asdict, replace

import pytest
import torch

from src.tasks.ball_refiner.refiner_2d import (
    Refiner2DTarget,
    build_ball_refiner_2d,
    refiner_2d_nll,
)
from src.tasks.ball_refiner.refiner_2d.config import (
    CandidateAnchoredConfig,
    Refiner2DConfig,
    parse_model_config,
)
from src.tasks.ball_refiner.refiner_2d.mean_anchors import candidate_mean_logits
from tests.integration.tasks.ball_refiner.test_refiner_2d import config, inputs


def anchored_config() -> CandidateAnchoredConfig:
    return CandidateAnchoredConfig(**asdict(config(use_pose=False, use_court=False)),
                                   mean_parameterization='candidate_residual_v1', anchored_components=2, max_offset_uv=.02)


def test_complete_schema_roundtrip_preserves_legacy_without_defaults() -> None:
    legacy, anchored = config(), anchored_config()
    assert type(parse_model_config(asdict(legacy))) is Refiner2DConfig
    assert parse_model_config(asdict(anchored)) == anchored
    for missing in ('max_offset_uv', 'mean_parameterization', 'use_court'):
        values = asdict(anchored)
        del values[missing]
        with pytest.raises(TypeError):
            parse_model_config(values)
    with pytest.raises(ValueError, match='mean_parameterization'):
        replace(anchored, mean_parameterization='typo')
    with pytest.raises(ValueError, match='free component'):
        replace(anchored, anchored_components=anchored.components)


def test_initial_means_keep_subpixel_peaks_and_state_dict_restores() -> None:
    cfg, batch = anchored_config(), inputs()
    pair = build_ball_refiner_2d(cfg)
    pair.model.eval()
    predicted = pair.run(batch)
    order = batch.candidates.scores.argsort(dim=-1, descending=True)[..., :2]
    expected = batch.candidates.coords.gather(-2, order[..., None].expand(-1, -1, -1, 2))
    torch.testing.assert_close(predicted.means[..., :2, :], expected)
    restored = build_ball_refiner_2d(parse_model_config(asdict(cfg)))
    restored.model.load_state_dict(pair.model.state_dict(), strict=True)
    restored.model.eval()
    torch.testing.assert_close(restored.run(batch).means, predicted.means, rtol=0, atol=0)


def test_missing_means_are_absolute_and_invalid_coordinates_do_not_anchor() -> None:
    raw = torch.tensor([[[[-2., 2.], [1., 0.], [3., -3.]]]], requires_grad=True)
    features = torch.tensor([[[[.23, .71, .8], [.9, .1, .9]]]])
    empty = torch.zeros(1, 1, 2, dtype=torch.bool)
    torch.testing.assert_close(candidate_mean_logits(raw, features, empty, count=2, max_offset_uv=.02), raw)
    valid = torch.tensor([[[True, False]]])
    result = candidate_mean_logits(raw, features, valid, count=2, max_offset_uv=.02).sigmoid()
    torch.testing.assert_close(result[..., 0, :], features[..., 0, :2] + .02 * raw[..., 0, :].tanh())
    torch.testing.assert_close(result[..., 1:, :], raw[..., 1:, :].sigmoid())
    result.sum().backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad != 0).all()


def test_tied_candidates_permute_without_changing_internal_component_assignment() -> None:
    raw = torch.zeros(1, 1, 3, 2)
    features = torch.tensor([[[[.7, .4, .8], [.3, .4, .8], [.5, .6, .9]]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    first = candidate_mean_logits(raw, features, valid, count=2, max_offset_uv=.02)
    second = candidate_mean_logits(raw, features.flip(-2), valid.flip(-1), count=2, max_offset_uv=.02)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first.sigmoid()[0, 0, :2], torch.tensor([[.5, .6], [.3, .4]]))


def test_anchored_model_has_finite_nll_gradients_and_free_component_can_leave_peaks() -> None:
    cfg, batch = anchored_config(), inputs()
    pair = build_ball_refiner_2d(cfg)
    target = Refiner2DTarget(uv=torch.rand(2, 5, 2), position_valid=torch.ones(2, 5, dtype=torch.bool),
                            presence=torch.ones(2, 5, dtype=torch.bool), presence_valid=torch.ones(2, 5, dtype=torch.bool),
                            weight=torch.ones(2, 5))
    refiner_2d_nll(pair.run(batch), target).loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in pair.model.parameters() if p.grad is not None)
    assert pair.model.head.weight.grad is not None
    assert pair.model.head.weight.grad[:2].abs().sum() > 0
    raw = torch.tensor([[[[100., -100.], [0., 0.], [5., 5.]]]])
    out = candidate_mean_logits(raw, torch.tensor([[[[.2, .3, .9], [.4, .5, .8]]]]),
                                torch.ones(1, 1, 2, dtype=torch.bool), count=2, max_offset_uv=.02).sigmoid()
    torch.testing.assert_close(out[0, 0, 0], torch.tensor([.22, .28]))
    assert (out[0, 0, 2] > .99).all()


def test_anchored_model_gap_and_amp_preserve_distribution_contract() -> None:
    from src.tasks.ball_refiner.data.gaps import mask_detector_evidence

    pair, batch = build_ball_refiner_2d(anchored_config()), inputs()
    empty = mask_detector_evidence(batch, torch.ones(2, 5, dtype=torch.bool))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = pair.run(empty)
    assert result.means.dtype == torch.float32
    assert result.means.shape == (2, 5, 4, 2)
    assert torch.isfinite(result.means).all()
    assert torch.linalg.eigvalsh(result.covariance).min() > 0


def test_compile_captures_candidate_sort_and_residual_means() -> None:
    pair, batch = build_ball_refiner_2d(anchored_config()), inputs()
    pair.model.eval()
    call = pair.build_call(batch)
    expected = pair.model(*call.args)
    # Graph capture is CPU-only here; CUDA/Inductor performance belongs to the job.
    compiled = torch.compile(pair.model, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(*call.args), expected)
