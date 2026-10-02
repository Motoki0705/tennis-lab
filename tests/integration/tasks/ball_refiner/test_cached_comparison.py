"""Cached variants must reuse exact reference frames without detector inference."""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.tasks.ball_refiner.evaluation.cached_comparison as cached
import src.tasks.ball_refiner.evaluation.pilot_comparison as original
from src.tasks.ball_refiner.deployment import export_pilot_bundle, load_inference_bundle
from src.tasks.ball_refiner.refiner_2d.config import CandidateAnchoredConfig
from src.tasks.ball_refiner.training.runner import run_training
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_refiner.test_training import config_for
from tests.integration.tasks.ball_refiner.test_training import (
    pilot_inputs as pilot_inputs,
)


@pytest.mark.parametrize('training_seed', [42, 43])
def test_cached_evaluation_trains_restores_and_exports_candidate_variant(pilot_inputs, tmp_path, monkeypatch, training_seed):
    for module in (original, cached):
        monkeypatch.setattr(module, 'validation_clips', lambda store: tuple(r for r in store.clips if r.split == 'val'))
    baseline = run_training(config_for(pilot_inputs, tmp_path / 'baseline', model='comparison/absolute'))
    settings = original.ComparisonSettings(levels=(.5, .9), samples=32, seed=1729, chunk_size=4, uniform_weight=.001)
    reference = original.run_comparison(baseline, baseline, tmp_path / 'reference', settings=settings, device=torch.device('cpu'))
    cfg = config_for(pilot_inputs, tmp_path / 'anchored')
    OmegaConf.set_struct(cfg, False)
    cfg.model.mean_parameterization = 'candidate_residual_v1'
    cfg.model.anchored_components = 2
    cfg.model.max_offset_uv = .02
    cfg.run.seed = training_seed
    variant = run_training(cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError('A cached comparison must not run the detector')

    monkeypatch.setattr(original, 'infer_clip_evidence', forbidden)
    monkeypatch.setattr(original, 'load_ball_checkpoint', forbidden)
    result = cached.run_cached_comparison(variant, reference, tmp_path / 'result', settings=settings, device=torch.device('cpu'))
    assert json.loads((result / 'run_state.json').read_text()) == {'status': 'complete', 'clips': 6, 'files': 12}
    metrics = json.loads((result / 'metrics.json').read_text())
    old_metrics = json.loads((reference / 'metrics.json').read_text())
    for key, value in metrics.items():
        if key in old_metrics:
            assert {k: v for k, v in value.items() if k != 'p90_error_px'} == old_metrics[key]
    assert metrics['variant/meiji/observed/observed']['frames'] == 24
    assert metrics['variant/meiji/evidence_gap/observed']['frames'] < 24
    assert metrics['variant/meiji/observed/no_instance_unknown']['mean_nll_px'] is None
    manifest = json.loads((result / 'manifest.json').read_text())
    assert manifest['training_seed'] == training_seed
    assert manifest['reference_training_seed'] == 42
    bundle = export_pilot_bundle(variant, tmp_path / 'bundle')
    loaded = load_inference_bundle(bundle.directory)
    assert isinstance(loaded.model_config, CandidateAnchoredConfig)
    loaded.load_model()


def test_reference_reuse_rejects_changed_timeline_and_bytes(tmp_path):
    path = tmp_path / 'reference.npz'
    np.savez(path, pts=np.array([10, 20]))
    digest = dual_sha256(path)
    with pytest.raises(ValueError, match='identity'):
        cached.paired_reference(path, digest, {'pts': np.array([11, 20])})
    with pytest.raises(ValueError, match='hash'):
        cached.paired_reference(path, '0' * 64, {'pts': np.array([10, 20])})
