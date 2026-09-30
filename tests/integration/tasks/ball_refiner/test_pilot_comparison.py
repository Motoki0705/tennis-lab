"""CPU end-to-end paired metrics from real tiny checkpoints and native heatmaps."""

import json

import numpy as np
import pytest
import torch

import src.tasks.ball_refiner.evaluation.pilot_comparison as comparison
from src.tasks.ball_refiner.training.runner import run_training
from tests.integration.tasks.ball_refiner.test_training import config_for
from tests.integration.tasks.ball_refiner.test_training import (
    pilot_inputs as pilot_inputs,
)


def test_paired_comparison_writes_all_sources_and_preserves_fixed_inputs(pilot_inputs, tmp_path, monkeypatch):
    # This fixture uses generic group names; production selection still requires video_000/game9.
    monkeypatch.setattr(comparison, 'validation_clips', lambda store: tuple(r for r in store.clips if r.split == 'val'))
    training = run_training(config_for(pilot_inputs, tmp_path / 'training'))
    before = (training / 'best.json').read_bytes()
    settings = comparison.ComparisonSettings(levels=(.5, .9), samples=32, seed=1729, chunk_size=4, uniform_weight=.001)
    result = comparison.run_comparison(training, training, tmp_path / 'comparison', settings=settings, device=torch.device('cpu'))
    assert (training / 'best.json').read_bytes() == before
    state = json.loads((result / 'run_state.json').read_text())
    assert state == {'status': 'complete', 'clips': 6, 'files': 48}
    manifest = json.loads((result / 'manifest.json').read_text())
    assert all('/val/' in item['clip_id'] for item in manifest['artifacts'])
    metrics = json.loads((result / 'metrics.json').read_text())
    for key, value in metrics.items():
        if key.startswith('old_'):
            assert metrics['new_' + key[4:]] == value
    observed = metrics['new_refiner/meiji/observed/observed']
    assert observed['error_px_frames'] == 24
    assert observed['coverage_0.9_frames'] == 24
    detector = metrics['new_detector/meiji/observed/observed']
    assert detector['nll_px_frames'] == 24
    gap = metrics['new_detector/meiji/evidence_gap/observed']
    assert gap['mean_nll_uv'] == 0 and gap['coverage_0.9'] == 1
    assert gap['mean_error_px'] is None
    assert detector['mean_presence_nll'] is None
    empty = metrics['new_refiner/meiji/observed/no_instance_unknown']
    assert empty['frames'] == 2 and empty['nll_px_frames'] == 0
    artifacts = [item for item in manifest['artifacts'] if item['method'] == 'new_refiner']
    for item in artifacts:
        with np.load(result / item['path'], allow_pickle=False) as data:
            assert data['means'].shape[0] == item['frames']
            assert data['frame_index'].shape == data['target_reason'].shape
    with pytest.raises(FileExistsError):
        comparison.run_comparison(training, training, result, settings=settings, device=torch.device('cpu'))


def test_modified_selected_checkpoint_is_rejected(pilot_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'validation_clips', lambda store: tuple(r for r in store.clips if r.split == 'val'))
    training = run_training(config_for(pilot_inputs, tmp_path / 'training'))
    best = json.loads((training / 'best.json').read_text())
    best['checkpoint_sha256'] = '0' * 64
    (training / 'best.json').write_text(json.dumps(best))
    with pytest.raises(ValueError, match='identity'):
        comparison.load_pilot(training, torch.device('cpu'))
