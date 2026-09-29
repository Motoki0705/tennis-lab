"""CPU JPEG/checkpoint comparison, exact cache parity and validation isolation."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.tasks.ball_refiner.evaluation.detector_selection as selection
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_refiner.data.cache_identity import store_hashes
from src.tasks.ball_refiner.data.evidence_inference import (
    WINDOW_SELECTION,
    infer_clip_evidence,
)
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.mark.parametrize('requested,index', [('cuda', 2), ('cuda:1', 1)])
def test_cuda_budget_uses_explicit_device_index_without_a_gpu(monkeypatch, requested, index):
    calls = []
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)

    def properties(device):
        assert device == torch.device('cuda', index)
        return SimpleNamespace(total_memory=16 * 2**30)

    def fraction(value, device):
        assert device.index is not None  # torch rejects bare torch.device('cuda') here
        calls.append((value, device))

    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', fraction)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append(('reset', device)))
    assert selection.configure_comparison_device(requested, 5) == torch.device('cuda', index)
    assert calls == [(5 / 16, torch.device('cuda', index)), ('reset', torch.device('cuda', index))]


def test_cpu_budget_does_not_initialize_cuda(monkeypatch):
    def fail():
        raise AssertionError('CPU comparison must not initialize CUDA')

    monkeypatch.setattr(torch.cuda, 'current_device', fail)
    assert selection.configure_comparison_device('cpu', 5) == torch.device('cpu')


@pytest.fixture
def comparison_inputs(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = OmegaConf.create({
        'model': {'name': 'conv_next_unet', 'input_mode': 'rgb', 'in_channels': 3, 'num_classes': 1,
                  'num_frames': 4, 'input_layout': 'bcthw', 'dims': [4, 8, 16, 32], 'depth': 1,
                  'drop_path_prob': 0.0, 'mdd_a': .2, 'mdd_b': .15},
        'data': {'image_size': [32, 32], 'augmentation': {'normalize_imagenet': {'enabled': False}}},
    })
    pair = build_ball_detection_pair(cfg)
    checkpoint = tmp_path / 'detector.ckpt'
    torch.save({'hyper_parameters': {'config': OmegaConf.to_container(cfg)},
                'state_dict': {f'model.{key}': value for key, value in pair.model.state_dict().items()}}, checkpoint)
    directory = tmp_path / 'store'
    for source, group, split in [('meiji', 'video_000', 'val'), ('tracknet', 'game9', 'val'),
                                 ('chat_annotation', 'youtube1', 'val'), ('meiji', 'video_001', 'test')]:
        write_store_clip(directory, f'{source}/{group}/clip1', [frame(i, ball()) if i != 3 else frame(i) for i in range(9)],
                         source=source, split=split)
    path = directory / 'metadata.json'
    data = json.loads(path.read_text())
    for clip in data['clips']:
        clip['group_id'] = clip['clip_id'].split('/')[1]
        clip['camera_id'] = 'cam0' if clip['source'] == 'meiji' else None
        clip.update(source_width=128, source_height=96)
    path.write_text(json.dumps(data))
    store = BallFrameStore(directory)
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({
        'schema': selection.EVIDENCE_SCHEMA, 'status': 'complete', 'store': {'sha256': store_hashes(directory)},
        'selection': {'clip_ids': [c.clip_id for c in store.split_clips('val')]},
        'detector': {'candidates': asdict(BallCandidateConfig(max_candidates=8, nms_kernel=5, patch_size=5)),
                     'subpixel_refine': True, 'window_selection': WINDOW_SELECTION,
                     'tail_policy': 'backfill_real_frames_no_padding', 'window_length': 4, 'stride': 2, 'image_size_hw': [32, 32]},
    }))
    yield dict(store_directory=directory, cache_manifest=reference,
               checkpoints=tuple(selection.SelectionCheckpoint(name, checkpoint, dual_sha256(checkpoint)) for name in selection.CHECKPOINT_NAMES),
               output=tmp_path / 'comparison', device='cpu', batch_size=2, cuda_allocator_limit_gib=6)
    torch.set_num_threads(previous)


def test_real_comparison_matches_cache_inference_and_excludes_test(comparison_inputs, monkeypatch):
    args = comparison_inputs
    store = BallFrameStore(args['store_directory'])
    original = selection.infer_clip_evidence

    def validate_split(store, clip, predictor, **kwargs):
        assert clip.split == 'val'
        assert 'video_001' not in clip.clip_id
        return original(store, clip, predictor, **kwargs)

    monkeypatch.setattr(selection, 'infer_clip_evidence', validate_split)
    output = selection.compare_detectors(**args)
    result = json.loads((output / 'manifest.json').read_text())
    assert result['status'] == 'complete'
    assert result['decision']['winner'] == 'ft-e13'  # identical models use predeclared tie break
    assert not result['decision']['mixed_e11_beats_e0']
    assert len(result['results']) == 3
    first = result['results'][0]
    assert set(first['groups']) == {'meiji', 'meiji/cam0', 'tracknet', 'chat_annotation'}
    assert all(group['observed'] == 8 for group in first['groups'].values())
    clip = store.clips[0]
    loaded = load_ball_checkpoint(args['checkpoints'][0].path)
    expected = infer_clip_evidence(store, clip, BallDetectionPredictor(loaded.model_io, torch.device('cpu'), subpixel_refine=True,
                                   image_normalization=loaded.image_normalization), image_size_hw=(32, 32), stride=2, batch_size=2,
                                   config=BallCandidateConfig(max_candidates=8, nms_kernel=5, patch_size=5))
    targets = project_store_targets(store, clip)
    scale = np.array([127, 95], np.float32)
    record = first['clips'][0]
    assert dual_sha256(output / record['file']) == record['sha256']
    with np.load(output / record['file']) as saved:
        np.testing.assert_array_equal(saved['candidate_xy_source_px'], expected.candidates.coords[0].numpy() * scale)
        np.testing.assert_array_equal(saved['target_xy_source_px'], targets.uv * scale)
        np.testing.assert_array_equal(saved['window_start'], expected.window_start)
        np.testing.assert_array_equal(saved['pts'], targets.pts)
    assert 'recall@8' in (output / 'comparison.md').read_text()
    with pytest.raises(FileExistsError):
        selection.compare_detectors(**args)


@pytest.mark.parametrize('change', ['test_group', 'checkpoint', 'window', 'partial'])
def test_preflight_rejects_wrong_split_hash_or_cache_contract(comparison_inputs, change):
    args = comparison_inputs
    if change == 'test_group':
        path = args['store_directory'] / 'metadata.json'
        data = json.loads(path.read_text())
        data['clips'][0]['group_id'] = 'video_001'
        path.write_text(json.dumps(data))
    elif change == 'checkpoint':
        args['checkpoints'] = (replace(args['checkpoints'][0], sha256='0' * 64), *args['checkpoints'][1:])
    else:
        path = args['cache_manifest']
        data = json.loads(path.read_text())
        if change == 'window':
            data['detector']['window_selection'] = 'maximum_score'
        else:
            data['status'] = 'building'
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        selection.compare_detectors(**args)
    assert not args['output'].exists()


def test_source_weighting_cannot_change_primary_winner():
    results: list[dict[str, Any]] = [{'name': name, 'groups': {'meiji': {'observed': 100, 'recalled_at_k': hit},
                                       'tracknet': {'observed': 100000, 'recalled_at_k': 100000 - hit}}}
               for name, hit in zip(selection.CHECKPOINT_NAMES, [50, 60, 70], strict=True)]
    decision = selection.select_winner(results)
    assert decision['winner'] == 'mixed-e11'
    assert decision['mixed_e11_beats_e0']
    assert decision['evidence_cache_rebuild_required']
    results[2]['groups']['meiji']['observed'] = 99
    with pytest.raises(ValueError, match='denominators'):
        selection.select_winner(results)


def test_summary_pools_observed_frames_instead_of_averaging_clip_rates():
    records = [{"clip": {"source": "meiji", "camera_id": "cam0"},
                "counts": {"frames": n, "observed": n, "recalled_at_1": hit, "recalled_at_k": hit,
                           "wrong_ranked_above_true": 0, "wrong_strictly_higher_score": 0}}
               for n, hit in [(1, 1), (9, 0)]]
    groups = selection.summarize_clips(records)
    assert groups["meiji"]["recall_at_k"] == .1
    assert groups["meiji/cam0"] == groups["meiji"]


def test_shard_mutation_leaves_partial_unselected_output(comparison_inputs, monkeypatch):
    original = selection.infer_clip_evidence

    def mutate(store, clip, predictor, **kwargs):
        evidence = original(store, clip, predictor, **kwargs)
        with (store.directory / 'shards' / f'clip-{clip.index:05d}.bin').open('ab') as stream:
            stream.write(b'changed')
        return evidence

    monkeypatch.setattr(selection, 'infer_clip_evidence', mutate)
    with pytest.raises(ValueError, match='JPEG shard changed'):
        selection.compare_detectors(**comparison_inputs)
    manifest = json.loads((comparison_inputs['output'] / 'manifest.json').read_text())
    assert manifest['status'] == 'running'
    assert 'decision' not in manifest
    assert not (comparison_inputs['output'] / 'comparison.md').exists()


def test_cli_dry_run_declares_paths_without_loading_model_or_creating_output(comparison_inputs):
    args = comparison_inputs
    command = [sys.executable, '-m', 'src.tasks.ball_refiner.scripts.compare_detectors', '--store', str(args['store_directory']),
               '--cache-manifest', str(args['cache_manifest']), '--output', str(args['output']), '--device', 'cuda',
               '--batch-size', '2', '--cuda-allocator-limit-gib', '6', '--cpu-threads', '1', '--dry-run']
    for checkpoint in args['checkpoints']:
        command += [f'--{checkpoint.name}', str(checkpoint.path)]
    command += ['--expected-sha256', *[c.sha256 for c in args['checkpoints']]]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    assert result['status'] == 'prepared'
    assert len(result['selection']['clips']) == 3
    assert not args['output'].exists()
