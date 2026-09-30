"""Recompute stored dev predictions next to untrained baselines on matched val."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import evaluate_baselines
from .diffusion.dev_evaluation import RallyInput
from .diffusion.metrics import TrajectoryMetrics
from .synthetic.configuration import sha256
from .synthetic.dataset import SyntheticDataset
from .synthetic.generator import write_json


def assert_same_metrics(actual: Any, expected: Any) -> None:
    """A changed metric/support cannot silently replace the historical result."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Historical metric keys differ')
        for key in expected:
            assert_same_metrics(actual[key], expected[key])
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f'Historical metric differs: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'Historical support differs: {actual} != {expected}')


def comparison_markdown(methods: dict[str, Any]) -> str:
    def number(value: float | None) -> str:
        return 'N/A' if value is None else f'{value:.3f}'

    lines = ['# 同一validationラリーの軌道比較', '',
             'RMSEはframe加重の3D距離、再投影は画面内合成GTとの画素誤差。正depth条件付きの値にはbehind件数を併記する。',
             '加速度はm/s²、jerkはm/s³。freeは全差分stencilが自由飛行のもの。samplesは全標本をpoolし、最良標本を選ばない。', '',
             '| 方法 | N | RMSE m | gap m | no evidence m | event±5 m | accel p95 all/free | jerk p95 all/free | repro mean/p50/p95 px | behind |',
             '|---|---:|---:|---:|---:|---:|---|---|---|---:|']
    for name, result in methods.items():
        m = result['metrics']
        errors = [number(m['rmse_m_' + k]['value']) for k in ('overall', 'gap', 'no_evidence', 'event_pm5')]
        derivatives = [' / '.join(number(m[key + '_' + support]['p95']) for support in ('all', 'free_flight')) for key in ('acceleration', 'jerk')]
        repro = ' / '.join(number(m['reprojection_px_all'][key]) for key in ('mean', 'p50', 'p95'))
        behind = f"{m['behind_all']['invalid_count']}/{m['behind_all']['count']}"
        lines.append('| ' + ' | '.join([name, str(m['rmse_m_overall']['count']), *errors, *derivatives, repro, behind]) + ' |')
    lines.extend(['', '可視camera = occlusionもout_of_frameもないcamera数。存在確率とは別。',
                  '層別の差分は元の連続時系列で計算し、加速度は中央frame、jerkは左中央frameの層へ割り当てる。', '',
                  '| 可視camera数 | 方法 | N | RMSE m | gap m | event±5 m | accel free p95 | jerk free p95 | repro p95 px | behind |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for cameras in ('0', '1', '2', '3'):
        for name, result in methods.items():
            m = result['by_visible_cameras'][cameras]
            vals = [number(m['rmse_m_' + k]['value']) for k in ('overall', 'gap', 'event_pm5')]
            vals += [number(m[k]['p95']) for k in ('acceleration_free_flight', 'jerk_free_flight', 'reprojection_px_all')]
            behind = f"{m['behind_all']['invalid_count']}/{m['behind_all']['count']}"
            lines.append('| ' + ' | '.join([cameras, name, str(m['rmse_m_overall']['count']), *vals, behind]) + ' |')
    return '\n'.join(lines) + '\n'


def training_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    """Pool every scheduled validation next to baselines and identical GT support."""
    baselines = manifest['baselines']
    methods = dict(baselines['methods'])
    truth = methods['truth']
    for objective in ('flow', 'regression'):
        arm = manifest['arms'][objective]
        if arm['status'] != 'complete' or [r['update'] for r in arm['validation']] != list(manifest['config']['evaluate_updates']):
            raise ValueError('Comparison requires both complete scheduled learning curves')
        for row in arm['validation']:
            if row['frames'] != baselines['frames']:
                raise ValueError('Model and baseline frame counts differ')
            assert_same_metrics(row['metrics']['truth'], truth['metrics'])
            assert_same_metrics(row['by_visible_cameras']['truth'], truth['by_visible_cameras'])
            for kind in ('mean', 'samples'):
                methods[f"{objective}_{row['update']:05d}_{kind}"] = {
                    'metrics': row['metrics'][kind],
                    'by_visible_cameras': row['by_visible_cameras'][kind],
                }
    return {'methods': methods, 'frames': baselines['frames'], 'rallies': baselines['rallies'],
            'read_rallies': [r for r in manifest['read_rallies'] if r['rally_id'].startswith('val-')],
            'source_manifest_sha256': manifest['source_manifest_sha256'],
            'evaluate_updates': manifest['config']['evaluate_updates'],
            'primary_update': manifest['config']['updates'],
            'selection': 'final update primary; every scheduled update reported; no best checkpoint/sample selection'}


def compare_saved_dev(dataset: Path, training_output: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    training = json.loads((training_output / 'manifest.json').read_text())
    if training['status'] != 'complete' or training['source_manifest_sha256'] != sha256(dataset / 'manifest.json'):
        raise ValueError('Need completed dev training on the identical dataset')
    source = SyntheticDataset(dataset)
    records = sorted((r for r in source.records if r['split'] == 'val'), key=lambda r: r['rally_id'])
    if len(records) != training['config']['expected_counts']['val']:
        raise ValueError('Validation count mismatch')
    training_hashes = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies'] if r['rally_id'].startswith('val-')}
    if {r['rally_id']: r['npz_sha256'] for r in records} != training_hashes:
        raise ValueError('Validation identity/hash mismatch')
    rallies = [RallyInput(r, source.load(r)) for r in records]  # Never open train or test.
    output.mkdir(parents=True)
    result: dict[str, Any] = evaluate_baselines(rallies, predictions=output / 'predictions')
    methods = result['methods']
    for arm in ('flow', 'regression'):
        accumulated = {kind: TrajectoryMetrics() for kind in ('mean', 'samples')}
        directory = training_output / arm / 'predictions'
        if {p.stem for p in directory.glob('*.npz')} != {r['rally_id'] for r in records}:
            raise ValueError('Require exactly the saved validation prediction set')
        for rally in rallies:
            with np.load(directory / (rally.record['rally_id'] + '.npz'), allow_pickle=False) as saved:
                for field in ('positions_3d_m', 'timestamps_seconds', 'occlusion_mask', 'out_of_frame_mask',
                              'event_region_mask', 'free_flight_mask', 'camera_true_K', 'camera_true_R', 'camera_true_t'):
                    if not np.array_equal(saved[field], rally.arrays[field]):
                        raise ValueError(f'Prediction/GT alignment mismatch: {field}')
                if not np.array_equal(saved['mean_m'], saved['samples_m'].mean(0)):
                    raise ValueError('Stored mean is not the sample mean')
                accumulated['mean'].add(saved['mean_m'][None], rally.arrays)
                accumulated['samples'].add(saved['samples_m'], rally.arrays)
        for kind, values in accumulated.items():
            summary = values.summarize()
            assert_same_metrics(summary['metrics'], training['arms'][arm]['validation'][-1]['metrics'][kind])
            methods[arm + '_' + kind] = summary
    result.update(dataset=str(dataset), source_manifest_sha256=sha256(dataset / 'manifest.json'),
                  training_output=str(training_output), training_manifest_sha256=sha256(training_output / 'manifest.json'),
                  read_rallies=training_hashes, historical_metrics_match=True)
    write_json(output / 'comparison.json', result)
    (output / 'comparison.md').write_text(comparison_markdown(methods))
    return result
