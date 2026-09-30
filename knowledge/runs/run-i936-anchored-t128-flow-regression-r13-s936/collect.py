"""CPU collection of run 13, including every scheduled prediction and rule axis.

Run from the recorded checkout with PYTHONPATH=. and OMP/MKL/OPENBLAS threads=1.
Output is a new collected/ directory beside this script; no test arrays are read.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.baseline_comparison import (
    assert_same_metrics,
    comparison_markdown,
)
from src.tasks.ball_refiner.refiner_3d.baselines import evaluate_baselines
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_evaluation import RallyInput
from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import TrajectoryMetrics
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset


def rule(candidate: dict, reference: dict) -> dict:
    axes = {}
    for key in ('overall', 'gap', 'no_evidence', 'event_pm5'):
        name = 'rmse_m_' + key
        axes[name] = (candidate['metrics'][name]['value'], reference['metrics'][name]['value'])
    for cameras in ('0', '1', '2', '3'):
        axes['camera_' + cameras + '_rmse'] = tuple(r['by_visible_cameras'][cameras]['rmse_m_overall']['value'] for r in (candidate, reference))
    for metric in ('acceleration_all', 'acceleration_free_flight', 'jerk_all', 'jerk_free_flight'):
        axes[metric + '_p95'] = tuple(r['metrics'][metric]['p95'] for r in (candidate, reference))
    for quantile in ('mean', 'p50', 'p95'):
        axes['reprojection_' + quantile] = tuple(r['metrics']['reprojection_px_all'][quantile] for r in (candidate, reference))
    decisions = {}
    for key, (value, baseline) in axes.items():
        if value is None or baseline is None:
            raise ValueError('Primary rule has unsupported axis: ' + key)
        tolerance = 1e-6 + 1e-6 * abs(baseline)
        decisions[key] = {'candidate': value, 'reference': baseline, 'tolerance': tolerance,
                          'status': 'worse' if value > baseline + tolerance else 'better' if value < baseline - tolerance else 'equal'}
    behind = candidate['metrics']['behind_all']['invalid_count']
    return {'pass': behind == 0 and all(r['status'] != 'worse' for r in decisions.values())
            and any(r['status'] == 'better' for r in decisions.values()),
            'behind_count': behind, 'axes': decisions}


def main() -> None:
    torch.set_num_threads(1)
    repo = Path.cwd()
    bundle = Path(__file__).resolve().parent
    output = bundle / 'collected'
    output.mkdir()
    job = '1790771078312809159_1781052_i936-anchored-t128-flow-regression-r13-s936-20260930'
    root = Path('/home/kamimura/projects/tennis-lab')
    source = root / 'outputs/ball_refiner/train/h-dev/r13-anchored-s936-t128-20k'
    queue = root / '.training_queue'
    training = json.loads((source / 'manifest.json').read_text())
    assert training['status'] == 'complete' and (queue / 'done' / (job + '.job')).is_file()
    dataset = SyntheticDataset(Path(training['dataset']))
    assert sha256(dataset.directory / 'manifest.json') == training['source_manifest_sha256']
    assert sha256(repo / 'src/tasks/ball_refiner/refiner_3d/training_dev_anchored_t128.yaml') == training['config_sha256']
    records = sorted((r for r in dataset.records if r['split'] == 'val'), key=lambda r: r['rally_id'])
    expected = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies'] if r['rally_id'].startswith('val-')}
    assert {r['rally_id']: r['npz_sha256'] for r in records} == expected and len(records) == 16
    rallies = [RallyInput(r, dataset.load(r)) for r in records]
    baselines = evaluate_baselines(rallies, predictions=output / 'baseline_predictions')
    assert_same_metrics(baselines['methods'], training['baselines']['methods'])
    prior = json.loads((repo / 'knowledge/runs/run-i936-anchored-dev-comparison-r12-s936/anchored/manifest.json').read_text())
    assert_same_metrics(baselines['methods'], prior['baselines']['methods'])
    methods = baselines['methods']
    fields = ('positions_3d_m', 'timestamps_seconds', 'occlusion_mask', 'out_of_frame_mask',
              'event_region_mask', 'free_flight_mask', 'camera_true_K', 'camera_true_R', 'camera_true_t')
    for arm in ('flow', 'regression'):
        original = training['arms'][arm]
        assert original['status'] == 'complete' and original['updates'] == 20000
        assert sha256(source / arm / 'dev-only.pt') == original['checkpoint_sha256']
        assert sha256(source / arm / 'initial-state.pt') == original['initial_state_sha256']
        assert [r['update'] for r in original['validation']] == [0, 2000, 5000, 10000, 20000]
        for row in original['validation']:
            prediction_dir = source / arm / 'predictions' / f"update-{row['update']:05d}"
            assert {p.stem for p in prediction_dir.glob('*.npz')} == set(expected)
            accumulators = {kind: TrajectoryMetrics() for kind in ('mean', 'samples')}
            for rally in rallies:
                with np.load(prediction_dir / (rally.record['rally_id'] + '.npz'), allow_pickle=False) as saved:
                    for key in fields:
                        np.testing.assert_array_equal(saved[key], rally.arrays[key])
                    np.testing.assert_array_equal(saved['mean_m'], saved['samples_m'].mean(0))
                    accumulators['mean'].add(saved['mean_m'][None], rally.arrays)
                    accumulators['samples'].add(saved['samples_m'], rally.arrays)
            for kind, accumulator in accumulators.items():
                value = accumulator.summarize()
                assert_same_metrics(value['metrics'], row['metrics'][kind])
                assert_same_metrics(value['by_visible_cameras'], row['by_visible_cameras'][kind])
                methods[f"{arm}_{row['update']:05d}_{kind}"] = value
        shutil.copytree(source / arm / 'predictions/update-20000', output / arm / 'predictions')
        shutil.copy2(source / arm / 'curves.png', output / arm / 'curves.png')
    stored = json.loads((source / 'comparison.json').read_text())
    assert_same_metrics(methods, stored['methods'])
    assert comparison_markdown(methods) == (source / 'comparison.md').read_text()
    rules = {candidate: {reference: rule(methods[candidate], methods[reference])
                        for reference in ('mixture_mean', 'mixture_mean_rts', 'regression_20000_mean') if candidate != reference}
             for candidate in ('flow_20000_mean', 'flow_20000_samples', 'regression_20000_mean')}
    for name in ('manifest.json', 'comparison.json', 'comparison.md', 'baselines.json'):
        shutil.copy2(source / name, output / name)
    shutil.copytree(queue / 'repro' / job, output / 'repro')
    shutil.copy2(queue / 'done' / (job + '.job'), output / 'done.job')
    shutil.copy2(queue / 'logs' / (job + '.log'), output / 'queue.log')
    hashes = {str(p.relative_to(source)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
              for p in sorted(source.rglob('*')) if p.is_file()}
    report = {'collected_at': datetime.now().astimezone().isoformat(), 'job': job, 'state': 'done',
              'output': str(source), 'resources': training['resources'], 'output_bytes_actual': sum(v['bytes'] for v in hashes.values()),
              'source_sha256': hashes, 'val_rally_sha256': expected, 'test_arrays_read': 0,
              'all_160_prediction_files_recomputed': True, 'all_metrics_match': True,
              'run12_baselines_match': True, 'frames': sum(r.record['frames'] for r in rallies),
              'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5911005227',
              'primary_update': 20000, 'rule_result': rules}
    (output / 'collection.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('state', 'resources', 'output_bytes_actual', 'all_metrics_match', 'run12_baselines_match', 'frames')}, indent=2))
    print({c: {b: r['pass'] for b, r in refs.items()} for c, refs in rules.items()})


if __name__ == '__main__':
    main()
