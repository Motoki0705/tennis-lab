"""CPU collection of reprojection-weight x3 run 17, including every scheduled prediction and rule axis.

Run from the recorded checkout with PYTHONPATH=. and OMP/MKL/OPENBLAS threads=1.
Output is a new collected/ directory beside this script; no test arrays are read.
"""
from __future__ import annotations

import gzip
import json
import shutil
import resource
import time
from collections import defaultdict
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
from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import preflight
from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import TrajectoryMetrics
from src.tasks.ball_refiner.refiner_3d.diffusion.roughness import magnitude_summary, stencil_masks, window_owners
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset


import runpy

rule = runpy.run_path(str(Path(__file__).resolve().parent.parent / 'run-i936-physics10-t128-r14-s936/collect.py'))['rule']


def diagnose(candidate: dict, reference: dict, inside_candidate: dict, inside_reference: dict) -> dict:
    formal = rule(candidate, reference)
    axes = {}
    for name, decision in formal['axes'].items():
        ratio = .8 if name == 'reprojection_p50' else .9 if name == 'reprojection_mean' else 1.05
        value, baseline = decision['candidate'], decision['reference']
        limit = ratio * baseline
        axes[name] = dict(candidate=value, control=baseline, ratio=value / baseline,
                          ratio_limit=ratio, limit=limit, tolerance=1e-6 + 1e-6 * abs(baseline),
                          passed=value <= limit + 1e-6 + 1e-6 * abs(baseline))
    for name in ('acceleration', 'jerk'):
        value, baseline = inside_candidate[name]['p95'], inside_reference[name]['p95']
        axes[name + '_inside_free_p95'] = dict(candidate=value, control=baseline, ratio=value / baseline,
            ratio_limit=1.05, limit=1.05*baseline, tolerance=1e-6+1e-6*abs(baseline),
            passed=value <= 1.05*baseline+1e-6+1e-6*abs(baseline))
    behind = candidate['metrics']['behind_all']['invalid_count']
    behind_control = reference['metrics']['behind_all']['invalid_count']
    failures = [name for name, v in axes.items() if not v['passed']]
    if behind > behind_control:
        failures.append('behind_count')
    assert len(axes) == 17
    return dict(passed=not failures, failed_axes=failures, axes=axes,
                behind_count=behind, behind_control=behind_control)


def inside_roughness(directory: Path, training: dict, rallies: list) -> dict:
    values = defaultdict(list)
    for arm in ('flow', 'regression'):
        for rally in rallies:
            arrays, rid = rally.arrays, rally.record['rally_id']
            owners = window_owners(training['arms'][arm]['validation'][-1]['windows'][rid], rally.record['frames'])
            dt = float(np.diff(arrays['timestamps_seconds'])[0])
            with np.load(directory / arm / 'predictions/update-20000' / (rid + '.npz'), allow_pickle=False) as saved:
                for kind, value in (('mean', saved['mean_m'][None]), ('samples', saved['samples_m'])):
                    for order, name in ((2, 'acceleration'), (3, 'jerk')):
                        mask = stencil_masks(arrays['free_flight_mask'], owners, order)['inside_free']
                        vectors = np.diff(value.astype(np.float64), n=order, axis=1) / dt**order
                        values[(arm, kind, name)].append(vectors[:, mask].reshape(-1, 3))
    return {f'{arm}_20000_{kind}': {name: magnitude_summary(np.concatenate(values[(arm, kind, name)]))
              for name in ('acceleration', 'jerk')} for arm in ('flow', 'regression') for kind in ('mean', 'samples')}


def main() -> None:
    torch.set_num_threads(1)
    started = time.perf_counter()
    repo = Path.cwd()
    bundle = Path(__file__).resolve().parent
    output = bundle / 'collected'
    output.mkdir()
    job = '1790810853371946846_2694673_i936-repro3-512-physics10-r17-s936-20261001'
    root = Path('/home/kamimura/projects/tennis-lab')
    source = root / 'outputs/ball_refiner/train/h-dev/r17-repro3-512-physics10-s936-t128-20k'
    queue = root / '.training_queue'
    training = json.loads((source / 'manifest.json').read_text())
    assert training['status'] == 'complete' and (queue / 'done' / (job + '.job')).is_file()
    plan = json.loads((bundle / 'plan.json').read_text())
    saved_preflight = json.loads(Path(plan['preflight_output']).read_text())
    repeated_preflight = preflight(plan)
    assert repeated_preflight == {k: v for k, v in saved_preflight.items() if k not in ('plan_path', 'plan_sha256')}
    assert saved_preflight['plan_sha256'] == sha256(bundle / 'plan.json')
    shutil.copy2(plan['preflight_output'], output / 'preflight.json')
    control = json.loads(Path(plan['control_manifest']).read_text())
    assert training['windows'] == control['windows']
    assert training['read_rallies'] == control['read_rallies']
    dataset = SyntheticDataset(Path(training['dataset']))
    assert sha256(dataset.directory / 'manifest.json') == training['source_manifest_sha256']
    assert sha256(repo / 'src/tasks/ball_refiner/refiner_3d/training_pilot512_physics10_repro3_t128.yaml') == training['config_sha256']
    expected = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies'] if r['rally_id'].startswith('val-')}
    records = sorted((r for r in dataset.records if r['rally_id'] in expected), key=lambda r: r['rally_id'])
    assert {r['rally_id']: r['npz_sha256'] for r in records} == expected and len(records) == 16
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available >= 6*1024**3
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
        assert original['initial_state_sha256'] == control['arms'][arm]['initial_state_sha256']
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
        shutil.copytree(source / arm / 'predictions', output / arm / 'predictions')
        updates = [json.loads(line) for line in (source / arm / 'updates.jsonl').read_text().splitlines()]
        old_updates = [json.loads(line) for line in (Path(plan['control_manifest']).parent / arm / 'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in updates] == list(range(1, 20001))
        assert [(r['window_indices'], r['real_frames']) for r in updates] == [(r['window_indices'], r['real_frames']) for r in old_updates]
        with gzip.open(output / arm / 'updates.jsonl.gz', 'wb') as stream:
            stream.write((source / arm / 'updates.jsonl').read_bytes())
        assert_same_metrics(methods[f'{arm}_00000_mean'], {'metrics': control['arms'][arm]['validation'][0]['metrics']['mean'], 'by_visible_cameras': control['arms'][arm]['validation'][0]['by_visible_cameras']['mean']})
        shutil.copy2(source / arm / 'curves.png', output / arm / 'curves.png')
    stored = json.loads((source / 'comparison.json').read_text())
    assert_same_metrics(methods, stored['methods'])
    assert comparison_markdown(methods) == (source / 'comparison.md').read_text()
    rules = {candidate: {reference: rule(methods[candidate], methods[reference])
                        for reference in ('mixture_mean', 'mixture_mean_rts', 'regression_20000_mean') if candidate != reference}
             for candidate in ('flow_20000_mean', 'flow_20000_samples', 'regression_20000_mean')}
    control_dir = Path(plan['control_manifest']).parent
    control_methods = json.loads((control_dir / 'comparison.json').read_text())['methods']
    inside = inside_roughness(source, training, rallies)
    inside_control = inside_roughness(control_dir, control, rallies)
    diagnostic = {candidate: diagnose(methods[candidate], control_methods[candidate], inside[candidate], inside_control[candidate])
        for candidate in ('flow_20000_mean', 'flow_20000_samples', 'regression_20000_mean')}
    for name in ('manifest.json', 'comparison.json', 'comparison.md', 'baselines.json'):
        shutil.copy2(source / name, output / name)
    shutil.copytree(queue / 'repro' / job, output / 'repro')
    shutil.copy2(queue / 'done' / (job + '.job'), output / 'done.job')
    shutil.copy2(queue / 'logs' / (job + '.log'), output / 'queue.log')
    lifecycle = [line for line in (queue / 'worker.log').read_text(errors='replace').splitlines() if job in line]
    (output / 'worker-lifecycle.log').write_text('\n'.join(lifecycle) + '\n')
    repro = json.loads((output / 'repro/run.json').read_text())
    assert repro['commit'] == '4a882fcaf9a5e0e74121fb697d9ca1e20bfa0b3e' and repro['resource'] == 'all'
    assert not (output / 'repro/uncommitted.patch').read_text().strip()
    assert not (output / 'repro/git_status.txt').read_text().strip()
    hashes = {str(p.relative_to(source)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
              for p in sorted(source.rglob('*')) if p.is_file()}
    print('lifecycle', lifecycle)
    report = {'collected_at': datetime.now().astimezone().isoformat(), 'job': job, 'state': 'done',
              'output': str(source), 'resources': training['resources'], 'output_bytes_actual': sum(v['bytes'] for v in hashes.values()),
              'clock_note': 'Raw queue lifecycle stamps preserved separately; trainer monotonic duration reported without equating the clocks',
              'source_sha256': hashes, 'val_rally_sha256': expected, 'test_arrays_read': 0,
              'all_160_prediction_files_recomputed': True, 'all_metrics_match': True,
              'run12_baselines_match': True, 'preflight_recomputed_matches': True,
              'initial_weights_and_initial_mean_metrics_match_control': True, 'all_40000_training_windows_match_control': True,
              'frames': sum(r.record['frames'] for r in rallies),
              'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5911005227',
              'primary_update': 20000, 'rule_result': rules, 'diagnostic_rule_url': plan['rule_url'],
              'diagnostic': diagnostic, 'inside_free_roughness': {'candidate': inside, 'control': inside_control},
              'flow_diagnostic_pass': all(diagnostic[k]['passed'] for k in ('flow_20000_mean','flow_20000_samples')),
              'common_improvement': all(v['passed'] for v in diagnostic.values()),
              'pinned_files_verified': len(plan['pinned_files']), 'extra_val_arrays_read': 0,
              'cpu_resources': {'seconds': time.perf_counter()-started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
              'collection_code_sha256': sha256(Path(__file__))}
    (output / 'collection.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('state', 'resources', 'output_bytes_actual', 'all_metrics_match', 'run12_baselines_match', 'frames')}, indent=2))
    print({c: {b: r['pass'] for b, r in refs.items()} for c, refs in rules.items()})
    print({c: r['failed_axes'] for c,r in diagnostic.items()})


if __name__ == '__main__':
    main()
