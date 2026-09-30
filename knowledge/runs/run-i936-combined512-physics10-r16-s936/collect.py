"""CPU collection of combined 512-train/physics10 run 16, including every scheduled prediction and rule axis.

Run from the recorded checkout with PYTHONPATH=. and OMP/MKL/OPENBLAS threads=1.
Output is a new collected/ directory beside this script; no test arrays are read.
"""
from __future__ import annotations

import gzip
import json
import runpy
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
from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import preflight
from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import TrajectoryMetrics
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset

rule = runpy.run_path(str(Path(__file__).resolve().parent.parent / 'run-i936-physics10-t128-r14-s936/collect.py'))['rule']


def main() -> None:
    torch.set_num_threads(1)
    repo = Path.cwd()
    bundle = Path(__file__).resolve().parent
    output = bundle / 'collected'
    output.mkdir()
    job = '1790800194473949395_4059977_i936-combined512-physics10-r16-s936-20261001'
    root = Path('/home/kamimura/projects/tennis-lab')
    source = root / 'outputs/ball_refiner/train/h-dev/r16-combined512-physics10-s936-t128-20k'
    queue = root / '.training_queue'
    training = json.loads((source / 'manifest.json').read_text())
    assert training['status'] == 'complete' and (queue / 'done' / (job + '.job')).is_file()
    plan_path = bundle / 'plan.json'
    plan = json.loads(plan_path.read_text())
    saved_preflight = json.loads(Path(plan['preflight_output']).read_text())
    repeated_preflight = preflight(plan)
    assert repeated_preflight == {k: v for k, v in saved_preflight.items() if k not in ('plan_path', 'plan_sha256')}
    assert saved_preflight['plan_sha256'] == sha256(plan_path)
    shutil.copy2(plan['preflight_output'], output / 'preflight.json')
    control = json.loads(Path(plan['control_manifest']).read_text())
    assert len(training['windows']) > len(control['windows'])
    assert len(training['read_rallies']) == 528
    a = json.loads((Path(plan['measured_factor_references']['a']) / 'manifest.json').read_text())
    assert training['windows'] == a['windows']
    assert training['read_rallies'] == a['read_rallies']
    dataset = SyntheticDataset(Path(training['dataset']))
    assert sha256(dataset.directory / 'manifest.json') == training['source_manifest_sha256']
    assert sha256(repo / 'src/tasks/ball_refiner/refiner_3d/training_pilot512_physics10_t128.yaml') == training['config_sha256']
    expected = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies'] if r['rally_id'].startswith('val-')}
    records = sorted((r for r in dataset.records if r['rally_id'] in expected), key=lambda r: r['rally_id'])
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
        old_updates = [json.loads(line) for line in (source / 'flow' / 'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in updates] == list(range(1, 20001))
        assert [(r['window_indices'], r['real_frames']) for r in updates] == [(r['window_indices'], r['real_frames']) for r in old_updates]
        a_updates = [json.loads(line) for line in (Path(plan['measured_factor_references']['a']) / arm / 'updates.jsonl').read_text().splitlines()]
        assert [(r['window_indices'], r['real_frames']) for r in updates] == [(r['window_indices'], r['real_frames']) for r in a_updates]
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
    for name in ('manifest.json', 'comparison.json', 'comparison.md', 'baselines.json'):
        shutil.copy2(source / name, output / name)
    shutil.copytree(queue / 'repro' / job, output / 'repro')
    shutil.copy2(queue / 'done' / (job + '.job'), output / 'done.job')
    shutil.copy2(queue / 'logs' / (job + '.log'), output / 'queue.log')
    lifecycle = [line for line in (queue / 'worker.log').read_text(errors='replace').splitlines() if job in line]
    (output / 'worker-lifecycle.log').write_text('\n'.join(lifecycle) + '\n')
    assert not (output / 'repro/uncommitted.patch').read_text().strip()
    assert not (output / 'repro/git_status.txt').read_text().strip()
    hashes = {str(p.relative_to(source)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
              for p in sorted(source.rglob('*')) if p.is_file()}
    report = {'collected_at': datetime.now().astimezone().isoformat(), 'job': job, 'state': 'done',
              'output': str(source), 'resources': training['resources'], 'output_bytes_actual': sum(v['bytes'] for v in hashes.values()),
              'queue_wall_seconds': 2973,
              'clock_difference_seconds': training['resources']['elapsed_seconds'] - 2973,
              'clock_note': 'queue ISO wall stamps and trainer perf_counter disagree; both recorded, cause not established',
              'source_sha256': hashes, 'val_rally_sha256': expected, 'test_arrays_read': 0,
              'all_160_prediction_files_recomputed': True, 'all_metrics_match': True,
              'run12_baselines_match': True, 'preflight_recomputed_matches': True,
              'initial_weights_and_initial_mean_metrics_match_control': True, 'paired_arms_all_40000_windows_match_each_other': True, 'all_training_windows_match_a': True, 'training_windows_differ_from_64_control_by_design': True, 'unused_extra_val_arrays_read': 0,
              'frames': sum(r.record['frames'] for r in rallies),
              'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5911005227',
              'primary_update': 20000, 'rule_result': rules}
    (output / 'collection.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('state', 'resources', 'output_bytes_actual', 'all_metrics_match', 'run12_baselines_match', 'frames')}, indent=2))
    print({c: {b: r['pass'] for b, r in refs.items()} for c, refs in rules.items()})


if __name__ == '__main__':
    main()
