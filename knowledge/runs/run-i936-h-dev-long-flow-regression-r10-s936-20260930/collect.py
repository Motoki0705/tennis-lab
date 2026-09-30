"""Collect immutable r10 outputs; run from the active checkout with PYTHONPATH=."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import TrajectoryMetrics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source, previous = args.source, args.previous
    job = '1790745177452488394_4095209_i936-h-dev-long-flow-regression-r10-s936-20260930'
    job_file = args.queue / 'done' / (job + '.job')
    assert job_file.is_file()
    manifest = json.loads((source / 'manifest.json').read_text())
    old = json.loads((previous / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['source_manifest_sha256'] == old['source_manifest_sha256']
    assert manifest['read_rallies'] == old['read_rallies']
    assert len(manifest['read_rallies']) == 80
    assert not any(r['rally_id'].startswith('test-') for r in manifest['read_rallies'])
    assert manifest['windows'] == old['windows']
    train = {}
    curves = []
    for arm in ('flow', 'regression'):
        info = manifest['arms'][arm]
        assert info['status'] == 'complete' and info['updates'] == 20000
        assert info['initial_state_sha256'] == old['arms'][arm]['initial_state_sha256']
        assert sha(source / arm / 'initial-state.pt') == info['initial_state_sha256']
        assert sha(source / arm / 'dev-only.pt') == info['checkpoint_sha256']
        rows = [json.loads(s) for s in (source / arm / 'updates.jsonl').read_text().splitlines()]
        old_rows = [json.loads(s) for s in (previous / arm / 'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == list(range(1, 20001))
        assert len(old_rows) == 2000
        for a, b in zip(rows, old_rows, strict=False):
            assert all(a[k] == b[k] for k in ('window_indices', 'real_frames', 'loss', 'x0', 'reprojection', 'physics', 'event'))
        train[arm] = rows
        assert [r['update'] for r in info['validation']] == [0, 2000, 5000, 10000, 15000, 20000]
        for val in info['validation']:
            step = val['update']
            if step in (0, 2000):
                old_val = next(r for r in old['arms'][arm]['validation'] if r['update'] == step)
                assert val['loss'] == old_val['loss'] and val['metrics'] == old_val['metrics']
            metrics = {name: TrajectoryMetrics() for name in ('mean', 'samples', 'truth')}
            paths = sorted((source / arm / 'predictions' / f'update-{step:05d}').glob('*.npz'))
            assert len(paths) == 16
            for path in paths:
                with np.load(path, allow_pickle=False) as archive:
                    arrays = dict(archive)
                for name, values in (('mean', arrays['mean_m'][None]), ('samples', arrays['samples_m']), ('truth', arrays['positions_3d_m'][None])):
                    metrics[name].add(values, arrays)
            for name, accumulator in metrics.items():
                summary = accumulator.summarize()
                assert summary['metrics'] == val['metrics'][name]
                assert summary['by_visible_cameras'] == val['by_visible_cameras'][name]
            window = rows[max(0, step - 500):step]
            row = {'arm': arm, 'update': step, 'train_first_update': window[0]['update'] if window else None,
                   'train_updates': len(window), 'train_frames': sum(r['real_frames'] for r in window),
                   'val_frames': val['frames']}
            for key in ('loss', 'x0', 'reprojection', 'physics', 'event'):
                row['train_' + key] = float(np.average([r[key] for r in window], weights=[r['real_frames'] for r in window])) if window else None
                row['val_' + key] = val['loss'][key]
            row['val_rmse_m'] = val['metrics']['mean']['rmse_m_overall']['value']
            curves.append(row)
    assert all(a['window_indices'] == b['window_indices'] for a, b in zip(train['flow'], train['regression'], strict=True))
    files = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        keep = path.suffix != '.pt'
        files.append({'path': str(rel), 'bytes': path.stat().st_size, 'sha256': sha(path), 'promoted': keep})
        if keep:
            target = root / 'output' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2(args.queue / 'logs' / (job + '.log'), root / 'queue.log')
    collection = {'job': job, 'queue_state': 'done', 'queue_job_sha256': sha(job_file),
                  'source': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
                  'previous_manifest_sha256': sha(previous / 'manifest.json'),
                  'historical_repro_unchanged': True, 'same_initialization_and_all_window_orders': True,
                  'first_2000_train_losses_and_0_2000_val_exactly_equal_to_r9': True,
                  'all_saved_val_metrics_and_camera_strata_recomputed_exactly': True,
                  'test_rally_reads_in_training_manifest': 0, 'files': files,
                  'source_bytes': sum(r['bytes'] for r in files),
                  'promoted_output_bytes': sum(r['bytes'] for r in files if r['promoted']),
                  'train_aggregation': 'preceding 500 online pre-update batches, weighted by real frames; changing weights/noise, not a fixed train evaluation',
                  'val_aggregation': 'fixed checkpoint, whole rallies, frame-weighted loss; fixed noisy x_t/t for flow, not generated-sample RMSE',
                  'curves': curves}
    (root / 'collection.json').write_text(json.dumps(collection, indent=2, allow_nan=False) + '\n')
    with (root / 'train-val.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, arm in enumerate(('flow', 'regression')):
        values = [r for r in curves if r['arm'] == arm and r['update'] > 0]
        for j, key in enumerate(('x0', 'physics')):
            for split in ('train', 'val'):
                axes[i, j].plot([r['update'] for r in values], [r[split + '_' + key] for r in values], 'o-', label=split)
            axes[i, j].set(title=arm + ' / ' + key, xlabel='update', yscale='log')
            axes[i, j].legend()
        axes[i, 2].plot([r['update'] for r in values], [r['val_rmse_m'] for r in values], 'o-', label='val generated mean')
        for name in ('mixture_mean', 'mixture_mean_rts'):
            axes[i, 2].axhline(manifest['baselines']['methods'][name]['metrics']['rmse_m_overall']['value'], linestyle='--', label=name)
        axes[i, 2].set(title=arm + ' / validation RMSE', xlabel='update', ylabel='m')
        axes[i, 2].legend()
    fig.suptitle('Train: preceding 500 online batches; val: fixed whole-rally evaluation (different contexts)')
    fig.savefig(root / 'train-val.png', dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in collection.items() if k not in ('files', 'curves')}, indent=2))


if __name__ == '__main__':
    main()
