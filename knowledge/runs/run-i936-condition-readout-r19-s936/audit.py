"""Recompute read-out tables from saved predictions; never refit or run an encoder."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256


def describe(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.linalg.norm(prediction - target, axis=-1)
    assert np.isfinite(error).all()
    if not len(error):
        return dict(frames=0, rmse_m=None, p50_m=None, p95_m=None, maximum_m=None)
    return dict(frames=len(error), rmse_m=float(np.sqrt(np.mean(error**2))),
                p50_m=float(np.quantile(error, .5)), p95_m=float(np.quantile(error, .95)),
                maximum_m=float(error.max()))


def main() -> None:
    started = time.perf_counter()
    bundle = Path(__file__).resolve().parent
    plan = json.loads((bundle / 'plan.json').read_text())
    assert all(sha256(Path(path)) == digest for path, digest in plan['input_hashes'].items())
    training = json.loads((Path(plan['training_output']) / 'manifest.json').read_text())
    dataset = Path(plan['dataset'])
    source = json.loads((dataset / 'manifest.json').read_text())
    expected = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies']}
    records = sorted((r for r in source['rallies'] if r['rally_id'] in expected), key=lambda r: r['rally_id'])
    assert len(records) == 528
    assert sum(r['split'] == 'train' for r in records) == 512
    assert [r['rally_id'] for r in records if r['split'] == 'val'] == plan['validation_reference']['rallies']
    targets = {'train': [], 'val': []}
    visible = {'train': [], 'val': []}
    minimum_ram = 2**63 - 1
    for record in records:
        available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        minimum_ram = min(minimum_ram, available)
        assert available >= 6 * 1024**3
        path = dataset / (record['rally_id'] + '.npz')
        assert sha256(path) == expected[record['rally_id']]
        with np.load(path, allow_pickle=False) as stored:
            # Read only condition moments/masks: no truth coordinate access.
            means, weights = stored['gmm3d_means_m'], stored['gmm3d_weights']
            assert means.shape == (record['frames'], 125, 3)
            assert weights.shape == (record['frames'], 125)
            targets[record['split']].append((means.astype(np.float64) * weights.astype(np.float64)[..., None]).sum(-2))
            visible[record['split']].append((~(stored['occlusion_mask'] | stored['out_of_frame_mask'])).sum(0))
    target_arrays = {s: np.concatenate(a) for s, a in targets.items()}
    visible_arrays = {s: np.concatenate(a) for s, a in visible.items()}
    assert len(target_arrays['val']) == 6383
    table = ['| arm | split / visible cameras | frames | RMSE m | p50 m | p95 m | maximum m | ≤0.10 m (val all only) |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    report = {'status': 'complete', 'arms': {}, 'source_hashes_verified': len(plan['input_hashes']),
              'rally_hashes_verified': len(records), 'refits': 0, 'encoder_runs': 0,
              'extra_val_or_test_arrays_opened': 0}
    for arm in plan['arms']:
        directory = bundle / arm
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['encoder']['objective'] == arm and manifest['encoder']['updates'] == 20000
        assert manifest['encoder']['frozen']
        assert manifest['solver']['driver'] == 'gelsd' and manifest['solver']['ridge'] == 0
        assert manifest['solver']['rcond'] == 1e-12 and manifest['solver']['intercept']
        assert manifest['solver']['dtype'] == 'float64'
        assert [{k: r[k] for k in ('rally_id', 'npz_sha256')} for r in manifest['read_rallies']] == training['read_rallies']
        assert all(r['components'] == 125 for r in manifest['read_rallies'])
        assert manifest['unused_val_rallies'] == plan['validation_reference']['unused_val_rallies']
        for name, spec in manifest['artifacts'].items():
            assert sha256(directory / name) == spec['sha256']
            assert (directory / name).stat().st_size == spec['bytes']
        with np.load(directory / 'head.npz', allow_pickle=False) as layer:
            rank = int(layer['rank'])
            singular = layer['singular_values']
            coefficients = layer['coefficients']
            assert coefficients.dtype == np.float64 and coefficients.shape == (129, 3)
            assert np.isfinite(coefficients).all() and np.isfinite(singular).all()
            assert rank == manifest['solver']['rank'] == int((singular > singular[0] * 1e-12).sum())
            layer_summary = dict(rank=rank, columns=129, singular_values=singular.tolist(),
                                 condition_number=float(singular[0] / singular[-1]),
                                 coefficients_sha256=manifest['artifacts']['head.npz']['sha256'])
        statistics = {}
        for split in ('train', 'val'):
            with np.load(directory / (split + '.npz'), allow_pickle=False) as saved:
                target = saved['target_m']
                np.testing.assert_allclose(target, target_arrays[split], rtol=0, atol=1e-13)
                np.testing.assert_array_equal(saved['visible_cameras'], visible_arrays[split])
                prediction = saved['prediction_m']
                stats = {'overall': describe(prediction, target),
                         'by_visible_cameras': {str(i): describe(prediction[visible_arrays[split] == i],
                                                                target[visible_arrays[split] == i]) for i in range(4)}}
            for group, values in [('overall', stats['overall']), *stats['by_visible_cameras'].items()]:
                stored_stats = (manifest['results'][split]['overall'] if group == 'overall'
                                else manifest['results'][split]['by_visible_cameras'][group])
                for key, value in values.items():
                    np.testing.assert_allclose(value, stored_stats[key], rtol=1e-11, atol=1e-13)
                rule = ('PASS' if values['rmse_m'] <= .1 else 'FAIL') if (split, group) == ('val', 'overall') else '—'
                table.append(f"| {arm} | {split} / {group} | {values['frames']} | "
                             f"{values['rmse_m']:.9f} | {values['p50_m']:.9f} | {values['p95_m']:.9f} | "
                             f"{values['maximum_m']:.9f} | {rule} |")
            statistics[split] = stats
        passed = statistics['val']['overall']['rmse_m'] <= .1
        assert passed == manifest['meets_diagnostic_tolerance']
        report['arms'][arm] = dict(solver=layer_summary, results=statistics, passed=passed,
                                    resources=manifest['resources'])
    assert all(sha256(Path(path)) == digest for path, digest in plan['input_hashes'].items())
    report['resources'] = dict(elapsed_seconds=time.perf_counter() - started,
                               peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                               minimum_available_host_bytes=minimum_ram)
    (bundle / 'tables.md').write_text('\n'.join(table) + '\n')
    (bundle / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({a: {k:v for k,v in r.items() if k in ('passed', 'resources')}
                      for a, r in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
