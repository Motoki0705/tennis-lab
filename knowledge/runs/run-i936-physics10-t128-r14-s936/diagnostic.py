"""Apply the preregistered physics rule to fixed 20k saved predictions on CPU."""
from __future__ import annotations

import json
import runpy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import Array
from src.tasks.ball_refiner.refiner_3d.diffusion.roughness import (
    magnitude_summary,
    stencil_masks,
    window_owners,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256


def roughness(source: Path, *, scheduled: bool) -> dict[str, Any]:
    manifest = json.loads((source / 'manifest.json').read_text())
    derivatives: dict[str, list[Array]] = defaultdict(list)
    hashes = {}
    windows = manifest['arms']['flow']['validation'][-1]['windows']
    assert windows == manifest['arms']['regression']['validation'][-1]['windows']
    for rally_id, spans in sorted(windows.items()):
        for arm in ('flow', 'regression'):
            directory = source / arm / 'predictions'
            if scheduled:
                directory /= 'update-20000'
            path = directory / (rally_id + '.npz')
            hashes[str(path.relative_to(source))] = sha256(path)
            with np.load(path, allow_pickle=False) as saved:
                free = saved['free_flight_mask']
                owners = window_owners(spans, len(free))
                dt = float(np.diff(saved['timestamps_seconds'])[0])
                for kind in ('mean', 'samples') if arm == 'flow' else ('mean',):
                    prediction = saved['mean_m'][None] if kind == 'mean' else saved['samples_m']
                    for order, label in ((2, 'acceleration'), (3, 'jerk')):
                        values = np.diff(prediction.astype(np.float64), n=order, axis=1) / dt**order
                        for support, mask in stencil_masks(free, owners, order).items():
                            derivatives[f'{arm}_{kind}/{label}/{support}'].append(values[:, mask].reshape(-1, 3))
    return {'prediction_sha256': hashes, 'derivative': {
        key: magnitude_summary(np.concatenate(parts)) for key, parts in derivatives.items()}}


def main() -> None:
    bundle = Path(__file__).resolve().parent
    source = bundle / 'collected'
    control = bundle.parent / 'run-i936-anchored-t128-flow-regression-r13-s936/collected'
    current = json.loads((source / 'comparison.json').read_text())['methods']
    prior = json.loads((control / 'comparison.json').read_text())['methods']
    before = roughness(control, scheduled=False)
    after = roughness(source, scheduled=True)
    registered = json.loads((bundle / 'plan.json').read_text())['diagnostic_rule']
    formal_rule = runpy.run_path(str(bundle / 'collect.py'))['rule']
    old_diagnosis = json.loads((bundle.parent / 'run-i936-roughness-r14-s936/diagnosis.json').read_text())
    results: dict[str, Any] = {}
    for name in ('flow_mean', 'flow_samples', 'regression_mean'):
        arm, kind = name.split('_')
        method = f'{arm}_20000_{kind}'
        # Reuse the exact 15 formal axes; diagnostic allows 5% on 8 RMSE + 3 reprojection axes.
        formal = formal_rule(current[method], prior[method])
        axes = {}
        for key, row in formal['axes'].items():
            if key.startswith(('acceleration_', 'jerk_')):
                continue
            limit = registered['rmse_8_axes_and_reprojection_3_axes_ratio_max'] * row['reference']
            axes[key] = {**row, 'maximum_ratio': 1.05, 'ratio': row['candidate'] / row['reference'],
                         'pass': row['candidate'] <= limit + row['tolerance']}
        assert len(axes) == 11
        old_name = 'regression' if name == 'regression_mean' else name
        for label in ('acceleration', 'jerk'):
            for support in ('free', 'inside_free'):
                key = f'{name}/{label}/{support}'
                candidate = after['derivative'][key]
                reference = before['derivative'][key]
                assert candidate['count'] == reference['count']
                np.testing.assert_allclose(reference['p95'], old_diagnosis['derivative'][f'{old_name}/{label}/{support}']['p95'], rtol=1e-12)
                if support == 'free':
                    for observed, metrics in ((candidate, current), (reference, prior)):
                        np.testing.assert_allclose(observed['p95'], metrics[method]['metrics'][label + '_free_flight']['p95'], rtol=1e-6)
                tolerance = registered['abs_tolerance'] + registered['relative_tolerance'] * abs(reference['p95'])
                limit = registered['free_acceleration_and_jerk_full_and_inside_ratio_max'] * reference['p95']
                axes[f'{label}_{support}_p95'] = {
                    'candidate': candidate['p95'], 'reference': reference['p95'], 'count': candidate['count'],
                    'ratio': candidate['p95'] / reference['p95'], 'maximum_ratio': 0.75,
                    'pass': candidate['p95'] <= limit + tolerance}
        behind = {key: table[method]['metrics']['behind_all'] for key, table in (('candidate', current), ('reference', prior))}
        assert behind['candidate']['count'] == behind['reference']['count']
        behind['pass'] = behind['candidate']['invalid_count'] <= behind['reference']['invalid_count']
        results[name] = {'pass': all(row['pass'] for row in axes.values()) and behind['pass'], 'axes': axes, 'behind': behind}
    report = {'primary_update': 20000, 'test_arrays_read': 0,
              'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5915164321',
              'registered_rule': registered, 'control': before, 'candidate': after, 'results': results,
              'flow_pass': results['flow_mean']['pass'] and results['flow_samples']['pass'],
              'regression_pass': results['regression_mean']['pass'],
              'common_improvement': all(row['pass'] for row in results.values())}
    (source / 'diagnostic-rule.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, result in results.items():
        print(name, 'pass=', result['pass'], 'failed_axes=', [key for key, row in result['axes'].items() if not row['pass']], 'behind=', result['behind'])


if __name__ == '__main__':
    main()
