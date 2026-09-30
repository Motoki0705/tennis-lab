"""Apply the registered combined-candidate rule without choosing a checkpoint."""
from __future__ import annotations

import json
import runpy
from pathlib import Path


def main() -> None:
    bundle = Path(__file__).resolve().parent
    output = bundle / 'collected'
    plan = json.loads((bundle / 'plan.json').read_text())
    formal = runpy.run_path(str(bundle / 'collect.py'))['rule']
    roughness = runpy.run_path(str(bundle.parent / 'run-i936-physics10-t128-r14-s936/diagnostic.py'))['roughness']
    current = json.loads((output / 'comparison.json').read_text())['methods']
    a = json.loads((Path(plan['measured_factor_references']['a']) / 'comparison.json').read_text())['methods']
    b = json.loads((Path(plan['measured_factor_references']['b']) / 'comparison.json').read_text())['methods']
    rough = {name: roughness(path, scheduled=True) for name, path in (
        ('candidate', output), ('b', Path(plan['measured_factor_references']['b'])))}
    registered = plan['diagnostic_rule']
    results = {}
    for name in ('flow_mean', 'flow_samples', 'regression_mean'):
        arm, kind = name.split('_')
        method = f'{arm}_20000_{kind}'
        axes = {}
        for key, row in formal(current[method], a[method])['axes'].items():
            if key.startswith(('acceleration_', 'jerk_')):
                continue
            maximum = registered['rmse_8_and_reprojection_3_axes_ratio_vs_a_max']
            axes[key] = {**row, 'reference_run': 'a', 'maximum_ratio': maximum,
                'ratio': row['candidate'] / row['reference'],
                'pass': row['candidate'] <= maximum * row['reference'] + row['tolerance']}
        assert len(axes) == 11
        for label in ('acceleration', 'jerk'):
            for support in ('free', 'inside_free'):
                key = f'{name}/{label}/{support}'
                candidate, reference = (rough[r]['derivative'][key] for r in ('candidate', 'b'))
                assert candidate['count'] == reference['count'] and candidate['count'] > 0
                maximum = registered['full_and_inside_free_accel_jerk_p95_ratio_vs_b_max']
                tolerance = registered['abs_tolerance'] + registered['relative_tolerance'] * abs(reference['p95'])
                axes[f'{label}_{support}_p95'] = {
                    'candidate': candidate['p95'], 'reference': reference['p95'], 'reference_run': 'b',
                    'ratio': candidate['p95'] / reference['p95'], 'maximum_ratio': maximum,
                    'count': candidate['count'], 'tolerance': tolerance,
                    'pass': candidate['p95'] <= maximum * reference['p95'] + tolerance}
        behind = {key: rows[method]['metrics']['behind_all'] for key, rows in (('candidate', current), ('a', a), ('b', b))}
        assert len({row['count'] for row in behind.values()}) == 1
        behind['pass'] = behind['candidate']['invalid_count'] <= min(behind[r]['invalid_count'] for r in ('a', 'b'))
        results[name] = {'axes': axes, 'behind': behind, 'pass': all(row['pass'] for row in axes.values()) and behind['pass']}
    expected = {
        'mixture_mean_rts': ['4.408', '669', '158', '87998', '7133', '16.4', '2.47', '74.8', '1'],
        'flow_20000_mean': ['1.953', '412', '183', '42809', '14299', '20.7', '12.1', '64.4', '4'],
        'regression_20000_mean': ['2.055', '753', '196', '91523', '13783', '18.6', '9.6', '60.0', '4'],
    }
    verified = []
    for name, displayed in expected.items():
        m = current[name]['metrics']
        values = [m['rmse_m_overall']['value']]
        values += [m[key]['p95'] for key in ('acceleration_all', 'acceleration_free_flight', 'jerk_all', 'jerk_free_flight')]
        values += [m['reprojection_px_all'][key] for key in ('mean', 'p50', 'p95')]
        values += [m['behind_all']['invalid_count']]
        for value, text in zip(values, displayed, strict=True):
            digits = len(text.split('.')[1]) if '.' in text else 0
            assert f'{value:.{digits}f}' == text, (name, value, text)
        verified.append({'method': name, 'reported': displayed, 'actual': values})
    report = {'primary_update': 20000, 'rule_url': plan['rule_url'], 'registered_rule': registered,
        'roughness': rough, 'results': results, 'common_improvement': all(r['pass'] for r in results.values()),
        'directive_all_27_values_match': True, 'directive_table': verified}
    (output / 'diagnostic-rule.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, row in results.items():
        print(name, 'pass=', row['pass'], 'failed=', [k for k, v in row['axes'].items() if not v['pass']], 'behind=', row['behind'])


if __name__ == '__main__':
    main()
