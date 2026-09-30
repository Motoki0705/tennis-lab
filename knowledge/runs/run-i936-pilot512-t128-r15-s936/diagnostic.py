"""Apply the original data-size rule and verify every run-16 directive number."""
from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any


def main() -> None:
    bundle = Path(__file__).resolve().parent
    source = bundle / 'collected'
    control = bundle.parent / 'run-i936-anchored-t128-flow-regression-r13-s936/collected'
    current = json.loads((source / 'comparison.json').read_text())['methods']
    prior = json.loads((control / 'comparison.json').read_text())['methods']
    formal = runpy.run_path(str(bundle / 'collect.py'))['rule']
    results: dict[str, Any] = {}
    for name in ('flow_20000_mean', 'flow_20000_samples', 'regression_20000_mean'):
        result = formal(current[name], prior[name])
        behind = {key: rows[name]['metrics']['behind_all'] for key, rows in (('candidate', current), ('reference', prior))}
        assert behind['candidate']['count'] == behind['reference']['count']
        behind['pass'] = behind['candidate']['invalid_count'] <= behind['reference']['invalid_count']
        results[name] = {'axes': result['axes'], 'behind': behind,
                         'pass': behind['pass'] and all(row['status'] != 'worse' for row in result['axes'].values())
                         and any(row['status'] == 'better' for row in result['axes'].values())}
    expected = [
        (current, 'mixture_mean_rts', ['4.408', '3.020', '669', '158', '16.4', '2.47', '74.8', '1']),
        (prior, 'flow_20000_mean', ['3.651', '2.074', '3550', '1848', '26.7', '13.6', '74.1', '6']),
        (current, 'flow_20000_mean', ['2.018', '2.093', '3437', '1985', '21.9', '11.6', '74.6', '4']),
        (current, 'regression_20000_mean', ['2.412', '2.311', '3330', '2065', '21.1', '11.5', '72.9', '4']),
    ]
    verified = []
    for rows, method, expected_row in expected:
        metrics = rows[method]['metrics']
        values = [metrics[key]['value'] for key in ('rmse_m_overall', 'rmse_m_gap')]
        values += [metrics[key]['p95'] for key in ('acceleration_all', 'acceleration_free_flight')]
        values += [metrics['reprojection_px_all'][key] for key in ('mean', 'p50', 'p95')]
        values += [metrics['behind_all']['invalid_count']]
        for value, display in zip(values, expected_row, strict=True):
            decimals = len(display.split('.')[1]) if '.' in display else 0
            assert f'{value:.{decimals}f}' == display, (method, value, display)
        verified.append({'method': method, 'reported': expected_row, 'actual': values})
    assert f"{current['flow_10000_mean']['metrics']['rmse_m_overall']['value']:.3f}" == '2.089'
    report = {'primary_update': 20000, 'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5915164321',
              'results': results, 'common_improvement': all(r['pass'] for r in results.values()),
              'directive_all_32_table_values_match': True, 'directive_table': verified,
              'flow_10000_rmse_m': current['flow_10000_mean']['metrics']['rmse_m_overall']['value'],
              'flow_20000_rmse_m': current['flow_20000_mean']['metrics']['rmse_m_overall']['value']}
    (source / 'diagnostic-rule.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, row in results.items():
        print(name, 'pass=', row['pass'], 'worse=', [k for k, v in row['axes'].items() if v['status'] == 'worse'], 'behind=', row['behind'])


if __name__ == '__main__':
    main()
