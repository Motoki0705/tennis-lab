"""Boundary checks for the preregistered 17-axis diagnostic, using recorded controls."""
from copy import deepcopy
import json
from pathlib import Path
import runpy


def main() -> None:
    bundle = Path(__file__).resolve().parent
    namespace = runpy.run_path(str(bundle / 'collect.py'))
    diagnose = namespace['diagnose']
    formal = namespace['rule']
    plan = json.loads((bundle / 'plan.json').read_text())
    control = json.loads((Path(plan['control_manifest']).parent / 'comparison.json').read_text())['methods']['flow_20000_mean']
    inside = json.loads((bundle / 'collected/collection.json').read_text())['inside_free_roughness']['control']['flow_20000_mean']
    candidate = deepcopy(control)
    candidate['metrics']['reprojection_px_all']['mean'] *= .9
    candidate['metrics']['reprojection_px_all']['p50'] *= .8
    base = diagnose(candidate, control, inside, inside)
    assert base['passed'] and not formal(candidate, control)['pass']  # behind=4 is diagnostic-only admissible.
    paths = {f'rmse_m_{name}': ('metrics', f'rmse_m_{name}', 'value') for name in ('overall','gap','no_evidence','event_pm5')}
    paths.update({f'camera_{v}_rmse': ('by_visible_cameras', str(v), 'rmse_m_overall', 'value') for v in range(4)})
    paths.update({name+'_p95': ('metrics', name, 'p95') for name in ('acceleration_all','acceleration_free_flight','jerk_all','jerk_free_flight')})
    paths.update({'reprojection_'+q: ('metrics', 'reprojection_px_all', q) for q in ('mean','p50','p95')})
    paths.update({name+'_inside_free_p95': (name, 'p95') for name in ('acceleration','jerk')})
    checks = 2
    for name, path in paths.items():
        for multiple, passed in ((0., True), (.5, True), (2., False)):
            c, i = deepcopy(candidate), deepcopy(inside)
            target = i if 'inside_free' in name else c
            for field in path[:-1]:
                target = target[field]
            target[path[-1]] = base['axes'][name]['limit'] + multiple * base['axes'][name]['tolerance']
            result = diagnose(c, control, i, inside)
            assert result['passed'] == passed, (name, multiple, result)
            assert result['failed_axes'] == ([] if passed else [name])
            checks += 1
    candidate['metrics']['behind_all']['invalid_count'] += 1
    assert diagnose(candidate, control, inside, inside)['failed_axes'] == ['behind_count']
    checks += 1
    print(f'{checks} checks passed: all 17 axes, inclusive threshold/tolerance, behind and formal separation')


if __name__ == '__main__':
    main()
