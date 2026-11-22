"""CPU counterfactual diagnosis of existing changes; production files stay fixed."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'court-acceptance-implementation-20261116'
CURRENT_SOURCE = Path('src/tasks/court_detection/geometry/hybrid_homography.py')
LEGACY_SOURCE = Path('knowledge/runs/run-court-acceptance-implementation-20261116/legacy_geometry/hybrid_homography.py')
MODULES = {}


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])


def module_for(observations, spatial, visibility):
    key = f'o{observations}s{spatial}v{visibility}'
    if key in MODULES:
        return MODULES[key]
    source = CURRENT_SOURCE.read_text()
    if not observations:
        source = replace_once(source, 'observed[~support.eligible] = np.nan', '# Analysis intervention: retain the pre-change valid observations.')
    if not spatial:
        source = replace_once(source, function_source(source, 'select_keypoints'), function_source(LEGACY_SOURCE.read_text(), 'select_keypoints'))
    if not visibility:
        source = replace_once(source, 'required_visible=required_visible,', 'required_visible=None,')
    # Instrument existing solve outcomes without changing their arithmetic or budgets.
    source = replace_once(source, 'for seed in seeds:', 'for seed_index, seed in enumerate(seeds):')
    source = replace_once(source, 'for field in fields:', 'for field_index, field in enumerate(fields):')
    source = replace_once(source, 'for _ in range(config.max_selection_rounds):', 'for round_index in range(config.max_selection_rounds):')
    marker = '                updated = evaluate(unpack(optimized.x), tuple(history))'
    source = replace_once(source, marker, marker + '''
                TRACE.append({
                    "seed": seed_index, "field": field_index, "round": round_index,
                    "success": bool(optimized.success), "optimizer_status": int(optimized.status),
                    "nfev": int(optimized.nfev), "cost": float(optimized.cost),
                    "fixed": np.flatnonzero(fixed).tolist(),
                    "updated": None if updated is None else np.flatnonzero(updated.selected).tolist(),
                })''')
    name = 'src.tasks.court_detection.geometry.analysis_' + key
    module = types.ModuleType(name)
    module.__package__ = 'src.tasks.court_detection.geometry'
    sys.modules[name] = module
    exec(compile(source, str(BASE / ('counterfactual_' + key + '.py')), 'exec'), module.__dict__)
    module.TRACE = []
    module.analysis_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    MODULES[key] = module
    return module


def solve(task):
    record, dataset, group, variant = task
    name = ''.join(f'{prefix}{value}' for prefix, value in zip('osv', variant))
    target = BASE / 'regression-factorial' / dataset / record['clip_id'].replace(':', '_') / (name + '.json')
    if target.exists():
        return {'cached': str(target)}
    cv2.setNumThreads(1)
    cv2.setRNGSeed(2190)
    np.random.seed(2190)
    module = module_for(*variant)
    module.TRACE.clear()
    raw = np.load(OLD / (dataset + '_raw') / record['key'] / 'raw_00.npz')
    points = raw['raw_xy'].astype(float)
    points[~raw['valid']] = np.nan
    start = time.monotonic()
    result = module.estimate_hybrid_homography(court_template_xy().astype(float), points, raw['scores'], raw['line_probability'],
        edges=np.array(COURT_HOMOGRAPHY_EDGES), image_size_hw=tuple(raw['image_size_hw']), config=module.HybridHomographyConfig())
    elapsed = time.monotonic() - start
    gt = np.array(record['gt_xy']); mask = np.array(record['gt_valid'], bool)
    errors = np.linalg.norm(result.projected[mask] - gt[mask], axis=1) if result.matrix is not None else np.array([])
    by_seed = {}
    for row in module.TRACE:
        by_seed[row['seed']] = row
    failures = []
    for row in by_seed.values():
        if not row['success']:
            reason = 'optimizer_budget' if row['optimizer_status'] == 0 else 'optimizer_failed'
        elif row['updated'] is None:
            reason = 'postsolve_geometry_or_support_rejected'
        elif row['fixed'] != row['updated']:
            reason = 'selection_did_not_stabilize'
        else:
            reason = 'seed_completed'
        failures.append({'seed': row['seed'], 'reason': reason, 'last_field': row['field'], 'last_round': row['round']})
    entry = {'key': record['key'], 'dataset': dataset, 'group': group, 'variant': name,
        'variant_source_sha256': module.analysis_source_sha256, 'raw_sha256': hashlib.sha256((OLD / (dataset + '_raw') / record['key'] / 'raw_00.npz').read_bytes()).hexdigest(),
        'status': result.status, 'valid': result.matrix is not None, 'seconds': elapsed,
        'mean_error_px': float(errors.mean()) if len(errors) else None,
        'max_error_px': float(errors.max()) if len(errors) else None,
        'selected': np.flatnonzero(result.selected).tolist(), 'seed_outcomes': failures,
        'trace': module.TRACE.copy(), 'observation_diagnostics': result.observation_diagnostics}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entry, allow_nan=False))
    return {'key': record['key'], 'variant': name, 'status': result.status, 'seconds': round(elapsed, 3)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-cases', type=int); args = parser.parse_args()
    cases = []; tasks = []
    for dataset in ('val', 'test'):
        records = sorted(json.loads((OLD / (dataset + '_raw') / 'manifest.json').read_text())['records'], key=lambda r: r['key'])
        unchanged = []
        for record in records:
            previous = [json.loads((OLD / (dataset + '_' + mode) / record['key'] / 'geometry.json').read_text())['candidates'][0] for mode in ('legacy', 'current')]
            success = [c['status'] == 'ok' and c['matrix'] is not None for c in previous]
            group = 'lost' if success == [True, False] else 'gained' if success == [False, True] else 'unchanged_control'
            entry = (record, dataset, group, previous)
            if success[0] != success[1]: cases.append(entry)
            elif success[0]: unchanged.append(entry)
        # Fixed 16 evenly spaced unchanged-success controls per dataset.
        cases.extend(unchanged[i] for i in np.linspace(0, len(unchanged) - 1, min(16, len(unchanged))).round().astype(int))
    if args.max_cases: cases = cases[:args.max_cases]
    manifest = []
    for record, dataset, group, previous in cases:
        diagnostics = previous[1]['observation_diagnostics']
        inactive = not diagnostics['boundary_censored_indices'] and not diagnostics['duplicate_peak_indices']
        variants = list(itertools.product((1,) if inactive else (0, 1), (0, 1), (0, 1)))
        tasks.extend((record, dataset, group, variant) for variant in variants)
        manifest.append({'key': record['key'], 'dataset': dataset, 'group': group,
            'frozen_status': {name: c['status'] for name, c in zip(('legacy', 'current'), previous)},
            'observation_filter_inactive': inactive, 'variants': variants,
            'o0_equivalent_to_o1_when_inactive': inactive})
    out = BASE / 'regression-factorial'; out.mkdir(exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps({'source_head': '7b59c4730dfa934fb8e2323ae0553b96f737f412',
        'meaning': {'o': 'boundary/duplicate observation censoring', 's': 'spatial point selection', 'v': 'seed-visible lines fixed during solve'},
        'optimizer_parameters_unchanged': True, 'production_sources_unmodified': True, 'cases': manifest}, indent=2))
    started = time.monotonic()
    print('cases', len(cases), 'solves', len(tasks), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve, task) for task in tasks]
        for future in as_completed(futures): print(json.dumps(future.result()), flush=True)
    (out / 'complete.json').write_text(json.dumps({'cases': len(cases), 'solves': len(tasks), 'seconds': time.monotonic() - started}))


if __name__ == '__main__': main()
