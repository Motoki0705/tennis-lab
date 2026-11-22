"""Repeat factor interventions with one immutable LINE evidence object per image.

This isolates O/S/V effects from observed repeated distanceTransform variation.
It is an offline diagnostic; no production function or setting is edited.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import time

import cv2
import numpy as np

from src.tasks.court_detection.geometry.line_evidence import LineEvidence
from regression_factorial import module_for
from common import BASE, OLD, WORLD, EDGES, write_json


def solve_case(task):
    case, record = task
    cv2.setNumThreads(1); cv2.setRNGSeed(2190)
    raw = np.load(OLD / (case['dataset'] + '_raw') / case['key'] / 'raw_00.npz')
    observed = raw['raw_xy'].astype(float); observed[~raw['valid']] = np.nan
    probability = raw['line_probability']; size = tuple(raw['image_size_hw'])
    # Build once, then require bitwise identical evidence for all interventions.
    frozen = LineEvidence.from_probability(probability, size, threshold=.5, max_observations=768)
    hashes = {name: hashlib.sha256(np.asarray(getattr(frozen, name)).tobytes()).hexdigest() for name in ('distance', 'points', 'tangents', 'weights', 'coherence')}
    descriptor = LineEvidence.__dict__['from_probability']
    def fixed(cls, values, dimensions, **kwargs):
        assert np.array_equal(values, probability) and dimensions == size
        assert kwargs == {'threshold': .5, 'max_observations': 768}
        return frozen
    LineEvidence.from_probability = classmethod(fixed)
    results = []
    try:
        # The reference order is predefined, not picked after observing correctness.
        for variant in case['variants']:
            name = ''.join(f'{letter}{value}' for letter, value in zip('osv', variant))
            module = module_for(*variant); module.TRACE.clear()
            cv2.setRNGSeed(2190); start = time.monotonic()
            result = module.estimate_hybrid_homography(WORLD, observed, raw['scores'], probability, edges=EDGES, image_size_hw=size, config=module.HybridHomographyConfig())
            valid = result.matrix is not None
            mask = np.array(record['gt_valid'], bool); truth = np.array(record['gt_xy'])
            error = np.linalg.norm(result.projected[mask] - truth[mask], axis=1) if valid else np.array([])
            ends = {}
            for row in module.TRACE: ends[row['seed']] = row
            reasons = []
            for row in ends.values():
                reason = 'optimizer_budget' if not row['success'] and row['optimizer_status'] == 0 else 'optimizer_failed' if not row['success'] else 'postsolve_geometry_or_support_rejected' if row['updated'] is None else 'selection_did_not_stabilize' if row['fixed'] != row['updated'] else 'seed_completed'
                reasons.append({'seed': row['seed'], 'reason': reason})
            results.append({'variant': name, 'status': result.status, 'valid': valid, 'seconds': time.monotonic() - start,
                'mean_error_px': float(error.mean()) if len(error) else None, 'max_error_px': float(error.max()) if len(error) else None,
                'selected': np.flatnonzero(result.selected).tolist(), 'seed_outcomes': reasons, 'trace': module.TRACE.copy()})
        assert hashes == {name: hashlib.sha256(np.asarray(getattr(frozen, name)).tobytes()).hexdigest() for name in hashes}
    finally:
        LineEvidence.from_probability = descriptor
    out = BASE / 'regression-paired' / case['dataset'] / (record['clip_id'].replace(':', '_') + '.json')
    write_json(out, {'case': case, 'line_evidence_sha256': hashes, 'line_evidence_unchanged': True,
        'probability_sha256': hashlib.sha256(probability.tobytes()).hexdigest(), 'results': results})
    return {'key': case['key'], 'statuses': {r['variant']: r['status'] for r in results}}


def main():
    cases = json.loads((BASE / 'regression-factorial/manifest.json').read_text())['cases']
    records = {r['key']: r for dataset in ('val', 'test') for r in json.loads((OLD / (dataset + '_raw') / 'manifest.json').read_text())['records']}
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(solve_case, (case, records[case['key']])) for case in cases]
        for f in as_completed(futures): print(json.dumps(f.result()), flush=True)
    write_json(BASE / 'regression-paired/complete.json', {'cases': len(cases), 'seconds': time.monotonic() - start, 'workers': 6})


if __name__ == '__main__': main()
