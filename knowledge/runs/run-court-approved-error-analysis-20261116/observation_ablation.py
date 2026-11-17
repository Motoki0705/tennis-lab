"""Separate border censoring from duplicate censoring on the O-sensitive cohort."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import time
import cv2
import numpy as np
from src.tasks.court_detection.geometry.line_evidence import LineEvidence
from src.tasks.court_detection.geometry.observations import keypoint_support
from regression_factorial import module_for
from common import BASE, OLD, WORLD, EDGES, write_json


def solve(task):
    case, record = task
    cv2.setNumThreads(1)
    raw = np.load(OLD / 'test_raw' / record['key'] / 'raw_00.npz')
    xy = raw['raw_xy'].astype(float); xy[~raw['valid']] = np.nan
    size = tuple(raw['image_size_hw']); probability = raw['line_probability']
    support = keypoint_support(xy, raw['scores'], image_size_hw=size, native_size_hw=probability.shape,
        min_score=.05, boundary_margin_cells=1., duplicate_distance_cells=1.)
    frozen = LineEvidence.from_probability(probability, size, threshold=.5, max_observations=768)
    descriptor = LineEvidence.__dict__['from_probability']
    def fixed(cls, values, dimensions, **kwargs):
        assert np.array_equal(values, probability) and dimensions == size
        return frozen
    LineEvidence.from_probability = classmethod(fixed)
    result_rows = []
    try:
        for name, removed in [('neither', np.zeros(14, bool)), ('boundary_only', support.boundary), ('duplicate_only', support.duplicate), ('both', ~support.eligible)]:
            observed = xy.copy(); observed[removed] = np.nan
            module = module_for(0, 1, 1); module.TRACE.clear(); cv2.setRNGSeed(2190)
            start = time.monotonic()
            result = module.estimate_hybrid_homography(WORLD, observed, raw['scores'], probability, edges=EDGES, image_size_hw=size)
            result_rows.append({'mode': name, 'status': result.status, 'valid': result.matrix is not None, 'removed': np.flatnonzero(removed).tolist(), 'seconds': time.monotonic() - start})
    finally:
        LineEvidence.from_probability = descriptor
    gt = np.array(record['gt_xy']); gt_valid = np.array(record['gt_valid'], bool); errors = np.linalg.norm(xy - gt, axis=1)
    per_point = [{'kp': i, 'boundary': bool(support.boundary[i]), 'duplicate': bool(support.duplicate[i]),
        'gt_valid': bool(gt_valid[i]), 'raw_gt_error_px': float(errors[i]) if np.isfinite(errors[i]) else None,
        'within_existing_homography_threshold': bool(errors[i] <= .005 * np.hypot(*size))} for i in range(14) if support.boundary[i] or support.duplicate[i]]
    entry = {'key': record['key'], 'group': case['group'], 'prior_attribution': case['single_factor_removal_explanation'], 'results': result_rows, 'removed_point_evidence': per_point}
    write_json(BASE / 'observation-ablation' / (record['clip_id'].replace(':', '_') + '.json'), entry)
    return {'key': record['key'], 'group': case['group'], 'statuses': {r['mode']: r['status'] for r in result_rows}}


def main():
    with (BASE / 'report/regression-cases.csv').open() as f: cases = [r for r in csv.DictReader(f) if r['dataset'] == 'test' and 'O' in r['single_factor_removal_explanation']]
    records = {r['key']: r for r in json.loads((OLD / 'test_raw/manifest.json').read_text())['records']}
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(solve, (case, records[case['key']])) for case in cases]
        for future in as_completed(futures): print(json.dumps(future.result()), flush=True)
    write_json(BASE / 'observation-ablation/complete.json', {'cases': len(cases), 'solves': 4 * len(cases), 'seconds': time.monotonic() - start})


if __name__ == '__main__': main()
