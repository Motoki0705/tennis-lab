"""Summarize the immutable stratified audit including every failed frame."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def statistics(rows):
    times = np.array([r['seconds'] for r in rows])
    return {'frames': len(rows), 'converged': sum(r['converged'] for r in rows),
            'rate': sum(r['converged'] for r in rows)/len(rows), 'failures': sum('error' in r for r in rows),
            'worker_seconds_mean': float(times.mean()), 'worker_seconds_p50': float(np.median(times)),
            'worker_seconds_p95': float(np.quantile(times, .95)), 'worker_seconds_max': float(times.max()),
            'cpu_seconds_mean': sum(r['cpu_seconds'] for r in rows)/len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sample = json.loads(args.sample.read_text())
    raw = json.loads(args.results.read_text())
    rows = raw['records']
    if {(r['rally_id'], r['frame']) for r in rows} != {(r['rally_id'], r['frame']) for r in sample['frames']} or len(rows) != len(sample['frames']):
        raise ValueError('Sample identity/count mismatch')
    strata = {key: statistics([r for r in rows if r['stratum'] == key]) for key in sample['strata']}
    total = sum(c['population'] for c in sample['strata'].values())
    weighted = {metric: sum(strata[key][metric]*count['population']/total for key, count in sample['strata'].items()) for metric in ('rate', 'worker_seconds_mean', 'cpu_seconds_mean')}
    avg = weighted['worker_seconds_mean']
    balanced_avg = statistics(rows)['worker_seconds_mean']
    result = {
        'balanced_sample': statistics(rows), 'all_missing': statistics([r for r in rows if r['observed_views']==0]),
        'strata': strata, 'stopped_data_population_weighted': weighted,
        'methods': dict(Counter(m for r in rows for m in r.get('methods', []))),
        'actual_wall_seconds': raw['wall_seconds'], 'ideal_four_worker_wall_seconds': sum(r['seconds'] for r in rows)/4,
        'failures': [r for r in rows if 'error' in r],
        'projections': {str(n): {'population_weighted_hours': avg*n*450/4/3600,
                               'balanced_stress_hours': balanced_avg*n*450/4/3600,
                               'population_weighted_hours_with_20pct_margin': avg*n*450/4/3600*1.2,
                               'balanced_512_frame_hours_with_20pct_margin': balanced_avg*n*512/4/3600*1.2} for n in (96,640)},
        'limits': 'train-only, completion-biased stopped-dev population; within-rally correlated sample; development audit, not held-out accuracy or full 640-rally guarantee; includes time spent in failures',
    }
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
