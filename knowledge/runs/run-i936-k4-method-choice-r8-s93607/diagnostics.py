"""Input integrity, prior-only scoring reference, and transparent generation costs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from src.utils.geometry.probabilistic_triangulation import GaussianMixture3D

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sample = json.loads((ROOT / 'knowledge/runs/run-i936-k4-audit-r7-s93607/sample.json').read_text())
    summary = json.loads((args.output / 'summary.json').read_text())
    truth, inputs = {}, {}
    for key, source in sample['sources'].items():
        path = Path(source['path'])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != source['sha256']:
            raise ValueError('Source changed')
        inputs[key] = sha
        with np.load(path, allow_pickle=False) as z:
            truth[key] = z['positions_3d_m'].astype(float)
    variance = np.array(sample['settings']['prior_covariance_diagonal_m2'])
    prior_mean = np.array(sample['settings']['prior_mean_m'])
    rows = []
    for row in sample['frames']:
        delta = truth[row['rally_id']][row['frame']] - prior_mean
        quadratic = float(np.sum(delta**2 / variance))
        rows.append(dict(stratum=row['stratum'], nll_nat=.5 * (quadratic + np.log(variance).sum() + 3*np.log(2*np.pi)),
            mean_error_m=float(np.linalg.norm(delta)), **{f'hdr{a}': quadratic <= chi2.ppf(a/100, 3) for a in (50,90,95)}))
    weights = np.array([sample['strata'][r['stratum']]['population']/sample['strata'][r['stratum']]['sample'] for r in rows])
    prior = {group: {k: float(np.average([r[k] for r in rows], weights=w)) for k in ('nll_nat', 'mean_error_m', 'hdr50', 'hdr90', 'hdr95')}
        for group, w in (('overall', np.ones(len(rows))), ('weighted', weights))}
    source_cost = json.loads((BUNDLE / 'generation-cost-source.json').read_text())
    sim = float(np.mean([r['simulation_seconds'] for r in source_cost]))
    other = float(np.mean([r['elapsed_seconds'] - r['triangulation_seconds'] - r['simulation_seconds'] for r in source_cost]))
    bytes_per_frame = sum(r['npz_bytes'] for r in source_cost) / sum(r['frames'] for r in source_cost)
    selected = summary['methods']['H']['weighted']
    worker_seconds = selected['cost_seconds']['mean'] + selected['mean_adapter_seconds']
    projections = []
    for rallies in (96,640):
        for frames in (450,512):
            wall = rallies * (frames*worker_seconds + sim + other) / 4
            projections.append(dict(rallies=rallies, frames_per_rally=frames, ideal_wall_hours=wall/3600,
                with_20pct_margin_hours=1.2*wall/3600, npz_bytes_estimate=rallies*frames*bytes_per_frame))
    decision = dict(selected_method='H', cost_ceiling_mean_worker_seconds=.6,
        eligible_by_mean_cost=[m for m,v in summary['methods'].items() if v['overall']['cost_seconds']['mean']<=.6 and v['weighted']['cost_seconds']['mean']<=.6],
        selection='Lower overall and weighted GT NLL and mean absolute HDR calibration error among cost-eligible methods; convergence diagnostic only',
        workers=4,native_threads=1,simulation_seconds_per_rally=sim,other_seconds_per_rally=other,
        old_npz_bytes_per_frame=bytes_per_frame,projections=projections,
        dev_proposal={'rallies': {'train':64,'val':16,'test':16}, 'schedule_minutes':90, 'cpu_processes':4,
            'estimated_peak_ram_gb':6,'disk_reservation_bytes':750000000,'gpu':False,'started':False,
            'calibration_report':str(ROOT/sample['settings']['calibration']['report']),
            'calibration_report_sha256':sample['settings']['calibration']['report_sha256'],
            'bank_sha256':sample['settings']['calibration']['bank_sha256']},
        pipeline_10minute_clip_solver_hours_4workers=600*60000/1001*selected['cost_seconds']['mean']/4/3600,
        limits='Completed train-only sample; new split/geometry/seed/calibration may change costs. 640 rallies await final #935. Pipeline number is arithmetic only, not pipeline integration.')
    # Validate the selected generator's float32 representation, independently
    # of the float64 solver scores. All 125 entries remain, including underflow.
    nll_deltas, mean_deltas = [], []
    zero_weights = 0
    with np.load(args.output/'H-components.npz', allow_pickle=False) as z:
        for j, index in enumerate(z['sample_index']):
            row = sample['frames'][int(index)]
            gt = truth[row['rally_id']][row['frame']]
            original = GaussianMixture3D(z['means'][j], z['covariance'][j], z['weights'][j])
            cov32 = z['covariance'][j].astype(np.float32)
            np.linalg.cholesky(cov32)
            exported = GaussianMixture3D(z['means'][j].astype(np.float32).astype(float), cov32.astype(float), z['weights'][j].astype(np.float32).astype(float))
            nll_deltas.append(abs(float(original.log_prob(gt)-exported.log_prob(gt))))
            mean_deltas.append(float(np.linalg.norm(original.moments()[0]-exported.moments()[0])))
            zero_weights += int((exported.weights == 0).sum())
    export_check = dict(frames=len(nll_deltas), components_per_frame=125, float32_covariance_spd=True,
        max_gt_nll_delta_nat=max(nll_deltas), max_mixture_mean_delta_m=max(mean_deltas), retained_zero_weight_components=zero_weights)
    (args.output/'selection-and-cost.json').write_text(json.dumps(decision,indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps({'source_hashes_unchanged':inputs, 'prior_reference':prior, 'H_float32_export':export_check},indent=2)+'\n')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    main()
