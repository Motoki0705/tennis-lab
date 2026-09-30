"""Package every fixed-sample result and compute explicit stratum-weighted tables."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / 'knowledge/runs/run-i936-k4-audit-r7-s93607/sample.json'
METHODS = ('A', 'H', 'ray', 'adaptive_ray', 'C', 'Q20')


def summarize(rows, weights):
    valid = np.asarray(['error' not in r for r in rows])
    w = np.asarray(weights, float)
    result = dict(frames=len(rows), failed=int((~valid).sum()), failed_weight=float(w[~valid].sum() / w.sum()))
    for key in ('nll_nat', 'mean_error_m', 'hdr50', 'hdr90', 'hdr95', 'prior_only_probability'):
        success_w = w[valid]
        result[key] = float(np.average([r[key] for r in rows if 'error' not in r], weights=success_w)) if success_w.sum() else None
    # Failed densities have no NLL/HDR. Conditional estimates above are labeled
    # as such; no finite unconditional score is invented or used for selection.
    result['unconditional_scores_available'] = bool(valid.all())
    result['mean_hdr_calibration_error'] = None if not valid.any() else float(np.mean([abs(result[f'hdr{p}'] - p/100) for p in (50, 90, 95)]))
    costs = np.asarray([r['seconds'] for r in rows])
    order = np.argsort(costs)
    cdf = np.cumsum(w[order]) / w.sum()
    result['cost_seconds'] = dict(mean=float(np.average(costs, weights=w)),
        p50=float(costs[order[np.searchsorted(cdf, .5)]]), p95=float(costs[order[np.searchsorted(cdf, .95)]]), max=float(costs.max()))
    result['mean_cpu_seconds'] = float(np.average([r['cpu_seconds'] for r in rows], weights=w))
    result['mean_adapter_seconds'] = float(np.average([r['adapter_seconds'] for r in rows], weights=w))
    checked = [r['converged'] for r in rows if 'error' not in r and r['converged'] is not None]
    result['converged'] = sum(checked) if checked else None
    result['converged_rate_all_frames'] = float(np.sum([weight * bool(r.get('converged', False)) for r, weight in zip(rows, w)]) / w.sum()) if checked else None
    optimizer = [r['optimizer_converged'] for r in rows if 'error' not in r and 'optimizer_converged' in r]
    result['optimizer_converged'] = sum(optimizer) if optimizer else None
    result['float32_spd_failures'] = sum(r.get('float32_spd', True) is False for r in rows)
    result['subset_mass_max_abs_error'] = max((r['subset_mass_error'] for r in rows if 'error' not in r), default=None)
    brier = [float(np.mean((np.array(r['mean_presence']) - np.array(r['amodal_present'])) ** 2)) for r in rows if 'error' not in r]
    result['amodal_presence_brier'] = float(np.average(brier, weights=w[valid])) if valid.any() else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    bundle = args.output if args.output is not None else args.results / 'collected'
    if not bundle.is_absolute():
        raise ValueError('Use an absolute collection output')
    bundle.mkdir(parents=True, exist_ok=True)
    if (bundle / 'summary.json').exists():
        raise FileExistsError(bundle / 'summary.json')
    sample = json.loads(SAMPLE.read_text())
    order = [(r['rally_id'], r['frame']) for r in sample['frames']]
    ids = {identity: i for i, identity in enumerate(order)}
    all_rows, all_components = {}, {}
    summary = dict(sample_sha256=hashlib.sha256(SAMPLE.read_bytes()).hexdigest(), strata=sample['strata'],
                   note='Scores conditional on valid full distributions when failures > 0; cost includes failed attempts. Quantiles use inverse empirical CDF. Weighted estimates use population/sample stratum weights.', methods={})
    for method in METHODS:
        directory = args.results / method
        result = json.loads((directory / 'results.json').read_text())
        rows = sorted(result['records'], key=lambda r: ids[(r['rally_id'], r['frame'])])
        if [(r['rally_id'], r['frame']) for r in rows] != order or result['sample_sha256'] != summary['sample_sha256']:
            raise ValueError('Incomplete or changed sample')
        all_rows[method] = rows
        w = [sample['strata'][r['stratum']]['population'] / sample['strata'][r['stratum']]['sample'] for r in rows]
        summary['methods'][method] = dict(overall=summarize(rows, np.ones(len(rows))), weighted=summarize(rows, w),
            strata={s: summarize([r for r in rows if r['stratum'] == s], np.ones(count['sample'])) for s, count in sample['strata'].items()},
            total_wall_seconds=result['wall_seconds'], budget=result['budget'], commit=result['commit'])
        arrays = {}
        components = {}
        for i, row in enumerate(rows):
            if 'error' in row:
                continue
            with np.load(directory / f"{row['rally_id']}-{row['frame']:05d}.npz", allow_pickle=False) as z:
                components[i] = {key: z[key] for key in z.files}
        all_components[method] = components
        if components:
            arrays['sample_index'] = np.array(list(components), dtype=np.int32)
            for key in next(iter(components.values())):
                arrays[key] = np.stack([c[key] for c in components.values()])
            np.savez_compressed(bundle / f'{method}-components.npz', **arrays)
        with gzip.open(bundle / f'{method}-results.json.gz', 'wt') as f:
            json.dump(result, f, allow_nan=False)
    reference = all_components['adaptive_ray']
    reference_indices = [i for i, r in enumerate(all_rows['adaptive_ray']) if r.get('converged')]
    for method in METHODS:
        differences = []
        for i in reference_indices:
            if i not in all_components[method]:
                continue
            a, b = all_components[method][i], reference[i]
            differences.append(dict(sample_index=i, mean_l2_m=float(np.linalg.norm(a['weights'] @ a['means'] - b['weights'] @ b['means'])),
                gt_nll_abs_nat=abs(all_rows[method][i]['nll_nat'] - all_rows['adaptive_ray'][i]['nll_nat']),
                weights_l1=float(np.abs(a['weights']-b['weights']).sum())))
        summary['methods'][method]['reference_deviation'] = dict(reference_converged_frames=len(reference_indices), compared_frames=len(differences),
            metrics={key: dict(mean=float(np.mean([r[key] for r in differences])), p95=float(np.quantile([r[key] for r in differences], .95)), max=float(max(r[key] for r in differences))) for key in ('mean_l2_m', 'gt_nll_abs_nat', 'weights_l1')} if differences else {})
    (bundle / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['N/fail = attempted frames / failed frames. NLL, HDR and mean error are conditional on successful distributions when fail > 0; cost includes all attempts.',
             'HDR uses the full mixture density (8192 fixed-seed draws/frame); error is mean Euclidean distance in metres. Weighted rows use stopped-population/sample stratum ratios.',
             'Time is solver wall seconds per frame with 4 workers and one native thread each; quantiles use the inverse empirical CDF. Full definitions and limitations: [knowledge node](../../nodes/ball_refiner_3d/000013-run-i936-k4-method-choice-r8-s93607.md).', '']
    for group in ('overall', *sample['strata'], 'weighted'):
        lines += [f'### {group}', '', '| Method | N/fail | NLL nat | HDR50/90/95 % | Mean error m | Cost mean/p50/p95/max s |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            x = summary['methods'][method][group] if group in ('overall', 'weighted') else summary['methods'][method]['strata'][group]
            coverage = '/'.join(f"{100*x[f'hdr{p}']:.2f}" for p in (50,90,95)) if x['nll_nat'] is not None else 'N/A'
            nll = f"{x['nll_nat']:.4f}" if x['nll_nat'] is not None else 'N/A'
            error = f"{x['mean_error_m']:.4f}" if x['mean_error_m'] is not None else 'N/A'
            cost = '/'.join(f"{x['cost_seconds'][k]:.3f}" for k in ('mean', 'p50', 'p95', 'max'))
            lines.append(f"| {method} | {x['frames']}/{x['failed']} | {nll} | {coverage} | {error} | {cost} |")
        lines += ['']
    (bundle / 'tables.md').write_text('\n'.join(lines))
    print(json.dumps({m: summary['methods'][m]['weighted'] for m in METHODS}, indent=2))


if __name__ == '__main__':
    main()
