"""All-frame/component diagnostics, including float32 export feasibility."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def quantiles(values):
    return dict(zip(('min','p50','p95','max'), np.quantile(values,[0,.5,.95,1]).tolist(),strict=True)) if values else None


def main():
    p=argparse.ArgumentParser()
    for name in ('before','after','bundle'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=json.loads((args.after/'results.json').read_text())
    records=result['records']
    changes, means, covariances, weights, frames, flags=[],[],[],[],[],[]
    unresolved=Counter(); methods=Counter(); fp32_failures=[]; mix_shift=[]; zero_weights=0
    errors=[]; metrics=Counter(); embedded_capped=0
    for r in records:
        if 'error'in r: continue
        name=r['rally_id']+f"-{r['frame']:05d}"
        with np.load(args.after/(name+'.npz')) as z:
            if z['means'].shape!=(125,3):raise ValueError('Dropped component')
            np.linalg.cholesky(z['covariance'])
            try: np.linalg.cholesky(z['covariance'].astype(np.float32))
            except np.linalg.LinAlgError:fp32_failures.append(name)
            zero_weights+=int((z['weights'].astype(np.float32)==0).sum())
            unresolved.update(int(x) for x in z['camera_subsets'][~z['component_converged']].sum(-1))
            methods.update(r['methods'])
            changes.append(z['component_changes']); means.append(z['means']);covariances.append(z['covariance']);weights.append(z['weights']);flags.append(z['component_converged']);frames.append(name)
            before=args.before/(name+'.npz')
            if before.exists():
                with np.load(before) as old:
                    mix_shift.append(float(np.linalg.norm(z['weights']@z['means']-old['weights']@old['means'])))
        for d in r['integration_diagnostics']:
            if d:
                errors.append(d['embedded_relative_error']);embedded_capped+=not d['embedded_converged']
                metrics['local_hessian' if d['metric_is_local_hessian'] else 'unit_ray_axes']+=1
    np.savez_compressed(args.bundle/'after-components.npz',frames=np.array(frames),component_changes=np.stack(changes),means=np.stack(means),covariance=np.stack(covariances),weights=np.stack(weights),component_converged=np.stack(flags))
    (args.bundle/'after-results.json.gz').write_bytes(gzip.compress((args.after/'results.json').read_bytes(),mtime=0))
    unresolved_rows=[r for r in records if 'error'not in r and not r['converged']]
    report={
        'methods':dict(methods),'unresolved_components_by_active_views':dict(unresolved),
        'nonconverged_frames_with_NLL_passing':sum(r['history'][-1]['nll_delta_nat']<=.05 for r in unresolved_rows),
        'nonconverged_frames':len(unresolved_rows),'nonconverged_component_mass':quantiles([r['nonconverged_mass'] for r in unresolved_rows]),
        'float32_covariance_failures':fp32_failures,'float32_zero_weight_components_retained':zero_weights,
        'mixture_mean_change_vs_baseline_m':quantiles(mix_shift),'matched_baseline_frames':len(mix_shift),
        'embedded_error':quantiles(errors),'embedded_target_not_met_components':embedded_capped,'chart_metrics':dict(metrics),
        'outer_passed_frames_with_embedded_target_unmet':sum(r['converged'] and any(d and not d['embedded_converged'] for d in r['integration_diagnostics']) for r in records if 'error'not in r),
        'embedded_note':'Embedded error is supplementary; acceptance still uses the preregistered four outer differences. It is not a rigorous bound.',
        'source_files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.before/'results.json',args.after/'results.json')},
        'bytes':{str(p):sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in (args.before,args.after,args.bundle)},
    }
    (args.bundle/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
