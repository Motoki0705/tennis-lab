"""Compare full-frame Court geometry on identical GT masks and frozen heads."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def stats(values):
    x=np.asarray(values,float)
    return {'count':len(x),'mean_px':float(x.mean()) if len(x) else None,
        'median_px':float(np.median(x)) if len(x) else None,
        'p90_px':float(np.percentile(x,90)) if len(x) else None,
        'p95_px':float(np.percentile(x,95)) if len(x) else None}


def main():
    parser=argparse.ArgumentParser()
    for name in ['raw','legacy','current','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();manifest=json.loads((args.raw/'manifest.json').read_text())
    rows=[];errors={'raw':[],'legacy':[],'current':[],'paired_legacy':[],'paired_current':[]}
    changed_points=[];raw_visible=0;raw_missing=0;no_gt=0
    for record in manifest['records']:
        assert len(record['regions'])==1
        raw=np.load(args.raw/record['key']/'raw_00.npz')
        gt=np.asarray(record['gt_xy']);mask=np.asarray(record['gt_valid'],bool)&np.isfinite(gt).all(axis=1)
        no_gt+=int(not mask.any());raw_visible+=int(mask.sum());raw_missing+=int((mask&~raw['valid']).sum())
        err=np.linalg.norm(raw['raw_xy'][mask&raw['valid']]-gt[mask&raw['valid']],axis=1)
        errors['raw'].extend(err.tolist())
        row={'id':record['clip_id'],'gt_points':int(mask.sum()),'raw_mean_px':float(err.mean()) if len(err) else None}
        candidates={}
        for name,root in [('legacy',args.legacy),('current',args.current)]:
            candidate=json.loads((root/record['key']/'geometry.json').read_text())['candidates'][0]
            candidates[name]=candidate
            valid=candidate['status']=='ok' and candidate['matrix'] is not None
            err=np.linalg.norm(np.array(candidate['projected'])[mask]-gt[mask],axis=1) if valid else np.array([])
            errors[name].extend(err.tolist())
            row.update({name+'_valid':valid,name+'_status':candidate['status'],
                name+'_mean_px':float(err.mean()) if len(err) else None,name+'_max_px':float(err.max()) if len(err) else None})
        if row['legacy_valid'] and row['current_valid']:
            for name in ['legacy','current']:
                errors['paired_'+name].extend(np.linalg.norm(np.array(candidates[name]['projected'])[mask]-gt[mask],axis=1).tolist())
            changed_points.extend(np.linalg.norm(np.array(candidates['current']['projected'])-np.array(candidates['legacy']['projected']),axis=1).tolist())
        rows.append(row)
    paired=[r for r in rows if r['legacy_mean_px'] is not None and r['current_mean_px'] is not None]
    report={'schema':'court_paired_regression_v1','samples':len(rows),'no_visible_gt':no_gt,
        'units':'original source pixels; same visible GT IDs in every condition',
        'raw_head_comparison':'identical frozen KP/LINE inputs; weights and decoding unchanged; no new dense-head accuracy claim',
        'visible_gt_points':raw_visible,'raw_missing_visible_gt_points':raw_missing,
        'solver_success':{n:sum(r[n+'_valid'] for r in rows) for n in ['legacy','current']},
        'lost_geometry':sum(r['legacy_valid'] and not r['current_valid'] for r in rows),
        'gained_geometry':sum(not r['legacy_valid'] and r['current_valid'] for r in rows),
        'paired_cases':len(paired),'paired_frame_mean_delta_px':float(np.mean([r['current_mean_px']-r['legacy_mean_px'] for r in paired])) if paired else None,
        'paired_worse_by_over_1px':sum(r['current_mean_px']-r['legacy_mean_px']>1 for r in paired),
        'paired_better_by_over_1px':sum(r['legacy_mean_px']-r['current_mean_px']>1 for r in paired),
        'errors':{k:stats(v) for k,v in errors.items()},'projected_points_change':stats(changed_points)}
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'paired_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for offset in range(0,len(rows),1000):
        with (args.output/f'paired-cases-{offset//1000:03d}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
            writer.writerows({k:f'{v:.6g}' if isinstance(v,float) else v for k,v in r.items()} for r in rows[offset:offset+1000])
    (args.output/'paired_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
