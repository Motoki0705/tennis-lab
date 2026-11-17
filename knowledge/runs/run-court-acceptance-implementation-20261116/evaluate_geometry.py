"""Run pinned legacy or current CPU geometry on frozen KP/LINE evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def run_case(task):
    raw_root,out,record,mode=task
    cv2.setNumThreads(1)
    if mode=='legacy':
        from legacy_geometry.hybrid_homography import estimate_hybrid_homography, HybridHomographyConfig
    else:
        from src.tasks.court_detection.geometry.hybrid_homography import estimate_hybrid_homography, HybridHomographyConfig
    destination=out/record['key']/'geometry.json'
    if destination.exists():return {'key':record['key'],'cached':True}
    template=court_template_xy().astype(float);edges=np.array(COURT_HOMOGRAPHY_EDGES)
    candidates=[];start=time.monotonic()
    for index,region in enumerate(record['regions']):
        data=np.load(raw_root/record['key']/f'raw_{index:02d}.npz')
        observed=data['raw_xy'].astype(float);observed[~data['valid']]=np.nan
        result=estimate_hybrid_homography(template,observed,data['scores'],data['line_probability'],
            edges=edges,image_size_hw=tuple(data['image_size_hw']),config=HybridHomographyConfig())
        candidate={'region_index':index,'region':region,'status':result.status,
            'matrix':result.matrix,'projected':result.projected,'selected':result.selected,
            'line':result.best.line if result.best else None,'score':result.best.score if result.best else None,
            'candidate_count':result.candidate_count,'refined_count':result.refined_count,
            'kp_only_status':result.kp_only.status,'alternative_score_gap':result.alternative_score_gap,
            'observation_diagnostics':getattr(result,'observation_diagnostics',None)}
        candidates.append(clean(candidate))
    result={'key':record['key'],'mode':mode,'candidates':candidates,'seconds':time.monotonic()-start}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result,allow_nan=False))
    temporary.replace(destination)
    return {'key':record['key'],'seconds':result['seconds']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['legacy','current'],required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--max-cases',type=int)
    args=parser.parse_args();manifest=json.loads((args.raw/'manifest.json').read_text())
    records=manifest['records'][:args.max_cases];args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,(args.raw,args.output,r,args.mode)) for r in records]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    (args.output/'complete.json').write_text(json.dumps({'mode':args.mode,'cases':len(records),
        'regions':sum(len(r['regions']) for r in records),'seconds':time.monotonic()-started},indent=2))
    print('complete',len(records),flush=True)


if __name__=='__main__':main()
