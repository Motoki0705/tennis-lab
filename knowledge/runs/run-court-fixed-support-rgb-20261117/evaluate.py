"""Paired legacy/(a)/(a+b), with frozen LINE arrays and all rejected diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

sys.path.insert(0,str(Path('knowledge/runs/run-court-acceptance-implementation-20261116').resolve()))
from legacy_geometry import hybrid_homography as legacy
from legacy_geometry.line_evidence import LineEvidence as LegacyEvidence
from summarize_final import truth_for
from src.tasks.court_detection.geometry import hybrid_homography as fixed
from src.tasks.court_detection.geometry.homography import court_template_xy, COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.line_evidence import LineEvidence
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig, assess_region_prediction
from src.tasks.court_detection.model_io.contracts import CourtKeypointPrediction, CourtLinePrediction

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
OLD=Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116')
REF=Path('knowledge/runs/run-court-white-line-reference-20261116/reference/references.json')
MODES=('legacy','fixed_support','fixed_support_rgb')


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def prepare(raw, destination):
    h,w=map(int,raw['image_size_hw'])
    if destination.exists():
        data=np.load(destination)
        return LineEvidence(w,h,float(np.hypot(w,h)),data['distance'],data['points'],data['weights'],data['tangents'],data['coherence'],cKDTree(data['all_points']))
    evidence=LineEvidence.from_probability(raw['line_probability'],(h,w))
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,distance=evidence.distance,points=evidence.points,weights=evidence.weights,
        tangents=evidence.tangents,coherence=evidence.coherence,all_points=evidence.tree.data)
    return evidence


def typed(raw, geometry):
    # Dense KP heatmaps / LINE logits were not cached and are not consumed by
    # this boundary. Empty tensors explicitly mark those unavailable fields.
    return CourtPrediction({'kp':CourtKeypointPrediction(torch.from_numpy(raw['raw_xy'][:,None]),
        torch.from_numpy(raw['scores'][:,None]),torch.from_numpy(raw['valid'][:,None]),torch.empty((14,0,0))),
        'line':CourtLinePrediction(torch.from_numpy(raw['line_probability']),torch.empty((0,0)))},
        tuple(map(int,raw['image_size_hw'])),tuple(map(int,raw['native_size_hw'])),geometry,'synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1')


def label(candidate,truth,mask,size):
    candidate.update(correct=False,mean_error_px=None,max_error_px=None,errors_px=None)
    points=candidate['points']
    if points is not None and truth is not None and mask.any():
        error=np.linalg.norm(np.asarray(points)[mask]-truth[mask],axis=1)
        scale=np.hypot(*size)/np.hypot(1080,1920)
        candidate.update(correct=bool(error.mean()<=10*scale and error.max()<=25*scale),mean_error_px=float(error.mean()),
                         max_error_px=float(error.max()),errors_px=error.tolist())
    return candidate


def area_ok(points,record):
    if points is None:return False
    h,w=record['source_size_hw'];left,top,right,bottom=record['content_region']
    polygon=np.asarray(points)[[0,1,3,2]].astype(np.float32)
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):return False
    content=np.array([[left,top],[right-1,top],[right-1,bottom-1],[left,bottom-1]],np.float32)
    return cv2.intersectConvexConvex(polygon,content)[0]/(h*w)>=.01


def process(record):
    cv2.setNumThreads(1);torch.set_num_threads(1)
    destination=OUT/'evaluation'/record['key']/'evaluation.json'
    if destination.exists():return {'key':record['key'],'cached':True}
    image=cv2.cvtColor(cv2.imread(str(OUT/'rgb'/record['key']/'frame.png')),cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(image.tobytes()).hexdigest()==record['rgb_sha256']
    raw_root=OLD/record['old_raw_group']/record['key']
    candidates={mode:[] for mode in MODES};seconds={mode:0. for mode in MODES}
    world=court_template_xy().astype(float);edges=np.asarray(COURT_HOMOGRAPHY_EDGES)
    start=time.monotonic()
    for index,region in enumerate(record['regions']):
        raw=dict(np.load(raw_root/f'raw_{index:02d}.npz'))
        evidence_path=OUT/'line-evidence'/record['key']/f'evidence_{index:02d}.npz'
        try:evidence=prepare(raw,evidence_path)
        except ValueError as error:
            if str(error)!='insufficient_or_diffuse_line_evidence':raise
            evidence=None
        def prepared(*args,**kwargs):
            if evidence is None:raise ValueError('insufficient_or_diffuse_line_evidence')
            return evidence
        observed=raw['raw_xy'].astype(float);observed[~raw['valid']]=np.nan
        with patch.object(LineEvidence,'from_probability',side_effect=prepared),patch.object(LegacyEvidence,'from_probability',side_effect=prepared):
            tick=time.monotonic()
            old=legacy.estimate_hybrid_homography(world,observed,raw['scores'],raw['line_probability'],edges=edges,
                image_size_hw=tuple(raw['image_size_hw']),config=legacy.HybridHomographyConfig())
            seconds['legacy']+=time.monotonic()-tick
            points=None if old.matrix is None else old.projected+np.array(region[:2])
            inliers=np.zeros(14,bool) if points is None else raw['valid']&(np.linalg.norm(old.projected-raw['raw_xy'],axis=1)<=.005*np.hypot(*record['source_size_hw']))
            left,top,right,bottom=record['content_region']
            inside=points is not None and bool(((points[:,0]>=left)&(points[:,0]<=right-1)&(points[:,1]>=top)&(points[:,1]<=bottom-1)).all())
            candidates['legacy'].append({'region_index':index,'region':region,'points':points,'accepted':bool(old.status=='ok' and inside and inliers.sum()>=8 and area_ok(points,record)),
                'geometry_valid':old.matrix is not None,'status':old.status,'quality':None,'inliers':int(inliers.sum()),'inlier_score':float(raw['scores'][inliers].sum()),
                'model_score':None if old.best is None else old.best.score,'selected':old.selected})
            tick=time.monotonic()
            new=fixed.estimate_hybrid_homography(world,observed,raw['scores'],raw['line_probability'],edges=edges,
                image_size_hw=tuple(raw['image_size_hw']),config=fixed.HybridHomographyConfig())
            geometry_seconds=time.monotonic()-tick
            prediction=typed(raw,new)
            for mode,method in (('fixed_support','model'),('fixed_support_rgb','rgb_paint')):
                tick=time.monotonic()
                assessed=assess_region_prediction(prediction,image,tuple(region),CourtRegionSearchConfig(refinement_method=method))
                seconds[mode]+=geometry_seconds+time.monotonic()-tick
                if method=='model':points=None if new.matrix is None else new.projected+np.array(region[:2])
                else:
                    matrix=assessed.paint.fitted_matrix
                    points=None if matrix is None else fixed.project_candidate(matrix,world)
                quality=assessed.quality.diagnostics()
                candidates[mode].append({'region_index':index,'region':region,'points':points,
                    'accepted':bool(new.status=='ok' and quality['accepted'] and area_ok(points,record)),
                    'geometry_valid':points is not None,'model_geometry_valid':new.matrix is not None,
                    'status':new.status if new.status!='ok' else ('ok' if quality['accepted'] else 'quality_rejected'),
                    'quality':quality,'model_score':None if new.best is None else new.best.score,'selected':new.selected,
                    'diagnostics':assessed.geometry_diagnostics()})
    # GT and reference are read only after every estimate and acceptance is fixed.
    refs=json.loads(REF.read_text());truth,mask,reference=truth_for(record,refs)
    conditions={}
    for mode in MODES:
        values=[label(c,truth,mask,record['source_size_hw']) for c in candidates[mode]]
        if mode=='legacy':chosen=max(values,key=lambda c:(c['accepted'],c['inliers'],c['inlier_score']))
        else:
            qualified=[c for c in values if c['accepted']]
            chosen=min(qualified or values,key=lambda c:(float('inf') if c['quality']['score'] is None else c['quality']['score'],c['region_index']))
        conditions[mode]={k:v for k,v in chosen.items() if k not in ('diagnostics','selected')}
        conditions[mode]['any_correct_candidate']=any(c['correct'] for c in values)
        conditions[mode]['any_correct_accepted_candidate']=any(c['correct'] and c['accepted'] for c in values)
    report={'key':record['key'],'dataset':record['dataset'],'camera':record['camera'],'clip_id':record['clip_id'],
        'reference_source':reference,'conditions':conditions,'candidates':candidates,'seconds':seconds,'wall_seconds':time.monotonic()-start}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(clean(report),allow_nan=False));temporary.replace(destination)
    return {'key':record['key'],'seconds':report['wall_seconds']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12);parser.add_argument('--dataset');parser.add_argument('--limit',type=int)
    args=parser.parse_args();protocol=json.loads((BASE/'protocol.json').read_text())
    for name,digest in protocol['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    records=json.loads((OUT/'manifest.json').read_text())['records']
    if args.dataset:records=[r for r in records if r['dataset']==args.dataset]
    if args.limit:records=records[:args.limit]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(process,record) for record in records]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    (OUT/f'complete-{args.dataset or "all"}.json').write_text(json.dumps({'cases':len(records),'seconds':time.monotonic()-started,'limit':args.limit}))


if __name__=='__main__':main()
