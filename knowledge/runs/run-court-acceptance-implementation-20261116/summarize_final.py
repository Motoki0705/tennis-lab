"""Three-condition acceptance tables on a fixed, approved reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from current_geometry.homography import court_template_xy
from current_geometry.quality import CourtQualityConfig, assess_court_quality

BASE=Path(__file__).resolve().parent


def truth_for(record, references):
    h,w=record['source_size_hw']
    if record['dataset']=='meiji_4cam':
        ref=next(r for r in references['cameras'] if r['camera']==record['camera'])
        assert ref['human_reviewed'] and ref['approved_for_evaluation']
        xy=cv2.perspectiveTransform(court_template_xy().astype(float)[None],np.array(ref['matrix_court_metres_to_image_pixels']))[0]
        valid=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        return xy,valid,'human_approved_meiji4cam'
    if 'gt_xy' in record:
        return np.asarray(record['gt_xy']),np.asarray(record['gt_valid'],bool),'dataset_ground_truth'
    if record['dataset']=='meiji_3cam' and record['clip_id']=='video_000/clip_000':
        path=Path('data/tennis_multiview/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/manual_court_kp_result.json')
        manual=json.loads(path.read_text());cam=int(record['camera'][-1])
        return np.array(manual['keypoints'])[cam,0]*[w-1,h-1],np.array(manual['visibility'])[cam,0]>0,'existing_manual_not_reconfirmed_in_this_run'
    return None,None,None


def judge(candidate, raw, record, truth, mask, quality):
    h,w=record['source_size_hw'];diagonal=np.hypot(w,h);region=candidate['region'];result=dict(candidate)
    result['quality']=quality
    result.update(legacy_accepted=False,new_accepted=False,inliers=0,inlier_score=0.,correct=False,
        mean_error_px=None,max_error_px=None,errors_px=None,points=None,reference_present=truth is not None)
    if candidate['matrix'] is None:return result
    points=np.asarray(candidate['projected'])+np.array(region[:2]);observed=raw['raw_xy']+np.array(region[:2])
    inliers=raw['valid']&(np.linalg.norm(points-observed,axis=1)<=.005*diagonal)
    polygon=points[[0,1,3,2]].astype(np.float32);convex=cv2.isContourConvex(polygon)
    left,top,right,bottom=record['content_region']
    inside=bool(((points[:,0]>=left)&(points[:,0]<=right-1)&(points[:,1]>=top)&(points[:,1]<=bottom-1)).all())
    fullarea=abs(cv2.contourArea(polygon))/(h*w)
    intersection=0.
    if convex:
        intersection=float(cv2.intersectConvexConvex(polygon,np.array([[left,top],[right-1,top],[right-1,bottom-1],[left,bottom-1]],np.float32))[0])/(h*w)
    result['legacy_accepted']=bool(candidate['status']=='ok' and inside and convex and fullarea>=.01 and inliers.sum()>=8)
    result['new_accepted']=bool(candidate['status']=='ok' and quality['accepted'] and convex and intersection>=.01)
    result['inliers']=int(inliers.sum());result['inlier_score']=float(raw['scores'][inliers].sum());result['points']=points.tolist()
    if truth is not None and mask.any():
        error=np.linalg.norm(points[mask]-truth[mask],axis=1)
        result['mean_error_px']=float(error.mean());result['max_error_px']=float(error.max());result['errors_px']=error.tolist()
        # The agreed thresholds are at 1920x1080. Other domains also report
        # original-pixel errors; this optional class uses the same diagonal ratio.
        scale=diagonal/np.hypot(1920,1080)
        result['correct']=bool(error.mean()<=10*scale and error.max()<=25*scale)
    return result


def process(task):
    record,raw_root,legacy_root,current_root,references,out=task
    cv2.setNumThreads(1);truth,mask,source=truth_for(record,references)
    arrays=[np.load(raw_root/record['key']/f'raw_{i:02d}.npz') for i in range(len(record['regions']))]
    result={'key':record['key'],'dataset':record['dataset'],'camera':record['camera'],'reference_source':source if mask is not None and mask.any() else None,'conditions':{}}
    for mode,root in [('legacy',legacy_root),('current',current_root)]:
        geometry=json.loads((root/record['key']/'geometry.json').read_text());candidates=[]
        for c,raw in zip(geometry['candidates'],arrays,strict=True):
            q=assess_court_quality(None if c['matrix'] is None else np.asarray(c['matrix']),raw['raw_xy'],
                raw['valid']&np.array(c['selected'],bool),raw['line_probability'],
                image_size_hw=tuple(raw['image_size_hw']),evaluation_size_hw=tuple(record['source_size_hw']),config=CourtQualityConfig())
            candidates.append(judge(c,raw,record,truth,mask,q.diagnostics()))
        conditions=['legacy','acceptance_only'] if mode=='legacy' else ['postprocess_and_acceptance']
        for condition in conditions:
            if condition=='legacy':
                chosen=max(candidates,key=lambda c:(c['legacy_accepted'],c['inliers'],c['inlier_score']))
                accepted=chosen['legacy_accepted']
            else:
                qualified=[c for c in candidates if c['new_accepted']]
                choices=qualified or candidates
                chosen=min(choices,key=lambda c:(float('inf') if c['quality']['score'] is None else c['quality']['score'],float('inf') if c['score'] is None else c['score'],c['region_index']))
                accepted=bool(qualified)
            result['conditions'][condition]={'accepted':accepted,'correct':chosen['correct'],
                'region_index':chosen['region_index'],'region':chosen['region'],'geometry_status':chosen['status'],
                'geometry_valid':chosen['matrix'] is not None,'points':chosen['points'],'quality':chosen['quality'],
                'mean_error_px':chosen['mean_error_px'],'max_error_px':chosen['max_error_px'],'errors_px':chosen['errors_px'],
                'any_correct_candidate':bool(truth is not None and any(c['correct'] for c in candidates))}
        result[mode+'_candidates']=candidates
    destination=out/record['key']/'evaluation.json';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,allow_nan=False))
    return result


def summarize(rows):
    result={}
    for dataset in sorted({r['dataset'] for r in rows}):
        result[dataset]={}
        for camera in ['all',*sorted({r['camera'] for r in rows if r['dataset']==dataset})]:
            subset=[r for r in rows if r['dataset']==dataset and (camera=='all' or r['camera']==camera)]
            result[dataset][camera]={}
            for name in ['legacy','acceptance_only','postprocess_and_acceptance']:
                values=[r['conditions'][name] for r in subset];labelled=[r['conditions'][name] for r in subset if r['reference_source']]
                counts={'TP':0,'FN':0,'FP':0,'TN':0}
                for v in labelled:counts['TP' if v['accepted'] and v['correct'] else 'FN' if v['correct'] else 'FP' if v['accepted'] else 'TN']+=1
                means=[v['mean_error_px'] for v in values if v['mean_error_px'] is not None]
                errors=[x for v in values if v['errors_px'] is not None for x in v['errors_px']]
                points=np.asarray([v['points'] for v in values if v['points'] is not None])
                dispersion=None
                if len(points)>1:dispersion=np.linalg.norm(points-np.median(points,axis=0),axis=-1)
                available=sum(v['any_correct_candidate'] for v in labelled)
                result[dataset][camera][name]={'cases':len(values),'labelled':len(labelled),'accepted':sum(v['accepted'] for v in values),
                    'geometry_valid':sum(v['geometry_valid'] for v in values),**counts,
                    'correct_acceptance_rate':counts['TP']/(counts['TP']+counts['FN']) if counts['TP']+counts['FN'] else None,
                    'wrong_rejection_rate':counts['TN']/(counts['TN']+counts['FP']) if counts['TN']+counts['FP'] else None,
                    'correct_candidate_available_cases':available,'correct_accepted_over_available':counts['TP']/available if available else None,
                    'frame_mean_error_median_px':float(np.median(means)) if means else None,
                    'point_error_p95_px':float(np.percentile(errors,95)) if errors else None,
                    'clip_dispersion_median_px':float(np.median(dispersion)) if dispersion is not None and dataset=='meiji_4cam' and camera!='all' and camera!='image' else None,
                    'clip_dispersion_p95_px':float(np.percentile(dispersion,95)) if dispersion is not None and dataset=='meiji_4cam' and camera!='all' and camera!='image' else None}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--max-cases',type=int)
    args=parser.parse_args();refs=json.loads((BASE/'references.json').read_text());protocol=json.loads((BASE/'protocol.json').read_text())
    assert refs['approved_for_evaluation'] and hashlib.sha256((BASE/'references.json').read_bytes()).hexdigest()==protocol['reference_sha256']
    records=json.loads((args.raw/'manifest.json').read_text())['records'][:args.max_cases]
    for r in records:
        for root in [args.legacy,args.current]:
            if not (root/r['key']/'geometry.json').exists():raise FileNotFoundError(root/r['key'])
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(process,[(r,args.raw,args.legacy,args.current,refs,args.output) for r in records]))
    (args.output/'summary.json').write_text(json.dumps({'protocol':protocol,'partial':args.max_cases is not None,'summary':summarize(rows)},indent=2,allow_nan=False))
    print(json.dumps(summarize(rows),indent=2),flush=True)


if __name__=='__main__':main()
