"""Change the evaluation reference only; keep predictions, gates and ranks frozen."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import numpy as np

from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

BASE=Path(__file__).resolve().parent
PREVIOUS=BASE.parent/'court-acceptance-implementation-20261116'
WORLD=court_keypoints_3d(STANDARD_COURT_CONFIG)[:14,:2].numpy().astype(float)
CONDITIONS=('legacy','acceptance_only','postprocess_and_acceptance')


def reference_points(camera):
    h=np.array(camera['matrix_court_metres_to_image_pixels']);p=np.c_[WORLD,np.ones(14)]@h.T;xy=p[:,:2]/p[:,2,None]
    valid=(xy[:,0]>=0)&(xy[:,0]<=1919)&(xy[:,1]>=0)&(xy[:,1]<=1079)
    return xy,valid


def error(points,truth,mask):
    if points is None:return {'correct':False,'mean_error_px':None,'max_error_px':None,'errors_px':None}
    d=np.linalg.norm(np.array(points)[mask]-truth[mask],axis=1)
    return {'correct':bool(d.mean()<=10 and d.max()<=25),'mean_error_px':float(d.mean()),'max_error_px':float(d.max()),'errors_px':d.tolist()}


def classification(v):
    return 'TP' if v['accepted'] and v['correct'] else 'FP' if v['accepted'] else 'FN' if v['correct'] else 'TN'


def summarize(rows):
    report={}
    for cam in ('all','cam0','cam1','cam2','cam3'):
        subset=[r for r in rows if cam=='all' or r['camera']==cam];report[cam]={}
        for reference in ('old','provisional','old_common_mask','provisional_common_mask'):
            report[cam][reference]={}
            for condition in CONDITIONS:
                values=[r[reference][condition] for r in subset];counts={k:0 for k in ('TP','FN','FP','TN')}
                for v in values:counts[classification(v)]+=1
                means=[v['mean_error_px'] for v in values if v['mean_error_px'] is not None]
                point_errors=[d for v in values if v['errors_px'] is not None for d in v['errors_px']]
                available=sum(v['any_correct_candidate'] for v in values)
                positions=np.array([v['points'] for v in values if v['points'] is not None])
                dispersion=np.linalg.norm(positions-np.median(positions,axis=0),axis=-1) if len(positions)>1 else None
                report[cam][reference][condition]={**counts,'cases':len(values),'accepted':sum(v['accepted'] for v in values),
                    'correct_acceptance_rate':counts['TP']/(counts['TP']+counts['FN']) if counts['TP']+counts['FN'] else None,
                    'wrong_rejection_rate':counts['TN']/(counts['TN']+counts['FP']) if counts['TN']+counts['FP'] else None,
                    'correct_candidate_available':available,'correct_accepted_over_available':counts['TP']/available if available else None,
                    'frame_mean_error_mean_px':float(np.mean(means)) if means else None,
                    'frame_mean_error_median_px':float(np.median(means)) if means else None,
                    'point_error_p95_px':float(np.percentile(point_errors,95)) if point_errors else None,
                    'clip_dispersion_p95_px':float(np.percentile(dispersion,95)) if dispersion is not None and cam!='all' else None}
    return report


def main():
    new_path=BASE/'reference/references.json';freeze=json.loads((BASE/'reference/freeze.json').read_text());assert hashlib.sha256(new_path.read_bytes()).hexdigest()==freeze['references.json']
    new=json.loads(new_path.read_text());assert new['approved_for_evaluation'] is False and new['evaluation_status']=='provisional_parent_authorized'
    old=json.loads((PREVIOUS/'references.json').read_text());out=BASE/'evaluation';out.mkdir(exist_ok=True)
    old_refs={r['camera']:reference_points(r) for r in old['cameras']};new_refs={r['camera']:reference_points(r) for r in new['cameras']}
    source_files=sorted((PREVIOUS/'meiji4-evaluation/meiji_4cam').glob('**/evaluation.json'));assert len(source_files)==276
    digest=hashlib.sha256();rows=[];table=[];flips=[];candidate_stats={}
    for path in source_files:
        digest.update(str(path.relative_to(PREVIOUS)).encode());digest.update(path.read_bytes());original=json.loads(path.read_text());cam=original['camera'];old_xy,old_mask=old_refs[cam];new_xy,new_mask=new_refs[cam];common=old_mask&new_mask
        row={'key':original['key'],'camera':cam,'references':{'old_visible_indices':np.flatnonzero(old_mask).tolist(),'new_visible_indices':np.flatnonzero(new_mask).tolist()}}
        for reference,xy,mask in [('old',old_xy,old_mask),('provisional',new_xy,new_mask),('old_common_mask',old_xy,common),('provisional_common_mask',new_xy,common)]:
            row[reference]={}
            for name in CONDITIONS:
                v=copy.deepcopy(original['conditions'][name]);v.update(error(v['points'],xy,mask))
                candidates=original['current_candidates' if name=='postprocess_and_acceptance' else 'legacy_candidates']
                v['any_correct_candidate']=any(error(c['points'],xy,mask)['correct'] for c in candidates)
                if reference=='old':
                    assert v['correct']==original['conditions'][name]['correct']
                    if v['mean_error_px'] is not None:assert abs(v['mean_error_px']-original['conditions'][name]['mean_error_px'])<1e-8
                row[reference][name]=v
                if reference in ('old','provisional'):
                    table.append({'key':row['key'],'reference':reference,'condition':name,'accepted':v['accepted'],'correct':v['correct'],'classification':classification(v),'mean_error_px':v['mean_error_px'],'max_error_px':v['max_error_px'],'correct_candidate_available':v['any_correct_candidate'],'selected_region_index':v['region_index']})
                    for c in candidates:
                        gate='legacy_accepted' if name=='legacy' else 'new_accepted';cc=error(c['points'],xy,mask)['correct'];label='TP' if c[gate] and cc else 'FP' if c[gate] else 'FN' if cc else 'TN'
                        stats=candidate_stats.setdefault(cam,{}).setdefault(reference,{}).setdefault(name,{k:0 for k in ('TP','FN','FP','TN')});stats[label]+=1
        for name in CONDITIONS:
            a,b=row['old'][name],row['provisional'][name]
            assert (a['accepted'],a['region_index'],a['points'],a['quality'])==(b['accepted'],b['region_index'],b['points'],b['quality'])
            if a['correct']!=b['correct']:flips.append({'key':row['key'],'condition':name,'accepted':a['accepted'],'old_correct':a['correct'],'new_correct':b['correct'],'old_mean_px':a['mean_error_px'],'new_mean_px':b['mean_error_px']})
        rows.append(row);target=out/row['key']/'evaluation.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(row,allow_nan=False))
    summary={'evaluation_status':'PROVISIONAL / 未再承認','approved_for_evaluation':False,'targets':{'false_accepts':0,'correct_acceptance_rate_min':.9},'tolerance':{'mean_error_px':10,'max_error_px':25},'reference_sha256':freeze['references.json'],'frozen_prediction_files_sha256':digest.hexdigest(),'predictions_acceptance_and_ranks_unchanged':True,'summary':summarize(rows),'candidate_confusion':candidate_stats}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(out/'label-changes.json').write_text(json.dumps(flips,indent=2))
    with (out/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    for cam in ('all','cam0','cam1','cam2','cam3'):
        for reference in ('old','provisional'):
            for name in CONDITIONS:
                v=summary['summary'][cam][reference][name];print(cam,reference,name,{k:v[k] for k in ('TP','FN','FP','TN','correct_acceptance_rate','correct_candidate_available','correct_accepted_over_available','frame_mean_error_median_px')},flush=True)


if __name__=='__main__':main()
