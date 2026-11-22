"""Aggregate fixed acceptance and regression contracts; never fit from labels."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import cv2

from src.tasks.court_detection.geometry.homography import court_template_xy

BASE=Path(__file__).resolve().parent
OUT=Path('outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117')
MODES=('legacy','fixed_support','fixed_support_rgb')


def stats(values):
    x=np.asarray(values,float)
    return {'n':int(x.size),'mean':float(x.mean()) if x.size else None,
        'median':float(np.median(x)) if x.size else None,'p95':float(np.percentile(x,95)) if x.size else None}


def summarize(rows):
    report={}
    for mode in MODES:
        values=[r['conditions'][mode] for r in rows]
        labelled=[r['conditions'][mode] for r in rows if r['reference_source']]
        counts=Counter('TP' if v['correct'] and v['accepted'] else 'FN' if v['correct'] else 'FP' if v['accepted'] else 'TN' for v in labelled)
        available=sum(v['any_correct_candidate'] for v in labelled)
        errors=[e for v in labelled if v['errors_px'] is not None for e in v['errors_px']]
        accepted_errors=[e for v in labelled if v['accepted'] and v['errors_px'] is not None for e in v['errors_px']]
        reasons=Counter(reason for v in values if v['quality'] and not v['accepted'] for reason in v['quality']['reasons'])
        reasons['visible_area_gate']+=sum(bool(v['quality'] and v['quality']['accepted'] and not v['accepted']) for v in values)
        reasons=+reasons
        frame_means=[v['mean_error_px'] for v in labelled if v['mean_error_px'] is not None]
        valid_points=np.asarray([v['points'] for v in values if v['points'] is not None])
        accepted_points=np.asarray([v['points'] for v in values if v['accepted'] and v['points'] is not None])
        dispersion=None;accepted_dispersion=None;visible_dispersion=None
        if rows and len({r['camera'] for r in rows})==1 and rows[0]['dataset']=='meiji_4cam':
            if len(valid_points)>1:dispersion=stats(np.linalg.norm(valid_points-np.median(valid_points,axis=0),axis=-1))
            if len(accepted_points)>1:accepted_dispersion=stats(np.linalg.norm(accepted_points-np.median(accepted_points,axis=0),axis=-1))
            truth=reference(rows[0]['camera']);mask=(truth[:,0]>=0)&(truth[:,0]<=1919)&(truth[:,1]>=0)&(truth[:,1]<=1079)
            if len(valid_points)>1:visible_dispersion=stats(np.linalg.norm(valid_points[:,mask]-np.median(valid_points[:,mask],axis=0),axis=-1))
        report[mode]={'cases':len(values),'labelled':len(labelled),'geometry_valid':sum(v['geometry_valid'] for v in values),
            'accepted':sum(v['accepted'] for v in values),**{k:counts[k] for k in ('TP','FN','FP','TN')},
            'correct_acceptance_rate':counts['TP']/(counts['TP']+counts['FN']) if counts['TP']+counts['FN'] else None,
            'wrong_rejection_rate':counts['TN']/(counts['TN']+counts['FP']) if counts['TN']+counts['FP'] else None,
            'correct_candidate_available':available,'rescued_over_available':counts['TP']/available if available else None,
            'rank_lost':sum(v['any_correct_accepted_candidate'] and not(v['accepted'] and v['correct']) for v in labelled),
            'all_correct_candidates_gated':sum(v['any_correct_candidate'] and not v['any_correct_accepted_candidate'] for v in labelled),
            'frame_mean_error_px':stats(frame_means),'all_point_errors_px':stats(errors),'accepted_point_errors_px':stats(accepted_errors),
            'pck_native_px':{str(t):float(np.mean(np.asarray(errors)<=t)) if errors else None for t in (5,10,20)},
            'clip_dispersion_all14_px':dispersion,'clip_dispersion_reference_visible_px':visible_dispersion,
            'accepted_clip_dispersion_all14_px':accepted_dispersion,'rejection_reasons':dict(reasons),
            'seconds_per_image':stats([r['seconds'][mode] for r in rows])}
    return report


def regression(rows):
    result={}
    for mode in MODES[1:]:
        common=[r for r in rows if r['conditions']['legacy']['geometry_valid'] and r['conditions'][mode]['geometry_valid']]
        lost=[r for r in rows if r['conditions']['legacy']['geometry_valid'] and not r['conditions'][mode]['geometry_valid']]
        gained=[r for r in rows if not r['conditions']['legacy']['geometry_valid'] and r['conditions'][mode]['geometry_valid']]
        def errors(name, subset):return [e for r in subset for e in (r['conditions'][name]['errors_px'] or [])]
        accepted_common=[r for r in rows if r['conditions']['legacy']['accepted'] and r['conditions'][mode]['accepted']]
        gt_common=[r for r in common if r['conditions']['legacy']['mean_error_px'] is not None and r['conditions'][mode]['mean_error_px'] is not None]
        deltas=[r['conditions'][mode]['mean_error_px']-r['conditions']['legacy']['mean_error_px'] for r in gt_common]
        change=[float(x) for r in common for x in np.linalg.norm(np.asarray(r['conditions'][mode]['points'])-np.asarray(r['conditions']['legacy']['points']),axis=1)]
        result[mode]={'paired_geometry_cases':len(common),'lost_geometry':len(lost),'gained_geometry':len(gained),
            'legacy_common_error_px':stats(errors('legacy',common)),'new_common_error_px':stats(errors(mode,common)),
            'lost_legacy_error_px':stats(errors('legacy',lost)),'gained_new_error_px':stats(errors(mode,gained)),
            'lost_status':dict(Counter(r['conditions'][mode]['status'] for r in lost)),
            'paired_gt_cases':len(gt_common),'paired_frame_mean_delta_px':stats(deltas),
            'paired_worse_over_1px':sum(d>1 for d in deltas),'paired_better_over_1px':sum(d < -1 for d in deltas),
            'paired_all14_projection_change_px':stats(change),
            'paired_accepted_cases':len(accepted_common),
            'paired_accepted_legacy_error_px':stats(errors('legacy',accepted_common)),
            'paired_accepted_new_error_px':stats(errors(mode,accepted_common)),
            'lost_acceptance':sum(r['conditions']['legacy']['accepted'] and not r['conditions'][mode]['accepted'] for r in rows),
            'gained_acceptance':sum(not r['conditions']['legacy']['accepted'] and r['conditions'][mode]['accepted'] for r in rows)}
    return result


def reference(camera):
    references=json.loads(Path('knowledge/runs/run-court-white-line-reference-20261116/reference/references.json').read_text())
    matrix=next(r['matrix_court_metres_to_image_pixels'] for r in references['cameras'] if r['camera']==camera)
    return cv2.perspectiveTransform(court_template_xy()[None].astype(float),np.asarray(matrix))[0]


def historical_rescue(rows):
    available=[]
    old=Path('outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116/meiji4-evaluation')
    for row in rows:
        truth=reference(row['camera']);mask=(truth[:,0]>=0)&(truth[:,0]<=1919)&(truth[:,1]>=0)&(truth[:,1]<=1079)
        frozen=json.loads((old/row['key']/'evaluation.json').read_text())
        for candidate in frozen['legacy_candidates']:
            if candidate['points'] is None:continue
            errors=np.linalg.norm(np.asarray(candidate['points'])[mask]-truth[mask],axis=1)
            if errors.mean()<=10 and errors.max()<=25:
                available.append(row);break
    assert len(available)==184
    result={}
    for camera in ('all','cam0','cam1','cam2','cam3'):
        subset=[r for r in available if camera=='all' or r['camera']==camera];result[camera]={}
        for mode in MODES:
            counts=Counter()
            for row in subset:
                value=row['conditions'][mode]
                reason=('recovered' if value['correct'] and value['accepted'] else 'ranking' if value['any_correct_accepted_candidate']
                        else 'gate' if value['any_correct_candidate'] else 'no_correct_candidate')
                counts[reason]+=1
            result[camera][mode]={'available':len(subset),**{k:counts[k] for k in ('recovered','ranking','gate','no_correct_candidate')},
                'rescue_rate':counts['recovered']/len(subset) if subset else None}
    return result


def main():
    records=json.loads((OUT/'manifest.json').read_text())['records']
    rows=[json.loads((OUT/'evaluation'/r['key']/'evaluation.json').read_text()) for r in records]
    for record,row in zip(records,rows,strict=True):
        if 'gt_valid' in record and not any(record['gt_valid']):row['reference_source']=None
    protocol=json.loads((BASE/'protocol.json').read_text())
    report={'protocol':'protocol.json','datasets':{},'regression':{},'holdout':{}}
    for dataset in sorted({r['dataset'] for r in rows}):
        subset=[r for r in rows if r['dataset']==dataset]
        report['datasets'][dataset]={'all':summarize(subset)}
        for camera in sorted({r['camera'] for r in subset}):report['datasets'][dataset][camera]=summarize([r for r in subset if r['camera']==camera])
        report['regression'][dataset]=regression(subset)
    excluded=set(protocol['reference_clips']+protocol['pilot_clips'])
    holdout=[r for r in rows if r['dataset']=='meiji_4cam' and int(r['clip_id'].split('_')[-1]) not in excluded]
    assert len(holdout)==224
    report['holdout']['all']=summarize(holdout)
    for camera in sorted({r['camera'] for r in holdout}):report['holdout'][camera]=summarize([r for r in holdout if r['camera']==camera])
    report['historical_approved_reference']={'legacy':{'TP':47,'FN':24,'FP':105,'TN':100},
        'acceptance_only':{'TP':66,'FN':12,'FP':116,'TN':82},'previous_postprocess':{'TP':64,'FN':15,'FP':116,'TN':81}}
    report['frozen_184_pool_rescue']=historical_rescue([r for r in rows if r['dataset']=='meiji_4cam'])
    meiji=[r for r in rows if r['dataset']=='meiji_4cam']
    clips={clip:[r for r in meiji if r['clip_id']==clip] for clip in sorted({r['clip_id'] for r in meiji})}
    assert len(clips)==69 and all(len(values)==4 for values in clips.values())
    report['meiji4_clip_counts']={mode:{'all4_accepted':sum(all(r['conditions'][mode]['accepted'] for r in values) for values in clips.values()),
        'all4_correct_diagnostic_H':sum(all(r['conditions'][mode]['correct'] for r in values) for values in clips.values())} for mode in MODES}
    out=OUT/'report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    table=[]
    for r in rows:
        for mode in MODES:
            v=r['conditions'][mode]
            table.append({'key':r['key'],'dataset':r['dataset'],'camera':r['camera'],'condition':mode,
                **{k:v[k] for k in ('accepted','correct','geometry_valid','region_index','mean_error_px','max_error_px','any_correct_candidate','any_correct_accepted_candidate')},
                'reasons':';'.join(v['quality']['reasons']) if v['quality'] else v['status']})
    with (out/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    for index in range(0,len(table),800):
        with (out/f'cases-{index//800:02d}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table[index:index+800])
    print(json.dumps(report['datasets']['meiji_4cam']['all'],indent=2))


if __name__=='__main__':main()
