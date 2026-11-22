"""Pool measured frame/line centres; retain every failed observation explicitly."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

import fit_reference as fit

BASE=Path(__file__).resolve().parent


def main():
    source=BASE/'fit-a2/references.json';working=json.loads(source.read_text());out=BASE/'reference';out.mkdir(exist_ok=True)
    result={k:v for k,v in working.items() if k!='cameras'}
    result['schema']='court_paint_reference_v2'
    result['config']={**fit.CFG,'final_stage_radius_px':12.,'final_stage_line_fit':'per_frame','minimum_frames_per_local_line':1,'max_samples_per_frame_line':24,'minimum_successful_frames_per_sampled_clip':2,'minimum_frames_per_pooled_line':8}
    result['provenance']={'parent_note':'https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6914138836','instruction_turn':'35c6b7b214bd4c70b7a180a6913a1867','authorization':'Provisional evaluation explicitly requested; reapproval is still pending.','model_outputs_used_for_reference':False}
    result['cameras']=[];residual_rows=[];motion=[]
    for old in working['cameras']:
        cam=old['camera'];path=BASE/'fit-final'/f'{cam}-evidence.json';e=json.loads(path.read_text());initial=np.array(old['initial_matrix']);groups=e['fit_groups'];models=e['frame_models']
        frames=sorted({g['frame'] for g in groups});coverage=Counter(e['frames'][i]['clip_id'] for i in frames)
        if len(coverage)!=5 or min(coverage.values())<2:
            result['cameras'].append({'camera':cam,'matrix_court_metres_to_image_pixels':None,'status':'insufficient_temporal_coverage','human_reviewed':False,'approved_for_evaluation':False});continue
        counts=Counter(g['line_index'] for g in groups)
        used_groups=[g for g in groups if counts[g['line_index']]>=8]
        if len({g['line_index'] for g in used_groups})<5:raise RuntimeError((cam,'insufficient_temporally_supported_lines'))
        points=np.concatenate([g['points'] for g in used_groups]);labels=np.concatenate([np.full(len(g['points']),g['line_index']) for g in used_groups])
        h,diagnostic=fit.fit_h(initial,points,labels)
        if not diagnostic['success'] or diagnostic['rank']<8 or not fit.valid_h(h):raise RuntimeError((cam,diagnostic))
        old_lines,new_lines=fit.image_lines(initial),fit.image_lines(h);reports=[];raw_points=[]
        for index,name in enumerate(fit.NAMES):
            accepted=[o for o in e['independent_lines'] if o['line_index']==index and o['supported_by_frame_geometry'] and models[o['frame']]['matrix'] is not None]
            pts=np.concatenate([o['points'] for o in accepted]) if accepted else np.empty((0,2))
            stats={mode:fit.distribution(np.c_[pts,np.ones(len(pts))]@lines[index]) for mode,lines in [('old',old_lines),('new',new_lines)]}
            signed=[]
            for o in accepted:
                frame=o['frame'];xy=np.asarray(o['points']);r_old=np.c_[xy,np.ones(len(xy))]@old_lines[index];r_new=np.c_[xy,np.ones(len(xy))]@new_lines[index]
                row={'camera':cam,'clip_id':e['frames'][frame]['clip_id'],'frame_index':e['frames'][frame]['frame_index'],'line':name,'points':len(xy),**{f'{mode}_{key}':value for mode,r in [('old',r_old),('new',r_new)] for key,value in fit.distribution(r).items()}}
                residual_rows.append(row);signed.append(float(np.median(r_new)))
                raw_points.extend([{'frame':frame,'line':index,'x':round(float(x),6),'y':round(float(y),6)} for x,y in xy])
            status='supported' if len(accepted)>=8 else 'insufficient_temporal_support' if accepted else 'not_measured'
            reports.append({'name':name,'status':status,'supported_for_fit':len(accepted)>=8,'frames':len(accepted),'residuals':stats,'frame_signed_median_range_px':float(np.ptp(signed)) if signed else None})
        ref_xy=fit.project(h);visible=(ref_xy[:,0]>=0)&(ref_xy[:,0]<=1919)&(ref_xy[:,1]>=0)&(ref_xy[:,1]<=1079)
        frame_reports=[]
        for i,m in enumerate(models):
            delta=np.linalg.norm(fit.project(np.array(m['matrix']))[visible]-ref_xy[visible],axis=1) if m['matrix'] is not None else None
            frame_reports.append({'clip_id':m['frame']['clip_id'],'frame_index':m['frame']['frame_index'],'status':m['diagnostic']['status'],'mean_query_displacement_px':float(delta.mean()) if delta is not None else None,'max_query_displacement_px':float(delta.max()) if delta is not None else None})
        failures=[{'clip_id':m['frame']['clip_id'],'frame_index':m['frame']['frame_index'],'reason':m['diagnostic']['status']} for m in models if m['matrix'] is None]
        entry={'camera':cam,'human_reviewed':False,'approved_for_evaluation':False,'evaluation_status':'provisional_parent_authorized','matrix_court_metres_to_image_pixels':h.tolist(),'initial_matrix':initial.tolist(),'diagnostic':diagnostic,'lines':reports,'successful_frames':len(frames),'failed_frames':failures,'frame_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'visible_reference_indices':np.flatnonzero(visible).tolist()}
        result['cameras'].append(entry);motion.append({'camera':cam,'frames':frame_reports})
        with (out/f'{cam}-measured-centres.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['frame','line','x','y'],lineterminator='\n');writer.writeheader();writer.writerows(raw_points)
        print(cam,'successful_frames',len(frames),'failed_frames',failures,'line ranges',[(r['name'],None if r['frame_signed_median_range_px'] is None else round(r['frame_signed_median_range_px'],2)) for r in reports],flush=True)
    with (out/'line-frame-residuals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(residual_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(residual_rows)
    (out/'references.json').write_text(json.dumps(result,indent=2,allow_nan=False));(out/'frame-stability.json').write_text(json.dumps(motion,indent=2,allow_nan=False))
    # Freeze before any prediction is read by the separate comparison script.
    identity={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [out/'references.json',BASE/'frames.json',BASE/'net-exclusions.json',BASE/'fit_reference.py',BASE/'balance_reference.py',Path(__file__)]}
    (out/'freeze.json').write_text(json.dumps(identity,indent=2))


if __name__=='__main__':main()
