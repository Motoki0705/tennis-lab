"""Measure each frame before pooling, so camera drift is not trimmed away.

Stage 1/2 working H comes from RGB-only fit_reference.py, initialized at the old
reference. Stage 3 uses 12px RGB strips and per-frame consensus, then equally
samples each frame/line. No model outputs or evaluation errors are inputs.
"""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np

import fit_reference as fit

BASE=Path(__file__).resolve().parent
RADIUS=12.
SAMPLES_PER_FRAME_LINE=24


def main():
    cv2.setNumThreads(1);out=BASE/'fit-final';out.mkdir(exist_ok=True)
    previous=json.loads((BASE/'fit-a2/references.json').read_text());manifest=json.loads((BASE/'frames.json').read_text());masks=json.loads((BASE/'net-exclusions.json').read_text())
    result={k:v for k,v in previous.items() if k!='cameras'};result['config']={**fit.CFG,'final_stage_radius_px':RADIUS,'final_stage_line_fit':'per_frame','minimum_frames_per_local_line':1,'max_samples_per_frame_line':SAMPLES_PER_FRAME_LINE};result['cameras']=[]
    result['working_reference_sha256']=hashlib.sha256((BASE/'fit-a2/references.json').read_bytes()).hexdigest()
    for previous_cam in previous['cameras']:
        cam=previous_cam['camera'];records=[r for r in manifest['records'] if r['camera']==cam];working=np.array(previous_cam['matrix_court_metres_to_image_pixels']);initial=np.array(previous_cam['initial_matrix'])
        excluded=np.zeros((1080,1920),np.uint8);a,b=masks['cameras'][cam]['endpoints_xy'];cv2.line(excluded,tuple(a),tuple(b),1,2*masks['half_width_px']+1)
        independent=[];all_groups=[];frame_models=[]
        for frame,record in enumerate(records):
            im=cv2.imread(str(BASE/record['path'])).min(axis=2).astype(np.float32)
            observations=fit.extract([im],working,excluded,RADIUS)
            lines=[fit.fit_line(o,minimum_frames=1) for o in observations]
            h,supported,diagnostic=fit.court_consensus(lines,working)
            frame_models.append({'frame':record,'matrix':h.tolist() if h is not None else None,'diagnostic':diagnostic,'supported_lines':[fit.NAMES[i] for i in supported]})
            # Preserve failed frame/line measurements rather than replacing them
            # with the pooled H or copying another frame's correspondences.
            for index,line in enumerate(lines):
                values=np.asarray(line['samples'],float);ids=line['inliers'];xy=values[ids,:2] if ids else np.empty((0,2))
                independent.append({'frame':frame,'line_index':index,'status':line['status'],'supported_by_frame_geometry':index in supported,'points':xy.tolist(),'pair_candidates':len(values),'coverage':line.get('coverage')})
                if h is None or index not in supported:continue
                order=np.argsort(values[ids,3],kind='stable');chosen=order[np.linspace(0,len(order)-1,min(SAMPLES_PER_FRAME_LINE,len(order))).round().astype(int)]
                all_groups.append({'frame':frame,'line_index':index,'points':xy[chosen].tolist()})
            print(cam,record['clip_id'],record['frame_index'],diagnostic,flush=True)
        if len({g['frame'] for g in all_groups})!=len(records):
            entry={'camera':cam,'matrix_court_metres_to_image_pixels':None,'status':'some_frames_have_no_identifiable_court','human_reviewed':False,'approved_for_evaluation':False}
        else:
            points=np.concatenate([g['points'] for g in all_groups]);labels=np.concatenate([np.full(len(g['points']),g['line_index']) for g in all_groups])
            h,diagnostic=fit.fit_h(initial,points,labels)
            if not diagnostic['success'] or diagnostic['rank']<8 or not fit.valid_h(h):raise RuntimeError((cam,diagnostic))
            line_reports=[]
            for index,name in enumerate(fit.NAMES):
                relevant=[g for g in all_groups if g['line_index']==index]
                xy=np.concatenate([g['points'] for g in relevant]) if relevant else np.empty((0,2))
                residuals={mode:fit.distribution(np.c_[xy,np.ones(len(xy))]@fit.image_lines(matrix)[index]) for mode,matrix in [('old',initial),('new',h)]}
                line_reports.append({'name':name,'status':'measured' if relevant else 'not_measured','frames':len(relevant),'residuals':residuals})
            entry={'camera':cam,'matrix_court_metres_to_image_pixels':h.tolist(),'initial_matrix':initial.tolist(),'human_reviewed':False,'approved_for_evaluation':False,'evaluation_status':'provisional_parent_authorized','diagnostic':diagnostic,'lines':line_reports}
        entry['frame_geometry_failures']=sum(f['matrix'] is None for f in frame_models)
        (out/f'{cam}-evidence.json').write_text(json.dumps({'frames':records,'frame_models':frame_models,'independent_lines':independent,'fit_groups':all_groups},allow_nan=False))
        result['cameras'].append(entry);(out/'references.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print('final',cam,entry.get('diagnostic',entry.get('status')),flush=True)


if __name__=='__main__':main()
