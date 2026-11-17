"""Post-hoc failure explanation only; never promotes a rejected frame.

Re-extract the same stripes, then show the motion implied by the already
reported consensus subset. Omitted lines remain failures in the official report.
These conditional fits are not acceptance results or fallback artifacts.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from src.tasks.court_detection.geometry.paint_observations import PaintConfig,PAINT_NAMES,extract_paint_lines,image_lines
from src.tasks.court_detection.geometry.homography import court_template_xy,COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.hybrid_homography import project_candidate
from src.tasks.court_detection.geometry.video_drift.motion import DriftConfig,solve_offsets,_system,_anchors
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.utils.video import read_video_frame
from evidence import assert_sources


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def inspect(job):
    dataset,output,camera,clip,h,supported,f,example=job
    cv2.setNumThreads(1)
    image=read_video_frame(dataset/f'videos/video_000/clips/{clip.split("/")[1]}/media/{camera}.mp4',f['frame_index']).frame
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Input is the initial BGR-checksum evaluation, kept separately.
    assert hashlib.sha256(image.tobytes()).hexdigest()==f['rgb_sha256']
    size=(image.shape[1],image.shape[0]);scale=np.hypot(*size)/np.hypot(1920,1080)
    p=project_candidate(h,court_template_xy().astype(float))
    paint=replace(PaintConfig(),line_inlier_distance_px=1.5*scale,ambiguity_separation_px=3*scale,sample_spacing_px=8*scale)
    lines=tuple(r for r in extract_paint_lines(rgb.min(axis=2).astype(float),p,48*scale,paint) if PAINT_NAMES[r.index] in supported)
    full=solve_offsets(h,lines,size,DriftConfig())
    assert all(f[k]==v for k,v in full.items()), (camera,clip,f['frame_index'])
    subset=tuple(r for r in lines if PAINT_NAMES[r.index] in f['supported_lines'])
    conditional=solve_offsets(h,subset,size,DriftConfig())
    conflicts=[]
    common=image_lines(h)
    for line in lines:
        if PAINT_NAMES[line.index] not in f['unsupported_observed_lines']:
            continue
        values=[]
        for points in (line.points,*line.alternatives):
            a,b=_system(_anchors(points),common[line.index],size)
            residual=None if 'inverse_similarity_parameters' not in conditional else (a@np.asarray(conditional['inverse_similarity_parameters'])-b).tolist()
            values.append({'anchor_residual_px':residual,'signed_median_px':float(np.median(points@common[line.index,:2]+common[line.index,2]))})
        conflicts.append({'line':PAINT_NAMES[line.index],'options':values})
    if example:
        canvas=image.copy()
        for a,b in COURT_HOMOGRAPHY_EDGES:
            ok,x,y=cv2.clipLine((0,0,*size),tuple(np.rint(p[a]).astype(int)),tuple(np.rint(p[b]).astype(int)))
            if ok:cv2.line(canvas,x,y,(0,255,0),1,cv2.LINE_AA)
        for line in lines:
            bad=PAINT_NAMES[line.index] in f['unsupported_observed_lines']
            colour=(0,0,255) if bad else (255,180,0)
            for xy in line.points:
                cv2.circle(canvas,tuple(np.rint(xy).astype(int)),2,colour,-1)
            if len(line.points):
                cv2.putText(canvas,PAINT_NAMES[line.index],tuple(np.rint(line.points[len(line.points)//2]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1,cv2.LINE_AA)
            if bad:
                centre=np.median(line.points,axis=0).astype(int)
                x,y=centre;box=(max(x-180,0),max(y-150,0),min(x+181,size[0]),min(y+151,size[1]))
                left,top,right,bottom=box
                target=output/'causes'/f'{camera}-{clip.split("/")[1]}-{f["frame_index"]}-{PAINT_NAMES[line.index]}'
                target.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(target.with_suffix('.png')),canvas[top:bottom,left:right])
                cv2.imwrite(str(target.with_name(target.name+'-raw').with_suffix('.png')),image[top:bottom,left:right])
        target=output/'causes'/f'{camera}-{clip.split("/")[1]}-{f["frame_index"]}.jpg'
        cv2.imwrite(str(target),canvas,[cv2.IMWRITE_JPEG_QUALITY,98])
    return {'camera':camera,'clip':clip,'frame':f['frame_index'],'official_status':'failed','official_reason':f['reason'],
        'conditional_status':conditional['status'],'conditional_reason':conditional['reason'],
        'conditional_mean_px':conditional['mean_displacement_px'],'conditional_max_px':conditional['max_displacement_px'],
        'conditional_std_px':conditional['max_visible_projection_std_px'],'conflicts':conflicts,'example':example}


def main(args):
    started=time.monotonic()
    summary=json.loads((args.output/'evaluation-summary-bgr.json').read_text())
    frozen=json.loads((args.output/'frozen-profile.json').read_text());assert_sources(frozen)
    base=VideoCourtStore(Path('outputs/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store')).load(summary['base_bundle_digest'])
    jobs=[]
    for camera,digest in summary['drift_camera_digests'].items():
        doc=json.loads((args.output/'drift-store/objects'/f'{digest}.json').read_text())
        seen=set()
        for clip,row in doc['clips'].items():
            for f in row['frames']:
                if f['reason']!='drift_unobservable':continue
                signature=tuple(f.get('unsupported_observed_lines',[]));example=signature not in seen
                seen.add(signature)
                jobs.append((args.dataset,args.output,camera,clip,np.array(base.cameras[camera]['matrix']),
                    base.cameras[camera]['diagnostics']['quality']['supported_lines'],f,example))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(inspect,jobs))
    save(args.output/'causes-detail.json',rows)
    result={'posthoc_only':True,'used_for_acceptance':False,'official_verdicts_unchanged':True,'frames':len(rows),'seconds':time.monotonic()-started,'cameras':{}}
    for camera in base.cameras:
        selected=[r for r in rows if r['camera']==camera]
        result['cameras'][camera]={'frames':len(selected),'unmatched_lines':dict(Counter(c['line'] for r in selected for c in r['conflicts'])),
            'conditional_status_counts':dict(Counter(r['conditional_status'] for r in selected)),
            'conditional_reasons':dict(Counter(str(r['conditional_reason']) for r in selected)),
            'conditional_mean_range_px':[min([r['conditional_mean_px'] for r in selected if r['conditional_mean_px'] is not None],default=None),
                                          max([r['conditional_mean_px'] for r in selected if r['conditional_mean_px'] is not None],default=None)]}
    save(args.output/'causes-summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('data/tennis_multiview/processed/meiji_4cam/dataset'))
    p.add_argument('--workers',type=int,default=4);main(p.parse_args())
