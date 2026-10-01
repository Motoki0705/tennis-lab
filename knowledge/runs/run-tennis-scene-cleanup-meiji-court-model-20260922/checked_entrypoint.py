"""Replay existing manual identities after inspection of fresh tracking outputs."""
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json, runpy, sys, time
import cv2
import numpy as np
from src.tennis_scene.pipeline.components.player_association import PlayerAssociationModule
from types import SimpleNamespace

request=json.loads(Path(sys.argv[1]).read_text())
review=Path(request['review'])
review.mkdir(parents=True,exist_ok=True)
if request['module'].endswith('.generate_dataset'):
    previous=json.loads((review.parent/'pipeline.json').read_text())
    if not (Path(previous['output'])/'scene.npz').is_file() or not (Path(previous['output'])/'scene.metadata.json').is_file():
        raise RuntimeError('Independent pipeline scene must finish before dataset regeneration')
clip=Path(request['clip'])
original=PlayerAssociationModule.process

def inspect_identities(self, **kwargs):
    comparisons=[]
    contact_rows=[]
    for camera_id, result, video_path in zip(kwargs['camera_ids'],kwargs['gvhmr_results'],kwargs['video_paths'],strict=True):
        with np.load(Path(request['manual_reference']).parent/f'identity_reference_{camera_id}.npz') as reference:
            old=SimpleNamespace(**{key:reference[key] for key in reference.files})
        pairs=[]
        for old_axis in range(2):
            row=[]
            for new_axis in range(2):
                scale=np.maximum(old.bbx_xys[old_axis,:,2],1)
                center=np.linalg.norm(old.bbx_xys[old_axis,:,:2]-result.bbx_xys[new_axis,:,:2],axis=-1)/scale
                pose=np.linalg.norm(old.human_kp_2d[old_axis]-result.human_kp_2d[new_axis],axis=-1).mean(axis=-1)/scale
                valid=(old.human_kp_vis[old_axis].mean(axis=-1)>0.15)&(result.human_kp_vis[new_axis].mean(axis=-1)>0.15)
                row.append({'bbox_median':float(np.median(center[valid])) if valid.any() else None,'pose_median':float(np.median(pose[valid])) if valid.any() else None,'supported_frames':int(valid.sum())})
            pairs.append(row)
        comparisons.append({'camera':camera_id,'old_track_ids':old.track_ids.tolist(),'new_track_ids':result.track_ids.tolist(),'old_to_new_pair_costs':pairs})
        cap=cv2.VideoCapture(str(video_path))
        panels=[]
        for t in (0,252,505,757,1009):
            cap.set(cv2.CAP_PROP_POS_FRAMES,t)
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Cannot read {camera_id} frame {t}')
            for label,data,colors in [('old',old,[(255,255,0),(255,0,255)]),('new',result,[(0,255,0),(0,128,255)])]:
                for p in range(2):
                    x,y,size=data.bbx_xys[p,t]
                    left,top=int(x-size/2),int(y-size/2)
                    cv2.rectangle(frame,(left,top),(int(x+size/2),int(y+size/2)),colors[p],3)
                    cv2.putText(frame,f'{label} axis{p} track{data.track_ids[p]}',(left,top+(20 if label=='old' else 45)),cv2.FONT_HERSHEY_SIMPLEX,.8,colors[p],2)
                    for point in data.human_kp_2d[p,t]:
                        if np.isfinite(point).all(): cv2.circle(frame,tuple(point.astype(int)),3,colors[p],-1)
            cv2.putText(frame,f'{camera_id} frame {t}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            panels.append(cv2.resize(frame,(640,360)))
        cap.release()
        contact_rows.append(np.concatenate(panels,axis=1))
        del old
    (review/'association_comparison.json').write_text(json.dumps(comparisons,indent=2))
    cv2.imwrite(str(review/'identity_contact_sheet.jpg'),np.concatenate(contact_rows,axis=0))
    (review/'ready.json').write_text(json.dumps({'frames':1010,'cameras':kwargs['camera_ids']}))
    print(f'Fresh identities await inspection: {review}',flush=True)
    deadline=time.monotonic()+3600
    verified=review/'verified_mapping.json'
    while not verified.exists():
        if time.monotonic()>deadline: raise RuntimeError('Fresh track identity inspection was not completed within one hour')
        time.sleep(1)
    decision=json.loads(verified.read_text())
    mapping=json.loads(Path(request['manual_reference']).read_text())
    if decision['camera_ids']!=kwargs['camera_ids']: raise ValueError('Verification camera order mismatch')
    mapping['segments'][0]['assignments']=decision['assignments']
    self.config.load_path.parent.mkdir(parents=True,exist_ok=True)
    self.config.load_path.write_text(json.dumps(mapping,indent=2))
    return original(self,**kwargs)

PlayerAssociationModule.process=inspect_identities
sys.argv=[request['module'],*request['overrides']]
runpy.run_module(request['module'],run_name='__main__')
