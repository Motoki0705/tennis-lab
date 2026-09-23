"""Source-video evidence around each phase's largest predicted 3D step."""
from pathlib import Path
import json,sys
import cv2,numpy as np
root=Path(__file__).resolve().parent
phase=sys.argv[1]
request=json.loads((root/f'{phase}.json').read_text())
clip=Path(request['clip']);scene_path=Path(request['output'])/'scene.npz' if phase=='pipeline' else clip/'annotations/tennis_scene/scene.npz'
out=root/'figures';out.mkdir(exist_ok=True)
summary={}
with np.load(scene_path) as scene:
 for event,key in [('ball','ball_3d'),('player','player_position')]:
  positions=scene[key];speed=np.linalg.norm(np.diff(positions,axis=-2),axis=-1)*float(scene['fps'])
  axis=np.unravel_index(np.argmax(speed),speed.shape);center=int(axis[-1]+1);player=int(axis[0]) if speed.ndim==2 else None
  frames=list(range(max(0,center-1),min(int(scene['num_frames']),center+2)))
  rows=[];crops=[];records=[]
  for c,camera in enumerate(('cam0','cam1','cam2')):
   annotations=json.loads((clip/'outsource'/f'{camera}_annotations.json').read_text())['frames']
   cap=cv2.VideoCapture(str(clip/'media'/f'{camera}.mp4'));panels=[];crop_panels=[]
   for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read();assert ok
    if event=='ball':
     observed=annotations[f]['status']=='observed';visible=bool(scene['ball_vis'][c,f]);uv=scene['ball_uv'][c,f]*[1919,1079]
     target=np.array([annotations[f]['center_px']['x'],annotations[f]['center_px']['y']]) if observed else None
     if observed:cv2.circle(image,tuple(np.rint(target).astype(int)),15,(0,0,255),2)
     if visible:cv2.drawMarker(image,tuple(np.rint(uv).astype(int)),(0,255,0),cv2.MARKER_CROSS,25,3)
     if observed and visible:cv2.line(image,tuple(np.rint(target).astype(int)),tuple(np.rint(uv).astype(int)),(255,255,255),2)
     label=f'{camera} f{f}: red=observed green=prediction'
     records.append({'camera':camera,'frame':f,'observed':observed,'predicted':visible,'prediction_px':uv.tolist(),'reference_px':target.tolist() if target is not None else None})
    else:
     points=scene['human_kp_2d'][player,c,f]*[1919,1079];valid=scene['human_kp_vis'][player,c,f]>=.15
     lo=np.maximum(np.floor(points.min(axis=0)-20).astype(int),[0,0]);hi=np.minimum(np.ceil(points.max(axis=0)+20).astype(int),[1920,1080])
     for point,v in zip(points,valid,strict=True):cv2.circle(image,tuple(np.rint(point).astype(int)),3,(0,255,0) if v else (0,0,255),-1)
     crop=image[lo[1]:hi[1],lo[0]:hi[0]];scale=min(240/crop.shape[1],240/crop.shape[0]);small=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),interpolation=cv2.INTER_NEAREST)
     panel=np.zeros((275,240,3),np.uint8);panel[35:35+small.shape[0],:small.shape[1]]=small
     cv2.putText(panel,f'{camera} P{player} f{f} vis{valid.sum()}/17',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);crop_panels.append(panel)
     label=f'{camera} P{player} f{f}: green=valid red=masked'
     records.append({'camera':camera,'frame':f,'visible_joints_at_015':int(valid.sum())})
    panel=cv2.resize(image,(640,360));cv2.putText(panel,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
   cap.release();rows.append(np.concatenate(panels,axis=1))
   if crop_panels:crops.append(np.concatenate(crop_panels,axis=1))
  path=out/f'{phase}-{event}-max-step-source.jpg';cv2.imwrite(str(path),np.concatenate(rows,axis=0))
  if crops:cv2.imwrite(str(out/f'{phase}-{event}-max-step-crops.jpg'),np.concatenate(crops,axis=0))
  summary[event]={'frame':center,'player':player,'speed_m_s':float(speed[axis]),'source_contact':str(path),'records':records}
(root/f'{phase}-anomaly-source.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in summary.items()},indent=2))

