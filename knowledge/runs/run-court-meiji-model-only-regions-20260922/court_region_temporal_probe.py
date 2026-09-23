"""Image-only region selection and temporal stability; manual labels score only."""
from pathlib import Path
from dataclasses import replace
import json, time
import cv2, numpy as np, torch
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.utils.configuration import PathResolver

torch.set_num_threads(4)
root=Path(__file__).resolve().parent
out=root/'court_region_temporal_probe';out.mkdir(exist_ok=False)
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
resolver=PathResolver(replace(runtime.roots,checkpoint_root=Path('/home/kamimura/projects/tennis-lab/outputs')))
checkpoint=Path('/home/kamimura/projects/tennis-lab/outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt')
predictor=CourtPredictor.load_from_checkpoint(checkpoint,device='cpu',resolver=resolver,hybrid_config=runtime.court_kp.postprocess)
clip=Path(json.loads((root/'pipeline.json').read_text())['clip'])
truth=np.asarray(json.loads((clip/'annotations/manual_court_kp_result.json').read_text())['keypoints'])[:,0]*[1919,1079]
records=[];selected={};temporal=[]
def predict(rgb,box):
 x0,y0,x1,y1=box
 pred=predictor.predict(np.ascontiguousarray(rgb[y0:y1,x0:x1]),postprocess='hybrid',heads=('kp','line'))
 points,valid=pred.downstream_keypoints();points=points+[x0,y0]
 raw=pred.raw_heads['kp'];raw_points=raw.keypoints[:,0].numpy()+[x0,y0];scores=raw.scores[:,0].numpy()
 inliers=(np.linalg.norm(points-raw_points,axis=-1)<=.005*np.hypot(rgb.shape[1],rgb.shape[0]))&raw.valid[:,0].numpy()
 poly=points[[0,1,3,2]].astype(np.float32);area=abs(cv2.contourArea(poly))/(rgb.shape[0]*rgb.shape[1]);convex=bool(cv2.isContourConvex(poly))
 inside=bool(((points[:,0]>=0)&(points[:,0]<rgb.shape[1])&(points[:,1]>=0)&(points[:,1]<rgb.shape[0])).all())
 record={'box':box,'status':pred.homography.status,'inliers':int(inliers.sum()),'inlier_score':float(scores[inliers].sum()),'area_fraction':area,'convex':convex,'inside_image':inside}
 record['accepted']=pred.homography.status=='ok' and inside and convex and area>=.01 and int(inliers.sum())>=8
 return points,record,pred
def save():
 (out/'metrics.json').write_text(json.dumps({'selection':'fixed grid, original image geometry and raw inlier evidence; no annotation-derived proposal or ranking','records':records,'selected':selected,'temporal':temporal},indent=2,allow_nan=False))
for c,camera in enumerate(('cam0','cam1','cam2')):
 cap=cv2.VideoCapture(str(clip/'media'/f'{camera}.mp4'));ok,bgr=cap.read();assert ok
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 ys,xs=np.nonzero(np.any(rgb!=0,axis=-1));left,top,right,bottom=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1;width,height=right-left,bottom-top
 boxes=[(0,0,rgb.shape[1],rgb.shape[0])]
 if (left,top,right,bottom)!=boxes[0]:boxes.append((left,top,right,bottom))
 for fy in (0.,.25):boxes.append((left,top+round(fy*height),right,top+round((fy+.75)*height)))
 for fx in (0.,.125,.25):
  for fy in (0.,.25,.5):boxes.append((left+round(fx*width),top+round(fy*height),left+round((fx+.75)*width),top+round((fy+.5)*height)))
 candidates=[]
 for i,box in enumerate(boxes):
  points,record,pred=predict(rgb,box);record.update(camera=camera,proposal=i,manual_mean_error_px=float(np.linalg.norm(points-truth[c],axis=-1).mean()) if record['status']=='ok' else None)
  records.append(record);save();print(record,flush=True)
  if record['accepted']:candidates.append((record,points))
 if not candidates:raise RuntimeError(f'No accepted image-only region for {camera}')
 best,points=max(candidates,key=lambda pair:(pair[0]['inliers'],pair[0]['inlier_score']))
 selected[camera]=best;save()
 for f in np.linspace(0,1009,9,dtype=int):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read();assert ok
  points,record,pred=predict(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),best['box'])
  record.update(camera=camera,frame=int(f),manual_mean_error_px=float(np.linalg.norm(points-truth[c],axis=-1).mean()) if record['status']=='ok' else None)
  temporal.append(record);save();print('temporal',record,flush=True)
  np.savez_compressed(out/f'{camera}_{f:04d}.npz',projected=points,raw_kp=pred.raw_heads['kp'].keypoints.numpy(),raw_scores=pred.raw_heads['kp'].scores.numpy())
  x0,y0,x1,y1=best['box'];cv2.rectangle(bgr,(x0,y0),(x1,y1),(255,255,0),3)
  if record['status']=='ok':
   for k,p in enumerate(points):
    xy=tuple(np.rint(p).astype(int));cv2.circle(bgr,xy,6,(0,255,255),-1);cv2.putText(bgr,str(k),(xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
  cv2.imwrite(str(out/f'{camera}_{f:04d}.jpg'),bgr)
 cap.release()
