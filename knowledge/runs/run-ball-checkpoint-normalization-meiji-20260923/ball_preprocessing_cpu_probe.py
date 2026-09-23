"""CPU-only comparison of the checkpoint's saved normalization on real windows."""
from pathlib import Path
import hashlib,json,time
from dataclasses import replace
import cv2,numpy as np,torch
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.model_io.normalization import IDENTITY_NORMALIZATION
from src.utils.configuration import PathResolver
from src.utils.video import BgrToTensorTransform

torch.set_num_threads(1)
root=Path(__file__).resolve().parent
out=root/'ball_preprocessing_cpu_probe';out.mkdir(exist_ok=False)
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
resolver=PathResolver(replace(runtime.roots,project_root=Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-ball-preprocessing')))
checkpoint=Path('/home/kamimura/projects/tennis-lab/ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt')
predictor=BallDetectionPredictor.load_from_checkpoint(checkpoint,resolver=resolver,device='cpu',subpixel_refine=True,strict=True,weights_only=False)
normalization=predictor.image_normalization
assert normalization.enabled
transform=BgrToTensorTransform(image_size=(288,512),normalize_imagenet=False)
clip=Path(json.loads((root/'pipeline.json').read_text())['clip'])
starts=[0,96,232,400,504,528,768,824,992]
records=[];summary={}
for camera in ('cam0','cam1','cam2'):
 annotations=json.loads((clip/'outsource'/f'{camera}_annotations.json').read_text())['frames']
 cap=cv2.VideoCapture(str(clip/'media'/f'{camera}.mp4'))
 mode_results={'missing_normalization':[],'saved_normalization':[]};panels=[]
 for start in starts:
  cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
  for _ in range(8):
   ok,frame=cap.read();assert ok;frames.append(frame)
  images=torch.stack([transform(frame) for frame in frames]).unsqueeze(0)
  predictions={}
  for mode,policy in [('missing_normalization',IDENTITY_NORMALIZATION),('saved_normalization',normalization)]:
   predictor.image_normalization=policy
   begun=time.perf_counter();prediction=predictor.predict(images);elapsed=time.perf_counter()-begun
   uv=prediction.coords[0].numpy()*[1919,1079];confidence=prediction.confidence[0].numpy()
   predictions[mode]=(uv,confidence)
   for offset in range(8):
    f=start+offset;label=annotations[f];observed=label['status']=='observed'
    target=np.array([label['center_px']['x'],label['center_px']['y']]) if observed else None
    detected=bool(confidence[offset]>=.5)
    row={'camera':camera,'window_start':start,'frame':f,'mode':mode,'observed':observed,'detected':detected,'confidence':float(confidence[offset]),'uv_px':uv[offset].tolist(),'distance_px':float(np.linalg.norm(uv[offset]-target)) if observed and detected else None,'window_seconds':elapsed}
    mode_results[mode].append(row);records.append(row)
  for offset in (0,4):
   f=start+offset;panel=frames[offset].copy();label=annotations[f]
   if label['status']=='observed':cv2.circle(panel,(round(label['center_px']['x']),round(label['center_px']['y'])),13,(0,0,255),2)
   for mode,color in [('missing_normalization',(0,128,255)),('saved_normalization',(0,255,0))]:
    uv,confidence=predictions[mode]
    if confidence[offset]>=.5:cv2.drawMarker(panel,tuple(np.rint(uv[offset]).astype(int)),color,cv2.MARKER_CROSS,25,3)
   panel=cv2.resize(panel,(640,360));cv2.putText(panel,f'{camera} f{f}: red=GT orange=old green=saved',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
  print(camera,start,'done',flush=True)
 cap.release()
 summary[camera]={}
 for mode,rows in mode_results.items():
  observed=[r for r in rows if r['observed']];matched=[r for r in observed if r['detected']];errors=[r['distance_px'] for r in matched]
  summary[camera][mode]={'observed':len(observed),'detected_on_observed':len(matched),'missing_rate':1-len(matched)/len(observed),'distance_px':{'mean':float(np.mean(errors)),'median':float(np.median(errors)),'p95':float(np.percentile(errors,95))} if errors else None}
 cv2.imwrite(str(out/f'{camera}-comparison.jpg'),np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in range(0,len(panels),3)],axis=0))
 (out/'metrics.json').write_text(json.dumps({'scope':'Selected diagnostic 8-frame windows; includes difficult frames; not full-pipeline overlap/gating or a held-out benchmark. Both modes use the same model weights and raw video windows on CPU.','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'normalization':{'enabled':normalization.enabled,'mean':normalization.mean,'std':normalization.std},'starts':starts,'summary':summary,'records':records},indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))

