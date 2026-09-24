"""Model-only crop proposals: fixed image grid, geometry scoring; references only score outputs."""
from pathlib import Path
from dataclasses import replace
import cv2,json,time,numpy as np
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tennis_scene.pipeline.court_reference import CourtReferenceRuntimeConfig,prepare_court_reference
from src.utils.configuration import PathResolver
root=Path(__file__).resolve().parent;out=root/'court_grid_probe';out.mkdir(exist_ok=False)
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
resolver=PathResolver(replace(runtime.roots,checkpoint_root=Path('/home/kamimura/projects/tennis-lab/outputs')))
checkpoint=Path('/home/kamimura/projects/tennis-lab/outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt')
predictor=CourtPredictor.load_from_checkpoint(checkpoint,device='cpu',resolver=resolver,hybrid_config=runtime.court_kp.postprocess)
clip=Path(json.loads((root/'pipeline.json').read_text())['clip']);manifest=json.loads((clip/'clip.json').read_text())
cap=cv2.VideoCapture(str(clip/'media/cam0.mp4'));ok,bgr=cap.read();cap.release();assert ok
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
letterbox=manifest['cameras'][0]['letterbox'];left=int(letterbox['pad_x']);top=int(letterbox['pad_y']);width=int(letterbox['scaled_width']);height=int(letterbox['scaled_height'])
# Proposals depend only on image dimensions / recorded letterbox, never annotations.
boxes=[]
for fy in (0.,.25):boxes.append((left,top+round(fy*height),left+width,top+round((fy+.75)*height)))
for fx in (0.,.125,.25):
 for fy in (0.,.25,.5):boxes.append((left+round(fx*width),top+round(fy*height),left+round((fx+.75)*width),top+round((fy+.5)*height)))
other=[]
for camera in ('cam1','cam2'):
 with np.load(root/'court_cpu_probe'/f'meiji_b863_{camera}_full.npz') as a:other.append(a['projected'])
truth=np.asarray(json.loads((clip/'annotations/manual_court_kp_result.json').read_text())['keypoints'])[0,0]*[1919,1079]
records=[]
for i,(x0,y0,x1,y1) in enumerate(boxes):
 image=np.ascontiguousarray(rgb[y0:y1,x0:x1]);start=time.monotonic();pred=predictor.predict(image,postprocess='hybrid');geometry=pred.homography
 raw=pred.raw_heads['kp'];points=np.asarray(geometry.projected)+[x0,y0] if geometry.matrix is not None else np.zeros((14,2))
 scores=raw.scores[:,0].numpy();raw_points=raw.keypoints[:,0].numpy()+[x0,y0]
 residual=np.linalg.norm(points-raw_points,axis=-1);inliers=(residual<=.005*np.hypot(1920,1080))&raw.valid[:,0].numpy()
 poly=points[[0,1,3,2]].astype(np.float32);area=abs(cv2.contourArea(poly))/(1920*1080);convex=bool(cv2.isContourConvex(poly))
 inside=bool(((points[:,0]>=left)&(points[:,0]<left+width)&(points[:,1]>=top)&(points[:,1]<top+height)).all())
 accepted=geometry.status=='ok' and inside and convex and area>=.01 and int(inliers.sum())>=8
 calibration_error=None
 if accepted:
  try:
   combined=np.stack([points,*other])[:,None]/[1919,1079]
   prepare_court_reference(camera_ids=('cam0','cam1','cam2'),keypoints=combined.astype(np.float32),visibility=np.ones((3,1,14),np.float32),contract=resolve_court_keypoint_contract('camera_view_v2'),config=CourtReferenceRuntimeConfig(reference_camera='cam0',view_half_turns=(False,False,True)),size=(1920,1080),frame_index=0)
  except Exception as error:calibration_error=f'{type(error).__name__}: {error}';accepted=False
 record={'proposal':i,'box':[x0,y0,x1,y1],'status':geometry.status,'accepted':accepted,'inliers':int(inliers.sum()),'inlier_score':float(scores[inliers].sum()),'area_fraction':area,'convex':convex,'inside_content':inside,'calibration_error':calibration_error,'raw_median_error_px':float(np.median(np.linalg.norm(raw_points-truth,axis=-1))),'projected_mean_error_px':float(np.linalg.norm(points-truth,axis=-1).mean()) if geometry.status=='ok' else None,'seconds':time.monotonic()-start,'diagnostics':pred.geometry_diagnostics()}
 records.append(record)
 np.savez_compressed(out/f'proposal_{i:02d}.npz',points=points,raw_points=raw_points,scores=scores,inliers=inliers,heatmaps=raw.heatmaps.numpy(),line_probability=pred.raw_heads['line'].probability.numpy())
 overlay=bgr.copy();cv2.rectangle(overlay,(x0,y0),(x1,y1),(255,255,0),3)
 if geometry.status=='ok':
  for k,p in enumerate(points):
   if np.isfinite(p).all():cv2.circle(overlay,tuple(np.rint(p).astype(int)),7,(0,255,255),-1)
 cv2.imwrite(str(out/f'proposal_{i:02d}.jpg'),overlay)
 (out/'metrics.json').write_text(json.dumps({'selection':'geometry, coverage and raw model evidence only; manual errors are evaluation-only','records':records},indent=2,allow_nan=False))
 print({k:v for k,v in record.items() if k!='diagnostics'},flush=True)
accepted=[r for r in records if r['accepted']]
if accepted:
 best=max(accepted,key=lambda r:(r['inliers'],r['inlier_score']))
 print('MODEL_ONLY_SELECTED',best['proposal'],'GT_EVALUATION_ERROR',best['projected_mean_error_px'],flush=True)
else:print('NO_ACCEPTED_MODEL_ONLY_PROPOSAL',flush=True)
