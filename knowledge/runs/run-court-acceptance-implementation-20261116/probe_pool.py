import json,pathlib,cv2,numpy as np
from src.tasks.court_detection.geometry.observations import keypoint_support
from src.tasks.court_detection.geometry.hybrid_homography import estimate_hybrid_homography
from src.tasks.court_detection.geometry.homography import court_template_xy,COURT_HOMOGRAPHY_EDGES
from src.tasks.court_detection.geometry.quality import assess_court_quality,CourtQualityConfig
p=pathlib.Path(__file__).resolve().parent
refs=json.loads((p/'references.json').read_text())
manifest=json.loads((p/'raw/manifest.json').read_text())
cv2.setNumThreads(1);template=court_template_xy().astype(float)
records=[r for r in manifest['records'] if r['dataset']=='meiji_4cam' and (r['clip_id'].endswith('clip_000') or (r['clip_id'].endswith('clip_013') and r['camera'] in ['cam0','cam3']) or (r['clip_id'].endswith('clip_035') and r['camera']=='cam3'))]
result=[]
for r in records:
 h,w=r['source_size_hw'];numerator=np.zeros((h,w),np.float64);denominator=np.zeros((h,w),np.float64);kps=[[] for _ in range(14)];scores=[[] for _ in range(14)];scales=[]
 for i,box in enumerate(r['regions']):
  z=np.load(p/'raw'/r['key']/f'raw_{i:02d}.npz');x0,y0,x1,y1=box;ch,cw=y1-y0,x1-x0;shape=z['line_probability'].shape
  weight=np.ones((ch,cw))
  if i:
   yy,xx=np.mgrid[:ch,:cw];mx=16*(cw-1)/(shape[1]-1);my=16*(ch-1)/(shape[0]-1)
   weight=np.minimum(np.clip(np.minimum(xx,cw-1-xx)/mx,0,1),np.clip(np.minimum(yy,ch-1-yy)/my,0,1))
  numerator[y0:y1,x0:x1]+=weight*cv2.resize(z['line_probability'],(cw,ch));denominator[y0:y1,x0:x1]+=weight
  support=keypoint_support(z['raw_xy'],z['scores'],image_size_hw=(ch,cw),native_size_hw=shape,min_score=.05,boundary_margin_cells=1,duplicate_distance_cells=1)
  for k in np.flatnonzero(support.eligible):
   kps[k].append(z['raw_xy'][k]+[x0,y0]);scores[k].append(z['scores'][k])
  if i==0:native=shape
 prob=cv2.resize((numerator/denominator).astype(np.float32),(native[1],native[0]))
 xy=np.array([np.median(x,axis=0) if x else [np.nan,np.nan] for x in kps]);sc=np.array([np.median(s) if s else 0. for s in scores])
 fit=estimate_hybrid_homography(template,xy,sc,prob,edges=np.array(COURT_HOMOGRAPHY_EDGES),image_size_hw=(h,w))
 quality=assess_court_quality(fit.matrix,xy,np.isfinite(xy).all(axis=1)&fit.selected,prob,image_size_hw=(h,w),config=CourtQualityConfig())
 cam=int(r['camera'][-1]);gt=cv2.perspectiveTransform(template[None],np.array(refs['cameras'][cam]['matrix_court_metres_to_image_pixels']))[0];valid=(gt[:,0]>=0)&(gt[:,0]<w)&(gt[:,1]>=0)&(gt[:,1]<h)
 errors=np.linalg.norm(fit.projected[valid]-gt[valid],axis=1);mean=float(errors.mean());mx=float(errors.max())
 row={'key':r['key'],'status':fit.status,'accepted':quality.accepted,'mean_error_px':mean,'max_error_px':mx,'quality':quality.diagnostics()}
 result.append(row);print(r['key'],fit.status,quality.accepted,quality.reasons,mean,mx,flush=True)
(p/'pool-probe-a1.json').write_text(json.dumps(result,indent=2))
