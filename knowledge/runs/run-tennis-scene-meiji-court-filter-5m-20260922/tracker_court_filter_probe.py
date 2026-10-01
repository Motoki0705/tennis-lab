"""Queue-only full-clip tracker check with the existing model-derived court gate."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,time,hashlib
import cv2,numpy as np
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.court_reference import prepare_court_reference,court_footpoint_polygon_px
from src.submodules.models.tracker.dino_tracker import DinoPersonTracker
from src.submodules.models.tracker.common import TrackRequest

root=Path(__file__).resolve().parent;out=root/'court_filtered_tracking';out.mkdir(exist_ok=True)
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
request=json.loads((root/'pipeline.json').read_text());court_path=Path(request['output'])/'court_kp_result.json';court=json.loads(court_path.read_text())
context=prepare_court_reference(camera_ids=runtime.camera_ids,keypoints=np.asarray(court['keypoints'],np.float32),visibility=np.asarray(court['visibility'],np.float32),contract=runtime.plcs.court_keypoint_contract,config=runtime.court_reference,size=(1920,1080),frame_index=0)
model=runtime.gvhmr.runtime.dino_detector
tracker=DinoPersonTracker(runtime.gvhmr.dino_checkpoint,runtime.gvhmr.dino_repository,device='cuda',confidence=model.confidence,short_side=model.short_side,max_long_side=model.max_long_side)
records=[]
for c,(camera,video) in enumerate(zip(runtime.camera_ids,runtime.video_paths,strict=True)):
    polygon=court_footpoint_polygon_px(context.keypoints[c,0],size=(1920,1080),sideline_margin_m=runtime.gvhmr.court_footpoint_filter.sideline_margin_m,baseline_margin_m=runtime.gvhmr.court_footpoint_filter.baseline_margin_m)
    begin=time.monotonic();result=tracker.predict(TrackRequest(video_path=video,num_tracks=2,interactive=False,footpoint_polygon_px=polygon))
    boxes=np.stack([result.bbx_xys(t,base_enlarge=runtime.gvhmr.runtime.tracking.bbox_enlarge).numpy() for t in result.track_ids])
    observed=np.stack([result.observed_mask(t).numpy() for t in result.track_ids])
    with np.load(root/f'identity_reference_{camera}.npz') as ref:old_boxes=ref['bbx_xys'];old_ids=ref['track_ids']
    pair=np.stack([np.stack([np.linalg.norm(old_boxes[a,:,:2]-boxes[b,:,:2],axis=-1)/np.maximum(old_boxes[a,:,2],1) for b in range(2)]) for a in range(2)])
    costs=np.stack([pair[0,0]+pair[1,1],pair[0,1]+pair[1,0]])
    selected=int(np.argmin(np.median(costs,axis=1)));mapping=[0,1] if selected==0 else [1,0]
    record={'camera':camera,'seconds':time.monotonic()-begin,'polygon_px':polygon,'old_ids':old_ids.tolist(),'new_ids':result.track_ids,'old_to_new_proposal':mapping,'bbox_costs_median':np.median(pair,axis=-1).tolist(),'matched_bbox_median':[float(np.median(pair[a,b])) for a,b in enumerate(mapping)],'matched_bbox_p95':[float(np.percentile(pair[a,b],95)) for a,b in enumerate(mapping)],'opposite_order_better_frames':np.flatnonzero(costs[1-selected]+1e-6<costs[selected]).tolist(),'observed_counts':observed.sum(axis=-1).tolist()}
    records.append(record);np.savez_compressed(out/f'{camera}.npz',bbx_xys=boxes,track_ids=np.array(result.track_ids),observed=observed)
    cap=cv2.VideoCapture(str(video));panels=[]
    for f in (0,252,505,757,1009):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read();assert ok
        cv2.polylines(image,[np.rint(polygon).astype(np.int32)],True,(255,255,255),3)
        for label,values,colors in [('old',old_boxes,[(255,255,0),(255,0,255)]),('new',boxes,[(0,255,0),(0,128,255)])]:
            for p in range(2):
                x,y,size=values[p,f];left,top=int(x-size/2),int(y-size/2);cv2.rectangle(image,(left,top),(int(x+size/2),int(y+size/2)),colors[p],3);cv2.putText(image,f'{label} axis{p}',(left,top+(20 if label=='old' else 45)),cv2.FONT_HERSHEY_SIMPLEX,.8,colors[p],2)
        panel=cv2.resize(image,(960,540));cv2.putText(panel,f'{camera} frame {f}',(15,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(panel)
    cap.release();cv2.imwrite(str(out/f'{camera}_comparison.jpg'),np.concatenate(panels,axis=1))
    (out/'metrics.json').write_text(json.dumps({'court_input':str(court_path),'court_sha256':hashlib.sha256(court_path.read_bytes()).hexdigest(),'filter':'existing court_footpoint_filter enabled; margins 1m sideline / 5m baseline; model-derived Court only','records':records},indent=2))
    print(json.dumps(record),flush=True)
