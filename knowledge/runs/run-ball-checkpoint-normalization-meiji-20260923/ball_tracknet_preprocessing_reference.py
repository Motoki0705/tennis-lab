"""Compare TrackNet UI inference with the independent dataset normalization."""
from pathlib import Path
import json,numpy as np,torch
from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.ball_detection.visualization.inference.loader import load_ball_model
from src.tasks.ball_detection.visualization.inference.peaks import decode_frame_peaks
from src.tasks.ball_detection.model_io.normalization import IDENTITY_NORMALIZATION
from src.utils.data.augmentation import normalize_frames_imagenet
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
repo=Path('/home/kamimura/projects/tennis-lab')
service=DetectionService(repo)
checkpoint=repo/'ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt'
info=next(i for i in service.checkpoints().values() if i.path.resolve()==checkpoint.resolve())
scene=service.scenes('tracknet',limit=1)['items'][0]['id']
resolved=service._resolve_scene(scene);plan=service._plan_window(info,resolved,start=20,count=8)
loaded=load_ball_model(info.path,device='cpu');policy=loaded.image_normalization
raw=service._window_tensor(resolved.frames,plan,size=loaded.image_size_hw)
frames=list(raw[0].permute(0,2,3,1).numpy())
normalized=normalize_frames_imagenet(frames,mean=np.asarray(policy.mean,dtype=np.float32).reshape(1,1,3),std=np.asarray(policy.std,dtype=np.float32).reshape(1,1,3))
prepared=torch.from_numpy(np.stack(normalized)).permute(0,3,1,2).unsqueeze(0).contiguous()
preview=service.preview(scene,start=20,count=8)
targets={r['index']:r['gt']['points'] for r in preview['items']}
records={};inputs={}
for mode,images,normalization,preprocessed in [('old_raw',raw,IDENTITY_NORMALIZATION,False),('saved_normalization',raw,policy,False),('dataset_reference',prepared,policy,True)]:
 with torch.no_grad():
  call=loaded.adapter.prepare_model_call(images,image_normalization=normalization,preprocessed=preprocessed)
  inputs[mode]=call.model_input
  heatmaps=loaded.adapter.probability_heatmaps(loaded.model(*call.model_args),call)[0]
 peaks=decode_frame_peaks(heatmaps,original_size=resolved.frames.original_size(20),threshold=.5,nms_kernel=info.metrics.nms_kernel,max_peaks=info.metrics.max_predictions_per_frame,subpixel_refine=info.metrics.subpixel_refine)
 distances=[min(np.hypot(x-t['x'],y-t['y']) for x,y in row.points for t in targets[20+i]) for i,row in enumerate(peaks)]
 records[mode]={'points':[r.points for r in peaks],'distances_px':distances,'matched_at_checkpoint_threshold':int(sum(d<=info.metrics.ball_distance_threshold for d in distances)),'median_distance_px':float(np.median(distances))}
max_input_diff=float((inputs['saved_normalization']-inputs['dataset_reference']).abs().max())
assert max_input_diff==0
assert records['saved_normalization']==records['dataset_reference']
result={'scene':scene,'start':20,'count':8,'threshold_px':info.metrics.ball_distance_threshold,'max_model_input_difference_saved_vs_dataset':max_input_diff,'results':records}
(root/'ball_tracknet_preprocessing_reference.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

