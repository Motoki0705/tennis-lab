"""CPU diagnostic: two fixed checkpoints, full images and metadata-only unletterbox."""
from pathlib import Path
from dataclasses import replace
import gc,json,time,traceback
import cv2,numpy as np,torch
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.utils.configuration import PathResolver

root=Path(__file__).resolve().parent
out=root/'court_cpu_probe';out.mkdir(exist_ok=False)
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
clip=Path(json.loads((root/'pipeline.json').read_text())['clip']);manifest=json.loads((clip/'clip.json').read_text())
truth=np.asarray(json.loads((clip/'annotations/manual_court_kp_result.json').read_text())['keypoints'])[:,0]*[1919,1079]
models={'current_dd3':runtime.court_kp.checkpoint,'meiji_b863':Path('/home/kamimura/projects/tennis-lab/outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt')}
records=[];errors={}
for model,path in models.items():
    try:
        resolver=runtime.resolver if model=='current_dd3' else PathResolver(replace(runtime.roots,checkpoint_root=Path('/home/kamimura/projects/tennis-lab/outputs')))
        begin=time.monotonic();predictor=CourtPredictor.load_from_checkpoint(path,device='cpu',resolver=resolver,hybrid_config=runtime.court_kp.postprocess)
        print(model,'loaded seconds',round(time.monotonic()-begin,2),flush=True)
        (out/f'{model}_checkpoint.json').write_text(json.dumps(predictor.checkpoint_identity,indent=2))
        for n,camera in enumerate(manifest['camera_ids']):
            cap=cv2.VideoCapture(str(clip/'media'/f'{camera}.mp4'));ok,bgr=cap.read();cap.release();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            variants=[('full',rgb,0,0)]
            letterbox=manifest['cameras'][n]['letterbox']
            if letterbox is not None:
                x,y=int(letterbox['pad_x']),int(letterbox['pad_y']);width,height=int(letterbox['scaled_width']),int(letterbox['scaled_height'])
                variants.append(('unletterbox',np.ascontiguousarray(rgb[y:y+height,x:x+width]),x,y))
            for variant,image,dx,dy in variants:
                ident=f'{model}_{camera}_{variant}';begin=time.monotonic()
                pred=predictor.predict(image,postprocess='hybrid')
                kp,line,seg=pred.raw_heads['kp'],pred.raw_heads['line'],pred.raw_heads['seg']
                projected,valid=pred.downstream_keypoints();projected=projected+[dx,dy]
                raw=kp.keypoints[:,0].numpy()+[dx,dy]
                error=np.linalg.norm(raw-truth[n],axis=-1)
                geometry=pred.geometry_diagnostics()
                record={'id':ident,'seconds':time.monotonic()-begin,'offset_xy':[dx,dy],'original_size_hw':list(pred.original_size_hw),'native_size_hw':list(pred.native_size_hw),'raw_mean_error_px':float(error.mean()),'raw_median_error_px':float(np.median(error)),'projected_visible':int(valid.sum()),'projected_mean_error_px':float(np.linalg.norm(projected-truth[n],axis=-1).mean()) if pred.homography.status=='ok' else None,**geometry}
                records.append(record)
                np.savez_compressed(out/f'{ident}.npz',raw_kp=raw,scores=kp.scores.numpy(),raw_valid=kp.valid.numpy(),heatmaps=kp.heatmaps.numpy(),line_probability=line.probability.numpy(),seg_mask=seg.mask.numpy(),seg_logits=seg.logits.numpy(),projected=projected,projected_valid=valid)
                cv2.imwrite(str(out/f'{ident}_input.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                line_map=cv2.resize(line.probability.numpy(),(image.shape[1],image.shape[0]));overlay=image.copy();mask=line_map>=.5;overlay[mask]=(0.35*overlay[mask]+0.65*np.array([0,255,255])).astype(np.uint8)
                for k,point in enumerate(kp.keypoints[:,0].numpy()):
                    xy=tuple(np.rint(point).astype(int));cv2.circle(overlay,xy,6,(255,0,0),-1);cv2.putText(overlay,str(k),(xy[0]+7,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,0),2)
                cv2.imwrite(str(out/f'{ident}_overlay.jpg'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
                (out/'metrics.json').write_text(json.dumps({'device':'cpu','records':records,'errors':errors},indent=2,allow_nan=False))
                print(ident,'status',pred.homography.status,'raw median',record['raw_median_error_px'],'visible',int(valid.sum()),'seconds',round(record['seconds'],2),flush=True)
        del predictor;gc.collect()
    except Exception as error:
        errors[model]=f'{type(error).__name__}: {error}';traceback.print_exc()
        (out/'metrics.json').write_text(json.dumps({'device':'cpu','records':records,'errors':errors},indent=2,allow_nan=False))
if errors:raise SystemExit(1)
