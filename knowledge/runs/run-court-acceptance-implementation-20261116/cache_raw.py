"""Freeze scene frame-0 KP/LINE once, without any homography fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.tasks.court_detection.inference.regions import image_content_region, region_proposals
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tennis_scene.configuration import PROJECT_ROOT, _roots
from src.utils.configuration import PathRole
from src.utils.video import read_video_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',default=['meiji_4cam','meiji_3cam','tennis_clip'])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    paths=OmegaConf.load(PROJECT_ROOT/'src/tennis_scene/configs/paths/default.yaml')
    paths.external_asset_root='/home/kamimura/projects/tennis-lab/third_party'
    _,resolver=_roots(OmegaConf.to_container(paths,resolve=True))
    ckpt=resolver.resolve(PathRole.CHECKPOINT,'court_detection/multiscale_depth3/b863df1f01f0.ckpt')
    predictor=CourtPredictor.load_from_checkpoint(ckpt,resolver=resolver,device='cuda',subpixel_refine=True)
    header={'schema':'court_raw_cache_v1','checkpoint':predictor.checkpoint_identity,'postprocess':'none','frame':0}
    header_path=out/'header.json'
    if header_path.exists():
        assert json.loads(header_path.read_text())==header
    else:header_path.write_text(json.dumps(header,indent=2))
    records=[]
    for dataset in args.datasets:
        root=Path('data/tennis_multiview/processed')/dataset/'dataset'
        manifest=json.loads((root/'dataset.json').read_text())
        for clip in manifest['clips']:
            for camera in range(clip['num_cameras']):
                key=f'{dataset}/{clip["clip_id"]}/cam{camera}'
                target=out/key;target.mkdir(parents=True,exist_ok=True)
                video=root/clip['path']/'media'/f'cam{camera}.mp4'
                meta_path=target/'record.json'
                if meta_path.exists():
                    meta=json.loads(meta_path.read_text())
                    if all((target/f'raw_{i:02d}.npz').exists() for i in range(len(meta['regions']))):
                        records.append(meta);continue
                start=time.monotonic()
                rgb=cv2.cvtColor(read_video_frame(video,0).frame,cv2.COLOR_BGR2RGB)
                regions=region_proposals(rgb)
                for i,region in enumerate(regions):
                    destination=target/f'raw_{i:02d}.npz'
                    if destination.exists():continue
                    x0,y0,x1,y1=region
                    prediction=predictor.predict(np.ascontiguousarray(rgb[y0:y1,x0:x1]),heads=('kp','line'),postprocess='none')
                    kp,line=prediction.raw_heads['kp'],prediction.raw_heads['line']
                    temporary=target/f'.raw_{i:02d}.npz'
                    np.savez_compressed(temporary,raw_xy=kp.keypoints[:,0].numpy(),scores=kp.scores[:,0].numpy(),
                        valid=kp.valid[:,0].numpy(),line_probability=line.probability.numpy(),
                        image_size_hw=np.array(prediction.original_size_hw),native_size_hw=np.array(prediction.native_size_hw))
                    os.replace(temporary,destination)
                cv2.imwrite(str(target/'frame.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                meta={'key':key,'dataset':dataset,'clip_id':clip['clip_id'],'camera':f'cam{camera}',
                    'source_video':str(video),'source_size_hw':list(rgb.shape[:2]),'content_region':list(image_content_region(rgb)),
                    'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'regions':[list(r) for r in regions],
                    'seconds':time.monotonic()-start}
                meta_path.write_text(json.dumps(meta,indent=2));records.append(meta)
                print(json.dumps({'key':key,'regions':len(regions),'seconds':meta['seconds']}),flush=True)
    (out/'manifest.json').write_text(json.dumps({'header':header,'records':records},indent=2))
    print('complete',len(records),'images',sum(len(r['regions']) for r in records),'regions',flush=True)


if __name__=='__main__':main()
