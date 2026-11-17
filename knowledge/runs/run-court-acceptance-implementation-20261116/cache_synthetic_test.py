"""Cache explicitly published synthetic test records using the existing loader."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from src.tasks.court_detection.configuration import SyntheticCourtSourceConfig
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.inference.regions import image_content_region
from src.tennis_scene.configuration import PROJECT_ROOT,_roots
from src.utils.configuration import PathRole

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path('data/synthetic_data_generation/scenes');scenes=('B00','B01','B02','B03')
    source=SyntheticCourtInput(SyntheticCourtSourceConfig('synthetic_court','v3','target_court',root,scenes))
    paths=OmegaConf.load(PROJECT_ROOT/'src/tennis_scene/configs/paths/default.yaml');paths.external_asset_root='/home/kamimura/projects/tennis-lab/third_party'
    _,resolver=_roots(OmegaConf.to_container(paths,resolve=True))
    predictor=CourtPredictor.load_from_checkpoint(resolver.resolve(PathRole.CHECKPOINT,'court_detection/multiscale_depth3/b863df1f01f0.ckpt'),resolver=resolver,device='cuda',subpixel_refine=True)
    records=[]
    for index,record in enumerate(source.records('test')):
        key='synthetic_court_test/'+record.sample_id.replace(':','_');target=args.output/key;target.mkdir(parents=True,exist_ok=True)
        if (target/'record.json').exists():records.append(json.loads((target/'record.json').read_text()));continue
        sample=source.load(record);rgb=np.asarray(sample.image).copy();h,w=rgb.shape[:2]
        prediction=predictor.predict(rgb,heads=('kp','line'),postprocess='none');kp,line=prediction.raw_heads['kp'],prediction.raw_heads['line']
        np.savez_compressed(target/'raw_00.npz',raw_xy=kp.keypoints[:,0].numpy(),scores=kp.scores[:,0].numpy(),valid=kp.valid[:,0].numpy(),line_probability=line.probability.numpy(),image_size_hw=np.array(prediction.original_size_hw),native_size_hw=np.array(prediction.native_size_hw))
        meta={'key':key,'dataset':'synthetic_court_test','clip_id':record.sample_id,'camera':'image','scene':sample.metadata.scene_id,'source_size_hw':[h,w],'content_region':list(image_content_region(rgb)),'regions':[[0,0,w,h]],'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'gt_xy':sample.keypoint_channels.points_xy[:,0].tolist(),'gt_valid':sample.keypoint_channels.point_visible[:,0].tolist()}
        (target/'record.json').write_text(json.dumps(meta));records.append(meta)
        if index%100==0:print('cached',index+1,record.sample_id,flush=True)
    header={'schema':'court_raw_cache_v1','checkpoint':predictor.checkpoint_identity,'source_split':'test','split_counts':{k:len(source.records(k)) for k in source.available_splits},'source_descriptors':{s:hashlib.sha256((root/s/'datasets/court/dataset.json').read_bytes()).hexdigest() for s in scenes},'checkpoint_training_overlap':'not established; regression test, not an independent generalization claim'}
    (args.output/'manifest.json').write_text(json.dumps({'header':header,'records':records},indent=2));print('complete',len(records),flush=True)

if __name__=='__main__':main()
