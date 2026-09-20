"""Bounded diagnosis of observed receipt mismatches around real CUDA pose inference."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from src.tennis_scene.scripts import build_slcs_dataset
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.dataset_pipeline.people import _detections
from src.tennis_scene.generate_dataset.manifest import ClipManifest,load_dataset_manifest
from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models import ViTPosePose2D,Pose2DRequest
from src.submodules.models.tracker.common import TrackResult

root=Path.cwd();out=root/'outputs/tennis_scene/analyze/meiji_hash_probe/s42-001';out.mkdir(parents=True,exist_ok=True)
with initialize_config_dir(config_dir=str(root/'src/tennis_scene/configs'),version_base='1.3'):
    cfg=compose(config_name='build_slcs_dataset',overrides=['paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party'])
paths=ReferenceClipPaths.from_config(cfg)
source=root/'data/tennis_multivew/processed/meiji_3cam/dataset';manifest=load_dataset_manifest(source)
check_clip=ClipManifest.load(source/manifest.clips['video_000/clip_001'].path)
cache=root/'outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/video_000/clip_001/cam1_detections.npz'
receipt=json.loads(cache.with_suffix('.metadata.json').read_text())
files=[(paths.dino_checkpoint,receipt['checkpoint_sha256']),(paths.vitpose_checkpoint,'50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc'),(Path('/home/kamimura/projects/tennis-lab/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),'73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'),(check_clip.media_path('cam1'),receipt['video_sha256'])]
results=[];errors=[]
def checkpoint(phase):
    for p,expected in files:
        before=p.stat()
        with p.open('rb') as f: automatic=hashlib.file_digest(f,'sha256').hexdigest()
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
        manual=digest.hexdigest()
        system=subprocess.run(['sha256sum',str(p)],check=True,capture_output=True,text=True).stdout.split()[0]
        after=p.stat()
        row={'phase':phase,'path':str(p),'expected':expected,'file_digest':automatic,'manual':manual,'sha256sum':system,'match':automatic==manual==system==expected,'stat_unchanged':(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)}
        results.append(row);print(json.dumps(row),flush=True)
    try:
        settings=OmegaConf.create(OmegaConf.to_container(cfg.people,resolve=True));settings.device=cfg.device
        _detections(settings,paths,check_clip.media_path('cam1'),cache,check_clip.num_frames)
    except Exception as exc:
        errors.append({'phase':phase,'detection_identity_error':repr(exc)})
    (out/'results.json').write_text(json.dumps({'read_checks':results,'errors':errors,'scope':'same files read by three methods before/after real CUDA inference; stable results cannot establish historical failure cause'},indent=2)+'\n')
checkpoint('before_cuda_pose')
clip=ClipManifest.load(source/manifest.clips['video_000/clip_007'].path)
obs=root/'outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/video_000/clip_007/cam0_people.npz'
with np.load(obs,allow_pickle=False) as n: boxes=n['boxes'][0]
tracks=TrackResult({0:torch.from_numpy(boxes)},clip.num_frames)
pose=ViTPosePose2D(paths.vitpose_checkpoint,device='cuda',flip_test=True,batch_size=8,precision='bfloat16',head_config=ViTPoseHeadConfig(1280,17,2,(256,256),(4,4),1,0,()))
try:
    pose.predict(Pose2DRequest(clip.media_path('cam0'),tracks.bbx_xys(0,base_enlarge=1.2)))
    checkpoint('after_cuda_pose')
finally:
    pose.unload();torch.cuda.empty_cache()
checkpoint('after_unload')
assert not errors and all(r['match'] and r['stat_unchanged'] for r in results), 'Observed an identity/hash discrepancy; see results.json'
