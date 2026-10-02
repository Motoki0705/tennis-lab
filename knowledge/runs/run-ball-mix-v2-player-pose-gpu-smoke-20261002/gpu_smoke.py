import json
from pathlib import Path
import torch
from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models.dino.person_detector import DinoPersonDetector, PersonDetectionRequest
from src.submodules.models.vitpose.pose2d import ViTPosePose2D
from src.submodules.models.dino.extension import validate_dino_extension
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tasks.player_association.appearance.encoders import ClipReIDEncoder
from src.tasks.ball_detection.data.store import BallFrameStore

ROOT = Path('/home/kamimura/projects/tennis-lab')
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.65)
print({'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'extension':str(validate_dino_extension())},flush=True)
store=BallFrameStore(ROOT/'data/ball_detection/ball-mix-v2')
image=store.read_bgr(0)
detector=DinoPersonDetector(ROOT/'ckpt/dino/checkpoint0029_4scale_swin.pth',ROOT/'third_party/DINO',device='cuda',confidence=.3,short_side=800,max_long_side=1333)
det=detector.predict(PersonDetectionRequest(image))
torch.cuda.synchronize()
print({'stage':'DINO','detections':len(det.scores)},flush=True)
detector.unload(); del detector
torch.cuda.empty_cache()
pose=ViTPosePose2D(ROOT/'third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth',device='cuda',flip_test=True,batch_size=4,head_config=ViTPoseHeadConfig(in_channels=1280,out_channels=17,num_deconv_layers=2,num_deconv_filters=(256,256),num_deconv_kernels=(4,4),final_conv_kernel=1,num_conv_layers=0,num_conv_kernels=()),precision='float32')
encoder=ClipReIDEncoder('clipreid_vitb16_market1501',ROOT/'ckpt/player_association/person_vit_clip_reid.pth','cuda')
import numpy as np
features=FeatureExtractor(pose,UnpromptedEncoder(encoder,1280)).extract(0,image,np.arange(len(det.scores),dtype=np.int64),det.boxes_xyxy,det.scores)
torch.cuda.synchronize()
result={'status':'passed','detected_people':len(det.scores),'pose_shape':list(features.poses.shape),'embedding_shape':list(features.embeddings.shape),'court_executed':False}
(ROOT/'outputs/chat_annotation/player_pose').mkdir(parents=True,exist_ok=True)
(ROOT/'outputs/chat_annotation/player_pose/gpu-smoke-20261002.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
