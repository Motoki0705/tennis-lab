"""Current production Court region path and Ball loader on fresh source frames."""
from pathlib import Path
from dataclasses import replace, asdict
import json, logging
import numpy as np
import torch
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.court_reference import prepare_court_reference
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.utils.configuration import PathResolver

torch.set_num_threads(4)
logging.basicConfig(level=logging.INFO)
root=Path(__file__).resolve().parent
config=OmegaConf.load(root/'pipeline.expanded_pipeline.yaml')
config.court_kp.region_search=asdict(CourtRegionSearchConfig(enabled=True))
runtime=PipelineRuntimeConfig.from_config(config)
resolver=PathResolver(replace(runtime.roots,checkpoint_root=Path('/home/kamimura/projects/tennis-lab/outputs')))
court_config=replace(runtime.court_kp,device='cpu',checkpoint=Path('/home/kamimura/projects/tennis-lab/outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt'),resolver=resolver,output_path=root/'current_court_preflight.json')
module=CourtKPModule(court_config)
result=module.process(runtime.video_paths,max_frames=1)
reference=prepare_court_reference(camera_ids=runtime.camera_ids,keypoints=result.keypoints,visibility=result.visibility,contract=resolve_court_keypoint_contract('camera_view_v2'),config=runtime.court_reference,size=(1920,1080),frame_index=0)
print('CURRENT_COURT_CALIBRATION_PASSED',flush=True)
del module
ball_config=replace(runtime.ball_detection,device='cpu',pin_memory=False,output_path=root/'current_ball_preflight.json',save_result=True)
ball=BallDetectionModule(ball_config).process(runtime.video_paths,max_frames=32,image_width=1920,image_height=1080)
summary={'court_shape':list(result.keypoints.shape),'court_visibility':result.visibility.tolist(),'ball_shape':list(ball.ball_uv_px.shape),'ball_finite':bool(np.isfinite(ball.ball_uv_px).all()),'ball_visible':np.asarray(ball.visibility).sum(axis=-1).tolist()}
(root/'current_inference_preflight.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
