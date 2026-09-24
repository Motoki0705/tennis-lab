"""Queue-only extension numerics and a short fresh DINO→ViTPose→HMR2→GVHMR chain."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import importlib.util,json,time
import numpy as np
import torch
from omegaconf import OmegaConf
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.model_io.gvhmr import build_gvhmr_chain,GVHMRChainRequest
import MultiScaleDeformableAttention as native

root=Path(__file__).resolve().parent
expected=root/'dino_extension/lib/MultiScaleDeformableAttention.so'
assert Path(native.__file__).resolve()==expected.resolve(),native.__file__
runtime=PipelineRuntimeConfig.from_config(OmegaConf.load(root/'pipeline.expanded_pipeline.yaml'))
source=runtime.gvhmr.dino_repository/'models/dino/ops/functions/ms_deform_attn_func.py'
spec=importlib.util.spec_from_file_location('dino_math_reference',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.manual_seed(0)
value=torch.rand((2,7,2,4),device='cuda',requires_grad=True)
shapes=torch.tensor([[2,2],[1,3]],device='cuda',dtype=torch.int64)
starts=torch.tensor([0,4],device='cuda',dtype=torch.int64)
locations=torch.rand((2,5,2,2,3,2),device='cuda',requires_grad=True)
weights=torch.rand((2,5,2,2,3),device='cuda',requires_grad=True)
expected_output=module.ms_deform_attn_core_pytorch(value,shapes,locations,weights)
actual=native.ms_deform_attn_forward(value,shapes,starts,locations,weights,64)
torch.testing.assert_close(actual,expected_output,rtol=1e-5,atol=1e-6)
gradient=torch.rand_like(actual)
expected_grads=torch.autograd.grad(expected_output,(value,locations,weights),gradient)
actual_grads=native.ms_deform_attn_backward(value,shapes,starts,locations,weights,gradient,64)
for a,b in zip(actual_grads,expected_grads,strict=True):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
summary={'extension':str(native.__file__),'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'forward_max_abs_difference':float((actual-expected_output).abs().max()),'backward_max_abs_differences':[float((a-b).abs().max()) for a,b in zip(actual_grads,expected_grads,strict=True)]}
print(summary,flush=True)
del value,shapes,starts,locations,weights,expected_output,actual,gradient,expected_grads,actual_grads
torch.cuda.empty_cache()
begin=time.monotonic();chain=build_gvhmr_chain(runtime.gvhmr)
result=chain.predict(GVHMRChainRequest(video_path=root/'smoke_cam0.mp4',max_frames=None,num_tracks=2,interactive=False,bbox_enlarge=runtime.gvhmr.runtime.tracking.bbox_enlarge,static_cam=runtime.gvhmr.runtime.static_cam,footpoint_polygon_px=None))
arrays={key:value for key,value in vars(result).items() if isinstance(value,np.ndarray)}
summary['chain']={'seconds':time.monotonic()-begin,'arrays':{k:{'shape':list(v.shape),'finite':bool(np.isfinite(v).all())} for k,v in arrays.items()}}
assert all(value['finite'] for value in summary['chain']['arrays'].values())
(root/'dino_gvhmr_gpu_preflight.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
