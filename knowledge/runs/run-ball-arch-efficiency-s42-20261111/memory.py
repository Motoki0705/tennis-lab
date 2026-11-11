"""Exact-shape joint-gradient feasibility; no scientific training updates.

Fixed B=1,T64,720p,2 microbatches/update, BF16+FP32 MSDA, no checkpointing.
This diagnostic deliberately exercises an unfrozen CNN; the main 600-update
freeze schedule is unchanged. Synthetic targets test memory, not accuracy.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path.cwd()))
import torch
from protocol import ROOT, model_config, write, read
from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model
from src.tasks.ball_detection.training.runtime.compilation import compile_coordinate_model, coordinate_compile_scope, coordinate_compilation_report

p=argparse.ArgumentParser();p.add_argument('--case',choices=['T1','T2'],required=True);p.add_argument('--levels',type=int,choices=[3,4],required=True)
a=p.parse_args()
levels=(8,16,32,64) if a.levels==4 else (16,32,64)
path=ROOT/'qualification'/f'memory-{a.case}-{a.levels}.json'
result=dict(case=a.case,levels=levels,shape=[1,64,3,720,1280],microbatches=2,input_frames_per_update=128,
            precision='bf16 with complete official MSDA FP32',cnn_frozen=False,activation_checkpointing=False,
            msda_chunk_size=4,compile_mode='default',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            status='running',samples=[])
write(path,result)
phase='initialization'
try:
    if not read(ROOT/'data-verified.json')['complete']: raise RuntimeError('Fixed data must be verified first')
    if torch.cuda.get_device_name()!='NVIDIA L4': raise RuntimeError('L4 required')
    torch.set_num_threads(2);torch.manual_seed(42)
    model=build_temporal_model(model_config(a.case,levels=levels)).cuda().train()
    compile_coordinate_model(model,mode='default',fullgraph=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=7.360330616529733e-5,weight_decay=.01)
    rgb=torch.randint(256,(1,64,3,720,1280),device='cuda',dtype=torch.uint8)
    times=torch.arange(64,device='cuda')[None].float()/30
    for update in range(3):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with coordinate_compile_scope(model):
            for micro in range(2):
                phase=f'forward-{micro}'
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(rgb,times)
                    loss=torch.nn.functional.smooth_l1_loss(prediction.float(),torch.full_like(prediction,.5),beta=.01)/2
                phase=f'backward-{micro}';loss.backward()
            phase='optimizer'
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        sample=dict(update=update+1,loss=float(loss.detach()),gradient=float(gradient),
                    allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
                    seconds=time.perf_counter()-started)
        result['samples'].append(sample);write(path,result);print(json.dumps(sample),flush=True)
    result.update(status='passed',compilation=coordinate_compilation_report(model));write(path,result)
except BaseException as e:
    result.update(status='oom' if isinstance(e,torch.OutOfMemoryError) else 'failed',phase=phase,error=str(e),
                  peak_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                  peak_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None)
    write(path,result);raise
