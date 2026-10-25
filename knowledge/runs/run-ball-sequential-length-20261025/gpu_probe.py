"""Synthetic compute/memory preflight; excludes data loading and augmentation time."""

import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)

out = Path(
    "/content/tennis-lab/outputs/ball_detection/precompute/sequential_length/gpu-preflight-20261025"
)
out.mkdir(parents=True, exist_ok=True)
base = MDDHeatmapConfig.load(
    Path("src/tasks/ball_detection/configs/model/mdd_dpt_heatmap.yaml")
)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
results = []
for frames, batch in ((32, 1), (64, 1), (128, 1), (32, 4)):
    row = dict(
        T=frames,
        batch_size=batch,
        resolution=[720, 1280],
        compile=True,
        activation_checkpointing=False,
        frame_chunk_size=None,
        precision="bf16",
        input="synthetic_rgb; no JPEG decoding or augmentation",
        torch=torch.__version__,
    )
    model = rgb = optimizer = loss = logits = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_temporal_model(replace(base, frames=frames)).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        rgb = torch.randint(
            256, (batch, frames, 3, 720, 1280), dtype=torch.uint8, device="cuda"
        )
        target = dict(
            uv=torch.full((batch, frames, 2), 0.5, device="cuda"),
            position_valid=torch.ones((batch, frames), dtype=torch.bool, device="cuda"),
            heatmap_valid=torch.ones((batch, frames), dtype=torch.bool, device="cuda"),
        )
        compile_coordinate_model(model, mode="default")
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with coordinate_compile_scope(model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(rgb)
                    loss, _ = heatmap_objective(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            print(
                json.dumps(
                    dict(
                        T=frames,
                        batch=batch,
                        iteration=i,
                        seconds=times[-1],
                        loss=float(loss.detach()),
                        peak=torch.cuda.max_memory_allocated(),
                    )
                ),
                flush=True,
            )
        row.update(
            success=True,
            iteration_seconds=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except torch.OutOfMemoryError as e:
        row.update(
            success=False,
            error="CUDA_OOM",
            message=str(e),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as e:
        row.update(success=False, error=type(e).__name__, message=str(e))
    results.append(row)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    model = rgb = optimizer = loss = logits = None
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
