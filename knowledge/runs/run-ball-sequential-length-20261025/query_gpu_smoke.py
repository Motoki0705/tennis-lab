"""Native 720p/T64 query AMP+compile and CNN freeze/unfreeze diagnostic.

Uses synthetic inputs and a one-update freeze only to exercise both compiled
backward graphs. The production freeze remains 600 updates, covered separately
by the schedule tests and recorded in each real stage's recipe.
"""

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch

from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDTemporalQueryConfig,
)
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = replace(
        MDDTemporalQueryConfig.load(
            Path("src/tasks/ball_detection/configs/model/mdd_temporal_query.yaml")
        ),
        frames=64,
    )
    model = DeepMDDQueryDetector(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    rgb = torch.randint(256, (1, 64, 3, 720, 1280), device="cuda", dtype=torch.uint8)
    pts = torch.arange(64, device="cuda")[None].float() / 30
    initial = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
    compile_coordinate_model(model, mode="default")
    records = []
    for update in range(2):
        frozen = update == 0
        model.train()
        model.freeze_encoder(frozen)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with coordinate_compile_scope(model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                uv = model(rgb, pts)
                loss = torch.nn.functional.smooth_l1_loss(
                    uv, torch.full_like(uv, 0.4), beta=0.01
                )
            loss.backward()
            assert (
                model.ball_query.grad is not None
                and torch.isfinite(model.ball_query.grad).all()
            )
            if frozen:
                assert all(p.grad is None for p in model.encoder.parameters())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        torch.cuda.synchronize()
        changed = any(
            not torch.equal(initial[n], p) for n, p in model.encoder.named_parameters()
        )
        assert changed is (not frozen)
        record = dict(
            update=update + 1,
            encoder_frozen=frozen,
            encoder_changed=changed,
            loss=float(loss.detach()),
            seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    output = Path(
        "/content/tennis-lab/outputs/ball_detection/precompute/sequential_length/query-gpu-smoke-20261025.json"
    )
    output.write_text(
        json.dumps(
            dict(
                success=True,
                torch=torch.__version__,
                config=asdict(cfg),
                records=records,
                scope="synthetic backend diagnostic; one-update freeze; production freeze is600; no training metrics or cost extrapolation",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
