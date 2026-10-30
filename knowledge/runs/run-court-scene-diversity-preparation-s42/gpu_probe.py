"""Check three real production-shape optimizer steps without starting a full run."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# This standalone reproduction entry must import its own checkout.
# ruff: noqa: E402
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import pytorch_lightning as pl
import torch
from campaign import CAMPAIGN, load_config, preflight

from src.tasks.base.training.compilation import compile_modules
from src.tasks.court_detection.ablation.evaluation import parameter_counts
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
    _move_to_device,
)
from src.utils.data.image_record_store import file_sha256


def main():
    if not os.environ.get("TENNIS_RUN_ID") or not torch.cuda.is_available():
        raise RuntimeError("The GPU probe must run through the shared training queue.")
    preflight()
    config = load_config(1)
    runtime = CourtTrainingConfig.from_config(config)
    destination = CAMPAIGN / "gpu-probe.json"
    if destination.exists():
        raise FileExistsError(destination)
    evidence = {
        "schema": "court_scene_gpu_probe_v1",
        "is_full_training_result": False,
        "plan_sha256": file_sha256(CAMPAIGN / "plan.json"),
        "ready_sha256": file_sha256(CAMPAIGN / "ready.json"),
        "status": "preparing",
        "gpu": torch.cuda.get_device_name(0),
        "steps": [],
    }
    try:
        pl.seed_everything(42, workers=True)
        torch.set_float32_matmul_precision(config.training.matmul_precision)
        dm = CourtDetectionDataModule(config)
        dm.setup("fit")
        module = (
            CourtDetectionLightningModule(config, target_bundle=dm.target_bundle_spec)
            .cuda()
            .train()
        )
        evidence["parameters"] = parameter_counts(module)
        compile_modules(module.compilation_targets(), runtime.shared.training.compile)
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad],
            lr=module.learning_rate,
            weight_decay=module.weight_decay,
            betas=tuple(config.training.optimizer.betas),
        )
        batches = iter(dm.train_dataloader())
        torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            batch = _move_to_device(next(batches), device=torch.device("cuda"))
            if batch["image"].shape[0] != 8 or not batch["pose_supervision_mask"].all():
                raise ValueError("Expected eight synthetic pose-supervised images.")
            torch.cuda.synchronize()
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                call = module.model_io.prepare_training_batch(batch)
                result = module.model_io.training_result(
                    module.model(*call.model_call.model_args), call
                )
            if not torch.isfinite(result.loss):
                raise ValueError("Non-finite training loss.")
            result.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            evidence["steps"].append(
                {
                    "step": step + 1,
                    "loss": float(result.loss.detach()),
                    "gradient_norm": float(norm),
                    "seconds": time.monotonic() - start,
                    "image_shape": list(batch["image"].shape),
                    "sample_ids": batch["sample_id"],
                }
            )
            print(json.dumps(evidence["steps"][-1]), flush=True)
        evidence.update(
            status="completed",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except BaseException as error:
        evidence.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
