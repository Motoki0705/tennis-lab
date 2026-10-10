"""Measure the actual frozen ViT-L training path on a Colab L4 before long runs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.training.compilation import compile_modules
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.evaluation.ablation import parameter_counts
from src.tasks.court_detection.model_io.contracts import CourtPoseTrainingResult
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
    _move_to_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError(
            "At least two steps are needed to separate compile/warmup time"
        )
    rclone_config = Path(os.environ.get("RCLONE_CONFIG", ""))
    if (
        Path("/content/.tennis-colab") not in rclone_config.parents
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Run this preflight through the Colab skill with an allocated L4"
        )
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu:
        raise RuntimeError(f"Expected L4, observed {gpu}")
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name="train_i983_l")
    runtime = CourtTrainingConfig.from_config(cfg)
    pl.seed_everything(int(cfg.run.seed), workers=True)
    torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))
    args.output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    evidence: dict[str, Any] = {
        "status": "preparing",
        "gpu": gpu,
        "torch": torch.__version__,
        "steps": [],
        "is_full_training_result": False,
    }

    def save() -> None:
        temporary = args.output / "profile.partial"
        temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "profile.json")

    save()
    try:
        data = CourtDetectionDataModule(cfg)
        data.setup("fit")
        module = (
            CourtDetectionLightningModule(cfg, target_bundle=data.target_bundle_spec)
            .cuda()
            .train()
        )
        evidence["parameters"] = parameter_counts(module)
        compile_modules(module.compilation_targets(), runtime.shared.training.compile)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=module.learning_rate,
            weight_decay=module.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
        )
        batches = iter(data.train_dataloader())
        torch.cuda.reset_peak_memory_stats()
        evidence["status"] = "running"
        for step in range(args.steps):
            loading = time.perf_counter()
            batch = cast(
                dict[str, Any],
                _move_to_device(next(batches), device=torch.device("cuda")),
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                call = module.model_io.prepare_training_batch(batch)
                output = module.model(*call.model_call.model_args)
                result = module.model_io.training_result(output, call)
            if not isinstance(result, CourtPoseTrainingResult) or not bool(
                torch.isfinite(result.loss)
            ):
                raise RuntimeError(
                    "The five-output training path must produce a finite pose training loss"
                )
            result.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                float(cfg.training.trainer.gradient_clip_val),
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize()
            evidence["steps"].append(
                {
                    "step": step,
                    "sample_ids": batch["sample_id"],
                    "image_shape": list(batch["image"].shape),
                    "content_size_hw": batch["content_size_hw"].tolist(),
                    "pose_supervised_count": int(batch["pose_supervision_mask"].sum()),
                    "pose_intrinsics": batch["pose_target"]["intrinsics"].tolist(),
                    "data_wait_seconds": started - loading,
                    "compute_seconds": time.perf_counter() - started,
                    "loss": float(result.loss.detach()),
                    "gradient_norm": float(norm),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            save()
            print(json.dumps(evidence["steps"][-1]), flush=True)
        evidence["status"] = "completed"
        evidence["steady_step_median_seconds"] = statistics.median(
            item["compute_seconds"] for item in evidence["steps"][1:]
        )
    except Exception as error:
        evidence.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
