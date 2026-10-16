"""Short, isolated L4 timing diagnostic; never resumes or replaces an ablation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.training.compilation import compile_modules
from src.tasks.court_detection.ablation.evaluation import parameter_counts
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
    _move_to_device,
)


class TimedModule(CourtDetectionLightningModule):
    """Preserve training losses/metrics while replacing the external log sink."""

    loss_ready_at: float = 0.0

    def log(self, *args: Any, **kwargs: Any) -> None:
        # No Lightning trainer/TensorBoard in this timing diagnostic.
        pass

    def _log_training_result(self, stage: str, result: Any) -> None:
        torch.cuda.synchronize()
        self.loss_ready_at = time.perf_counter()
        super()._log_training_result(stage, result)


def summarize(rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {"steps": len(rows), "batch_size": batch_size}
    fields = [key for key in rows[0] if key.endswith("_seconds")]
    for key in fields:
        values = [row[key] for row in rows]
        result[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }
    result["steps_per_second"] = len(rows) / sum(row["total_seconds"] for row in rows)
    result["images_per_second"] = batch_size * result["steps_per_second"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cached-steps", type=int, default=12)
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 16 or args.cached_steps < 4:
        raise ValueError("Use enough warmup and steps to exhaust DataLoader prefetch")
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This diagnostic requires the allocated Colab L4")
    config_path = Path(os.environ.get("RCLONE_CONFIG", "")).resolve()
    if Path("/content/.tennis-colab") not in config_path.parents:
        raise RuntimeError("Run through the tennis-colab managed job")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != "c4b12f923b243da6b502149aa6f3912c706045e7":
        raise RuntimeError(f"Unexpected scientific source: {commit}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence: dict[str, Any] = {
        "schema": "i983_throughput_v1",
        "status": "preparing",
        "is_full_training_result": False,
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "torch_cpu_threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "scope": (
            "Fresh seed42 S+ weights; real local mixed data, original losses and train metrics, "
            "bf16/compile/AdamW/gradient clipping. Constant initial learning rate; "
            "no Lightning log sink, validation, checkpoint, or TensorBoard. "
            "Synchronized phase boundaries may reduce overlap. Cached phase repeats "
            "the last real GPU batch solely to remove input work; it is not an accuracy run."
        ),
        "steps": [],
    }

    def save() -> None:
        temporary = output / "profile.partial"
        temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "profile.json")

    save()
    with (output / "gpu-samples.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
            ],
            stdout=gpu_log,
            stderr=subprocess.DEVNULL,
        )
        try:
            with initialize_config_dir(
                config_dir=str(Path("src/tasks/court_detection/configs").resolve()),
                version_base="1.3",
            ):
                cfg = compose(config_name="train_i983_splus")
            cfg.data.batch_size = args.batch_size
            cfg.mixed.train_batch_counts.synthetic_court = args.batch_size // 2
            cfg.mixed.train_batch_counts.tennis_court_detector = args.batch_size // 2
            runtime = CourtTrainingConfig.from_config(cfg)
            pl.seed_everything(int(cfg.run.seed), workers=True)
            if cfg.training.trainer.deterministic != "warn":
                raise RuntimeError("Expected the same deterministic=warn setting as training")
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = bool(cfg.training.trainer.benchmark)
            torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.training.allow_tf32)
            OmegaConf.save(cfg, output / "config.yaml", resolve=True)
            evidence["config_sha256"] = hashlib.sha256(
                (output / "config.yaml").read_bytes()
            ).hexdigest()
            data = CourtDetectionDataModule(cfg)
            data.setup("fit")
            loader = data.train_dataloader()
            evidence["loader_batches"] = len(loader)
            evidence["loader_num_workers"] = loader.num_workers
            evidence["loader_pin_memory"] = loader.pin_memory
            evidence["loader_prefetch_factor"] = loader.prefetch_factor
            evidence["loader_persistent_workers"] = loader.persistent_workers
            module = (
                TimedModule(cfg, target_bundle=data.target_bundle_spec).cuda().train()
            )
            evidence["parameters"] = parameter_counts(module)
            compile_modules(
                module.compilation_targets(), runtime.shared.training.compile
            )
            optimizer = torch.optim.AdamW(
                [p for p in module.parameters() if p.requires_grad],
                lr=module.learning_rate,
                weight_decay=module.weight_decay,
                betas=tuple(cfg.training.optimizer.betas),
            )
            batches = iter(loader)
            torch.cuda.reset_peak_memory_stats()
            evidence["status"] = "running"
            batch: dict[str, Any] | None = None
            count = args.warmup + args.steps + args.cached_steps
            for index in range(count):
                cached = index >= args.warmup + args.steps
                torch.cuda.synchronize()
                started_at_unix = time.time()
                began = time.perf_counter()
                if not cached:
                    cpu_batch = next(batches)
                    loaded = time.perf_counter()
                    batch = _move_to_device(cpu_batch, device=torch.device("cuda"))
                    torch.cuda.synchronize()
                    transferred = time.perf_counter()
                else:
                    loaded = transferred = began
                assert batch is not None
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = module._shared_step(batch, "train")
                torch.cuda.synchronize()
                metrics_ready = time.perf_counter()
                if not bool(torch.isfinite(result.loss)):
                    raise RuntimeError("Nonfinite profile loss")
                result.loss.backward()
                torch.cuda.synchronize()
                backward_done = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    float(cfg.training.trainer.gradient_clip_val),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                row = {
                    "index": index,
                    "started_at_unix": started_at_unix,
                    "finished_at_unix": time.time(),
                    "phase": "cached"
                    if cached
                    else ("warmup" if index < args.warmup else "streaming"),
                    "image_shape": list(batch["image"].shape),
                    "sample_ids": batch["sample_id"],
                    "data_wait_seconds": loaded - began,
                    "h2d_seconds": transferred - loaded,
                    "forward_loss_seconds": module.loss_ready_at - transferred,
                    "train_metrics_seconds": metrics_ready - module.loss_ready_at,
                    "backward_seconds": backward_done - metrics_ready,
                    "clip_optimizer_seconds": finished - backward_done,
                    "total_seconds": finished - began,
                    "loss": float(result.loss.detach()),
                    "gradient_norm": float(norm),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                evidence["steps"].append(row)
                save()
                print(
                    json.dumps({k: v for k, v in row.items() if k != "sample_ids"}),
                    flush=True,
                )
            evidence["summary"] = {
                phase: summarize(
                    [row for row in evidence["steps"] if row["phase"] == phase],
                    args.batch_size,
                )
                for phase in ("streaming", "cached")
            }
            evidence["status"] = "completed"
        except torch.cuda.OutOfMemoryError as error:
            evidence.update(
                status="out_of_memory",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        except Exception as error:
            evidence.update(
                status="failed", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            evidence["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            evidence["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            save()


if __name__ == "__main__":
    main()
