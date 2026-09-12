"""Measure power-of-two batches in fresh processes using the actual training loop."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.ball_detection.configuration import validate_training
from src.tasks.ball_detection.data.multiview_datamodule import MultiviewBallDataModule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from src.tasks.base.training.qualitative_callback import QualitativeLoggingCallback


def require_l4() -> str:
    """Check assignment without creating a CUDA context in the probe coordinator."""
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if name != "NVIDIA L4":
        raise RuntimeError(
            f"Requested NVIDIA L4, received {name!r}; change the Colab runtime."
        )
    return name


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize CUDA OOMs even when torch.compile wraps the original exception."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        for cause in (
            current.__cause__,
            current.__context__,
            getattr(current, "inner_exception", None),
        ):
            if isinstance(cause, BaseException):
                pending.append(cause)
    return False


def largest_power_of_two(fits: Callable[[int], bool]) -> int:
    """Return the last successful power only after the next power has failed."""
    batch = 1
    while fits(batch):
        batch *= 2
    if batch == 1:
        raise RuntimeError("Batch size 1 does not fit; no valid training batch exists.")
    return batch // 2


def run_trial(config: DictConfig, batch_size: int) -> dict[str, Any]:
    """Exercise optimizer state, backward, metrics and validation at identical precision."""
    gpu = require_l4()
    config.data.batch_size = batch_size
    validate_training(config)
    runner = BallDetectionTrainingRunner()
    runtime = runner.validate_runtime_config(config)
    runner.seed_everything(runtime)
    runner.apply_runtime_settings(runtime)
    options = runtime.training.trainer
    accumulation = options.accumulate_grad_batches
    microbatches = 3 * accumulation
    dm = MultiviewBallDataModule(config)
    dm.setup("fit")
    if dm.train_dataset is None or len(dm.train_dataset) < batch_size * microbatches:
        raise RuntimeError(
            "Calibration needs three optimizer updates with full accumulated batches; cannot claim the GPU maximum."
        )
    module = BallDetectionLightningModule(config)
    # Preserve the full-run schedule while the probe Trainer limits execution
    # to three updates. Otherwise warmup=200 is rejected against only 3 steps.
    module.steps_per_epoch = math.ceil(len(dm.train_dataloader()) / accumulation)
    runner.maybe_load_init_weights(runtime, module)
    runner.maybe_compile_models(runtime, module)
    qualitative = runtime.training.qualitative_logging
    callbacks: list[pl.Callback] = []
    if qualitative.enabled:
        callbacks.append(
            QualitativeLoggingCallback(
                every_n_epochs=1,
                num_samples=qualitative.num_samples,
                enabled=True,
                selection_mode="fixed_indices",
                selected_indices=[0],
            )
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=options.precision,
        max_epochs=1,
        limit_train_batches=microbatches,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=str(runtime.run.output_dir / "calibration"),
            name=f"batch_{batch_size}",
        ),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=options.gradient_clip_val,
        accumulate_grad_batches=options.accumulate_grad_batches,
        deterministic=options.deterministic,
        benchmark=options.benchmark,
        callbacks=callbacks,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(module, datamodule=dm)
    torch.cuda.synchronize()
    return {
        "status": "fits",
        "batch_size": batch_size,
        "accumulate_grad_batches": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "training_microbatches": microbatches,
        "optimizer_updates": trainer.global_step,
        "gpu": gpu,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def calibrate(config: DictConfig, output_dir: Path) -> int:
    """Keep per-trial logs/results and the resolved config next to checkpoints."""
    require_l4()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "probe_config.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    trials: list[dict[str, Any]] = []

    def fits(batch: int) -> bool:
        result_path = output_dir / f"batch_{batch}.json"
        log_path = output_dir / f"batch_{batch}.log"
        result_path.unlink(missing_ok=True)
        print(f"[L4 calibration] trial batch_size={batch}; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    __name__,
                    str(config_path),
                    str(batch),
                    str(result_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Calibration failed (not an accepted CUDA OOM); inspect {log_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] not in {"fits", "cuda_oom"}:
            raise RuntimeError(f"Invalid calibration result: {result}")
        trials.append(result)
        return bool(result["status"] == "fits")

    selected = largest_power_of_two(fits)
    (output_dir / "batch_size.json").write_text(
        json.dumps(
            {
                "batch_size": selected,
                "accumulate_grad_batches": int(config.training.trainer.accumulate_grad_batches),
                "effective_batch_size": selected * int(config.training.trainer.accumulate_grad_batches),
                "next_power_failed": selected * 2,
                "trials": trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def main() -> None:
    """Subprocess entry point; propagate every error except CUDA memory exhaustion."""
    config_path, batch, result_path = sys.argv[1:]
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError("Expected a config mapping")
    try:
        result = run_trial(config, int(batch))
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        result = {"status": "cuda_oom", "batch_size": int(batch), "error": str(error)}
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
