"""Reproducible bounded training, validation selection, and held-out evaluation."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.tasks.ball_refiner.coordinates.config import (
    CorruptionConfig,
    ModelConfig,
    TrainingConfig,
    parse_section,
    training_config,
)
from src.tasks.ball_refiner.coordinates.data import (
    PreparedRally,
    SharedDataset,
    prepare,
    sample_batch,
)
from src.tasks.ball_refiner.coordinates.evaluation import evaluate
from src.tasks.ball_refiner.coordinates.inference import (
    checkpoint_metadata,
    load_checkpoint,
)
from src.tasks.ball_refiner.coordinates.model import (
    CoordinateRefiner,
    TrajectoryDiscriminator,
)
from src.tasks.base.configuration import CompileConfig
from src.tasks.base.training.compilation import compile_modules
from src.tasks.base.training.gan_loss import LSGANLoss
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.device import resolve_device


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)


def corruption_audit(data: list[PreparedRally]) -> dict[str, float]:
    noises = np.concatenate([np.linalg.norm(r.corrupted.noise_px, axis=-1)[~r.corrupted.missing_2d] for r in data])
    eligible = sum(np.count_nonzero(r.source.visible & ~np.any(r.corrupted.missing_2d & ~r.corrupted.isolated & r.source.visible, axis=0)[None]) for r in data)
    event_count = sum(np.count_nonzero(r.source.events) for r in data)
    return {
        "noise_p95_px": float(np.quantile(noises, 0.95)),
        "noise_p50_px": float(np.median(noises)),
        "frame_missing_rate_2d": float(np.concatenate([r.corrupted.missing_2d.ravel() for r in data]).mean()),
        "frame_missing_rate_3d": float(np.concatenate([r.corrupted.missing_3d for r in data]).mean()),
        "isolated_probability_realized": sum(int(r.corrupted.isolated.sum()) for r in data) / eligible if eligible else 0.0,
        "selected_event_fraction": sum(len(r.corrupted.intervals) for r in data) / event_count if event_count else 0.0,
    }


def run_training(config: DictConfig) -> Path:
    raw, dataset_path, output = training_config(config)
    model_config = parse_section(ModelConfig, raw["model"])
    train = parse_section(TrainingConfig, raw["training"])
    corruption = parse_section(CorruptionConfig, raw["corruption"])
    seed = int(raw["run"]["seed"])
    device = resolve_device(raw["run"]["device"])
    if device.type == "cuda" and not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("GPU training must be launched through the shared training queue")
    if output.exists() and any(p.name != "hydra" for p in output.iterdir()):
        raise FileExistsError(f"New training requires a new run directory: {output}")
    dataset = SharedDataset(dataset_path)
    if min(len(r.xyz) for r in dataset.rallies) < model_config.window_length:
        raise ValueError("Training window exceeds the shortest shared rally")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(raw), output / "config.yaml")
    write_json_atomic(output / "state.json", {"status": "preparing"})
    checkpoint_dir = output / "logs" / "version_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    writer = SummaryWriter(str(output / "logs" / "version_0"))
    torch.set_num_threads(train.cpu_threads)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = CoordinateRefiner(model_config).to(device)
    discriminator = TrajectoryDiscriminator(model_config.dimensions, model_config.width).to(device) if train.gan_weight else None
    compile_modules({"refiner": model, **({"discriminator": discriminator} if discriminator else {})}, CompileConfig.from_mapping(raw["compile"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay) if discriminator else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train.steps, eta_min=train.learning_rate * 0.05)
    adversarial = LSGANLoss()
    # Discriminator initialization must not change the generator dropout stream.
    torch.manual_seed(seed + 100)
    flow_rng = torch.Generator(device=device).manual_seed(seed + 200)
    sampling = np.random.default_rng(seed + 300)
    eval_config = replace(corruption, event_probability=float(raw["data"]["evaluation_event_probability"]))
    evaluation_seed = int(raw["data"]["evaluation_seed"])
    validation = prepare(dataset.split("val"), model_config.dimensions, eval_config, evaluation_seed)
    write_json_atomic(output / "data_contract.json", {
        "dataset": str(dataset_path), "manifest_sha256": dataset.manifest_hash, "fps": dataset.fps,
        "train_ids": [r.name for r in dataset.split("train")], "val_ids": [r.name for r in dataset.split("val")],
        "test_ids": [r.name for r in dataset.split("test")], "evaluation_corruption": asdict(eval_config),
        "evaluation_seed": evaluation_seed, "validation_audit": corruption_audit(validation),
        "selection": "lowest all-frame physical-coordinate RMSE on fixed validation corruption",
        "augmentation_refresh": "entire training split refreshed every evaluate_every updates; seed+10000+block",
    })
    best = math.inf
    best_step = 0
    start = time.monotonic()
    training_data: list[PreparedRally] = []
    sums: dict[str, float] = {"reconstruction": 0, "generator_gan": 0, "discriminator": 0}
    try:
        for step in range(1, train.steps + 1):
            if (step - 1) % train.evaluate_every == 0:
                block = (step - 1) // train.evaluate_every
                training_data = prepare(dataset.split("train"), model_config.dimensions, corruption, seed + 10000 + block)
                audit = corruption_audit(training_data)
                write_json_atomic(output / f"corruption-block-{block:03d}.json", audit)
                for key, value in audit.items():
                    writer.add_scalar(f"corruption/{key}", value, step)
            coordinates, missing, target = sample_batch(training_data, train.batch_size, model_config.window_length, sampling, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gan_loss = coordinates.new_zeros(())
            disc_loss = coordinates.new_zeros(())
            if model_config.architecture == "flow":
                reconstruction = model.flow_loss(coordinates, missing, target, flow_rng)
            else:
                prediction = model(coordinates, missing)
                reconstruction = F.smooth_l1_loss(prediction, target, beta=0.02)
                if discriminator is not None and disc_optimizer is not None and step > train.gan_warmup_steps:
                    discriminator.requires_grad_(True)
                    disc_optimizer.zero_grad(set_to_none=True)
                    disc_loss = adversarial.discriminator_loss(discriminator(target, coordinates, missing), discriminator(prediction.detach(), coordinates, missing))
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), train.gradient_clip, error_if_nonfinite=True)
                    disc_optimizer.step()
                    discriminator.requires_grad_(False)
                    gan_loss = adversarial.generator_loss(discriminator(prediction, coordinates, missing))
            loss = reconstruction + train.gan_weight * gan_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for key, value in (("reconstruction", reconstruction), ("generator_gan", gan_loss), ("discriminator", disc_loss)):
                sums[key] += float(value.detach())
            if step % train.log_every == 0:
                row = {key: value / train.log_every for key, value in sums.items()}
                row.update(step=step, seconds=time.monotonic() - start)
                for key, value in row.items():
                    writer.add_scalar(f"train/{key}", value, step)
                with (output / "learning_curve.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                sums = dict.fromkeys(sums, 0.0)
            if step % train.evaluate_every == 0 or step == train.steps:
                report, _ = evaluate(model, validation, device, seed=evaluation_seed, batch_size=train.batch_size)
                score = float(report["all"]["rmse"])
                writer.add_scalar("val/rmse", score, step)
                for name in ("observed", "missing", "event"):
                    if report[name]["rmse"] is not None:
                        writer.add_scalar(f"val/{name}_rmse", report[name]["rmse"], step)
                payload = {**checkpoint_metadata(model), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "step": step, "seed": seed, "fps": dataset.fps,
                           "manifest_sha256": dataset.manifest_hash, "validation_rmse": score,
                           "discriminator": discriminator.state_dict() if discriminator else None,
                           "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer else None,
                           "torch_rng": torch.get_rng_state(), "flow_rng": flow_rng.get_state(),
                           "sampling_rng": sampling.bit_generator.state}
                save_checkpoint(checkpoint_dir / "last.ckpt", payload)
                if score < best:
                    best, best_step = score, step
                    save_checkpoint(checkpoint_dir / "best.ckpt", payload)
                write_json_atomic(output / f"validation-{step:06d}.json", report)
                write_json_atomic(output / "state.json", {"status": "training", "step": step, "best_step": best_step, "best_val_rmse": best})
                print(json.dumps({"step": step, "validation_rmse": score, "best_step": best_step}), flush=True)
        # Test is only opened after checkpoint selection is complete.
        del training_data, validation
        selected, _ = load_checkpoint(checkpoint_dir / "best.ckpt", device)
        test = prepare(dataset.split("test"), model_config.dimensions, eval_config, evaluation_seed)
        report, predictions = evaluate(selected, test, device, seed=evaluation_seed, batch_size=train.batch_size)
        report.update(best_step=best_step, validation_rmse=best, training_seconds=time.monotonic() - start,
                      peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                      test_corruption_audit=corruption_audit(test), dataset_manifest_sha256=dataset.manifest_hash)
        prediction_dir = output / "predictions"
        prediction_dir.mkdir()
        with (prediction_dir / "pred_test.npz").open("xb") as stream:
            np.savez_compressed(stream, allow_pickle=False, **predictions)
        unit = report["unit"]
        metrics = {f"test_rmse_{unit}": report["all"]["rmse"], f"test_missing_rmse_{unit}": report["missing"]["rmse"],
                   f"test_event_rmse_{unit}": report["event"]["rmse"], "test_frame_missing_rate": report["frame_missing_rate"],
                   "inference_ms_per_frame": report["milliseconds_per_frame"], "best_step": best_step}
        write_json_atomic(prediction_dir / "metrics.json", metrics)
        write_json_atomic(prediction_dir / "diagnostic_metrics.json", report)
        from src.tasks.ball_refiner.coordinates.visualization import plot_predictions
        plot_predictions(predictions, prediction_dir / "examples.png", dimensions=model_config.dimensions)
        if os.environ.get("TENNIS_REPRO_DIR"):
            repro = Path(os.environ["TENNIS_REPRO_DIR"])
            shutil.copytree(prediction_dir, repro / "predictions", dirs_exist_ok=True)
            (repro / "output_dir.txt").write_text(str(checkpoint_dir) + "\n")
        write_json_atomic(output / "state.json", {"status": "complete", "step": train.steps, "best_step": best_step, **metrics})
        print(json.dumps({"output": str(output), **metrics}), flush=True)
    except BaseException as error:
        write_json_atomic(output / "state.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        writer.close()
    return output
