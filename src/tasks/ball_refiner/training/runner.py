"""Bounded detector-only NLL pilot with fixed validation and durable epoch outputs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache, select_clips
from src.tasks.ball_refiner.data.gaps import (
    fixed_gap_mask,
    mask_detector_evidence,
    random_gap_mask,
    validation_partition,
)
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import (
    BalancedSourceSampler,
    LoadedClip,
    RefinerBatch,
    RefinerWindowDataset,
    collate_windows,
)
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d, refiner_2d_nll
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import evaluate_selection
from src.tasks.base.training.compilation import compile_modules
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.device import resolve_device


def prepare_data(config: PilotConfig) -> tuple[RefinerWindowDataset, tuple[LoadedClip, ...], dict[str, Any]]:
    store = BallFrameStore(config.store)
    cache = EvidenceCache(config.evidence, store)
    manifest_hash = dual_sha256(config.evidence / "manifest.json")
    records = select_clips(store, splits=("train", "val"), sources=config.sources)
    if not {r.clip_id for r in records} <= set(cache.clip_ids):
        raise ValueError("Evidence cache is missing requested source/split clips")
    partition = validation_partition(tuple(r for r in records if r.source == "meiji" and r.split == "val"), config.partition_seed)
    training, selection = [], []
    for record in records:
        if record.split == "train" or record.clip_id in partition["selection"]:
            clip = LoadedClip(record, cache.load(record.clip_id), project_store_targets(store, record))
            if record.split == "train":
                training.append(clip)
            else:
                if record.frame_count < config.window_length:
                    raise ValueError("Selection clips must support the configured real window length")
                selection.append(clip)
    dataset = RefinerWindowDataset(training, length=config.window_length, stride=config.stride, config=config.model)
    gap_intervals: dict[str, list[list[int]]] = {}
    for clip in selection:
        gap = fixed_gap_mask(clip.record.frame_count, clip_id=clip.record.clip_id, block_length=config.window_length,
                             lengths=config.training.gap_lengths, seed=config.partition_seed)
        edges = np.diff(np.r_[False, gap, False].astype(np.int8))
        gap_intervals[clip.record.clip_id] = [[int(a), int(b)] for a, b in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True,
        )]
    manifest = {
        "schema": "ball_refiner_pilot_data.v1", "evidence_manifest_sha256": manifest_hash,
        "store_sha256": cache.manifest["store"]["sha256"], "detector": cache.manifest["detector"],
        "model_input": "detector_only; pose/court disabled explicitly, not missing observations",
        "train_clip_ids": [c.record.clip_id for c in training],
        "source_windows": {key: len(value) for key, value in dataset.source_indices.items()},
        "train_frames": sum(c.record.frame_count for c in training),
        "excluded": dataset.excluded, "validation": partition,
        "partition_policy": "sha256(seed:meiji/video/clip), alternating groups, every camera kept together",
        "unused_val_clip_ids": [r.clip_id for r in records if r.split == "val" and r.source != "meiji"],
        "selection_gap_intervals": gap_intervals,
        "gap_semantics": "all candidate fields removed on fixed source frames; not RGB occlusion",
        "window_policy": "nearest centre; tie earlier start; real tail backfill; no temporal padding",
        "sampling": "equal source counts (+/-1), uniform windows within source, seeded per epoch, with replacement",
        "validation_scope": "selection only; calibration clips and test excluded from model selection",
        "metrics_scope": "pilot NLL/error/presence/total variance; HDR coverage/bootstrap not yet evaluated",
    }
    if dual_sha256(config.evidence / "manifest.json") != manifest_hash:
        raise ValueError("Evidence manifest changed while preparing training data")
    return dataset, tuple(selection), manifest


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Publish an immutable epoch file only after its bytes are complete."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)


def run_training(config_value: DictConfig) -> Path:
    config = PilotConfig.from_config(config_value)
    # Hydra may already have made its metadata subdirectory, but a prior run is never reused.
    config.output.mkdir(parents=True, exist_ok=True)
    if any(path.name != "hydra" for path in config.output.iterdir()):
        raise FileExistsError(f"Pilot output already contains a run: {config.output}")
    with (config.output / "run_state.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "preparing"}, stream)
    OmegaConf.save(config=OmegaConf.create(config.resolved), f=config.output / "config.yaml")
    dataset, selection, manifest = prepare_data(config)
    write_json_atomic(config.output / "data_manifest.json", manifest)
    if config.dry_run:
        write_json_atomic(config.output / "run_state.json", {"status": "dry_run_complete"})
        print(json.dumps({"status": "dry_run_complete", "windows": len(dataset), "sources": manifest["source_windows"],
                          "exclusion_records": len(dataset.excluded), "selection_clips": len(selection),
                          "manifest": str(config.output / "data_manifest.json")}), flush=True)
        return config.output
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    pair = build_ball_refiner_2d(config.model)
    pair.model.to(device)
    compiled = compile_modules({"refiner_2d": pair.model}, config.compilation)
    train = config.training
    optimizer = torch.optim.AdamW(pair.model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
    sampler = BalancedSourceSampler(dataset.source_indices, draws=train.steps_per_epoch * train.batch_size, seed=config.seed)
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=train.batch_size, sampler=sampler,
                                       collate_fn=collate_windows, num_workers=train.num_workers,
                                       generator=torch.Generator().manual_seed(config.seed))
    gaps = torch.Generator().manual_seed(config.seed + 1)
    best, steps, best_epoch = math.inf, 0, -1
    best_report: dict[str, Any] | None = None
    data_hash = dual_sha256(config.output / "data_manifest.json")
    write_json_atomic(config.output / "run_state.json", {"status": "training", "compiled": list(compiled)})
    for epoch in range(train.epochs):
        sampler.epoch = epoch
        pair.model.train()
        sums: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        for batch in loader:
            if not isinstance(batch, RefinerBatch):
                raise TypeError("Unexpected refiner DataLoader batch")
            mask = random_gap_mask(batch.target.weight.shape[0], config.window_length, lengths=train.gap_lengths,
                                   probability=train.gap_probability, generator=gaps)
            batch = RefinerBatch(mask_detector_evidence(batch.inputs, mask), batch.target).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = refiner_2d_nll(pair.run(batch.inputs), batch.target)
            loss.loss.backward()
            torch.nn.utils.clip_grad_norm_(pair.model.parameters(), train.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            sums += [float(value.detach()) for value in (
                loss.position_nll_sum, loss.presence_bce_sum, loss.position_weight, loss.presence_weight,
            )]
            steps += 1
            if steps % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": steps, "batch_nll": float(loss.loss.detach())}), flush=True)
            if steps >= train.max_steps:
                break
        report, predictions = evaluate_selection(pair, selection, config, device)
        score = float(report["selection_nll_uv"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation selection NLL")
        row = {"epoch": epoch, "step": steps, "train_joint_nll": float((sums[0] + sums[1]) / sums[3]),
               "train_position_weight": float(sums[2]), "train_presence_weight": float(sums[3]), **report}
        with (config.output / "learning_curve.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        checkpoint = config.output / f"epoch-{epoch:03d}.pt"
        _save_checkpoint(checkpoint, {
            "schema": "ball_refiner_2d_checkpoint.v1", "model_config": asdict(config.model),
            "state_dict": pair.model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": steps, "selection_nll_uv": score, "data_manifest_sha256": data_hash,
            "torch_rng": torch.get_rng_state(), "gap_rng": gaps.get_state(),
        })
        if score < best:
            best, best_epoch, best_report = score, epoch, row
            destination = config.output / f"validation-epoch-{epoch:03d}"
            destination.mkdir(exist_ok=False)
            for name, arrays in predictions.items():
                with (destination / f"{name}.npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
            write_json_atomic(destination / "metrics.json", row)
            write_json_atomic(config.output / "best.json", {
                "epoch": epoch, "checkpoint": checkpoint.name, "checkpoint_sha256": dual_sha256(checkpoint),
                "predictions": destination.name, "selection_nll_uv": score,
            })
        write_json_atomic(config.output / "run_state.json", {"status": "training", "epoch": epoch,
                                                           "step": steps, "best_epoch": best_epoch})
        print(json.dumps({"epoch": epoch, "step": steps, "selection_nll_uv": score, "best_epoch": best_epoch}), flush=True)
        if steps >= train.max_steps:
            break
    if best_report is None:
        raise RuntimeError("No validation checkpoint was selected")
    write_json_atomic(config.output / "metrics.json", best_report)
    write_json_atomic(config.output / "run_state.json", {"status": "complete", "steps": steps,
                                                       "best_epoch": best_epoch, "selection_nll_uv": best})
    print(f"Published pilot {config.output}", flush=True)
    return config.output
