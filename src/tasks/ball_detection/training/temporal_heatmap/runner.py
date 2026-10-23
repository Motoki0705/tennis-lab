"""Train complete shared packs, normalizing over an entire optimizer update."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal_heatmap import (
    CHECKPOINT_SCHEMA,
    RUN_SCHEMA,
    initialize_finetuning,
    json_bytes,
    log_identity,
    read_resume,
    restore_log,
    restore_rng,
    rng_state,
    save_json,
    save_torch,
)
from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.data.windows.all_frames import LABEL_KINDS
from src.tasks.ball_detection.data.windows.dataset import (
    FrozenBallPackDataset,
    FrozenBallWindowDataset,
    collate_pack_samples,
)
from src.tasks.ball_detection.evaluation.temporal_heatmap import (
    evaluate_temporal_heatmap,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import (
    HeatmapModel,
    MDDHeatmapAdapter,
    build_mdd_heatmap_detector,
    heatmap_model_payload,
    restore_heatmap_model,
)
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig, MDDHeatmapHybridConfig
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.execution import (
    CoordinateRuntime,
    coordinate_worker_init,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
    UpdateBudget,
)
from src.tasks.ball_detection.training.temporal_heatmap.packs import (
    PackSampler,
    augment_pack,
    microbatch,
    supervised_loss_sum,
)
from src.utils.checksum import dual_sha256

IDENTITY_KEYS = ("common_intervals_sha256", "target_identity_sha256", "teacher_identity_sha256",
                 "scored_frame_sha256", "training_packs_sha256")


def _data_recipe(train: FrozenBallWindowDataset, validation: FrozenBallWindowDataset) -> dict[str, Any]:
    if (not train.all_frames or not validation.all_frames or train.manifest["mode"] != "train"
            or validation.manifest["mode"] != "evaluation" or train.manifest["config"] != validation.manifest["config"]
            or train.manifest["config"]["pack_length"] != 128):
        raise ValueError("Training/evaluation must use matching all-frame manifests with 128-frame packs")
    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = {key: train.manifest["summaries"][split][key] for key in IDENTITY_KEYS}
        if any(validation.manifest["summaries"][split][key] != value for key, value in summaries[split].items()):
            raise ValueError("Train and evaluation manifests disagree on the common frame/teacher/pack sets")
    if train.manifest["ball_store"] != validation.manifest["ball_store"]:
        raise ValueError("Train and validation must reference the same immutable snapshot")
    rgb = {r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in train.manifest["clips"]}
    if rgb != {r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in validation.manifest["clips"]}:
        raise ValueError("Train and validation RGB shard identities differ")
    splits = {clip.clip_id: clip.split for clip in train.store.clips}
    return dict(snapshot=train.manifest["ball_store"], rgb_shards_sha256=rgb, summaries=summaries, splits=splits,
                split_sha256=hashlib.sha256(json_bytes(splits)).hexdigest())


def train_temporal_heatmap(
    *, architecture: str, phase: str, manifest: Path, evaluation_manifest: Path, model_config: MDDHeatmapConfig,
    output: Path, budget: UpdateBudget, runtime: CoordinateRuntime, device: torch.device,
    augmentation: AugmentationConfig | None, distance_px: float,
    initial_artifact: Path | None = None, initial_artifact_sha256: str | None = None,
    parent_update: int | None = None, resume: Path | None = None, stop_after_update: int | None = None,
) -> dict[str, Any]:
    expected_type = dict(dpt=MDDHeatmapConfig, hybrid=MDDHeatmapHybridConfig).get(architecture)
    if type(model_config) is not expected_type or phase not in {"pretrain", "finetune"}:
        raise ValueError("Declare architecture and training phase explicitly")
    if phase == "pretrain" and model_config.frames != 32:
        raise ValueError("Direct pretraining requires T=32")
    if model_config.frames not in {32, 64, 128}:
        raise ValueError("Shared-pack training supports T=32/64/128")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pack training path requires one device; distributed updates are unsupported")
    if device.type == "cuda" and runtime.precision != "bf16" or device.type == "cpu" and runtime.precision != "fp32":
        raise ValueError("Use BF16 on CUDA or FP32 on CPU explicitly")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Unsupported training device")
    if stop_after_update is not None and not 1 <= stop_after_update <= budget.updates:
        raise ValueError("Diagnostic stop must be within the original update budget")
    if not np.isfinite(distance_px) or distance_px <= 0:
        raise ValueError("Distance threshold must be positive source pixels")
    for path in (manifest, evaluation_manifest, output):
        if not path.is_absolute():
            raise ValueError("Manifest and output paths must be absolute")
    supplied_parent = (initial_artifact, initial_artifact_sha256, parent_update)
    if resume is not None and any(value is not None for value in supplied_parent):
        raise ValueError("Resume uses saved ancestry; do not provide a new initial artifact")
    if phase == "pretrain" and any(value is not None for value in supplied_parent):
        raise ValueError("Pretraining starts directly from the common random initialization rule")
    if phase == "finetune" and resume is None and any(value is None for value in supplied_parent):
        raise ValueError("Fine-tuning requires an initial heatmap artifact, its hash and the explicit parent update")
    runtime.configure(device)
    packs = FrozenBallPackDataset(manifest, split="train", jpeg_decoder=runtime.jpeg_decoder)
    dataset = packs.window_dataset
    validation = FrozenBallWindowDataset(evaluation_manifest, split="val", requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    if dataset.window_length != model_config.frames:
        raise ValueError("Training manifest T differs from model T")
    data = _data_recipe(dataset, validation)
    if not validation.manifest["summaries"]["val"]["counts"]["observed"] or not (
        validation.manifest["summaries"]["val"]["counts"]["reviewed_empty"] + validation.manifest["summaries"]["val"]["counts"]["out_of_frame"]
    ):
        raise ValueError("Validation needs both observed and confirmed-negative frames before training")
    random.seed(budget.seed)
    np.random.seed(budget.seed)
    torch.manual_seed(budget.seed)
    parent = None
    if resume is not None:
        parent = json.loads((output / "run.json").read_text())["recipe"]["initialization"]
        model = cast(HeatmapModel, build_mdd_heatmap_detector(model_config).model)
    elif phase == "finetune":
        assert initial_artifact is not None and initial_artifact_sha256 is not None and parent_update is not None
        model, parent = initialize_finetuning(initial_artifact, expected_sha256=initial_artifact_sha256,
                                             parent_update=parent_update, config=model_config)
        origin = parent["training"]
        if (origin["data"] != data or origin["image_decode"] != jpeg_decoder_contract(runtime.jpeg_decoder)):
            raise ValueError("Fine-tuning must retain the pretrained snapshot, common sets, splits and decoder")
    else:
        model = cast(HeatmapModel, build_mdd_heatmap_detector(model_config).model)
    recipe = dict(architecture=architecture, phase=phase, model=asdict(model_config), budget=budget.payload(),
                  runtime=asdict(runtime), device=str(device), augmentation=asdict(augmentation) if augmentation else None,
                  distance_threshold_source_px=distance_px, data=data, manifest_sha256=dual_sha256(manifest),
                  evaluation_manifest_sha256=dual_sha256(evaluation_manifest), image_decode=jpeg_decoder_contract(runtime.jpeg_decoder),
                  initialization=parent, initialization_rule="torch_manual_seed; direct_complete_architecture; no_pretrained_modules",
                  input_contract=model.mdd.input_contract(), sampler="shared-pack-v1_counter_seeded_uniform_replacement")
    code = source_identity()
    saved = read_resume(resume, output=output, recipe=recipe, code=code) if resume is not None else None
    if saved is not None:
        restored = restore_heatmap_model(saved["model_artifact"])
        if asdict(restored.config) != asdict(model_config) or type(restored.config) is not type(model_config):
            raise ValueError("Resume model T/config differs from the requested model")
        model = restored
    elif output.exists() and any(output.iterdir()):
        raise ValueError("New runs require an empty output directory")
    model.to(device)
    compile_coordinate_model(model, mode=runtime.compile_mode, recompile_limit=runtime.compile_recompile_limit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay,
                                 betas=(.9, .999), eps=1e-8)
    completed, position = 0, 0
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        completed, position = saved["updates_completed"], saved["sample_position"]
    output.mkdir(parents=True, exist_ok=True)
    log = output / "train.jsonl"
    if saved is not None:
        restore_log(log, saved["train_log"], completed)
    else:
        shutil.copyfile(manifest, output / "train_manifest.json")
        shutil.copyfile(evaluation_manifest, output / "evaluation_manifest.json")
        save_json(output / "run.json", dict(schema=RUN_SCHEMA, recipe=recipe, code=code))
        log.touch(exist_ok=False)
    if runtime.input_verification == "upfront":
        dataset.verify_image_files(runtime.num_workers)
    target = stop_after_update if stop_after_update is not None else budget.updates
    if target < completed:
        raise ValueError("Diagnostic stop precedes the latest completed update")
    per_update = budget.input_frames // 128
    sampler = PackSampler(len(packs), seed=budget.seed, position=position, count=(target - completed) * per_update)
    loader = DataLoader(packs, batch_size=1, sampler=sampler, collate_fn=collate_pack_samples,
                        num_workers=runtime.num_workers, pin_memory=runtime.pin_memory,
                        prefetch_factor=runtime.prefetch_factor if runtime.num_workers else None,
                        persistent_workers=runtime.num_workers > 0, worker_init_fn=coordinate_worker_init,
                        generator=torch.Generator().manual_seed(budget.seed))
    batches = iter(coordinate_batches(loader, device, prefetch=runtime.image_prefetch))
    augmenter = VideoAugmenter(augmentation, budget.seed) if augmentation else None
    adapter = MDDHeatmapAdapter(model_config)
    if saved is not None:
        restore_rng(saved["rng"], device)
    for update in range(completed + 1, target + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = budget.lr(update - 1)
        numerator, valid, input_frames, child_windows, microbatches = 0., 0, 0, 0, 0
        counts, original_counts = dict.fromkeys(LABEL_KINDS, 0), dict.fromkeys(LABEL_KINDS, 0)
        draw_ids, draw_indices, augmentation_hashes = [], [], []
        data_wait, augmentation_seconds = 0., 0.
        with coordinate_compile_scope(model):
            for _ in range(per_update):
                waiting = time.perf_counter()
                batch = next(batches)
                rgb = image_input(batch, device)
                data_wait += time.perf_counter() - waiting
                augment_started = time.perf_counter()
                for i, name in enumerate(LABEL_KINDS):
                    original_counts[name] += int((batch["label_kind"] == i).sum())
                rgb, batch, audit = augment_pack(batch, rgb, augmenter, position=position)
                augmentation_seconds += time.perf_counter() - augment_started
                for i, name in enumerate(LABEL_KINDS):
                    counts[name] += int((batch["label_kind"] == i).sum())
                draw_ids.append(batch["pack_id"][0])
                draw_indices.append(batch["pack_index"][0])
                augmentation_hashes.append(audit)
                position += 1
                child_windows += rgb.shape[0]
                input_frames += rgb.shape[0] * rgb.shape[1]
                for first in range(0, rgb.shape[0], budget.microbatch_windows):
                    last = min(first + budget.microbatch_windows, rgb.shape[0])
                    part = microbatch(batch, first, last)
                    call = adapter.build_call(MDDQueryInput(rgb[first:last], part["timestamps"].to(device)))
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=runtime.precision == "bf16"):
                        logits = model(*call.args)
                        loss_sum, teachers = supervised_loss_sum(logits, part, sigma_ratio=budget.sigma_ratio, gamma=budget.gamma)
                    loss_sum.backward()
                    numerator += float(loss_sum.detach())
                    valid += teachers
                    microbatches += 1
            if valid < 1 or input_frames != budget.input_frames:
                raise ValueError("An optimizer update requires its complete frame budget and at least one teacher")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(valid)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), budget.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        record = dict(update=update, sample_position=position, pack_ids=draw_ids, pack_indices=draw_indices,
                      input_frames=input_frames, cumulative_input_frames=update * input_frames, child_windows=child_windows,
                      microbatches=microbatches, valid_teachers=valid, teachers_by_kind=counts, original_teachers_by_kind=original_counts,
                      augmentation_sha256=augmentation_hashes, loss=numerator / valid, loss_sum=numerator,
                      learning_rate=optimizer.param_groups[0]["lr"], gradient_norm=float(gradient),
                      seconds=seconds, frames_per_second=input_frames / seconds, data_wait_seconds=data_wait,
                      augmentation_seconds=augmentation_seconds,
                      peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
        with log.open("ab") as stream:
            stream.write(json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
        evaluate = update % budget.evaluation_interval == 0 or update == budget.updates
        checkpoint = update % budget.checkpoint_interval == 0 or update == target
        artifact = None
        artifact_hash = None
        report = None
        if evaluate or checkpoint:
            artifact = heatmap_model_payload(model)
            artifact["training"] = dict(schema=RUN_SCHEMA, phase=phase, update=update, data=data,
                                        image_decode=recipe["image_decode"], runtime=asdict(runtime), seed=budget.seed, code=code,
                                        recipe_sha256=hashlib.sha256(json_bytes(recipe)).hexdigest(), parent=parent)
            artifact_path = output / f"weights-update-{update:08d}.pt"
            save_torch(artifact_path, artifact)
            artifact_hash = dual_sha256(artifact_path)
        if evaluate:
            assert artifact_hash is not None
            report = evaluate_temporal_heatmap(model, evaluation_manifest, split="val", runtime=runtime, device=device,
                                               artifact_sha256=artifact_hash, distance_px=distance_px,
                                               output=output / f"validation-update-{update:08d}.json")
        if checkpoint:
            checkpoint_path = output / f"checkpoint-update-{update:08d}.pt"
            save_torch(checkpoint_path, dict(schema=CHECKPOINT_SCHEMA, recipe=recipe, code=code,
                updates_completed=update, sample_position=position, model_artifact=artifact, artifact_sha256=artifact_hash,
                optimizer=optimizer.state_dict(), schedule=dict(updates_completed=update, **budget.payload()),
                rng=rng_state(device), validation=report, train_log=log_identity(log)))
            save_json(output / "latest.json", dict(checkpoint=checkpoint_path.name, sha256=dual_sha256(checkpoint_path), update=update))
        completed = update
    result = dict(schema=RUN_SCHEMA, phase=phase, updates_completed=completed, sample_position=position,
                  complete=completed == budget.updates)
    save_json(output / ("COMPLETED.json" if result["complete"] else "STOPPED.json"), result)
    return result
