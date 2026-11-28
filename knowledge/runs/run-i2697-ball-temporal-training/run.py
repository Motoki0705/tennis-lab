"""Measure, budget and run the three paired training conditions via the GPU queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (
    CONDITIONS,
    LABEL_KINDS,
    MICROBATCH,
    PACK_SIZE,
    RUN,
    SEED,
    Block,
    FrozenInputs,
    RGBBlocks,
    anchor_block,
)
from src.tasks.ball_detection.checkpoints.temporal import (
    json_bytes,
    log_identity,
    restore_log,
    restore_rng,
    rng_state,
    save_json,
    save_torch,
)
from src.tasks.ball_detection.models.mdd import MDDDPTDetector
from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig
from src.tasks.ball_detection.training.runtime.execution import (
    CoordinateRuntime,
    coordinate_worker_init,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
from src.tasks.ball_detection.training.temporal.packs import (
    PackSampler,
    pack_index,
    supervised_loss_sum,
)
from src.utils.checksum import dual_sha256
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic

MODEL_CONFIG = Path(
    "src/tasks/ball_detection/configs/temporal_models/mdd_stage_dpt_heatmap.yaml"
)
STEM_WEIGHT = "encoder.stages.0.0.weight"
SIZES = ((1280, 720), (512, 288))
BUDGET_SECONDS = 6 * 3600
CUDA_FRACTION = 0.80
WORKERS = 4
PROBE_UPDATES = 12
PROBE_VAL_BATCHES = 48
PROBE_ANCHOR_BATCHES = 24
PROBE_WARMUP = 3
CHECKPOINT_INTERVAL = 250
DEVICE = torch.device("cuda:0")
SCHEMA = "ball_temporal_training_ablation.v1"


def event(**fields: Any) -> None:
    print(json_bytes(fields).decode(), end="", flush=True)


class RGBOnly(nn.Module):
    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[1:3] != (1, 3):
            raise ValueError("Expected independent RGB frames with shape B,1,3,H,W")
        return rgb.transpose(1, 2).float() / 255


def model_config(frames: int) -> MDDStageHeatmapConfig:
    return replace(MDDStageHeatmapConfig.load(MODEL_CONFIG), frames=frames)


def build_model(frames: int) -> MDDDPTDetector:
    """Draw one common 5ch state, then project its RGB stem for T=1."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = MDDDPTDetector(model_config(frames))
    if frames == 1:
        original = model.encoder.stages[0][0]
        stem = nn.Conv3d(
            3,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=True,
        )
        with torch.no_grad():
            stem.weight.copy_(original.weight[:, :3])
            stem.bias.copy_(original.bias)
        model.encoder.stages[0][0] = stem
        model.mdd = RGBOnly()
    return model


def initialization(model: MDDDPTDetector) -> dict[str, Any]:
    def digest(shared_only: bool) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            if shared_only and name == STEM_WEIGHT:
                tensor = tensor[:, :3]
            h.update(json_bytes([name, list(tensor.shape), str(tensor.dtype)]))
            h.update(tensor.contiguous().numpy().tobytes())
        return h.hexdigest()

    return dict(
        seed=SEED,
        method="shared_scratch_draw_then_slice_rgb_stem",
        shared_sha256=digest(True),
        full_sha256=digest(False),
        input_channels=model.state_dict()[STEM_WEIGHT].shape[1],
        parameters=sum(p.numel() for p in model.parameters()),
    )


def prepare_model(model: MDDDPTDetector) -> None:
    model.to(DEVICE)
    nn.utils.convert_conv3d_weight_memory_format(model.encoder, torch.channels_last_3d)
    nn.utils.convert_conv2d_weight_memory_format(model.decoder, torch.channels_last)
    nn.utils.convert_conv2d_weight_memory_format(model.head, torch.channels_last)


def budget(updates: int, frames: int) -> UpdateBudget:
    return UpdateBudget(
        updates=updates,
        input_frames=PACK_SIZE,
        microbatch_windows=MICROBATCH // frames,
        learning_rate=1e-4,
        initial_learning_rate=1e-6,
        warmup_updates=updates // 20,
        minimum_lr_ratio=0.01,
        weight_decay=0.01,
        gradient_clip=1.0,
        evaluation_interval=updates,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        seed=SEED,
        sigma_ratio=0.012,
        gamma=2.0,
        schedule_updates=updates,
        schedule_offset=0,
        freeze_updates=0,
        encoder_lr_ratio=1.0,
        optimizer_transfer="fresh",
    )


def rgb_blocks(
    inputs: FrozenInputs, blocks: list[Block], sampler: Iterable[int] | None = None
) -> Generator[dict[str, Any], None, None]:
    loader = DataLoader(
        RGBBlocks(inputs, blocks),
        batch_size=None,
        sampler=sampler,
        num_workers=WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=coordinate_worker_init,
        generator=torch.Generator().manual_seed(SEED),
    )
    return coordinate_batches(loader, DEVICE, prefetch=True)


def resize(rgb: Tensor, size: tuple[int, int]) -> Tensor:
    width, height = size
    if rgb.shape[-2:] == (height, width):
        return rgb
    # Bound the FP32 interpolation workspace independently of the decoder batch.
    return torch.cat(
        [
            F.interpolate(
                chunk.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
            for chunk in rgb.split(16)
        ]
    )


def train_views(
    rgb: Tensor, uv: Tensor, codes: Tensor, frames: int
) -> Iterator[tuple[Tensor, dict[str, Tensor]]]:
    for pixels, targets, kinds in zip(
        rgb.split(MICROBATCH),
        uv.split(MICROBATCH),
        codes.split(MICROBATCH),
        strict=True,
    ):
        yield (
            pixels.reshape(-1, frames, *rgb.shape[1:]),
            dict(
                uv=targets.reshape(-1, frames, 2),
                position_valid=(kinds == 1).reshape(-1, frames),
                heatmap_valid=(kinds != 0).reshape(-1, frames),
            ),
        )


def update(
    model: MDDDPTDetector,
    optimizer: torch.optim.Optimizer,
    rgb: Tensor,
    uv: Tensor,
    codes: Tensor,
    frames: int,
    recipe: UpdateBudget,
) -> dict[str, Any]:
    total = int(codes.count_nonzero())
    if len(rgb) != PACK_SIZE or len(uv) != PACK_SIZE or not total:
        raise ValueError("Every update needs one full frozen pack with a teacher")
    optimizer.zero_grad(set_to_none=True)
    numerators, valid, seen = [], 0, 0
    for pixels, targets in train_views(rgb, uv, codes, frames):
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            logits = model(pixels)
        loss, n = supervised_loss_sum(
            logits, targets, sigma_ratio=recipe.sigma_ratio, gamma=recipe.gamma
        )
        (loss / total).backward()
        numerators.append(loss.detach())
        valid += n
        seen += pixels.shape[0] * pixels.shape[1]
    if valid != total or seen != PACK_SIZE:
        raise ValueError("Gradients did not cover the complete paired update")
    grad = nn.utils.clip_grad_norm_(
        model.parameters(), recipe.gradient_clip, error_if_nonfinite=True
    )
    optimizer.step()
    return dict(
        loss=float(torch.stack(numerators).sum() / total),
        grad_norm=float(grad),
        input_frames=seen,
    )


def train_condition(
    inputs: FrozenInputs,
    name: str,
    size: tuple[int, int],
    updates: int,
    output: Path,
    *,
    limit: int | None = None,
    plan_sha: str | None = None,
) -> tuple[MDDDPTDetector, dict[str, Any]]:
    frames = CONDITIONS[name]
    target = updates if limit is None else limit
    started = time.perf_counter()
    model = build_model(frames)
    init = initialization(model)
    prepare_model(model)
    configuration = budget(updates, frames)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
    output.mkdir(exist_ok=True)
    recipe = dict(
        schema=SCHEMA,
        plan_sha256=plan_sha,
        name=name,
        width=size[0],
        height=size[1],
        model=dict(asdict(model.config), input_channels=init["input_channels"]),
        initialization=init,
        budget=configuration.payload(),
        runtime=environment(),
        code=code_identity(),
        augmentation=None,
    )
    # Normalize dataclass tuples to the JSON list representation for exact resume.
    recipe = json.loads(json_bytes(recipe))
    run_file, checkpoint, log = (
        output / file for file in ("recipe.json", "latest.pt", "train.jsonl")
    )
    completed = 0

    def persist(update_number: int) -> float:
        start = time.perf_counter()
        save_torch(
            checkpoint,
            dict(
                recipe=recipe,
                update=update_number,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(DEVICE),
                log=log_identity(log),
            ),
        )
        return time.perf_counter() - start

    save_seconds = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if saved["recipe"] != recipe or json.loads(run_file.read_bytes()) != recipe:
            raise ValueError("Checkpoint belongs to a different frozen recipe")
        completed = saved["update"]
        if not 0 <= completed <= target:
            raise ValueError("Checkpoint update is outside the fixed budget")
        restore_log(log, saved["log"], completed)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], DEVICE)
        del saved
    else:
        if log.exists() and log.stat().st_size:
            raise ValueError("A nonempty training log requires its matching checkpoint")
        save_json(run_file, recipe)
        log.touch()
        save_seconds.append(persist(0))

    sampler = PackSampler(
        len(inputs.train), seed=SEED, position=completed, count=target - completed
    )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    if completed < target:
        with log.open("ab") as stream:
            batches = rgb_blocks(inputs, inputs.train, sampler)
            try:
                for number in range(completed + 1, target + 1):
                    begin = time.perf_counter()
                    batch = next(batches)
                    index, block = batch["index"], batch["block"]
                    if (
                        index != pack_index(SEED, number - 1, len(inputs.train))
                        or block != inputs.train[index]
                    ):
                        raise ValueError("Training loader changed the frozen pack draw")
                    labels = inputs.labels[block.clip_index]
                    section = slice(block.start, block.stop)
                    uv, codes = (
                        torch.from_numpy(values[section])
                        for values in (labels.uv, labels.codes)
                    )
                    lr = configuration.lr(number - 1)
                    optimizer.param_groups[0]["lr"] = lr
                    result = update(
                        model,
                        optimizer,
                        resize(batch["rgb"][0], size),
                        uv,
                        codes,
                        frames,
                        configuration,
                    )
                    torch.cuda.synchronize()
                    row = dict(
                        result,
                        update=number,
                        pack=[index, block.clip_index, block.start],
                        teacher_counts=dict(
                            zip(
                                LABEL_KINDS,
                                np.bincount(codes.numpy(), minlength=4).tolist(),
                                strict=True,
                            )
                        ),
                        lr=lr,
                        seconds=time.perf_counter() - begin,
                        allocated_bytes=torch.cuda.max_memory_allocated(),
                        reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                    stream.write(json_bytes(row))
                    stream.flush()
                    if number % CHECKPOINT_INTERVAL == 0 or number == target:
                        save_seconds.append(persist(number))
                    if number <= 2 or number % 100 == 0 or number == target:
                        event(condition=name, size=size, **row)
            finally:
                batches.close()

    rows = [json.loads(line) for line in log.read_bytes().splitlines()]
    if len(rows) != target or any(row["input_frames"] != PACK_SIZE for row in rows):
        raise ValueError("Incomplete update log or incorrect frame budget")
    durations = [r["seconds"] for r in rows]
    steady = durations[PROBE_WARMUP:] or durations
    return model, dict(
        condition=name,
        updates=target,
        input_frames=target * PACK_SIZE,
        initialization=init,
        pack_draws_sha256=hashlib.sha256(
            json_bytes([[r["pack"], r["teacher_counts"]] for r in rows])
        ).hexdigest(),
        train_log_sha256=dual_sha256(log),
        optimizer_update_seconds=sum(durations),
        first_update_seconds=durations[0],
        steady_mean_update_seconds=statistics.mean(steady),
        steady_median_update_seconds=statistics.median(steady),
        wall_seconds=time.perf_counter() - started,
        max_checkpoint_seconds=max(save_seconds, default=0.0),
        peak_allocated_bytes=max(r["allocated_bytes"] for r in rows),
        peak_reserved_bytes=max(r["reserved_bytes"] for r in rows),
    )


def window_pixels(rgb: Tensor, block: Block, frames: int) -> Tensor:
    positions = torch.tensor(
        [w.start - block.start for w in block.windows], device=rgb.device
    )
    indices = (positions[:, None] + torch.arange(frames, device=rgb.device)).flatten()
    return rgb.index_select(0, indices).reshape(-1, frames, *rgb.shape[1:])


@torch.inference_mode()
def predict_blocks(
    inputs: FrozenInputs,
    blocks: list[Block],
    model: MDDDPTDetector,
    frames: int,
    size: tuple[int, int],
    path: Path,
) -> dict[str, Any]:
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    temporary = path.with_suffix(".tmp")
    seen: Counter[str] = Counter()
    timings = []
    with temporary.open("wb") as stream, closing(rgb_blocks(inputs, blocks)) as batches:
        for count in range(len(blocks)):
            begin = time.perf_counter()
            batch = next(batches)
            block: Block = batch["block"]
            if block != blocks[count]:
                raise ValueError("Evaluation loader changed the frozen window order")
            pixels = window_pixels(resize(batch["rgb"][0], size), block, frames)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                logits = model(pixels)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite ablation heatmap output")
            probability = logits.float().sigmoid()
            uv, scores = heatmaps_to_argmax(probability)
            uv = refine_peaks_log_parabolic(probability, uv).cpu().tolist()
            scores = scores.cpu().tolist()
            clip = inputs.store.clips[block.clip_index]
            labels = inputs.labels[clip.index]
            fps = str(Fraction(clip.fps))
            for j, window in enumerate(block.windows):
                for frame in range(window.owner_left, window.owner_right):
                    offset = frame - window.start
                    kind = LABEL_KINDS[labels.codes[frame]]
                    seen[kind] += 1
                    target = (
                        (labels.uv[frame] * labels.endpoint).tolist()
                        if kind == "observed"
                        else None
                    )
                    stream.write(
                        json_bytes(
                            dict(
                                clip_id=clip.clip_id,
                                frame_index=frame,
                                native_fps=fps,
                                source=clip.source,
                                kind=kind,
                                source_size=[clip.source_width, clip.source_height],
                                teacher_xy=target,
                                pred_uv=uv[j][offset],
                                score=scores[j][offset],
                            )
                        )
                    )
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - begin)
            if (count + 1) % 1000 == 0:
                event(
                    prediction=path.name,
                    batches=count + 1,
                    total=len(blocks),
                    seconds=time.perf_counter() - started,
                )
    temporary.replace(path)
    return dict(
        batches=len(blocks),
        predicted_frames=sum(seen.values()),
        support=seen,
        model_input_frames=sum(len(b.windows) * frames for b in blocks),
        decoded_frames=sum(b.stop - b.start for b in blocks),
        seconds=time.perf_counter() - started,
        first_batch_seconds=timings[0],
        steady_mean_batch_seconds=statistics.mean(timings[PROBE_WARMUP:] or timings),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        sha256=dual_sha256(path),
    )


def representative_blocks(blocks: list[Block], count: int) -> list[Block]:
    return [
        blocks[index]
        for index in np.linspace(0, len(blocks) - 1, min(count, len(blocks)), dtype=int)
    ]


def environment() -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(DEVICE)
    return dict(
        torch=str(torch.__version__),
        torchvision=torchvision.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=gpu.name,
        total_memory_bytes=gpu.total_memory,
        cuda_allocator_fraction=CUDA_FRACTION,
        allocator_config=os.environ.get("PYTORCH_ALLOC_CONF"),
        precision="bf16",
        compile="off",
        cudnn_benchmark=False,
        memory_format="channels_last_3d_and_2d",
        decoder=jpeg_decoder_contract("nvjpeg"),
        num_workers=WORKERS,
        augmentation=None,
        resize="native_or_gpu_fp32_antialiased_bilinear_round_to_uint8",
    )


def code_identity() -> dict[str, str]:
    return {
        p.name: dual_sha256(p) for p in (RUN / "data.py", RUN / "run.py", MODEL_CONFIG)
    }


def verify_shards(inputs: FrozenInputs, paths: Iterable[Path]) -> dict[str, Any]:
    current = sorted(set(paths))
    nbytes = sum(p.stat().st_size for p in current)
    start = time.perf_counter()
    event(stage="verify_shards", files=len(current), bytes=nbytes)
    inputs.verification.verify_all(WORKERS, paths=current)
    result = dict(files=len(current), bytes=nbytes, seconds=time.perf_counter() - start)
    event(stage="verified_shards", **result)
    return result


def probe_case(
    inputs: FrozenInputs,
    name: str,
    size: tuple[int, int],
    validation: list[Block],
    anchors: list[Block],
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    model, trained = train_condition(
        inputs, name, size, 3_000, output, limit=PROBE_UPDATES
    )
    frames = CONDITIONS[name]
    val = predict_blocks(
        inputs,
        representative_blocks(validation, PROBE_VAL_BATCHES),
        model,
        frames,
        size,
        output / "val-probe.jsonl",
    )
    anchor = predict_blocks(
        inputs,
        representative_blocks(anchors, PROBE_ANCHOR_BATCHES),
        model,
        frames,
        size,
        output / "anchor-probe.jsonl",
    )
    return dict(
        status="passed",
        train=trained,
        validation=val,
        anchors=anchor,
        validation_batches=len(validation),
        anchor_batches=len(anchors),
        seconds=time.perf_counter() - started,
    )


def probe(inputs: FrozenInputs, output: Path, started: float) -> None:
    output.mkdir(parents=True, exist_ok=False)
    val = {t: inputs.validation(t) for t in CONDITIONS.values()}
    anchors = {
        t: [anchor_block(a, t) for a in inputs.anchors] for t in CONDITIONS.values()
    }
    sample = [
        inputs.train[i]
        for i in PackSampler(
            len(inputs.train), seed=SEED, position=0, count=PROBE_UPDATES
        )
    ]
    for t in CONDITIONS.values():
        sample += representative_blocks(val[t], PROBE_VAL_BATCHES)
        sample += representative_blocks(anchors[t], PROBE_ANCHOR_BATCHES)
    verification = verify_shards(inputs, (inputs.paths[b.clip_index] for b in sample))
    full_bytes = sum(path.stat().st_size for path in inputs.paths.values())
    attempts: dict[str, Any] = {}
    for width, height in SIZES:
        cases: dict[str, Any] = {}
        attempts[f"{width}x{height}"] = cases
        for name, frames in CONDITIONS.items():
            out = output / f"{width}x{height}-{name}"
            try:
                cases[name] = probe_case(
                    inputs, name, (width, height), val[frames], anchors[frames], out
                )
            except torch.OutOfMemoryError as error:
                cases[name] = dict(status="cuda_oom", detail=str(error))
                event(condition=name, size=[width, height], **cases[name])
                # A common resolution is required, so the remaining T at this size are unnecessary.
                break
            finally:
                torch.cuda.empty_cache()
    save_json(
        output / "probe.json",
        dict(
            schema=SCHEMA,
            attempts=attempts,
            verification=verification,
            all_shard_bytes=full_bytes,
            inputs=inputs.audit(),
            runtime=environment(),
            code=code_identity(),
            queue_job=os.environ["TENNIS_RUN_ID"],
            seconds=time.perf_counter() - started,
        ),
    )


def estimate_plan(probe_path: Path) -> dict[str, Any]:
    evidence = json.loads(probe_path.read_bytes())
    estimates = {}
    for size in (f"{w}x{h}" for w, h in SIZES):
        cases = evidence["attempts"][size]
        if set(cases) != set(CONDITIONS) or any(
            case["status"] != "passed" for case in cases.values()
        ):
            continue
        inits = {
            case["train"]["initialization"]["shared_sha256"] for case in cases.values()
        }
        draws = {case["train"]["pack_draws_sha256"] for case in cases.values()}
        if len(inits) != 1 or len(draws) != 1:
            raise ValueError(
                "The probe did not preserve paired initial tensors and pack draws"
            )
        update_seconds = sum(
            case["train"]["steady_mean_update_seconds"] for case in cases.values()
        )
        evaluation_seconds = sum(
            case[subset]["first_batch_seconds"]
            + case[subset]["steady_mean_batch_seconds"] * (case[f"{key}_batches"] - 1)
            for case in cases.values()
            for subset, key in (("validation", "validation"), ("anchors", "anchor"))
        )
        verify_seconds = (
            evidence["verification"]["seconds"]
            * evidence["all_shard_bytes"]
            / evidence["verification"]["bytes"]
        )
        # Checkpoint time is charged once per 250 updates; first update and final save are fixed costs.
        updates_with_saves = (
            update_seconds
            + sum(case["train"]["max_checkpoint_seconds"] for case in cases.values())
            / CHECKPOINT_INTERVAL
        )
        startup = sum(
            case["train"]["first_update_seconds"]
            + 2 * case["train"]["max_checkpoint_seconds"]
            for case in cases.values()
        )
        fixed = evaluation_seconds + verify_seconds + startup + 120.0
        multiplier = 1.20
        affordable = (BUDGET_SECONDS - evidence["seconds"] - multiplier * fixed) / (
            multiplier * updates_with_saves
        )
        updates = max(0, min(6_000, math.floor(affordable / 100) * 100))
        estimates[size] = dict(
            updates=updates,
            summed_steady_update_seconds=update_seconds,
            evaluation_seconds=evaluation_seconds,
            verify_seconds=verify_seconds,
            fixed_seconds=fixed,
            per_update_seconds_with_checkpoints=updates_with_saves,
            estimated_seconds_including_probe_and_margin=evidence["seconds"]
            + multiplier * (fixed + updates_with_saves * updates),
            shared_initialization_sha256=inits.pop(),
        )
    if not estimates or max(e["updates"] for e in estimates.values()) < 100:
        raise ValueError(
            "No complete common resolution fits the measured resource envelope"
        )
    selected = next(
        (size for size, estimate in estimates.items() if estimate["updates"] >= 3_000),
        max(estimates, key=lambda size: estimates[size]["updates"]),
    )
    return dict(
        schema=SCHEMA,
        selected=selected,
        updates=estimates[selected]["updates"],
        training_budget={
            name: budget(estimates[selected]["updates"], frames).payload()
            for name, frames in CONDITIONS.items()
        },
        frames_per_update=PACK_SIZE,
        microbatch_frames=MICROBATCH,
        seed=SEED,
        probe_sha256=dual_sha256(probe_path),
        probe_job=evidence["queue_job"],
        margin_multiplier=1.20,
        estimates=estimates,
        inputs=evidence["inputs"],
        code=evidence["code"],
        runtime=evidence["runtime"],
    )


def run_training(
    inputs: FrozenInputs, output: Path, plan_path: Path, started: float, *, resume: bool
) -> None:
    plan = json.loads(plan_path.read_bytes())
    plan_sha = dual_sha256(plan_path)
    if (
        plan["schema"] != SCHEMA
        or plan["inputs"] != inputs.audit()
        or plan["code"] != code_identity()
        or plan["runtime"] != environment()
        or plan["frames_per_update"] != PACK_SIZE
        or plan["microbatch_frames"] != MICROBATCH
        or plan["seed"] != SEED
    ):
        raise ValueError(
            "Training must use the frozen probed runtime, code and input plan"
        )
    size = next(size for size in SIZES if f"{size[0]}x{size[1]}" == plan["selected"])
    updates = plan["updates"]
    if (
        not 100 <= updates <= 6_000
        or updates != plan["estimates"][plan["selected"]]["updates"]
        or plan["training_budget"]
        != {
            name: budget(updates, frames).payload()
            for name, frames in CONDITIONS.items()
        }
    ):
        raise ValueError("Training budget differs from the probe plan")
    output.mkdir(parents=True, exist_ok=resume)
    frozen = output / "plan.json"
    if frozen.exists() and dual_sha256(frozen) != plan_sha:
        raise ValueError("The on-disk training plan changed")
    save_json(frozen, plan)
    if dual_sha256(frozen) != plan_sha:
        raise ValueError("Provide the canonical plan emitted by the budget command")
    job = os.environ["TENNIS_RUN_ID"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    verification = verify_shards(inputs, inputs.paths.values())
    results = {}
    for name, frames in CONDITIONS.items():
        stage_start = time.perf_counter()
        directory = output / name
        finished = directory / "completed.json"
        if finished.exists():
            result = json.loads(finished.read_bytes())
            if (
                result["plan_sha256"] != plan_sha
                or result["train"]["updates"] != updates
                or result["train"]["input_frames"] != updates * PACK_SIZE
            ):
                raise ValueError("Completed stage does not match the planned budget")
            for path, digest in (
                ("final.pt", result["weights_sha256"]),
                ("train.jsonl", result["train"]["train_log_sha256"]),
                ("validation.jsonl", result["validation"]["sha256"]),
                ("anchors.jsonl", result["anchors"]["sha256"]),
            ):
                if dual_sha256(directory / path) != digest:
                    raise ValueError(f"Completed stage artifact changed: {name}/{path}")
        else:
            model, trained = train_condition(
                inputs, name, size, updates, directory, plan_sha=plan_sha
            )
            if (
                trained["initialization"]["shared_sha256"]
                != plan["estimates"][plan["selected"]]["shared_initialization_sha256"]
            ):
                raise ValueError(
                    "Training initialization differs from the planned shared tensors"
                )
            weights = directory / "final.pt"
            save_torch(
                weights,
                dict(
                    recipe=json.loads((directory / "recipe.json").read_bytes()),
                    updates=updates,
                    weights=model.state_dict(),
                ),
            )
            val = predict_blocks(
                inputs,
                inputs.validation(frames),
                model,
                frames,
                size,
                directory / "validation.jsonl",
            )
            anchor = predict_blocks(
                inputs,
                [anchor_block(a, frames) for a in inputs.anchors],
                model,
                frames,
                size,
                directory / "anchors.jsonl",
            )
            result = dict(
                condition=name,
                commit=commit,
                queue_job=job,
                train=trained,
                validation=val,
                anchors=anchor,
                weights_sha256=dual_sha256(weights),
                plan_sha256=plan_sha,
                wall_seconds=time.perf_counter() - stage_start,
            )
            save_json(finished, result)
            del model
            torch.cuda.empty_cache()
        results[name] = result
    if len({r["train"]["pack_draws_sha256"] for r in results.values()}) != 1:
        raise ValueError("Training conditions processed different frozen pack draws")
    save_json(
        output / "completed.json",
        dict(
            schema=SCHEMA,
            commit=commit,
            queue_job=job,
            plan_sha256=plan_sha,
            verification=verification,
            conditions=results,
            seconds=time.perf_counter() - started,
        ),
    )


def configure_gpu(output: Path) -> None:
    if os.environ.get("TENNIS_GPU_RESOURCE") != "all" or not os.environ.get(
        "TENNIS_RUN_ID"
    ):
        raise ValueError(
            "Submit GPU stages through the shared training queue with --resource all"
        )
    CoordinateRuntime(precision="bf16", jpeg_decoder="nvjpeg").configure(DEVICE)
    torch.cuda.set_per_process_memory_fraction(CUDA_FRACTION, DEVICE)
    repro = Path(os.environ["TENNIS_REPRO_DIR"])
    (repro / "output_dir.txt").write_text(str(output) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "probe", "budget", "train"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "budget":
        if args.probe is None or args.output is None:
            parser.error("budget requires --probe and --output")
        save_json(args.output, estimate_plan(args.probe))
        return
    if args.dataset_root is None:
        parser.error("audit, probe and train require --dataset-root")
    started = time.perf_counter()
    inputs = FrozenInputs(args.dataset_root.resolve())
    if args.mode == "audit":
        event(**inputs.audit())
        return
    if args.output is None or (args.mode == "train" and args.plan is None):
        parser.error("GPU stages require --output; training also requires --plan")
    output = args.output.resolve()
    configure_gpu(output)
    if args.mode == "probe":
        probe(inputs, output, started)
    else:
        run_training(inputs, output, args.plan.resolve(), started, resume=args.resume)


if __name__ == "__main__":
    main()
