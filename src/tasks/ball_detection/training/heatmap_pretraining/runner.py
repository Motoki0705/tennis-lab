from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.models.mdd_pretrain import (
    MDDDPTDetector,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.training.coordinate_compilation import (
    COMPILE_MODES,
    compile_coordinate_model,
    coordinate_compilation_report,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import selection_score
from src.tasks.ball_detection.training.coordinate_images import (
    coordinate_batches,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.coordinate_provenance import (
    coordinate_source_identity,
)
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

from .checkpoint import SCHEMA, save_pretraining
from .data import HeatmapWindowDataset
from .evaluation import evaluate
from .objective import heatmap_objective, image_input

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.pretrain_mdd_dpt", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("model_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
))


def source_identity() -> dict[str, Any]:
    code: dict[str, Any] = coordinate_source_identity()
    paths = [*PROJECT_ROOT.glob("src/tasks/ball_detection/models/mdd_pretrain/*.py"),
             *PROJECT_ROOT.glob("src/tasks/ball_detection/training/heatmap_pretraining/*.py"),
             PROJECT_ROOT / "src/tasks/ball_detection/scripts/pretrain_mdd_dpt.py",
             PROJECT_ROOT / "src/tasks/ball_detection/data/supervision.py",
             PROJECT_ROOT / "src/utils/models/dpt.py", PROJECT_ROOT / "src/utils/models/blocks.py",
             PROJECT_ROOT / "src/utils/models/components/ffn_layers.py",
             PROJECT_ROOT / "src/utils/data/heatmaps.py", PROJECT_ROOT / "src/tasks/base/training/losses.py"]
    code["source_sha256"].update({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return code


def learning_rate(step: int, *, peak: float, total: int, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1.))))


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model-config", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--windows-per-epoch", type=int, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--warmup-updates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cuda", "cpu"), required=True)
    p.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--jpeg-decoder", choices=("nvjpeg", "opencv"), default="nvjpeg")
    p.add_argument("--input-verification", choices=("upfront", "lazy"), default="upfront")
    p.add_argument("--image-prefetch", action="store_true")
    p.add_argument("--compile-mode", choices=COMPILE_MODES, default="default")
    p.add_argument("--selection-scope", choices=("common", "full"), default="common")
    p.add_argument("--sigma-ratio", type=float, default=.012)
    p.add_argument("--focal-gamma", type=float, default=2.)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--preview-clips", type=int, default=3)
    args = p.parse_args()
    if min(args.epochs, args.windows_per_epoch, args.batch_size, args.log_every) < 1:
        p.error("Budgets and log interval must be positive")
    if args.preview_clips < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Invalid learning rate or preview count")
    if not math.isfinite(args.sigma_ratio) or args.sigma_ratio <= 0 or not math.isfinite(args.focal_gamma) or args.focal_gamma < 0:
        p.error("Invalid heatmap objective")
    if args.warmup_updates < 0 or args.warmup_updates >= args.epochs * math.ceil(args.windows_per_epoch / args.batch_size):
        p.error("Warmup must be nonnegative and shorter than the run")
    for name in ("manifest", "model_config", "output", "resume"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            p.error(f"{name} must be an absolute path")
    if args.device == "cuda" and args.precision != "bf16":
        p.error("This pretraining recipe fixes CUDA precision to BF16")
    return args


def main() -> None:
    args = arguments()
    roots = RuntimePathRoots(project_root=args.model_config.parent, data_root=args.manifest.parent,
        artifact_root=args.manifest.parent, checkpoint_root=args.output.parent, cache_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent)
    declared = {name: getattr(args, name) for name in ("manifest", "model_config", "output", "resume")
                if getattr(args, name) is not None}
    paths = PATH_BOUNDARY.validate(declared, resolver=PathResolver(roots))
    for name in declared:
        setattr(args, name, paths.declared(name).path)
    config = MDDPretrainConfig.load(args.model_config)
    data = HeatmapWindowDataset(args.manifest, split="train", jpeg_decoder=args.jpeg_decoder)
    val = HeatmapWindowDataset(args.manifest, split="val", jpeg_decoder=args.jpeg_decoder)
    runtime = CoordinateRuntime(args.precision, args.num_workers, args.pin_memory, args.prefetch_factor,
                                args.cpu_threads, args.compile_mode, 8, args.jpeg_decoder,
                                args.input_verification, args.image_prefetch)
    sampler = FPSMixSampler(data, windows_per_epoch=args.windows_per_epoch, seed=args.seed)
    train_loader = runtime.loader(data, batch_size=args.batch_size, sampler=sampler, seed=args.seed)
    val_loader = runtime.loader(val, batch_size=args.batch_size, seed=args.seed)
    device = torch.device(args.device)
    runtime.configure(device)
    torch.manual_seed(args.seed)
    model = MDDDPTDetector(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    code = source_identity()
    recipe = dict(model=asdict(config), runtime=asdict(runtime), seed=args.seed, epochs=args.epochs,
                  windows_per_epoch=args.windows_per_epoch, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, warmup_updates=args.warmup_updates,
                  schedule="linear warmup then cosine to 0.1 peak", weight_decay=.01, gradient_clip=1.,
                  manifest_sha256=dual_sha256(args.manifest), selection_scope=args.selection_scope,
                  selection_metric="macro_mean_error_px", sigma_ratio=args.sigma_ratio, focal_gamma=args.focal_gamma,
                  image_decode=jpeg_decoder_contract(args.jpeg_decoder), input_contract=model.mdd.input_contract(),
                  supervision="observed positive; reviewed no-instance/out_of_frame negative; uncertain/multiple/reference ignored",
                  test_usage="none", stage="heatmap_pretraining", automatic_next_stage=False)
    epoch_start, step, best, best_epoch = 0, 0, float("inf"), -1
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(dict(recipe=recipe, code=code, parameters=sum(p.numel() for p in model.parameters()),
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), indent=2))
        (args.output / "data_manifest.json").write_bytes(args.manifest.read_bytes())
    else:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume.parent != args.output or saved.get("schema") != SCHEMA or saved["recipe"] != recipe:
            raise ValueError("Resume requires the same output directory, pretraining stage and exact recipe")
        if saved["code"]["source_sha256"] != code["source_sha256"]:
            raise ValueError("Resume implementation changed")
        epoch_start = saved["epoch"] + 1
        if epoch_start >= args.epochs or any(int(p.stem.split("-")[1]) >= epoch_start for p in args.output.glob("epoch-*.pt")):
            raise ValueError("Resume must use the latest completed epoch with remaining budget")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, best, best_epoch = saved["global_step"], saved["best_score"], saved["best_epoch"]
    compile_coordinate_model(model, mode=args.compile_mode)
    total_steps = args.epochs * len(train_loader)
    print(json.dumps(dict(phase="training_start", stage="heatmap_pretraining", parameters=sum(p.numel() for p in model.parameters()),
        train_clips=len(data.records), validation_clips=len(val.records), total_updates=total_steps,
        cnn_3d_layers=len(model.encoder.temporal), ffn_dim_for_transfer=config.ffn_dim)), flush=True)
    for epoch in range(epoch_start, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        begin = previous = time.perf_counter()
        loss_sum, frames, windows, input_wait, reader_wait = 0., 0, 0, 0., 0.
        for update, batch in enumerate(coordinate_batches(train_loader, device, prefetch=args.image_prefetch), 1):
            waited = time.perf_counter() - previous
            input_wait += waited
            reader_wait += batch.get("_reader_wait_seconds", waited)
            lr = learning_rate(step, peak=args.learning_rate, total=total_steps, warmup=args.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            rgb = image_input(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with coordinate_compile_scope(model):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    logits = model(rgb)
                loss, _ = heatmap_objective(logits, batch, sigma_ratio=args.sigma_ratio, gamma=args.focal_gamma)
                loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count = int(batch["heatmap_valid"].sum())
            loss_sum += float(loss.detach()) * count
            frames += count
            windows += len(batch["clip_id"])
            step += 1
            if step % args.log_every == 0 or update == len(train_loader):
                elapsed = time.perf_counter() - begin
                temporal_grad = [float(layer.conv.weight.grad.norm()) if layer.conv.weight.grad is not None else None
                                 for layer in model.encoder.temporal]
                progress = dict(epoch=epoch, global_step=step, windows=windows, train_loss=loss_sum / frames,
                    grad_norm=float(grad), temporal_gradient_norms=temporal_grad, learning_rate=lr,
                    train_seconds=elapsed, windows_per_second=windows / elapsed,
                    mean_input_ready_wait_seconds=input_wait / update, mean_reader_wait_seconds=reader_wait / update)
                if device.type == "cuda":
                    progress.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                print(json.dumps(progress), flush=True)
            previous = time.perf_counter()
        train_seconds = time.perf_counter() - begin
        report = evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
                          image_prefetch=args.image_prefetch, output=args.output / "previews" / f"epoch-{epoch:03d}",
                          preview_clips=args.preview_clips, sigma_ratio=args.sigma_ratio, gamma=args.focal_gamma)
        validation_seconds = time.perf_counter() - begin - train_seconds
        score = selection_score(report, args.selection_scope)
        if dual_sha256(args.manifest) != recipe["manifest_sha256"]:
            raise ValueError("Frozen manifest changed during training")
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        path = args.output / f"epoch-{epoch:03d}.pt"
        save_pretraining(path, model, optimizer, epoch=epoch, step=step, recipe=recipe, code=code,
                         report=report, best=best, best_epoch=best_epoch)
        if improved:
            best_record = dict(epoch=epoch, global_step=step, checkpoint=path.name, sha256=dual_sha256(path),
                               selection_error_px=score, selection_scope=args.selection_scope, **report)
            (args.output / "best.json").write_text(json.dumps(best_record, indent=2))
        row = dict(epoch=epoch, global_step=step, train_loss=loss_sum / frames, train_seconds=train_seconds,
                   validation_seconds=validation_seconds, compilation=coordinate_compilation_report(model), **report)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(dict(phase="epoch_complete", **row)), flush=True)
    (args.output / "COMPLETED.json").write_text(json.dumps(dict(stage="heatmap_pretraining", global_step=step,
        best_epoch=best_epoch, best_error_px=best, automatic_next_stage=False), indent=2))
