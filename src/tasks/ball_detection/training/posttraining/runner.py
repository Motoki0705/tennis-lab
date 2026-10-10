from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset
from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.models.mdd_pretrain import DeepMDDQueryDetector
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.training.coordinate_checkpoint import (
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.training.coordinate_compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import selection_score
from src.tasks.ball_detection.training.coordinate_images import (
    coordinate_batches,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
    transfer_encoder,
)
from src.tasks.ball_detection.training.heatmap_pretraining.objective import image_input
from src.tasks.ball_detection.training.heatmap_pretraining.runner import (
    learning_rate,
    source_identity,
)
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

from .augmentation import AugmentationConfig, VideoAugmenter
from .checkpoint import SCHEMA, completed_pretraining
from .evaluation import evaluate, save_query_preview

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.posttrain_mdd_query", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("pretraining_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
))


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen CNN warmup then joint fine-tuning; augmentation only in train")
    for key in ("manifest", "pretraining-run", "augmentation-config", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--windows-per-epoch", type=int, default=6000)
    p.add_argument("--learning-rate", type=float, default=1.e-4)
    p.add_argument("--encoder-lr-ratio", type=float, default=.1)
    p.add_argument("--warmup-updates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--jpeg-decoder", choices=("opencv", "nvjpeg"), default="nvjpeg")
    p.add_argument("--image-prefetch", action="store_true")
    p.add_argument("--compile-mode", choices=("off", "default"), default="default")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--preview-clips", type=int, default=3)
    p.add_argument("--stress-evaluation", action="store_true", help="Fixed val stress profiles, after selection of clean best")
    args = p.parse_args()
    if min(args.epochs, args.windows_per_epoch, args.batch_size, args.log_every) < 1 or args.preview_clips < 0:
        p.error("Invalid training budget")
    if not 1 <= args.freeze_epochs < args.epochs:
        p.error("Require at least one frozen epoch followed by joint fine-tuning")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 < args.encoder_lr_ratio <= 1:
        p.error("Invalid learning rates")
    if not 0 <= args.warmup_updates < args.epochs * math.ceil(args.windows_per_epoch / args.batch_size):
        p.error("Invalid warmup budget")
    if args.device == "cuda" and args.precision != "bf16":
        p.error("CUDA training precision is fixed to BF16")
    for name in ("manifest", "pretraining_run", "augmentation_config", "output", "resume"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            p.error(f"{name} must be absolute")
    return args


def post_source_identity() -> dict[str, Any]:
    code: dict[str, Any] = source_identity()
    paths = [*PROJECT_ROOT.glob("src/tasks/ball_detection/training/posttraining/*.py"),
             PROJECT_ROOT / "src/tasks/ball_detection/scripts/posttrain_mdd_query.py"]
    code["source_sha256"].update({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return code


def main() -> None:
    args = arguments()
    roots = RuntimePathRoots(project_root=args.augmentation_config.parent, data_root=args.manifest.parent,
        artifact_root=Path(os.path.commonpath((args.manifest.parent, args.pretraining_run.parent))),
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in ("manifest", "pretraining_run", "augmentation_config", "output", "resume")
                                   if getattr(args, k) is not None}, resolver=PathResolver(roots))
    for key in ("manifest", "pretraining_run", "augmentation_config", "output", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, paths.declared(key).path)
    parent, pretrained = completed_pretraining(args.pretraining_run, args.manifest)
    config = pretraining_config(pretrained)
    if pretrained["recipe"]["image_decode"] != jpeg_decoder_contract(args.jpeg_decoder):
        raise ValueError("Posttraining JPEG decoder must match pretraining exactly")
    augmentation = AugmentationConfig.load(args.augmentation_config)
    data = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    val = CoordinateWindowDataset(args.manifest, split="val", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    runtime = CoordinateRuntime(args.precision, args.num_workers, args.pin_memory, args.prefetch_factor,
        args.cpu_threads, args.compile_mode, 8, args.jpeg_decoder, "upfront", args.image_prefetch)
    sampler = FPSMixSampler(data, windows_per_epoch=args.windows_per_epoch, seed=args.seed)
    train_loader = runtime.loader(data, batch_size=args.batch_size, sampler=sampler, seed=args.seed)
    val_loader = runtime.loader(val, batch_size=args.batch_size, seed=args.seed)
    device = torch.device(args.device)
    runtime.configure(device)
    torch.manual_seed(args.seed)
    model = DeepMDDQueryDetector(config)
    model.mdd = RGBToMDD.from_contract(pretrained["recipe"]["input_contract"])
    transfer = transfer_encoder(parent, model)
    model.to(device)
    parameters = [dict(params=list(model.encoder.parameters()), role="encoder"),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith("encoder.")], role="decoder")]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    recipe = dict(parent=transfer, model=asdict(config), runtime=asdict(runtime), augmentation=asdict(augmentation),
        epochs=args.epochs, freeze_epochs=args.freeze_epochs, windows_per_epoch=args.windows_per_epoch,
        batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate, encoder_lr_ratio=args.encoder_lr_ratio,
        warmup_updates=args.warmup_updates, schedule="warmup then cosine to 0.1 peak", weight_decay=.01, gradient_clip=1.,
        manifest_sha256=dual_sha256(args.manifest), selection_scope="common", selection_profile="clean", test_usage="none",
        input_contract=model.mdd.input_contract(), image_decode=jpeg_decoder_contract(args.jpeg_decoder),
        stress_evaluation=args.stress_evaluation)
    code = post_source_identity()
    first_epoch, step, best, best_epoch = 0, 0, float("inf"), -1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume.parent != args.output or saved.get("schema") != SCHEMA or saved["recipe"] != recipe:
            raise ValueError("Resume requires identical posttraining recipe and output")
        if saved["code"]["source_sha256"] != code["source_sha256"]:
            raise ValueError("Resume source code changed")
        first_epoch = saved["epoch"] + 1
        if first_epoch >= args.epochs or any(int(p.stem.split("-")[1]) >= first_epoch for p in args.output.glob("epoch-*.pt")):
            raise ValueError("Resume requires latest completed epoch and remaining budget")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, best, best_epoch = saved["global_step"], saved["best_score"], saved["best_epoch"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(dict(recipe=recipe, code=code), indent=2))
        (args.output / "data_manifest.json").write_bytes(args.manifest.read_bytes())
    augmenter = VideoAugmenter(augmentation, args.seed)
    compile_coordinate_model(model, mode=args.compile_mode)
    for epoch in range(first_epoch, args.epochs):
        model.freeze_encoder(epoch < args.freeze_epochs)
        model.train()
        sampler.set_epoch(epoch)
        begin = time.perf_counter()
        loss_sum, observed, windows = 0., 0, 0
        for batch in coordinate_batches(train_loader, device, prefetch=args.image_prefetch):
            lr = learning_rate(step, peak=args.learning_rate, total=args.epochs * len(train_loader), warmup=args.warmup_updates)
            optimizer.param_groups[0]["lr"] = 0. if model.encoder_frozen else lr * args.encoder_lr_ratio
            optimizer.param_groups[1]["lr"] = lr
            rgb, transformed, audit = augmenter(image_input(batch, device), batch, epoch=epoch)
            optimizer.zero_grad(set_to_none=True)
            with coordinate_compile_scope(model):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    prediction = model(rgb, batch["timestamps"].to(device, non_blocking=True))
                    loss = coordinate_loss(prediction, transformed["uv"].to(device), transformed["position_valid"].to(device))
                loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count = int(transformed["position_valid"].sum())
            loss_sum += float(loss.detach()) * count
            observed += count
            windows += len(batch["clip_id"])
            step += 1
            if windows == len(batch["clip_id"]) and args.preview_clips:
                cpu_batch = dict(transformed, uv=transformed["uv"].detach().cpu(), position_valid=transformed["position_valid"].cpu())
                save_query_preview(args.output / "previews" / f"train-epoch-{epoch:03d}.gif", rgb,
                                   prediction.detach().cpu(), cpu_batch, audit.get("artificially_occluded"))
            if step % args.log_every == 0 or windows == args.windows_per_epoch:
                record = dict(epoch=epoch, global_step=step, windows=windows, train_loss=loss_sum / observed,
                    encoder_frozen=model.encoder_frozen, learning_rate=lr, encoder_learning_rate=optimizer.param_groups[0]["lr"],
                    grad_norm=float(grad), train_seconds=time.perf_counter() - begin,
                    augmentation={key: value.detach().cpu().tolist() for key, value in audit.items()})
                if device.type == "cuda":
                    record.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps({key: value for key, value in record.items() if key != "augmentation"}), flush=True)
        report = evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile="clean", output=args.output / "previews" / f"val-epoch-{epoch:03d}",
            preview_clips=args.preview_clips)
        score = selection_score(report, "common")
        if dual_sha256(args.manifest) != recipe["manifest_sha256"]:
            raise ValueError("Manifest changed during posttraining")
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        path = args.output / f"epoch-{epoch:03d}.pt"
        save_coordinate_checkpoint(dict(schema=SCHEMA, stage="query_posttraining", model_config=asdict(config),
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, global_step=step,
            best_epoch=best_epoch, best_score=best, validation=report, recipe=recipe, code=code,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None), path)
        if improved:
            (args.output / "best.json").write_text(json.dumps(dict(checkpoint=path.name, sha256=dual_sha256(path),
                epoch=epoch, global_step=step, selection_error_px=best, **report), indent=2))
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch, global_step=step, train_loss=loss_sum / observed, **report)) + "\n")
    if args.stress_evaluation:
        saved = torch.load(args.output / f"epoch-{best_epoch:03d}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        stress = {profile: evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile=profile,
            output=args.output / "previews" / profile, preview_clips=args.preview_clips)
            for profile in ("camera", "occlusion", "combined")}
        (args.output / "stress.json").write_text(json.dumps(stress, indent=2))
    (args.output / "COMPLETED.json").write_text(json.dumps(dict(stage="query_posttraining", global_step=step,
        best_epoch=best_epoch, best_error_px=best, pretrained_checkpoint=transfer["sha256"]), indent=2))
