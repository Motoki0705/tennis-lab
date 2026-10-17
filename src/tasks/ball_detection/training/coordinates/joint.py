from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.coordinates import (
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.checkpoints.posttraining import (
    SCHEMA,
    resume_posttraining,
)
from src.tasks.ball_detection.checkpoints.provenance import (
    source_identity as post_source_identity,
)
from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.ball_detection.evaluation.stress import evaluate
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.tasks.ball_detection.training.coordinates.loop import train_epoch
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import (
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.sampling import FPSMixSampler
from src.tasks.ball_detection.training.schedules import FineTuneSchedule
from src.utils.checksum import dual_sha256
from src.utils.models.state import state_dict_sha256


def add_training_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--windows-per-epoch", type=int, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--encoder-lr-ratio", type=float, default=.1)
    p.add_argument("--warmup-updates", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--jpeg-decoder", choices=("opencv", "nvjpeg"), required=True)
    p.add_argument("--input-verification", choices=("upfront", "lazy"), required=True)
    p.add_argument("--image-prefetch", action="store_true")
    p.add_argument("--compile-mode", choices=("off", "default"), required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--preview-clips", type=int, default=3)
    p.add_argument("--stress-evaluation", action="store_true", help="Fixed val stress profiles, after selection of clean best")


def validate_training_arguments(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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


def run_finetuning(args: argparse.Namespace, *, validation_manifest: Path,
                   initialize: Callable[[], tuple[DeepMDDQueryDetector, dict[str, Any]]]) -> None:
    if not args.resume and args.output.exists():
        raise FileExistsError("New fine-tuning run requires a new output directory")
    manifest_digest, validation_digest = dual_sha256(args.manifest), dual_sha256(validation_manifest)
    augmentation = AugmentationConfig.load(args.augmentation_config)
    data = FrozenBallWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    val = FrozenBallWindowDataset(validation_manifest, split="val", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    if dual_sha256(args.manifest) != manifest_digest or dual_sha256(validation_manifest) != validation_digest:
        raise ValueError("Manifest changed while preparing posttraining")
    validate_partitions(data, val)
    runtime = CoordinateRuntime(args.precision, args.num_workers, args.pin_memory, args.prefetch_factor,
        args.cpu_threads, args.compile_mode, 8, args.jpeg_decoder, args.input_verification, args.image_prefetch)
    sampler = FPSMixSampler(data, windows_per_epoch=args.windows_per_epoch, seed=args.seed)
    train_loader = runtime.loader(data, batch_size=args.batch_size, sampler=sampler, seed=args.seed)
    val_loader = runtime.loader(val, batch_size=args.batch_size, seed=args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, transfer = initialize()
    config = model.config
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    runtime.configure(device)
    model.to(device)
    parameters = [dict(params=list(model.encoder.parameters()), role="encoder"),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith("encoder.")], role="decoder")]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    recipe = dict(parent=transfer, model=asdict(config), runtime=asdict(runtime), augmentation=asdict(augmentation),
        epochs=args.epochs, freeze_epochs=args.freeze_epochs, windows_per_epoch=args.windows_per_epoch,
        batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate, encoder_lr_ratio=args.encoder_lr_ratio,
        warmup_updates=args.warmup_updates, schedule="warmup then cosine to 0.1 peak", weight_decay=.01, gradient_clip=1.,
        manifest_sha256=manifest_digest, validation_manifest_sha256=validation_digest,
        initial_state_sha256=initial_state_sha256, optimizer_initialization="fresh", schedule_initialization="step_zero", selection_scope="common", selection_profile="clean", test_usage="none",
        input_contract=model.mdd.input_contract(), image_decode=jpeg_decoder_contract(args.jpeg_decoder),
        stress_evaluation=args.stress_evaluation)
    code = post_source_identity()
    first_epoch, step, best, best_epoch = 0, 0, float("inf"), -1
    if args.resume:
        first_epoch, step, best, best_epoch = resume_posttraining(args.resume, args.output, model, optimizer,
                                                               recipe=recipe, code=code, device=device)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(dict(recipe=recipe, code=code), indent=2))
        (args.output / "data_manifest.json").write_bytes(args.manifest.read_bytes())
        (args.output / "validation_manifest.json").write_bytes(validation_manifest.read_bytes())
    augmenter = VideoAugmenter(augmentation, args.seed)
    compile_coordinate_model(model, mode=args.compile_mode)
    schedule = FineTuneSchedule(args.learning_rate, args.epochs * len(train_loader),
                                args.warmup_updates, args.freeze_epochs, args.encoder_lr_ratio)
    for epoch in range(first_epoch, args.epochs):
        sampler.set_epoch(epoch)
        result = train_epoch(model, optimizer, train_loader, device, runtime,
            epoch=epoch, global_step=step, window_budget=args.windows_per_epoch,
            output=args.output, log_every=args.log_every, augmenter=augmenter,
            schedule=schedule, preview=bool(args.preview_clips))
        step = result.global_step
        report = evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile="clean", output=args.output / "previews" / f"val-epoch-{epoch:03d}",
            preview_clips=args.preview_clips)
        score = selection_score(report, "common")
        if (dual_sha256(args.manifest) != recipe["manifest_sha256"]
                or dual_sha256(validation_manifest) != recipe["validation_manifest_sha256"]):
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
            stream.write(json.dumps(dict(epoch=epoch, global_step=step, train_loss=result.mean_loss, **report)) + "\n")
    if args.stress_evaluation:
        saved = torch.load(args.output / f"epoch-{best_epoch:03d}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        stress = {profile: evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile=profile,
            output=args.output / "previews" / profile, preview_clips=args.preview_clips)
            for profile in ("camera", "occlusion", "combined")}
        (args.output / "stress.json").write_text(json.dumps(stress, indent=2))
    (args.output / "COMPLETED.json").write_text(json.dumps(dict(stage="query_posttraining", global_step=step,
        best_epoch=best_epoch, best_error_px=best, pretrained_checkpoint=transfer["sha256"], initial_state_sha256=initial_state_sha256), indent=2))


def validate_partitions(train: FrozenBallWindowDataset, validation: FrozenBallWindowDataset) -> None:
    """Reject known clip/source leakage before initializing a model or using CUDA.

    These are store-level group identities. Annotation learning additionally
    verifies its canonical source identities through the published snapshot.
    """
    train_ids = {r["clip"]["clip_id"] for r in train.records}
    val_ids = {r["clip"]["clip_id"] for r in validation.records}
    train_groups = {(r["clip"]["source"], r["clip"]["group_id"]) for r in train.records}
    val_groups = {(r["clip"]["source"], r["clip"]["group_id"]) for r in validation.records}
    if train_ids & val_ids or train_groups & val_groups:
        raise ValueError("Training clips/sources overlap fixed validation")
    if not any(record["common_evaluation"] for record in validation.records):
        raise ValueError("Fixed validation must contain the declared Common selection subset")
