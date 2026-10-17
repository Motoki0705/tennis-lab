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
    query_checkpoint_config,
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.checkpoints.posttraining import (
    SCHEMA,
    resume_posttraining,
)
from src.tasks.ball_detection.checkpoints.provenance import (
    data_identity,
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
from src.tasks.ball_detection.training.replay import ReplayFPSMixSampler, replay_recipe
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
    p.add_argument("--stop-after-epoch", type=int, help="Diagnostic stop; preserve the full cosine budget")
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
    p.add_argument("--replay-manifest", type=Path, help="Immutable manifest used to train the initial model")
    p.add_argument("--untrained-fraction", type=float, help="Window draw fraction reserved for untrained clips")
    p.add_argument("--selection-scope", choices=("common", "balanced_existing_new"), default="common")


def validate_training_arguments(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(args.epochs, args.windows_per_epoch, args.batch_size, args.log_every) < 1 or args.preview_clips < 0:
        p.error("Invalid training budget")
    if args.stop_after_epoch is not None and not 0 <= args.stop_after_epoch < args.epochs:
        p.error("Diagnostic stop must lie within the full run budget")
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
    validate_partitions(json.loads(args.manifest.read_text()), json.loads(validation_manifest.read_text()))
    data = FrozenBallWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    val = FrozenBallWindowDataset(validation_manifest, split="val", requires_pose=False, jpeg_decoder=args.jpeg_decoder)
    if dual_sha256(args.manifest) != manifest_digest or dual_sha256(validation_manifest) != validation_digest:
        raise ValueError("Manifest changed while preparing posttraining")
    if (getattr(args, "selection_scope", "common") == "balanced_existing_new") != ("evaluation_cohorts" in val.manifest):
        raise ValueError("Selection scope must agree with the fixed validation cohorts")
    runtime = CoordinateRuntime(args.precision, args.num_workers, args.pin_memory, args.prefetch_factor,
        args.cpu_threads, args.compile_mode, 8, args.jpeg_decoder, args.input_verification, args.image_prefetch)
    replay_manifest = getattr(args, "replay_manifest", None)
    untrained_fraction = getattr(args, "untrained_fraction", None)
    sampling = replay_recipe(replay_manifest, untrained_fraction)
    sampler: FPSMixSampler
    if sampling is None:
        sampler = FPSMixSampler(data, windows_per_epoch=args.windows_per_epoch, seed=args.seed)
    else:
        assert replay_manifest is not None and untrained_fraction is not None
        sampler = ReplayFPSMixSampler(data, windows_per_epoch=args.windows_per_epoch, seed=args.seed,
            previously_trained_manifest=replay_manifest, untrained_fraction=untrained_fraction)
    train_loader = runtime.loader(data, batch_size=args.batch_size, sampler=sampler, seed=args.seed)
    val_loader = runtime.loader(val, batch_size=args.batch_size, seed=args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, transfer = initialize()
    if sampling is not None and transfer.get("parent_manifest_sha256") != sampling["previously_trained_manifest_sha256"]:
        raise ValueError("Replay membership does not belong to the initial model's training manifest")
    config = model.config
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    runtime.configure(device)
    model.to(device)
    parameters = [dict(params=list(model.encoder.parameters()), role="encoder"),
                  dict(params=[p for n, p in model.named_parameters() if not n.startswith("encoder.")], role="decoder")]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    recipe = finetuning_recipe(args, parent=transfer, model_config=asdict(config), runtime=runtime,
        augmentation=augmentation, manifest_sha256=manifest_digest, validation_manifest_sha256=validation_digest,
        initial_state_sha256=initial_state_sha256, input_contract=model.mdd.input_contract())
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
        if replay_manifest is not None:
            (args.output / "replay_manifest.json").write_bytes(replay_manifest.read_bytes())
    if args.stop_after_epoch is not None and args.stop_after_epoch < first_epoch:
        raise ValueError("Diagnostic stop precedes the first resumed epoch")
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
        if isinstance(sampler, ReplayFPSMixSampler):
            (args.output / f"sampling-epoch-{epoch:03d}.json").write_text(json.dumps(sampler.last_report, indent=2))
        report = evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile="clean", output=args.output / "previews" / f"val-epoch-{epoch:03d}",
            preview_clips=args.preview_clips)
        score = selection_score(report, recipe["selection_scope"])
        if (dual_sha256(args.manifest) != recipe["manifest_sha256"]
                or dual_sha256(validation_manifest) != recipe["validation_manifest_sha256"]
                or replay_recipe(replay_manifest, untrained_fraction) != sampling):
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
        if epoch == args.stop_after_epoch and epoch + 1 < args.epochs:
            (args.output / "DIAGNOSTIC_STOP.json").write_text(json.dumps(dict(
                epoch=epoch, global_step=step, complete=False, checkpoint=path.name, sha256=dual_sha256(path))))
            return
    if args.stress_evaluation:
        saved = torch.load(args.output / f"epoch-{best_epoch:03d}.pt", map_location="cpu", weights_only=True)
        saved_config, preprocessing = query_checkpoint_config(saved)
        if saved_config != model.config or preprocessing.input_contract() != model.mdd.input_contract():
            raise ValueError("Selected checkpoint model or input contract changed")
        model.load_state_dict(saved["state_dict"], strict=True)
        stress = {profile: evaluate(model, val_loader, device, frame_steps=val.frame_steps, precision=args.precision,
            prefetch=args.image_prefetch, augmenter=augmenter, profile=profile,
            output=args.output / "previews" / profile, preview_clips=args.preview_clips)
            for profile in ("camera", "occlusion", "combined")}
        (args.output / "stress.json").write_text(json.dumps(stress, indent=2))
    (args.output / "COMPLETED.json").write_text(json.dumps(dict(stage="query_posttraining", global_step=step,
        best_epoch=best_epoch, best_error_px=best, pretrained_checkpoint=transfer["sha256"], initial_state_sha256=initial_state_sha256), indent=2))


def finetuning_recipe(
    args: argparse.Namespace, *, parent: dict[str, Any], model_config: dict[str, Any],
    runtime: CoordinateRuntime, augmentation: AugmentationConfig, manifest_sha256: str,
    validation_manifest_sha256: str, initial_state_sha256: str, input_contract: dict[str, Any],
) -> dict[str, Any]:
    """One recipe authority for execution and pre-queue restart verification."""
    sampling = replay_recipe(getattr(args, "replay_manifest", None), getattr(args, "untrained_fraction", None))
    return dict(parent=parent, model=model_config, runtime=asdict(runtime), augmentation=asdict(augmentation),
        epochs=args.epochs, freeze_epochs=args.freeze_epochs, windows_per_epoch=args.windows_per_epoch,
        batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate, encoder_lr_ratio=args.encoder_lr_ratio,
        warmup_updates=args.warmup_updates, schedule="warmup then cosine to 0.1 peak", weight_decay=.01, gradient_clip=1.,
        manifest_sha256=manifest_sha256, validation_manifest_sha256=validation_manifest_sha256,
        initial_state_sha256=initial_state_sha256, optimizer_initialization="fresh", schedule_initialization="step_zero",
        selection_scope=getattr(args, "selection_scope", "common"), selection_profile="clean", test_usage="none", input_contract=input_contract,
        image_decode=jpeg_decoder_contract(args.jpeg_decoder), stress_evaluation=args.stress_evaluation,
        data_identity=data_identity(args.manifest), initialization=dict(mode=parent["mode"], seed=args.seed),
        **({"replay_sampling": sampling} if sampling is not None else {}))


def validate_partitions(train: dict[str, Any], validation: dict[str, Any]) -> None:
    """Reject clip/source leakage before binding manifests to their frame stores.

    These are store-level group identities. Annotation learning additionally
    verifies its canonical source identities through the published snapshot.
    """
    train_records = [r for r in train["clips"] if r["clip"]["split"] == "train"]
    val_records = [r for r in validation["clips"] if r["clip"]["split"] == "val"]
    train_ids = {r["clip"]["clip_id"] for r in train_records}
    val_ids = {r["clip"]["clip_id"] for r in val_records}
    train_groups = {(r["clip"]["source"], r["clip"]["group_id"]) for r in train_records}
    val_groups = {(r["clip"]["source"], r["clip"]["group_id"]) for r in val_records}
    if train_ids & val_ids or train_groups & val_groups:
        raise ValueError("Training clips/sources overlap fixed validation")
    if "evaluation_cohorts" not in validation and not any(record["common_evaluation"] for record in val_records):
        raise ValueError("Fixed validation must contain the declared Common selection subset")
