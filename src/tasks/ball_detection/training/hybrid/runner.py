"""Hydra entry point and adapter API for hybrid learning on frozen snapshots."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.checkpoints.hybrid import (
    HYBRID_NEW_MODULES,
    QUERY_TRANSFER_MODULES,
)
from src.tasks.ball_detection.checkpoints.hybrid_training import (
    initialize_hybrid,
    restore_hybrid_model,
    resume_hybrid,
    save_hybrid,
    write_hybrid_best,
)
from src.tasks.ball_detection.checkpoints.provenance import (
    data_identity,
    source_identity,
)
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.hybrid import (
    AbsencePolicy,
    build_hybrid_manifest,
)
from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.ball_detection.evaluation.hybrid import evaluate_hybrid
from src.tasks.ball_detection.model_io.mdd_coordinates import validate_rgb_timestamps
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.tasks.ball_detection.training.hybrid.configuration import resolve_recipe
from src.tasks.ball_detection.training.hybrid.objective import hybrid_objective
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.diagnostics import (
    batch_identity,
    record_failure,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.sampling import HybridWindowSampler
from src.tasks.ball_detection.training.schedules import learning_rate
from src.utils.checksum import dual_sha256
from src.utils.hydra import hydra_main


def parameter_role(name: str) -> str:
    root = name.split(".")[0]
    if root == "encoder":
        return "encoder"
    if root in QUERY_TRANSFER_MODULES:
        return "query"
    if root in HYBRID_NEW_MODULES:
        return "new"
    raise ValueError(f"Hybrid parameter has no optimizer/freeze role: {name}")


def prepare_run_directory(output: Path) -> None:
    """Allow Hydra's log directory, but never reuse existing training artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    if any(child.name != "hydra" or not child.is_dir() for child in output.iterdir()):
        raise FileExistsError(f"Hybrid run already contains artifacts: {output}")


def freeze_for_epoch(model: DeepMDDHybridDetector, *, epoch: int, freeze: dict[str, int]) -> None:
    model.train()
    model.freeze_encoder(epoch < freeze["encoder_epochs"])
    query_frozen = epoch < freeze["query_epochs"]
    for name, parameter in model.named_parameters():
        if parameter_role(name) == "query":
            parameter.requires_grad_(not query_frozen)
    for name in QUERY_TRANSFER_MODULES:
        if name not in {"encoder", "ball_query"}:
            getattr(model, name).train(not query_frozen)


def train_hybrid(cfg: DictConfig) -> dict[str, Any]:
    """Integration point for a continual-learning adapter; returns the run receipt.

    `initialization.mode=continual` starts a new optimizer on a new cumulative
    snapshot. `resume` restores the exact epoch state in the original run.
    """
    config = resolve_recipe(cfg)
    output, manifest = Path(config["output"]), Path(config["manifest"])
    resume = Path(config["resume"]) if config["resume"] is not None else None
    runtime = CoordinateRuntime(**config["runtime"])
    device = torch.device(config["device"])
    runtime.configure(device)
    torch.manual_seed(config["seed"])
    identity = data_identity(manifest)
    model_config = MDDHybridConfig(**config["model"])
    decoder = jpeg_decoder_contract(runtime.jpeg_decoder)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        model, previous = restore_hybrid_model(saved)
        if model.config != model_config:
            raise ValueError("Resume model configuration changed")
        parent = previous["parent"]
    else:
        initial = config["initialization"]
        model, parent = initialize_hybrid(Path(initial["checkpoint"]), model_config, mode=initial["mode"],
                                          data=identity, image_decode=decoder)
    policy = AbsencePolicy.from_mapping(config["absence"])
    effective_manifest = output / "data_manifest.json"
    if resume:
        manifest_text = effective_manifest.read_text()
        frozen_data = json.loads(manifest_text)
        if (frozen_data["base_manifest_sha256"] != identity["manifest_sha256"]
                or frozen_data["ball_store"] != identity["snapshot"]
                or AbsencePolicy.from_mapping(frozen_data["absence"]) != policy):
            raise ValueError("Resume frozen hybrid windows/data/absence policy changed")
    else:
        data_manifest = build_hybrid_manifest(manifest, policy)
        manifest_text = json.dumps(data_manifest, sort_keys=True, indent=2) + "\n"
    manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
    recipe = {k: v for k, v in config.items() if k not in {"output", "manifest", "resume", "stop_after_epoch"}}
    recipe.update(model=asdict(model_config), parent=parent, data_identity=identity,
                  manifest_sha256=identity["manifest_sha256"], hybrid_manifest_sha256=manifest_hash,
                  input_contract=model.mdd.input_contract(), image_decode=decoder,
                  weight_decay=.01, gradient_clip=1., schedule="linear warmup then cosine to 0.1 peak",
                  augmentation="none; no synthetic absent labels", selection_metric="macro_mean_error_px",
                  selection_population="all observed positions, independent of presence decisions", test_usage="none")
    code = source_identity()
    model.to(device)
    groups = [dict(params=[p for name, p in model.named_parameters() if parameter_role(name) == role], role=role)
              for role in ("encoder", "query", "new")]
    optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], weight_decay=.01)
    first_epoch, step, best, best_epoch = 0, 0, float("inf"), -1
    if resume:
        first_epoch, step, best, best_epoch = resume_hybrid(resume, output, model, optimizer,
            recipe=recipe, code=code, device=device)
    else:
        prepare_run_directory(output)
        with (output / "config.json").open("x") as stream:
            json.dump(dict(recipe=recipe, code=code), stream, indent=2)
        effective_manifest.write_text(manifest_text)
        (output / "transfer.json").write_text(json.dumps(parent, indent=2))
    stop = config["stop_after_epoch"]
    if stop is not None and stop < first_epoch:
        raise ValueError("Diagnostic stop precedes the first resumed epoch")
    train_data = FrozenBallWindowDataset(effective_manifest, split="train", requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    val_data = FrozenBallWindowDataset(effective_manifest, split="val", requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    sampler = HybridWindowSampler(train_data, windows_per_epoch=config["windows_per_epoch"], seed=config["seed"])
    train_loader = runtime.loader(train_data, batch_size=config["batch_size"], sampler=sampler, seed=config["seed"])
    val_loader = runtime.loader(val_data, batch_size=config["batch_size"], seed=config["seed"])
    compile_coordinate_model(model, mode=runtime.compile_mode, recompile_limit=runtime.compile_recompile_limit)
    total_updates = config["epochs"] * len(train_loader)
    context: dict[str, Any] = {}
    with record_failure(output, context):
        for epoch in range(first_epoch, config["epochs"]):
            sampler.set_epoch(epoch)
            freeze_for_epoch(model, epoch=epoch, freeze=config["freeze"])
            sums, supervised, windows = dict(total=0., heatmap=0., presence=0.), 0, 0
            begin = time.perf_counter()
            for batch in coordinate_batches(train_loader, device, prefetch=runtime.image_prefetch):
                context.update(epoch=epoch, completed_updates=step, attempted_update=step + 1,
                               batch=batch_identity(batch), surface_phase="input")
                rgb = image_input(batch, device)
                timestamps = batch["timestamps"].to(device)
                validate_rgb_timestamps(model.config, rgb, timestamps)
                lr = learning_rate(step, peak=config["learning_rate"], total=total_updates, warmup=config["warmup_updates"])
                for group in optimizer.param_groups:
                    role = group["role"]
                    frozen = role != "new" and epoch < config["freeze"][f"{role}_epochs"]
                    group["lr"] = 0. if frozen else lr * (1. if role == "new" else config[f"{role}_lr_ratio"])
                optimizer.zero_grad(set_to_none=True)
                with coordinate_compile_scope(model):
                    context["surface_phase"] = "forward"
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=runtime.precision == "bf16"):
                        prediction = model(rgb, timestamps)
                    loss, losses = hybrid_objective(prediction, batch, loss_config=config["loss"])
                    context["surface_phase"] = "backward"
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                context["surface_phase"] = "optimizer"
                optimizer.step()
                count = int(batch["heatmap_valid"].sum())
                supervised += count
                windows += len(batch["clip_id"])
                sums["total"] += float(loss.detach()) * count
                for name in ("heatmap", "presence"):
                    sums[name] += float(losses[name].detach()) * count
                step += 1
            train_seconds = time.perf_counter() - begin
            progress = dict(epoch=epoch, global_step=step, windows=windows, supervised_frames=supervised,
                            train_loss=sums["total"] / supervised, heatmap_loss=sums["heatmap"] / supervised,
                            presence_loss=sums["presence"] / supervised, train_seconds=train_seconds,
                            windows_per_second=windows / train_seconds,
                            learning_rates={g["role"]: g["lr"] for g in optimizer.param_groups},
                            window_quotas={f"{fps}/{source}": n for (fps, source), n in sampler.quotas(epoch).items()})
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
            context.update(surface_phase="validation", batch=None, attempted_update=None)
            report = evaluate_hybrid(model, val_loader, device, frame_steps=val_data.frame_steps,
                presence_threshold=config["presence_threshold"], loss_config=config["loss"],
                precision=runtime.precision, image_prefetch=runtime.image_prefetch)
            score = selection_score(report, config["selection_scope"])
            if data_identity(manifest) != identity or dual_sha256(effective_manifest) != manifest_hash:
                raise ValueError("Frozen data changed during hybrid training")
            improved = score < best
            if improved:
                best, best_epoch = score, epoch
            checkpoint = output / f"epoch-{epoch:03d}.pt"
            context["surface_phase"] = "checkpoint"
            save_hybrid(checkpoint, model, optimizer, recipe=recipe, code=code, epoch=epoch, step=step,
                        report=report, best=best, best_epoch=best_epoch)
            if improved:
                write_hybrid_best(output, torch.load(checkpoint, map_location="cpu", weights_only=True))
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(epoch=epoch, global_step=step, **report), allow_nan=False) + "\n")
            print(json.dumps(dict(phase="epoch_complete", **progress, validation=report)), flush=True)
            if epoch == stop and epoch + 1 < config["epochs"]:
                receipt = dict(stage="hybrid", complete=False, epoch=epoch, global_step=step,
                               checkpoint=checkpoint.name, sha256=dual_sha256(checkpoint), total_updates=total_updates)
                (output / "DIAGNOSTIC_STOP.json").write_text(json.dumps(receipt, indent=2))
                return receipt
    receipt = dict(stage="hybrid", complete=True, global_step=step, best_epoch=best_epoch, best_error_px=best,
                   initialization=config["initialization"], data_identity=identity)
    (output / "COMPLETED.json").write_text(json.dumps(receipt, indent=2))
    return receipt


@hydra_main(config_path="../../configs", config_name="train_mdd_hybrid", version_base="1.3", validation_boundary="ball.hybrid")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
    train_hybrid(cfg)
