"""Explicit four-stage experiment; T128 was excluded by the user on 2026-10-25.

Run from the recorded worktree, with the shared workspace root and mounted-Drive
rclone configuration supplied explicitly. No pretrained checkpoint is reused.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal import save_json
from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    decode_label_runs,
    training_packs,
)
from src.tasks.ball_detection.evaluation.temporal_heatmap import (
    evaluate_temporal_heatmap,
)
from src.tasks.ball_detection.model_io.mdd_temporal import restore_temporal_model
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig, MDDTemporalQueryConfig
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore
from src.utils.checksum import dual_sha256


def budget(*, architecture="dpt", frames=32, common=False, microbatch=1):
    values = dict(
        updates=10000 if common else 6000,
        input_frames=128,
        microbatch_windows=microbatch,
        learning_rate=1e-4,
        warmup_updates=500 if common else 0,
        minimum_lr_ratio=0.01,
        initial_learning_rate=1e-6,
        weight_decay=0.01,
        gradient_clip=1.0,
        evaluation_interval=10000 if common else 6000,
        checkpoint_interval=500,
        seed=42,
        sigma_ratio=0.012,
        gamma=2.0,
        schedule_updates=28000 if common else 6000 if architecture == "dpt" else 12000,
        schedule_offset=6000 if architecture == "query" and frames == 64 else 0,
        freeze_updates=0 if common else 600,
        encoder_lr_ratio=1.0 if common else 0.1,
        optimizer_transfer="fresh" if common else "inherit",
    )
    if not common:
        common_budget = budget(common=True)
        values.update(
            learning_rate=common_budget.lr(9999),
            initial_learning_rate=common_budget.lr(9999),
            minimum_lr_ratio=1e-6 / common_budget.lr(9999),
        )
    return UpdateBudget(**values)


def runtime():
    return CoordinateRuntime(
        precision="bf16",
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        cpu_threads=2,
        compile_mode="default",
        compile_recompile_limit=8,
        jpeg_decoder="nvjpeg",
        input_verification="upfront",
        image_prefetch=True,
    )


def model_config(architecture, frames):
    cls = MDDTemporalQueryConfig if architecture == "query" else MDDHeatmapConfig
    filename = (
        "mdd_temporal_query.yaml" if architecture == "query" else "mdd_dpt_heatmap.yaml"
    )
    # No frame splitting or activation checkpointing is adopted from the old pilot.
    return replace(
        cls.load(Path("src/tasks/ball_detection/configs/model") / filename),
        frames=frames,
        activation_checkpointing=False,
        frame_chunk_size=None,
    )


def persist_callback(campaign_root, output, config, remote_root):
    store = RcloneSnapshotStore(
        output,
        remote="mounted_drive",
        remote_root=remote_root + "/" + output.name,
        config_path=config,
        timeout_seconds=3600,
    )
    index = RcloneArtifactStore(
        campaign_root,
        remote="mounted_drive",
        remote_root=remote_root,
        config_path=config,
    )

    def persist(directory, stage):
        names = [
            "run.json",
            "train_manifest.json",
            "evaluation_manifest.json",
            "train.jsonl",
        ]
        update = 0
        if (directory / "latest.json").exists():
            latest = json.loads((directory / "latest.json").read_text())
            update = latest["update"]
            names += [
                "latest.json",
                latest["checkpoint"],
                f"weights-update-{update:08d}.pt",
            ]
            report = f"validation-update-{update:08d}.json"
            if (directory / report).exists():
                names += [report, report.replace(".json", ".frames.jsonl")]
        for name in ("COMPLETED.json", "STOPPED.json", "FAILED.json"):
            if (directory / name).exists():
                names.append(name)
        generation = f"{stage}-{update:08d}-{uuid.uuid4().hex[:8]}"
        receipt = store.publish(generation, include=names)
        path = campaign_root / "persistence" / output.name / (generation + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, receipt)
        index.publish_file(path)
        print(
            json.dumps(
                dict(
                    event="persisted",
                    run=output.name,
                    stage=stage,
                    update=update,
                    receipt=receipt,
                )
            ),
            flush=True,
        )

    return persist


def prepare(args):
    receipt = (
        args.workspace_root
        / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025/progress.json"
    )
    if not json.loads(receipt.read_text())["complete"]:
        raise RuntimeError(
            "Download must finish and verify before manifest preparation"
        )
    store = (
        args.workspace_root
        / "data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
    )
    output = (
        args.workspace_root
        / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
    )
    started = time.perf_counter()
    plan = prepare_all_frame_training(
        store,
        output,
        configs=[AllFrameWindowConfig(t) for t in (32, 64)],
        rgb_manifest=store / "distribution.json",
        progress=lambda stage, done, total: print(
            json.dumps(dict(stage=stage, done=done, total=total)), flush=True
        ),
    )
    audit = {}
    for length in (32, 64):
        raw = json.loads((output / f"train_T{length}.json").read_text())
        previous = accepted = 0
        counts = dict.fromkeys(
            ("masked", "observed", "reviewed_empty", "out_of_frame"), 0
        )
        edge_only = zero_coordinate_children = 0
        for record in raw["clips"]:
            if record["clip"]["split"] != "train":
                continue
            codes = decode_label_runs(
                record["label_runs"], record["clip"]["frame_count"]
            )
            cfg = AllFrameWindowConfig(length)
            previous += len(
                training_packs(
                    record["common_intervals"], codes, cfg, require_position=False
                )
            )
            accepted += len(record["packs"])
            coverage = np.zeros(len(codes), dtype=bool)
            for pack in record["packs"]:
                a = pack["start"]
                coverage[a : a + 128] = True
                edge_only += not bool((codes[a + 1 : a + 127] == 1).any())
                zero_coordinate_children += sum(
                    not bool((codes[s : s + length] == 1).any())
                    for s in range(a, a + 128, length)
                )
            for i, key in enumerate(counts):
                counts[key] += int(((codes == i) & coverage).sum())
        audit[str(length)] = dict(
            previous_packs=previous,
            accepted_packs=accepted,
            excluded_packs=previous - accepted,
            unique_training_coverage=counts,
            edge_only_packs=int(edge_only),
            zero_coordinate_child_occurrences=zero_coordinate_children,
        )
    if audit["32"]["previous_packs"] != 32509 or audit["32"]["accepted_packs"] != 30478:
        raise RuntimeError(
            f"Production snapshot pack counts differ from the fixed evidence: {audit}"
        )
    if (
        audit["32"]["unique_training_coverage"]
        != audit["64"]["unique_training_coverage"]
    ):
        raise RuntimeError("Training coverage differs across T")
    save_json(
        output / "audit.json",
        dict(audit=audit, seconds=time.perf_counter() - started, plan=plan),
    )
    print(json.dumps(audit), flush=True)


def run_one(
    args,
    architecture,
    frames,
    output,
    bud,
    parent=None,
    stop=None,
    resume=None,
    root=None,
):
    prepared = (
        args.workspace_root
        / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
    )
    if not (prepared / "audit.json").is_file():
        raise RuntimeError(
            "The production pack-count and coverage audit must complete before GPU execution"
        )
    initial = {}
    if resume is not None:
        initial = dict(resume=resume)
    elif parent is not None:
        latest = json.loads((parent / "latest.json").read_text())
        checkpoint = parent / latest["checkpoint"]
        initial = dict(
            initial_artifact=checkpoint,
            initial_artifact_sha256=dual_sha256(checkpoint),
            parent_update=latest["update"],
        )
    campaign = root or output.parent
    remote = "outputs/ball_detection/train/sequential_length/" + campaign.name
    persist = persist_callback(campaign, output, args.rclone_config, remote)
    try:
        return train_temporal_model(
            architecture=architecture,
            phase="pretrain" if architecture == "dpt" and frames == 32 else "finetune",
            manifest=prepared / f"train_T{frames}.json",
            evaluation_manifest=prepared / f"evaluation_T{frames}.json",
            model_config=model_config(architecture, frames),
            output=output,
            budget=bud,
            runtime=runtime(),
            device=torch.device("cuda"),
            augmentation=AugmentationConfig.load(
                Path(
                    "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
                )
            ),
            distance_px=20.0,
            stop_after_update=stop,
            persist=persist,
            **initial,
        )
    except Exception:
        if output.exists():
            save_json(
                output / "FAILED.json",
                dict(complete=False, traceback=traceback.format_exc()),
            )
            persist(output, "failed")
        raise


def benchmark(args):
    root = (
        args.workspace_root
        / "outputs/ball_detection/train/sequential_length"
        / args.run_id
    )
    root.mkdir(parents=True, exist_ok=False)
    metrics = []
    for microbatch in (1, 2):
        output = root / f"dpt-T32-b{microbatch}"
        started = time.perf_counter()
        try:
            run_one(
                args,
                "dpt",
                32,
                output,
                budget(common=True, microbatch=microbatch),
                stop=30,
                root=root,
            )
        except torch.OutOfMemoryError as error:
            metrics.append(
                dict(
                    microbatch_windows=microbatch,
                    success=False,
                    error="CUDA_OOM",
                    message=str(error),
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch._dynamo.reset()
            continue
        rows = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text().splitlines()
        ]
        steady = [row["seconds"] for row in rows[5:]]
        metrics.append(
            dict(
                microbatch_windows=microbatch,
                accumulation_microbatches=4 // microbatch,
                median_update_seconds=statistics.median(steady),
                mean_update_seconds=statistics.mean(steady),
                pretraining_compute_hours=10000 * statistics.median(steady) / 3600,
                first_update_seconds=rows[0]["seconds"],
                peak_vram_bytes=max(r["peak_vram_bytes"] for r in rows),
                wall_seconds=time.perf_counter() - started,
                rows=len(rows),
                input_frames=sum(r["input_frames"] for r in rows),
                success=True,
                input_and_persistence_overhead_seconds=time.perf_counter()
                - started
                - sum(r["seconds"] for r in rows),
            )
        )
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    selected = min(
        (m for m in metrics if m["success"]),
        key=lambda item: item["median_update_seconds"],
    )
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    selected_output = root / f"dpt-T32-b{selected['microbatch_windows']}"
    artifact = selected_output / "weights-update-00000030.pt"
    model = restore_temporal_model(
        torch.load(artifact, map_location="cpu", weights_only=True)
    ).cuda()
    compile_coordinate_model(model, mode="default")
    prepared = (
        args.workspace_root
        / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
    )
    report = evaluate_temporal_heatmap(
        model,
        prepared / "evaluation_T32.json",
        split="val",
        runtime=runtime(),
        device=torch.device("cuda"),
        artifact_sha256=dual_sha256(artifact),
        distance_px=20.0,
        output=root / "benchmark-validation.json",
    )
    selected["full_validation_seconds"] = report["seconds"]
    receipts = [
        json.loads(p.read_text())
        for p in (root / "persistence" / selected_output.name).glob("*.json")
    ]
    save_seconds = statistics.median(r["seconds"] for r in receipts)
    selected["estimated_pretraining_wall_hours"] = (
        selected["median_update_seconds"] * 10000
        + max(0.0, selected["first_update_seconds"] - selected["median_update_seconds"])
        + max(0.0, selected["input_and_persistence_overhead_seconds"])
        + report["seconds"]
        + 20 * save_seconds
    ) / 3600
    selected["estimated_wall_components"] = dict(
        final_full_validation_seconds=report["seconds"],
        estimated_checkpoint_publication_seconds=20 * save_seconds,
        extra_dataset_preparation_and_initial_drive_download_included=False,
    )
    save_json(
        root / "benchmark-complete.json",
        dict(
            metrics=metrics,
            selected=selected,
            scope="real production data, augmentation, BF16, nvJPEG, compile; evaluation timed separately; 30-update diagnostic, no convergence claim",
        ),
    )
    print(json.dumps(dict(benchmark=metrics, selected=selected)), flush=True)


def train(args):
    download = json.loads(
        (
            args.workspace_root
            / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025/progress.json"
        ).read_text()
    )
    if not download["complete"] or any(
        download[f"verified_{unit}"] != download[f"expected_{unit}"]
        for unit in ("files", "bytes")
    ):
        raise ValueError("The full dataset download must verify before training")
    root = (
        args.workspace_root
        / "outputs/ball_detection/train/sequential_length"
        / args.run_id
    )
    saved_plan = None
    if args.resume:
        saved_plan = json.loads((root / "plan.json").read_text())
        # The benchmark summary was not part of the old persistence selection.
        # Its hash and selected batch sizes survive in the original plan. Keep
        # that evidence intact instead of manufacturing a replacement benchmark.
        microbatch = saved_plan["microbatch_T32"]
        benchmark_sha256 = saved_plan["benchmark_sha256"]
        if (
            args.benchmark is not None
            and dual_sha256(args.benchmark) != benchmark_sha256
        ):
            raise ValueError("Resume benchmark differs from the original plan")
        current_code = source_identity()
        found = False
        for name, *_ in saved_plan["stages"]:
            path = root / name / "run.json"
            if path.exists():
                found = True
                previous_code = json.loads(path.read_text())["code"]
                for key in ("source_sha256", "torch_version"):
                    if previous_code[key] != current_code[key]:
                        raise ValueError(
                            "Campaign resume code or torch version changed"
                        )
        if not found:
            raise ValueError("Campaign resume requires a saved stage recipe")
    else:
        if args.benchmark is None:
            raise ValueError("New training requires a completed real-data benchmark")
        measured = json.loads(args.benchmark.read_text())
        microbatch = measured["selected"]["microbatch_windows"]
        benchmark_sha256 = dual_sha256(args.benchmark)
    stages = [
        ["common-dpt-T32", "dpt", 32, None],
        ["dpt-T64", "dpt", 64, "common-dpt-T32"],
        ["query-T32", "query", 32, "common-dpt-T32"],
        ["query-T64", "query", 64, "query-T32"],
    ]
    plan = dict(
        schema="ball_sequential_campaign.v1",
        seed=42,
        stages=stages,
        skipped_frames=[128],
        skipped_reason="User explicitly excluded T128 on 2026-10-25 after L4 OOM preflight",
        optimizer_updates=28000,
        physical_input_frames=28000 * 128,
        pretraining_updates=10000,
        pretraining_lr_horizon=28000,
        posttraining_horizons=dict(dpt=6000, query=12000),
        cnn_freeze_updates_each_post_stage=600,
        encoder_lr_ratio=0.1,
        optimizer_state="inherit matching parameter names",
        compile_mode="default",
        activation_checkpointing=False,
        frame_chunk_size=None,
        microbatch_T32=microbatch,
        microbatch_T64=1,
        benchmark_sha256=benchmark_sha256,
        commit=saved_plan["commit"]
        if saved_plan
        else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    if root.exists():
        if not args.resume or json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError(
                "Existing campaign requires --resume and the exact original plan"
            )
    else:
        root.mkdir(parents=True)
        save_json(root / "plan.json", plan)
    index = RcloneArtifactStore(
        root,
        remote="mounted_drive",
        remote_root="outputs/ball_detection/train/sequential_length/" + args.run_id,
        config_path=args.rclone_config,
    )
    index.publish_file(root / "plan.json")
    if args.benchmark is not None:
        retained_benchmark = root / "benchmark-complete.json"
        if args.benchmark.resolve() != retained_benchmark.resolve():
            shutil.copyfile(args.benchmark, retained_benchmark)
        if dual_sha256(retained_benchmark) != benchmark_sha256:
            raise ValueError("Retained benchmark differs from the original plan")
        index.publish_file(retained_benchmark)
    if args.resume:
        attempt = root / f"resume-{uuid.uuid4().hex}.json"
        save_json(
            attempt,
            dict(
                schema="ball_sequential_resume.v1",
                original_plan=plan,
                code=source_identity(),
                controller_sha256=dual_sha256(Path(__file__)),
                benchmark_file_available=args.benchmark is not None,
            ),
        )
        index.publish_file(attempt)
    for name, architecture, frames, parent_name in stages:
        output = root / name
        if (output / "COMPLETED.json").exists():
            latest = json.loads((output / "latest.json").read_text())
            completed = json.loads((output / "COMPLETED.json").read_text())
            expected_updates = 10000 if parent_name is None else 6000
            if (
                completed.get("complete") is not True
                or completed.get("updates_completed") != expected_updates
                or latest["update"] != expected_updates
                or dual_sha256(output / latest["checkpoint"]) != latest["sha256"]
            ):
                raise ValueError("Completed stage has an invalid terminal checkpoint")
            continue
        resume = None
        if (output / "latest.json").exists():
            if not args.resume:
                raise RuntimeError("Interrupted stage requires explicit --resume")
            resume = (
                output / json.loads((output / "latest.json").read_text())["checkpoint"]
            )
        run_one(
            args,
            architecture,
            frames,
            output,
            budget(
                architecture=architecture,
                frames=frames,
                common=parent_name is None,
                microbatch=microbatch if frames == 32 else 1,
            ),
            parent=root / parent_name if parent_name else None,
            resume=resume,
            root=root,
        )
        gc.collect()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        save_json(
            root / "progress.json",
            dict(completed_stage=name, complete=False, plan=plan),
        )
        index.publish_file(root / "progress.json")
    save_json(
        root / "COMPLETED.json", dict(complete=True, optimizer_updates=28000, plan=plan)
    )
    index.publish_file(root / "COMPLETED.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "benchmark", "train"))
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--rclone-config", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode != "prepare" and (not args.run_id or args.rclone_config is None):
        parser.error("run-id and rclone-config are required")
    if args.mode == "train" and args.benchmark is None and not args.resume:
        parser.error("train requires a completed real-data benchmark")
    dict(prepare=prepare, benchmark=benchmark, train=train)[args.mode](args)
