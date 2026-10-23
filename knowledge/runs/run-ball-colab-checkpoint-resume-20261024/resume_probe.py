"""Short, explicit 720p save/resume experiment; orchestration stays outside src."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.distribution import prepare_distribution
from src.tasks.ball_detection.data.distribution.manifest import distribution_files
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    common_intervals,
    verified_label_kinds,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
    UpdateBudget,
    load_model_config,
)
from src.tasks.ball_detection.training.temporal_heatmap.runner import (
    train_temporal_heatmap,
)
from src.utils.artifact_store import ArtifactStoreError, RcloneSnapshotStore
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version
from src.utils.models.state import state_dict_sha256


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def git_commit() -> str:
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(args, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def store(root: Path, remote_root: str) -> RcloneSnapshotStore:
    remote, fragment = remote_root.split(":", 1)
    return RcloneSnapshotStore(root, remote=remote, remote_root=fragment,
                               config_path=Path(os.environ["RCLONE_CONFIG"]))


def select_clips(source: BallFrameStore) -> list[str]:
    config = AllFrameWindowConfig(32)
    chosen = []
    for split, origin in (("train", "tracknet"), ("train", "meiji"),
                          ("train", "chat_annotation"), ("val", "chat_annotation")):
        candidates = sorted((c for c in source.clips if c.split == split and c.source == origin
                             and (c.width, c.height) == (1280, 720)), key=lambda c: (c.frame_count, c.clip_id))
        for clip in candidates:
            states = annotation_states(source, clip)
            spans = common_intervals(states, config)
            if not spans:
                continue
            codes, _ = verified_label_kinds(source, clip, states)
            common = np.concatenate([codes[a:b] for a, b in spans])
            if (common == 1).any() and (split != "val" or ((common == 2) | (common == 3)).any()):
                chosen.append(clip.clip_id)
                break
        else:
            raise ValueError(f"No native 1280x720 diagnostic clip for {split}/{origin}")
    return sorted(chosen)


def prepare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.root.exists() or args.receipts.exists():
        raise FileExistsError("Use new input and receipt directories")
    args.root.mkdir(parents=True)
    args.receipts.mkdir(parents=True)
    source = BallFrameStore(args.source_store)
    watched = {source.directory / name for name in ("index.npz", "metadata.json")}
    watched.update(source.image_path(c) for c in source.clips)
    watched.update(source.annotation_path(c) for c in source.clips if c.source == "chat_annotation")
    versions = {p: file_version(p) for p in watched}
    full_started = time.monotonic()
    full = prepare_all_frame_training(args.source_store, args.root / "full_windows", configs=[AllFrameWindowConfig(32)])
    full_seconds = time.monotonic() - full_started
    subset_started = time.monotonic()
    chosen = select_clips(source)
    subset = args.root / "diagnostic_store"
    prepare_distribution(args.source_store, subset, version="ball-720p-resume-diagnostic-v1", clip_ids=chosen, inputs={})
    for name, path in distribution_files(subset, source_directory=args.source_store).items():
        destination = subset / name
        if destination != path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    plan = prepare_all_frame_training(subset, args.root / "diagnostic_windows", configs=[AllFrameWindowConfig(32)])
    model = yaml.safe_load(Path("src/tasks/ball_detection/configs/model/mdd_heatmap_hybrid.yaml").read_text())
    model.update(frame_chunk_size=4, activation_checkpointing=True)
    (args.root / "model.yaml").write_text(yaml.safe_dump(model))
    runtime = CoordinateRuntime(precision="bf16", num_workers=0, pin_memory=True, prefetch_factor=1, cpu_threads=2,
                                compile_mode="off", compile_recompile_limit=8, jpeg_decoder="nvjpeg",
                                input_verification="upfront", image_prefetch=False)
    budget = UpdateBudget(updates=12, input_frames=128, microbatch_windows=1, learning_rate=.0001,
                          warmup_updates=2, minimum_lr_ratio=.1, weight_decay=.01, gradient_clip=1.,
                          evaluation_interval=12, checkpoint_interval=5, seed=42, sigma_ratio=.012, gamma=2.)
    recipe = dict(schema_version=1, code_commit=git_commit(), uv_lock_sha256=dual_sha256(Path("uv.lock")),
                  budget=asdict(budget), runtime=asdict(runtime), distance_px=20., stop_after_update=5,
                  loss_tolerance=dict(atol=1e-6, rtol=1e-4), chosen_clip_ids=chosen,
                  native_image_size=[1280, 720], purpose="save/resume continuity only; not a training budget or accuracy experiment")
    write(args.root / "recipe.json", recipe)
    if any(file_version(p) != version for p, version in versions.items()):
        raise ValueError("Full distribution changed during diagnostic preparation")
    summary = dict(schema_version=1, code_commit=recipe["code_commit"], chosen_clip_ids=chosen,
                   full_preparation_seconds=full_seconds, diagnostic_preparation_seconds=time.monotonic() - subset_started,
                   full_summaries=full["datasets"]["32"]["train"]["summaries"],
                   diagnostic_summaries=plan["datasets"]["32"]["train"]["summaries"],
                   input_files_checked=len(versions), source_versions_unchanged=True,
                   disk=shutil.disk_usage(args.root)._asdict())
    summary["input_snapshot"] = store(args.root, args.remote_root).publish("inputs-" + uuid.uuid4().hex)
    summary["total_seconds"] = time.monotonic() - started
    write(args.receipts / "preparation.json", summary)
    print(json.dumps(summary), flush=True)


class Tee:
    def __init__(self, original: Any, path: Path) -> None:
        self.original = original
        self.file = path.open("a")

    def write(self, text: str) -> int:
        self.file.write(text)
        return int(self.original.write(text))

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.original.flush()


def summarize(root: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    latest = json.loads((root / "latest.json").read_text())
    checkpoint = torch.load(root / latest["checkpoint"], map_location="cpu", weights_only=True)
    return dict(records=records, latest=latest, train_log_sha256=dual_sha256(root / "train.jsonl"),
                updates_completed=checkpoint["updates_completed"], sample_position=checkpoint["sample_position"],
                schedule=checkpoint["schedule"], model_state_sha256=state_dict_sha256(checkpoint["model_artifact"]["state_dict"]),
                optimizer_steps=sorted({int(v["step"]) for v in checkpoint["optimizer"]["state"].values()}),
                checkpoint_rng_sha256=hashlib.sha256(checkpoint["rng"]["torch"].numpy().tobytes()).hexdigest(),
                checkpoint_cuda_rng_sha256=[hashlib.sha256(state.numpy().tobytes()).hexdigest()
                                            for state in checkpoint["rng"]["cuda"]] if checkpoint["rng"]["cuda"] else [])


def train(args: argparse.Namespace) -> None:
    if args.receipts.exists():
        raise FileExistsError("Use a new receipt directory")
    args.receipts.mkdir(parents=True)
    recipe = json.loads((args.inputs / "recipe.json").read_text())
    if git_commit() != recipe["code_commit"] or dual_sha256(Path("uv.lock")) != recipe["uv_lock_sha256"]:
        raise ValueError("Use the exact committed code and locked dependencies from preparation")
    publisher = store(args.root, args.remote_root)
    stdout = sys.stdout
    tee = None
    receipts = []
    started = time.monotonic()

    def persist(output: Path, stage: str) -> None:
        nonlocal tee
        if tee is None:
            tee = Tee(stdout, output / "training.log")
            sys.stdout = tee
        tee.flush()
        update = json.loads((output / "latest.json").read_text())["update"] if (output / "latest.json").exists() else 0
        if args.mode == "failure" and stage == "checkpoint":
            def disconnected(*arguments: str, timeout: int | None = None) -> str:
                raise ArtifactStoreError("Injected Drive communication failure at checkpoint publication")
            publisher._output = disconnected  # type: ignore[method-assign]
        receipt = publisher.publish(f"{stage}-{update:08d}-{uuid.uuid4().hex}")
        receipt.update(stage=stage, update=update)
        receipts.append(receipt)
        write(args.receipts / "publications.json", receipts)
        print(json.dumps(dict(persistence=receipt)), flush=True)

    try:
        result = train_temporal_heatmap(
            architecture="hybrid", phase="pretrain", manifest=args.inputs / "diagnostic_windows/train_T32.json",
            evaluation_manifest=args.inputs / "diagnostic_windows/evaluation_T32.json",
            model_config=load_model_config(args.inputs / "model.yaml", "hybrid"), output=args.root,
            budget=UpdateBudget(**recipe["budget"]), runtime=CoordinateRuntime(**recipe["runtime"]),
            device=torch.device("cuda"), augmentation=None, distance_px=recipe["distance_px"],
            resume=(args.root / json.loads((args.root / "latest.json").read_text())["checkpoint"]) if args.mode == "resume" else None,
            stop_after_update=recipe["stop_after_update"] if args.mode == "interrupted" else 1 if args.mode == "failure" else None,
            persist=persist,
        )
        summary = dict(schema_version=1, mode=args.mode, result=result, seconds=time.monotonic() - started,
                       code_commit=recipe["code_commit"], kernel_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                       python=sys.version, torch_version=torch.__version__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                       publications=receipts, **summarize(args.root))
        write(args.receipts / "training.json", summary)
    except Exception as error:
        write(args.receipts / "failure.json", dict(schema_version=1, exception=type(error).__name__, message=str(error),
              seconds=time.monotonic() - started, completed_marker=(args.root / "COMPLETED.json").exists(),
              publications=receipts, latest=json.loads((args.root / "latest.json").read_text()) if (args.root / "latest.json").exists() else None))
        raise
    finally:
        sys.stdout = stdout
        if tee is not None:
            tee.flush()
            tee.file.close()


def compare(args: argparse.Namespace) -> None:
    expected = json.loads(args.reference.read_text())
    actual = json.loads(args.actual.read_text())
    keys = ("update", "sample_position", "pack_ids", "pack_indices", "input_frames", "cumulative_input_frames",
            "valid_teachers", "teachers_by_kind", "original_teachers_by_kind", "augmentation_sha256", "learning_rate")
    before, after = expected["records"], actual["records"]
    if len(before) != len(after) or any(any(a[k] != b[k] for k in keys) for a, b in zip(before, after, strict=True)):
        raise ValueError("Update/sample/teacher/LR sequence differs")
    if (actual["updates_completed"] != 12 or actual["sample_position"] != 12 or actual["optimizer_steps"] != [12]
            or actual["schedule"] != expected["schedule"] or not actual["result"]["complete"]):
        raise ValueError("Update, optimizer or original budget was reset")
    losses_a = np.array([r["loss"] for r in before])
    losses_b = np.array([r["loss"] for r in after])
    np.testing.assert_allclose(losses_a, losses_b, atol=1e-6, rtol=1e-4)
    result = dict(schema_version=1, status="passed", updates=len(after), sample_teacher_lr_sequence_equal=True,
                  loss_atol=1e-6, loss_rtol=1e-4, max_absolute_loss_difference=float(np.max(np.abs(losses_a-losses_b))),
                  model_state_equal=actual["model_state_sha256"] == expected["model_state_sha256"],
                  rng_equal=(actual["checkpoint_rng_sha256"] == expected["checkpoint_rng_sha256"]
                             and actual["checkpoint_cuda_rng_sha256"] == expected["checkpoint_cuda_rng_sha256"]),
                  original_budget=actual["schedule"], optimizer_steps=actual["optimizer_steps"],
                  compared_at=datetime.now(UTC).isoformat())
    write(args.output, result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "restore"):
        child = commands.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--remote-root", required=True)
        child.add_argument("--receipts", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--source-store", type=Path, required=True)
        elif command == "train":
            child.add_argument("--inputs", type=Path, required=True)
            child.add_argument("--mode", choices=("baseline", "interrupted", "resume", "failure"), required=True)
        else:
            child.add_argument("--generation", required=True)
            child.add_argument("--manifest-sha256", required=True)
    comparison = commands.add_parser("compare")
    for name in ("reference", "actual", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "compare":
        compare(args)
    else:
        result = store(args.root, args.remote_root).restore(args.generation, manifest_sha256=args.manifest_sha256)
        write(args.receipts / "restoration.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
