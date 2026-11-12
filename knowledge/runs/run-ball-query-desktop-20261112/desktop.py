"""Explicit path relocation and desktop execution of the frozen #1899 length run.

The historical training source is imported unchanged from a detached checkout.
Only the snapshot directory and the two resulting manifest digests are relocated
in a COPY of the resume metadata. Model, AdamW, schedule, RNG and sample state
are compared recursively before the copy is accepted.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

COMMIT = "929ad65645c1c7522e8921f1b439619d636a0128"
GPU = "NVIDIA GeForce RTX 5060 Ti"
REMOTE = "tennis_lab/outputs/ball_detection/train/query_controls/s42-20261111-main/desktop-attempts/20261112-a1"
HISTORICAL_STORE = (
    "/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1"
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def relocate_manifest(raw, dataset):
    result = copy.deepcopy(raw)
    if result["ball_store"]["directory"] != HISTORICAL_STORE:
        raise ValueError("Unexpected historical data root")
    result["ball_store"]["directory"] = str(dataset.resolve())
    return result


def relocate_recipe(raw, dataset, train_sha, evaluation_sha):
    result = copy.deepcopy(raw)
    if result["data"]["snapshot"]["directory"] != HISTORICAL_STORE:
        raise ValueError("Unexpected resume snapshot root")
    result["data"]["snapshot"]["directory"] = str(dataset.resolve())
    result["manifest_sha256"] = train_sha
    result["evaluation_manifest_sha256"] = evaluation_sha
    return result


def normalize_report_snapshot(report, dataset):
    snapshot = copy.deepcopy(report["snapshot"])
    if snapshot["directory"] not in {HISTORICAL_STORE, str(dataset.resolve())}:
        raise ValueError("Unrecognized evaluation snapshot")
    snapshot["directory"] = HISTORICAL_STORE
    return snapshot


def load_frozen(source, root):
    if Path.cwd().resolve() != source.resolve():
        raise ValueError("Run from the detached frozen source checkout")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Frozen source commit differs")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "knowledge/runs/run-ball-query-controls-20261111"))
    control = importlib.import_module("control")
    frozen = importlib.import_module("train")
    plan = read(root / "original/inputs/inputs-v1/plan.json")
    control.verify_plan_sources(plan)
    control.REFERENCE = root / "reference"
    return control, frozen, plan


def prepare(root, source):
    import torch

    control, frozen, plan = load_frozen(source, root)
    dataset = root / "dataset"
    if not read(root / "dataset-verified.json")["complete"]:
        raise ValueError("Complete byte-verified dataset is required")
    original = root / "original/length-s1-T64"
    latest = read(original / "latest.json")
    if (
        latest["update"] != 5500
        or sha(original / latest["checkpoint"]) != latest["sha256"]
    ):
        raise ValueError("Expected the pinned latest 5500 full checkpoint")
    inputs = root / "original/inputs/inputs-v1"
    receipt = read(root / "original/inputs-receipt.json")
    if sha(inputs / "_SNAPSHOT.json") != receipt["manifest_sha256"]:
        raise ValueError("Input snapshot marker differs from its receipt")
    snapshot = read(inputs / "_SNAPSHOT.json")
    for name, identity in snapshot["files"].items():
        if sha(inputs / name) != identity["sha256"]:
            raise ValueError("Pinned input changed: " + name)
    for model in ("queries", "cnn"):
        if not read(root / "original/job-status" / (model + ".json"))["complete"]:
            raise ValueError("Previously completed conditions are required")
    status = read(root / "original/stage-status/length-s1-T64.json")
    if status["update"] != 5500:
        raise ValueError("Remote stage is not at the selected newest update")
    checkpoint = torch.load(
        original / latest["checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint["schema"] != "ball_temporal_checkpoint.v2"
        or not checkpoint["optimizer"]["state"]
    ):
        raise ValueError("Full-state checkpoint required")
    if checkpoint["sample_position"] != 15500:
        raise ValueError("Saved sample position changed")
    if checkpoint["code"]["source_sha256"] != plan["code"]["source_sha256"]:
        raise ValueError("Checkpoint source identity differs from plan")
    localized = root / "prepared"
    localized.mkdir(exist_ok=True)
    originals = {}
    for name, field in (
        ("train_manifest.json", "manifest_sha256"),
        ("evaluation_manifest.json", "evaluation_manifest_sha256"),
    ):
        path = original / name
        if sha(path) != checkpoint["recipe"][field]:
            raise ValueError("Original prepared manifest differs")
        raw = read(path)
        originals[name] = sha(path)
        write(localized / name, relocate_manifest(raw, dataset))
    recipe = relocate_recipe(
        checkpoint["recipe"],
        dataset,
        sha(localized / "train_manifest.json"),
        sha(localized / "evaluation_manifest.json"),
    )
    modified = dict(checkpoint, recipe=recipe)
    active = root / "production/length-s1-T64"
    if active.exists():
        raise ValueError(
            "An active attempt already exists; inspect rather than overwrite"
        )
    active.mkdir(parents=True)
    for path in original.iterdir():
        if path.name != "_SNAPSHOT.json":
            shutil.copyfile(path, active / path.name)
    for name in originals:
        shutil.copyfile(localized / name, active / name)
    run = read(original / "run.json")
    run["recipe"] = recipe
    write(active / "run.json", run)
    torch.save(modified, active / latest["checkpoint"])
    changed = torch.load(
        active / latest["checkpoint"], map_location="cpu", weights_only=True
    )
    restored = dict(changed, recipe=checkpoint["recipe"])
    if not frozen.same(checkpoint, restored):
        raise ValueError("Path relocation modified scientific or training state")
    latest["sha256"] = sha(active / latest["checkpoint"])
    write(active / "latest.json", latest)
    control.validate_log(
        next(j for j in plan["jobs"] if j["model"] == "length")["stages"][0], original
    )
    for t in (32, 64):
        reference = root / f"reference/query-T{t}"
        parent = read(inputs / f"parent-receipts/baseline-query-T{t}.json")
        marker = reference / "_SNAPSHOT.json"
        if sha(marker) != parent["manifest_sha256"]:
            raise ValueError("Baseline marker differs")
        manifest = read(marker)
        for name in ("train.jsonl", "validation-update-00006000.json"):
            if sha(reference / name) != manifest["files"][name]["sha256"]:
                raise ValueError("Baseline payload differs")
    write(
        root / "relocation.json",
        dict(
            complete=True,
            source_commit=COMMIT,
            source_directory=str(source),
            original_checkpoint_sha256=read(original / "latest.json")["sha256"],
            adapted_checkpoint_sha256=latest["sha256"],
            changed_fields=[
                "recipe.data.snapshot.directory",
                "recipe.manifest_sha256",
                "recipe.evaluation_manifest_sha256",
                "run.recipe",
                "latest.sha256",
            ],
            manifests_before=originals,
            manifests_after={name: sha(localized / name) for name in originals},
            model_optimizer_schedule_rng_sample_exact=True,
            original_store=HISTORICAL_STORE,
            local_store=str(dataset),
            original_gpu="NVIDIA L4",
            local_gpu=GPU,
            bitwise_gpu_continuation_guaranteed=False,
            remote_attempt=REMOTE,
            parent_run="s42-20261111-main",
        ),
    )


@contextmanager
def audit_resume(frozen, checkpoint, output):
    import torch

    from src.tasks.ball_detection.training.temporal import runner

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    audit = dict(
        resume=True,
        parent_update=saved["updates_completed"],
        parent_sample_position=saved["sample_position"],
        checkpoint_sha256=sha(checkpoint),
    )
    original_model = runner.restore_temporal_model
    original_load = torch.optim.AdamW.load_state_dict
    original_batches = runner.coordinate_batches

    def model(payload):
        result = original_model(payload)
        if not frozen.same(result.state_dict(), saved["model_artifact"]["state_dict"]):
            raise ValueError("Resume model tensors differ")
        audit["model_exact"] = True
        return result

    def load(optimizer, state):
        result = original_load(optimizer, state)
        if not frozen.same(optimizer.state_dict(), saved["optimizer"]):
            raise ValueError("Resume AdamW moments/groups differ")
        audit["adamw_exact"] = True
        return result

    def batches(*args, **kwargs):
        from src.tasks.ball_detection.checkpoints.temporal import restore_rng, rng_state

        first = True
        for batch in original_batches(*args, **kwargs):
            if first:
                restore_rng(saved["rng"], torch.device("cuda"))
                if not frozen.same(saved["rng"], rng_state(torch.device("cuda"))):
                    raise ValueError("Full RNG could not transfer to desktop GPU")
                audit["rng_exact_before_first_batch"] = True
                if not audit.get("model_exact") or not audit.get("adamw_exact"):
                    raise ValueError("Resume state was not audited")
                write(output / "continuity.json", audit)
                first = False
            yield batch

    runner.restore_temporal_model = model
    torch.optim.AdamW.load_state_dict = load
    runner.coordinate_batches = batches
    try:
        yield audit
    finally:
        runner.restore_temporal_model = original_model
        torch.optim.AdamW.load_state_dict = original_load
        runner.coordinate_batches = original_batches


def validate_result(control, plan, root, output, stage):
    rows = control.validate_log(stage, output)
    if len(rows) != 6000 or read(output / "latest.json")["update"] != 6000:
        raise ValueError("Incomplete stage")
    report = read(output / "validation-update-00006000.json")
    reference = read(root / "reference/query-T64/validation-update-00006000.json")
    if (
        report["identities"] != reference["identities"]
        or normalize_report_snapshot(report, root / "dataset")
        != normalize_report_snapshot(reference, root / "dataset")
        or report["metrics"]["support"] != reference["metrics"]["support"]
    ):
        raise ValueError("Validation frame/teacher/snapshot identity changed")
    if (
        report["metrics"]["support"]["observed"] != 85473
        or report["metrics"]["support"]["frames"] != 128668
    ):
        raise ValueError("Unexpected validation population")
    predictions = output / report["predictions"]["file"]
    if (
        sha(predictions) != report["predictions"]["sha256"]
        or control.observed_identity(predictions)
        != plan["baseline_observed_identity_sha256"]
    ):
        raise ValueError("Observed position teacher IDs changed")
    write(
        output / "acceptance.json",
        dict(
            complete=True,
            baseline_inputs_exact=True,
            identities=report["identities"],
            metrics=report["metrics"],
            freeze_boundary=[rows[i] for i in (0, 599, 600, 5999)],
        ),
    )


def train(root, source, mode, config):
    from dataclasses import asdict

    import torch

    control, frozen, plan = load_frozen(source, root)
    from src.tasks.ball_detection.checkpoints.temporal import CHECKPOINT_SCHEMA
    from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
    from src.tasks.ball_detection.models.mdd.config import MDDTemporalQueryConfig
    from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
    from src.tasks.ball_detection.training.temporal import runner
    from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
    from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore

    if sys.version_info[:3] != (3, 11, 17):
        raise ValueError("The pinned Python 3.11.17 is required")
    if not os.environ.get("TENNIS_RUN_ID") or os.environ.get(
        "TRAINING_QUEUE_DIR"
    ) != str(source.parents[2] / ".training_queue"):
        raise ValueError("Shared queue ownership is required")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != GPU:
        raise ValueError(
            "Expected the explicitly authorized single desktop RTX 5060 Ti"
        )
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    write(
        root / (mode + "-environment.json"),
        dict(
            torch=str(torch.__version__),
            python=sys.version,
            gpu=torch.cuda.get_device_name(0),
            cuda=torch.version.cuda,
            allocator_fraction=0.80,
            compile_threads=os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            queue_id=os.environ["TENNIS_RUN_ID"],
            pid=os.getpid(),
            pgid=os.getpgrp(),
        ),
    )
    if mode == "production" and not read(root / "probe-acceptance.json")["complete"]:
        raise ValueError("A successful exact-recipe probe is required")
    if mode == "production" and (root / "production-FAILED.json").exists():
        raise ValueError("Failed attempts require diagnosis; no automatic retry")
    if not read(root / "relocation.json")["complete"]:
        raise ValueError("Verified explicit path relocation required")
    if mode == "production":
        progress = read(root / "progress.json")
        progress.update(
            phase="production_process_started",
            production_started=True,
            production_queue_id=os.environ["TENNIS_RUN_ID"],
            updated_at=time.time(),
        )
        write(root / "progress.json", progress)
    local = root / mode
    if mode == "probe":
        if local.exists():
            raise ValueError("Never retry or overwrite an existing probe attempt")
        shutil.copytree(root / "production", local)
    job = next(j for j in plan["jobs"] if j["model"] == "length")
    augmentation = AugmentationConfig.load(
        Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    )
    if json.loads(json.dumps(asdict(augmentation))) != plan["augmentation"]:
        raise ValueError("Augmentation changed")
    index = RcloneArtifactStore(
        local, remote="gdrive", remote_root=REMOTE + "/" + mode, config_path=config
    )
    first = True
    for stage in job["stages"]:
        output = local / stage["name"]
        budget = UpdateBudget(
            **{k: stage["budget"][k] for k in UpdateBudget.__dataclass_fields__}
        )
        values = dict(stage["model"])
        for key in ("stem_channels", "mixed_channels", "residual_blocks"):
            values[key] = tuple(values[key])
        model = MDDTemporalQueryConfig(**values)
        if first:
            latest = read(output / "latest.json")
            checkpoint = output / latest["checkpoint"]
            options = dict(resume=checkpoint)
            context = audit_resume(frozen, checkpoint, output)
        else:
            previous = local / job["stages"][0]["name"]
            latest = read(previous / "latest.json")
            checkpoint = previous / latest["checkpoint"]
            options = dict(
                initial_artifact=checkpoint,
                initial_artifact_sha256=latest["sha256"],
                parent_update=6000,
            )
            context = frozen.continuity_adapter(
                dict(sha256=latest["sha256"]), checkpoint, budget, output
            )
        if mode == "probe":
            options["stop_after_update"] = 5502
        with context as audit:
            store = RcloneSnapshotStore(
                output,
                remote="gdrive",
                remote_root=REMOTE + "/" + mode + "/" + stage["name"],
                config_path=config,
            )

            def persist(directory, event, stage=stage, store=store, audit=audit):
                rows = control.validate_log(stage, directory)
                write(directory / "continuity.json", audit)
                names = [
                    "run.json",
                    "train_manifest.json",
                    "evaluation_manifest.json",
                    "train.jsonl",
                    "continuity.json",
                ]
                update = 0
                if (directory / "latest.json").exists():
                    latest = read(directory / "latest.json")
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
                if mode == "production":
                    generation = f"{event}-{update:08d}-{uuid.uuid4().hex[:8]}"
                    receipt = store.publish(generation, include=names)
                    path = (
                        local / "persistence" / stage["name"] / (generation + ".json")
                    )
                    write(path, receipt)
                    index.publish_file(path)
                    if event in {"checkpoint", "completed"}:
                        # Every durable snapshot gets an independent byte readback.
                        restored = root / "readback" / stage["name"] / generation
                        result = RcloneSnapshotStore(
                            restored,
                            remote="gdrive",
                            remote_root=REMOTE + "/" + mode + "/" + stage["name"],
                            config_path=config,
                        ).restore(
                            generation, manifest_sha256=receipt["manifest_sha256"]
                        )
                        check = torch.load(
                            restored / latest["checkpoint"],
                            map_location="cpu",
                            weights_only=True,
                        )
                        actual = torch.load(
                            directory / latest["checkpoint"],
                            map_location="cpu",
                            weights_only=True,
                        )
                        if check["schema"] != CHECKPOINT_SCHEMA or not frozen.same(
                            check, actual
                        ):
                            raise ValueError(
                                "Independent Drive checkpoint readback differs"
                            )
                        write(
                            local
                            / "readback-receipts"
                            / (stage["name"] + "-" + generation + ".json"),
                            result,
                        )
                write(
                    root / (mode + "-status.json"),
                    dict(
                        event=event,
                        stage=stage["name"],
                        update=update,
                        rows=len(rows),
                        updated_at=time.time(),
                    ),
                )
                if mode == "production":
                    progress = read(root / "progress.json")
                    progress.update(
                        phase="production_" + event,
                        active_stage=stage["name"],
                        durable_update=update,
                        updated_at=time.time(),
                    )
                    write(root / "progress.json", progress)

            result = runner.train_temporal_model(
                architecture="query",
                phase="finetune",
                manifest=root / "prepared/train_manifest.json",
                evaluation_manifest=root / "prepared/evaluation_manifest.json",
                model_config=model,
                output=output,
                budget=budget,
                runtime=CoordinateRuntime(**plan["runtime"]),
                device=torch.device("cuda"),
                augmentation=augmentation,
                distance_px=20.0,
                persist=persist,
                **options,
            )
        if mode == "probe":
            rows = control.validate_log(stage, output)[5500:]
            write(
                root / "probe-acceptance.json",
                dict(
                    complete=len(rows) == 2 and result["updates_completed"] == 5502,
                    rows=rows,
                    continuity=read(output / "continuity.json"),
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    recipe_exact_except_documented_paths=True,
                    peak_allocated_bytes=max(r["peak_vram_bytes"] for r in rows),
                    peak_reserved_bytes=max(
                        r["peak_reserved_vram_bytes"] for r in rows
                    ),
                ),
            )
            return
        if not result["complete"]:
            raise ValueError("Stage returned before 6000 updates")
        validate_result(control, plan, root, output, stage)
        index.publish_file(output / "acceptance.json")
        first = False
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    write(local / "COMPLETED.json", dict(complete=True, model="length", updates=12000))
    index.publish_file(local / "COMPLETED.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "probe", "production"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    if args.mode == "prepare":
        prepare(root, source)
        return
    stop = threading.Event()

    def host_guard():
        minimum = 2**63
        while not stop.wait(1):
            available = (
                int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            minimum = min(minimum, available)
            write(
                root / (args.mode + "-host-memory.json"),
                dict(minimum_available_bytes=minimum, available_bytes=available),
            )
            if available < 4 * 1024**3:
                write(
                    root / (args.mode + "-RESOURCE_STOP.json"),
                    dict(
                        reason="Host available RAM below 4 GiB",
                        available_bytes=available,
                    ),
                )
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return

    thread = threading.Thread(target=host_guard, daemon=True)
    thread.start()
    try:
        with (root / "desktop-writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, source, args.mode, args.rclone_config)
    except BaseException:
        write(
            root / (args.mode + "-FAILED.json"),
            dict(complete=False, traceback=traceback.format_exc(), time=time.time()),
        )
        raise
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
