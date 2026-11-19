"""Prepare the pinned three-clip benchmark and run it through the shared GPU queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUN.parent / "run-i1751-runtime-gpu-a5"))

from health_guard import (  # noqa: E402
    HealthError,
    query_health,
    run_variant,
    utc,
    write,
)

SPEC = json.loads((RUN / "conditions.json").read_text())
CLIPS = tuple(SPEC["clips"])
RESOURCES = SPEC["resources"]
GIB = 1024**3


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def common_root():
    return Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def attempts(root):
    return [root / "attempts" / f"{index}-{regime}" for index, regime in enumerate(SPEC["regimes"], 1)]


def pinned_inputs():
    source = read(RUN / SPEC["input_manifest"])
    if (source["dataset_id"], source["video_id"], set(source["clip_order"])) != (
        SPEC["dataset_id"], SPEC["video_id"], set(CLIPS)
    ):
        raise ValueError("Input manifest differs from the benchmark scope")
    return source


def verify_dataset(dataset):
    from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest

    pin = pinned_inputs()
    manifest = load_dataset_manifest(dataset)
    if manifest.dataset_id != SPEC["dataset_id"] or set(manifest.clips) != set(CLIPS):
        raise ValueError(f"Wrong benchmark dataset registration: {dataset}")
    if tuple(manifest.clips) != CLIPS:
        raise ValueError(f"Dataset registration order changed: {dataset}")
    expected = {record["clip_id"]: record for record in pin["selected_records"]}
    if any(manifest.clips[key].to_dict() != expected[key] for key in CLIPS):
        raise ValueError(f"Pinned clip registration changed: {dataset}")
    for item in pin["files"]:
        path = dataset / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Pinned input changed: {path}")
    return manifest


def freeze(root):
    from dataclasses import replace

    from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest

    source = (REPO / SPEC["source_dataset"]).resolve(strict=True)
    manifest = load_dataset_manifest(source)
    pin = pinned_inputs()
    if manifest.dataset_id != SPEC["dataset_id"]:
        raise ValueError(f"Wrong source dataset: {source}")
    for record in pin["selected_records"]:
        if manifest.clips[record["clip_id"]].to_dict() != record:
            raise ValueError(f"Source registration changed: {record['clip_id']}")
    if shutil.disk_usage(root.parent).free < RESOURCES["min_free_disk_gib"] * GIB:
        raise RuntimeError("Insufficient disk space for benchmark stores")
    root.mkdir(exist_ok=False)
    snapshot = root / "input"
    snapshot.mkdir()
    for item in pin["files"]:
        relative = Path(item["path"])
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    replace(manifest, clips={key: manifest.clips[key] for key in CLIPS}).save(snapshot)
    verify_dataset(snapshot)
    for directory in [root / "probe", *attempts(root)]:
        shutil.copytree(snapshot, directory / "dataset")
    write(root / "input.json", {
        "source_dataset": SPEC["source_dataset"], "selected_records": pin["selected_records"],
        "ordered_clips": CLIPS, "files": pin["files"],
        "dataset_manifest_sha256": sha(snapshot / "dataset.json"),
        "pin_sha256": sha(RUN / SPEC["input_manifest"]),
    })


@contextmanager
def compose_config(directory):
    from hydra import compose, initialize_config_dir

    from src.tennis_scene.configuration import parse_video_batch_config
    from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config

    with initialize_config_dir(config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"):
        batch_cfg = compose(config_name="run_video_batch", overrides=[
            f"paths.data_root={directory}", f"paths.output_root={directory}",
            f"paths.artifact_root={common_root() / '.worktree-outputs'}",
            "dataset_directory=dataset", f"video_id={SPEC['video_id']}", "progress_directory=progress",
            f"pipeline_config={SPEC['pipeline_config']}", "overwrite=false", "continue_on_error=true",
            f"clip_ids={json.dumps(CLIPS)}", f"pipeline_overrides={json.dumps(SPEC['pipeline_overrides'])}",
        ])
        parsed = parse_video_batch_config(batch_cfg)
        pipeline_cfg, runtime = _compose_pipeline_config(parsed.generation, config_name=parsed.pipeline_config)
        if (pipeline_cfg.court_side.evidence != "player" or runtime.video_court is None
                or runtime.video_court_side is not None or runtime.ball_detection is None
                or not runtime.enabled["ball_reconstruction"]
                or runtime.people.runtime.dino_detector.precision != "fp16"
                or runtime.people.runtime.vitpose.precision != "fp16"
                or runtime.player_association.players_per_side != 2):
            raise ValueError("Composed settings differ from the pinned experiment")
        yield batch_cfg, pipeline_cfg, runtime


def host_resources(root):
    import psutil

    values = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    name, total, used, free, driver = next(csv.reader([values.strip()], skipinitialspace=True))
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(root)
    result = {"gpu": name, "device_total_mib": int(total), "device_used_mib": int(used),
              "device_free_mib": int(free), "driver": driver,
              "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
              "disk_free_bytes": disk.free}
    if (int(free) < RESOURCES["min_free_vram_mib"]
            or memory.available < RESOURCES["min_available_ram_gib"] * GIB
            or disk.free < RESOURCES["min_free_disk_gib"] * GIB):
        raise RuntimeError(f"Insufficient resources for the pinned benchmark: {result}")
    return result


def preflight(root, expected_head):
    from omegaconf import OmegaConf

    from src.submodules.models.dino.extension import validate_dino_extension
    from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs
    from src.tennis_scene.pipeline.definition import standard_definition
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator

    directory = attempts(root)[0]
    dataset = directory / "dataset"
    verify_dataset(dataset)
    native_dino = validate_dino_extension()
    plans = {}
    with compose_config(directory) as (_, pipeline_cfg, runtime):
        orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)
        shared_court = orchestrator.prepare_video_court()
        identity = orchestrator.publication_identity()
        _, clips = prepare_video_inputs(dataset, SPEC["video_id"], CLIPS)
        for clip in clips:
            source = clip.media.source
            if source.camera_ids != tuple(SPEC["camera_ids"]):
                raise ValueError(f"Unexpected cameras: {source.camera_ids}")
            nodes = standard_definition(runtime, source, code_identity=orchestrator.code_identity,
                                        video_court=shared_court)
            by_name = {node.name: node for node in nodes}
            side_inputs = sorted(by_name["court_side"].bindings.values())
            if (not any(name.startswith(("player_selection/", "person_tracking/", "person_detection/"))
                        for name in side_inputs) or any(name.startswith("ball_") for name in side_inputs)):
                raise ValueError(f"Player court_side route is missing: {side_inputs}")
            ball_nodes = [f"{stage}/{camera}" for stage in ("ball_detection", "ball_points")
                          for camera in SPEC["camera_ids"]] + ["ball_triangulation"]
            if any(by_name[name].source != "execute" for name in ball_nodes):
                raise ValueError("Benchmark requires all ball nodes to execute")
            plans[source.clip_id] = {"frames": source.num_frames, "camera_ids": source.camera_ids,
                                     "court_side_inputs": side_inputs, "nodes": list(by_name)}
        (root / "pipeline.yaml").write_text(OmegaConf.to_yaml(pipeline_cfg, resolve=True))
    write(root / "preflight.json", {
        "commit": expected_head, "conditions_sha256": sha(RUN / "conditions.json"),
        "tolerances_sha256": sha(RUN / "tolerances.json"),
        "input_sha256": sha(root / "input.json"), "publication_identity": identity, "plans": plans,
        "native_dino": {"path": str(native_dino), "sha256": sha(native_dino)}, "resources": host_resources(root),
    })


def verify_head(expected_head):
    if not expected_head or git("rev-parse", "HEAD") != expected_head:
        raise RuntimeError("Use --expect-commit with the complete SHA of the measured checkout")
    if git("status", "--porcelain", "--untracked-files=all", "--", "src", str(RUN.relative_to(REPO))):
        raise RuntimeError("Commit the complete source and benchmark before execution")


def verify_pins(root, expected_head):
    verify_head(expected_head)
    preflight = read(root / "preflight.json")
    if (preflight["commit"], preflight["conditions_sha256"], preflight["tolerances_sha256"],
            preflight["input_sha256"]) != (
            expected_head, sha(RUN / "conditions.json"), sha(RUN / "tolerances.json"), sha(root / "input.json")):
        raise ValueError("Run does not match the preflight commit and pins")
    extension = Path(os.environ["TENNIS_LAB_DINO_EXTENSION"]).resolve(strict=True)
    if preflight["native_dino"] != {"path": str(extension), "sha256": sha(extension)}:
        raise ValueError("DINO extension changed since CPU preflight")
    return preflight["publication_identity"]


def require_empty_store(dataset):
    manifest = verify_dataset(dataset)
    if any((dataset / record.path / "annotations").exists() for record in manifest.clips.values()):
        raise ValueError(f"Cold component stores are required: {dataset}")


def serial_run(directory, pipeline_cfg, orchestrator, identity):
    from omegaconf import OmegaConf

    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    from src.tennis_scene.generate_dataset.pseudo_annotation import (
        generate_pseudo_annotations,
    )
    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.dataset_publication import ANNOTATION_RELATIVE_DIR
    from src.tennis_scene.pipeline.errors import is_clip_failure
    from src.utils.configuration import PathRole
    from src.utils.inference_recording import recording_context

    dataset = directory / "dataset"
    config_yaml = OmegaConf.to_yaml(pipeline_cfg, resolve=True)
    outcomes = {}

    def clip_runner(paths, cameras, clip_directory):
        return orchestrator.run(paths, camera_ids=cameras, video_role=PathRole.DATA,
                                store_root=clip_directory / ANNOTATION_RELATIVE_DIR,
                                clip_id=ClipManifest.load(clip_directory).clip_id)

    for clip_id in CLIPS:
        with recording_context(clip_id=clip_id):
            try:
                outcome, = generate_pseudo_annotations(
                    dataset, clip_runner, clip_ids=[clip_id], publication_identity=identity,
                    pipeline_config_yaml=config_yaml, continue_on_error=False,
                    scene_export=orchestrator.completed_scene_export,
                )
            except Exception as error:
                if not is_clip_failure(error):
                    raise
                outcomes[clip_id] = {
                    "status": "failed", "reason": getattr(error, "reason", type(error).__name__),
                    "error": str(error), "diagnostics": json_value(getattr(error, "diagnostics", {})),
                }
            else:
                outcomes[clip_id] = {"status": outcome.status}
        write(directory / "outcomes.json", outcomes)
        print(f"{clip_id} {outcomes[clip_id]['status']}", flush=True)
    return outcomes


def batch_run(directory, batch_cfg, identity):
    from src.tennis_scene.pipeline import video_batch as pipeline
    from src.tennis_scene.scripts import run_video_batch as cli

    run = pipeline.run_video_batch
    progress_paths = []

    def recorded(*args, **kwargs):
        result = run(*args, **kwargs, recording_devices=("cuda:0",))
        progress_paths.append(result.progress_path)
        return result

    # Invoke the actual Hydra entry with its C1 GPU timing enabled through the public runner option.
    with patch.object(pipeline, "run_video_batch", recorded):
        try:
            cli.main(batch_cfg)
        except SystemExit as exit_code:
            if not progress_paths or exit_code.code != 1:
                raise
    progress_path, = progress_paths
    progress = read(progress_path)
    if progress["publication_identity"] != identity:
        raise ValueError("Batch run weights/configuration differ from CPU preflight")
    write(directory / "outcomes.json", progress["clips"])
    return progress["clips"], progress_path.with_name("recording.json")


def probe_run(directory, orchestrator):
    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs
    from src.tennis_scene.pipeline.definition import standard_definition
    from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
    from src.tennis_scene.pipeline.runner import PlanExecutor
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore

    clip_id = "video_000/clip_027"
    _, (clip,) = prepare_video_inputs(directory / "dataset", SPEC["video_id"], [clip_id])
    source = clip.media.source
    nodes = standard_definition(orchestrator.config, source, code_identity=orchestrator.code_identity,
                                video_court=orchestrator.prepare_video_court())
    store = ClipStore(directory / "probe-store", json_value(source), memory_entries=0)
    targets = ("ball_detection/cam0", "person_appearance_features/cam0", "person_pose_features/cam0")
    plan = ExecutionPlan([ClipExecution(clip_id, tuple(nodes), store, False, targets)])
    PlanExecutor(plan).run()
    if any(plan.statuses[(clip_id, node)] != "executed" for node in targets):
        raise ValueError("Every GPU probe target must run cold")
    return {node: status for (_, node), status in plan.statuses.items()}


@contextmanager
def observe(phase, active):
    import faulthandler
    import threading

    import src.utils.inference_recording as recording

    previous = 0.0
    owner = threading.get_ident()
    heartbeat_path = phase / "worker" / "heartbeat.json"

    with Path(os.environ["TENNIS_STACK_FILE"]).open("a") as stack:
        faulthandler.enable(file=stack, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)

        def heartbeat(operation):
            nonlocal previous
            now = time.monotonic()
            if threading.get_ident() == owner and now - previous >= 1:
                write(heartbeat_path, {"monotonic": now, "at_utc": utc(), "operation": operation})
                previous = now

        span = recording.span

        @contextmanager
        def observed_span(stage, **attrs):
            if stage == "publication" and not attrs:
                attrs["operation"] = "publish_annotation"
            heartbeat({"stage": stage, **attrs})
            with span(stage, **attrs) as event:
                yield event
            if stage == "inference" and "batch_shape" in attrs:
                path = active[0] / "first-inference.json"
                if not path.exists():
                    record = {"at_utc": utc(), **event}
                    write(path, record)
                    print(f"First inference completed: {record}", flush=True)
            heartbeat({"completed": stage})

        heartbeat("worker initialization")
        try:
            with patch.object(recording, "span", observed_span):
                yield
        finally:
            faulthandler.unregister(signal.SIGUSR1)
            faulthandler.disable()


def worker(root, phase, condition, identity):
    import random

    probe = phase.name == "probe"
    directories = [root / "probe"] if probe else attempts(root)
    active = [directories[0]]
    worker_started = time.perf_counter()
    with observe(phase, active):
        import cv2
        import numpy as np
        import psutil
        import torch

        from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
        from src.utils.inference_recording import InferenceRecorder, span

        device = "cuda:0"
        process = psutil.Process()
        cv2.setNumThreads(1)
        startup_seconds = time.perf_counter() - worker_started
        for index, directory in enumerate(directories):
            active[0] = directory
            condition_name = "probe" if probe else condition
            regime = "probe" if probe else SPEC["regimes"][index]
            require_empty_store(directory / "dataset")
            started_at, initial_io = utc(), process.io_counters()
            started = time.perf_counter()
            random.seed(SPEC["seed"])
            np.random.seed(SPEC["seed"])
            torch.manual_seed(SPEC["seed"])
            if index == 0:
                capacity = torch.cuda.get_device_properties(device).total_memory
                torch.cuda.set_per_process_memory_fraction(RESOURCES["allocator_limit_gib"] * GIB / capacity, device)
            print(f"Running {condition_name} {regime} in {directory}", flush=True)
            recording = directory / "recording.json"
            with compose_config(directory) as (batch_cfg, pipeline_cfg, runtime):
                if not probe and condition == "reuse":
                    outcomes, recording = batch_run(directory, batch_cfg, identity)
                else:
                    with InferenceRecorder(recording, devices=(device,), metadata={
                        "condition": condition_name, "regime": regime, "clip_ids": CLIPS,
                    }):
                        orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)
                        with span("measurement", operation="publication_identity"):
                            if orchestrator.publication_identity() != identity:
                                raise ValueError("Run weights/configuration differ from CPU preflight")
                        outcomes = (probe_run(directory, orchestrator) if probe else
                                    serial_run(directory, pipeline_cfg, orchestrator, identity))
            elapsed = time.perf_counter() - started
            final_io = process.io_counters()
            write(directory / "measurement.json", {
                "variant": condition_name, "regime": regime, "started_at": started_at, "finished_at": utc(),
                "wall_seconds": elapsed, "recording": str(recording.relative_to(directory)), "outcomes": outcomes,
                "worker_startup_seconds": startup_seconds if index == 0 else 0,
                "io": {field: getattr(final_io, field) - getattr(initial_io, field)
                       for field in ("read_chars", "write_chars", "read_bytes", "write_bytes")},
                "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
            })


def run(root, condition, action, expected_head):
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("Enqueue real GPU measurements in the shared training queue")
    verify_pins(root, expected_head)
    phase = root / action
    directories = [root / "probe"] if action == "probe" else attempts(root)
    for directory in directories:
        require_empty_store(directory / "dataset")
    if action == "measurement":
        terminal = read(root / "probe" / "terminal.json")
        if terminal["commit"] != expected_head or terminal["status"] != "complete":
            raise RuntimeError("This checkout's resource probe must complete before the benchmark")
    if (phase / "started.json").exists():
        raise FileExistsError(f"A measurement was already started: {phase}")
    phase.mkdir(exist_ok=True)
    deadline = time.monotonic() + RESOURCES["probe_limit_seconds" if action == "probe" else "condition_limit_seconds"]

    def query():
        if time.monotonic() > deadline:
            raise HealthError("time_limit", f"Benchmark time limit elapsed: {action} {condition}")
        return query_health()

    job = {"commit": expected_head, "job_id": os.environ["TENNIS_RUN_ID"],
           "condition": condition, "phase": action, "host": host_resources(root)}
    write(phase / "started.json", job)
    env = {"PYTHONHASHSEED": str(SPEC["seed"]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "TORCHINDUCTOR_CACHE_DIR": str(phase / "inductor"),
           "TRITON_CACHE_DIR": str(phase / "triton"), "PYTHONPYCACHEPREFIX": str(phase / "pycache")}
    command = ["env", *[f"{name}={value}" for name, value in env.items()],
               sys.executable, "-B", "-u", str(Path(__file__).resolve()), "worker",
               "--root", str(root), "--condition", condition, "--phase", action, "--expect-commit", expected_head]
    print(f"GPU {action} {condition}: {phase / 'worker.log'}", flush=True)
    result = run_variant(phase, "worker", command, REPO, query=query)
    status = result["status"]
    if status == "complete" and action == "measurement" and any(
        outcome["status"] != "generated" for directory in directories
        for outcome in read(directory / "measurement.json")["outcomes"].values()
    ):
        status = "clips_unavailable"
    write(phase / "terminal.json", {**job, "status": status, "health": result})
    print(f"{action} {condition}: {status}", flush=True)
    return 0 if status == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "probe", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--condition", choices=SPEC["conditions"], default="serial")
    parser.add_argument("--expect-commit")
    parser.add_argument("--phase", choices=("probe", "measurement"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to((REPO / "outputs").resolve()) or root == (REPO / "outputs").resolve():
        parser.error("--root must be below this worktree's outputs")
    if args.action == "freeze":
        root.parent.mkdir(parents=True, exist_ok=True)
        freeze(root)
        return 0
    dependency = SPEC["conditions"][args.condition].get("wait_for")
    if dependency:
        parser.error(f"{args.condition} requires #{dependency} and its explicit adapter configuration")
    verify_head(args.expect_commit)
    if args.action == "preflight":
        preflight(root, args.expect_commit)
        return 0
    if args.action == "worker":
        if not args.phase or not os.environ.get("TENNIS_STACK_FILE") or not os.environ.get("TENNIS_RUN_ID"):
            parser.error("worker is invoked by the queue supervisor")
        worker(root, root / args.phase, args.condition, verify_pins(root, args.expect_commit))
        return 0
    return run(root, args.condition, "measurement" if args.action == "run" else "probe", args.expect_commit)


if __name__ == "__main__":
    raise SystemExit(main())
