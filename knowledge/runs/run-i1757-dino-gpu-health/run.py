"""Run DINO on the pinned probe camera, first without and then with GPU polling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "run-i1757-video-batch-meiji"))

from benchmark import (  # noqa: E402
    REPO,
    RESOURCES,
    SPEC,
    common_root,
    compose_config,
    git,
    pinned_inputs,
    read,
    sha,
    verify_head,
    write,
)
from health_guard import (  # noqa: E402
    QUERY_TIMEOUT_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    HealthError,
    query_health,
    terminate_variant,
    utc,
)

CLIP = "video_000/clip_027"
CAMERA = "cam0"
HEARTBEAT_SECONDS = 60
CPU_WAIT_SECONDS = 30 * 60
CPU_SAMPLE_SECONDS = 15
DEVICE = "cuda:0"


def verify_checkout(commit):
    verify_head(commit)
    if git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        str(RUN.relative_to(REPO)),
    ):
        raise RuntimeError("Commit the DINO diagnostic before preparing or running it")


def detector(root):
    from src.tennis_scene.pipeline.components.person_detection import (
        PersonDetectionModule,
    )

    with compose_config(root) as (_, _, cfg):
        return PersonDetectionModule(
            cfg.people,
            enabled=cfg.enabled["person_observations"],
            merge_duplicates=cfg.merge_duplicate_person_boxes,
        )


def model_pin(specs):
    from src.tennis_scene.pipeline.inference_runtime import model_identities
    from src.tennis_scene.pipeline.model_assets import file_identity

    return model_identities(specs, file_identity)["detector"]


def prepare(root, previous, commit):
    from src.submodules.models.dino.extension import validate_dino_extension

    original = read(previous / "preflight.json")
    source = pinned_inputs()
    clip = next(
        record for record in source["selected_records"] if record["clip_id"] == CLIP
    )
    media_path = f"{clip['path']}/media/{CAMERA}.mp4"
    media = next(entry for entry in source["files"] if entry["path"] == media_path)
    original_video = previous / "probe" / "dataset" / media_path
    if (
        sha(original_video) != media["sha256"]
        or clip["num_frames"] != original["plans"][CLIP]["frames"]
    ):
        raise ValueError("Probe video or frame count disagrees with the frozen input")
    extension = validate_dino_extension()
    if {"path": str(extension), "sha256": sha(extension)} != original["native_dino"]:
        raise ValueError("DINO extension differs from the previous probe")
    expected = original["publication_identity"]["models"]["person_detection"][
        "detector"
    ]
    bound = detector(root)
    if (
        bound.config.runtime.dino_detector.precision != "fp16"
        or model_pin(bound.model_specs()) != expected
    ):
        raise ValueError("DINO declaration differs from the previous fp16 probe")
    if expected["device"] != "cuda" or expected["dtype"] != "float16":
        raise ValueError("This diagnostic requires the pinned CUDA fp16 declaration")
    root.mkdir(parents=True, exist_ok=False)
    video = root / "cam0.mp4"
    shutil.copyfile(original_video, video)
    if sha(video) != media["sha256"]:
        raise ValueError("Copied video differs from the probe")
    write(
        root / "pins.json",
        {
            "commit": commit,
            "probe_commit": original["commit"],
            "clip": clip,
            "media": media,
            "model": expected,
            "native_dino": original["native_dino"],
            "seed": SPEC["seed"],
        },
    )


def worker(root, condition):
    import random

    directory = root / condition
    progress = {
        "frame_index": None,
        "frames_completed": 0,
        "last_completed_frame": None,
        "forwards_completed": 0,
    }

    def heartbeat(stage):
        write(
            directory / "heartbeat.json",
            {
                "stage": stage,
                **progress,
                "at_utc": utc(),
                "monotonic": time.monotonic(),
            },
        )

    heartbeat("importing")
    import cv2
    import numpy as np
    import torch

    from src.submodules.models.dino import person_detector
    from src.tennis_scene.pipeline.components import person_detection
    from src.tennis_scene.pipeline.contracts import SourceVideo
    from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
    from src.utils.inference_recording import (
        InferenceRecorder,
        model_event,
        recording_context,
    )
    from src.utils.video import probe_video_info

    pins = read(root / "pins.json")
    clip = pins["clip"]
    video = root / "cam0.mp4"
    heartbeat("verifying_input_and_assets")
    cv2.setNumThreads(1)
    info = probe_video_info(video)
    if (
        sha(video) != pins["media"]["sha256"]
        or (info.frame_count, info.width, info.height)
        != (clip["num_frames"], clip["width"], clip["height"])
        or abs(info.fps - clip["fps"]) > 1e-6
    ):
        raise ValueError("Video no longer matches the probe timeline")
    source = SourceVideo(
        CAMERA,
        video,
        pins["media"]["sha256"],
        info.frame_count,
        info.fps,
        info.width,
        info.height,
    )
    bound = detector(root)
    specs = bound.model_specs()
    if model_pin(specs) != pins["model"]:
        raise ValueError("DINO assets or settings changed since preparation")
    key = (CLIP, f"person_detection/{CAMERA}")
    expected_assets = {
        Path(asset["path"]): asset["sha256"]
        for asset in pins["model"]["assets"].values()
    }

    random.seed(pins["seed"])
    np.random.seed(pins["seed"])
    torch.manual_seed(pins["seed"])
    heartbeat("cuda_initialization")
    total_memory = torch.cuda.get_device_properties(DEVICE).total_memory
    torch.cuda.set_per_process_memory_fraction(
        RESOURCES["allocator_limit_gib"] * 1024**3 / total_memory, DEVICE
    )
    write(
        directory / "device.json",
        {
            "name": torch.cuda.get_device_name(DEVICE),
            "total_bytes": total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    reader, span = person_detection.OpenCVVideoFrameReader, person_detector.span

    def memory():
        progress["memory_allocated_bytes"] = torch.cuda.memory_allocated(DEVICE)
        progress["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(DEVICE)

    def observe_frames(*args, **kwargs):
        progress["frame_index"] = 0
        heartbeat("decode")
        for packet in reader(*args, **kwargs):
            if packet.index != progress["frames_completed"]:
                raise ValueError("Decoder changed the frame sequence")
            progress["frame_index"] = packet.index
            heartbeat("predict")
            yield packet
            progress.update(
                frames_completed=packet.index + 1, last_completed_frame=packet.index
            )
            heartbeat("sampling_memory")
            memory()
            heartbeat("frame_complete")
            progress["frame_index"] = (
                packet.index + 1 if packet.index + 1 < info.frame_count else None
            )
            heartbeat("decode")

    @contextmanager
    def observe_span(stage, **attributes):
        heartbeat(f"dino_{stage}")
        with span(stage, **attributes) as event:
            yield event
        if stage == "inference":
            progress["forwards_completed"] += 1
            heartbeat("dino_forward_complete")

    @contextmanager
    def observe_model(action, model, **attributes):
        heartbeat(f"model_{action}")
        with model_event(action, model, **attributes):
            yield
        memory()
        heartbeat(f"model_{action}_complete")

    heartbeat("initializing_recorder")
    with (
        patch.object(person_detection, "OpenCVVideoFrameReader", observe_frames),
        patch.object(person_detector, "span", observe_span),
        InferenceRecorder(directory / "recording.json", devices=(DEVICE,)),
        recording_context(clip_id=CLIP, camera_id=CAMERA, node=key[1]),
        InferenceRuntime(
            {key: specs},
            expected_assets={key: expected_assets},
            model_event=observe_model,
        ) as runtime,
        runtime.node(key) as models,
    ):
        heartbeat("binding_detector")
        result = bound.process_with_models(
            person_detection.PersonDetectionInput(source), models
        )

    if (progress["frames_completed"], progress["forwards_completed"]) != (
        info.frame_count,
        info.frame_count,
    ):
        raise RuntimeError("DINO did not complete one forward for every frame")
    heartbeat("complete")
    write(
        directory / "completed.json",
        {**progress, "detections": len(result.confidence), "finished_at": utc()},
    )


def pytest_command(args):
    if not args:
        return False
    name = Path(args[0]).name
    if name in {"pytest", "py.test"}:
        return True
    return name.startswith("python") and (
        (len(args) >= 2 and Path(args[1]).name in {"pytest", "py.test"})
        or any(args[i : i + 2] == ["-m", "pytest"] for i in range(1, len(args) - 1))
    )


def cpu_load():
    import psutil

    tree_parent = common_root() / ".claude" / "worktrees"
    pids, other_worktree_pids = [], []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "status"]):
        info = process.info
        if info["status"] == psutil.STATUS_ZOMBIE or not pytest_command(
            info["cmdline"]
        ):
            continue
        pids.append(info["pid"])
        if info["cwd"]:
            cwd = Path(info["cwd"])
            if cwd.is_relative_to(tree_parent) and not cwd.is_relative_to(REPO):
                other_worktree_pids.append(info["pid"])
    return {
        "load_1m": os.getloadavg()[0],
        "pytest_count": len(pids),
        "pytest_pids": pids,
        "other_worktree_pytest_count": len(other_worktree_pids),
        "other_worktree_pytest_pids": other_worktree_pids,
    }


def wait_for_cpu(directory):
    start = time.monotonic()
    status = None
    with (directory / "cpu-load.jsonl").open("x") as log:
        while True:
            load = cpu_load()
            elapsed = time.monotonic() - start
            allowed = load["load_1m"] < 2 and load["pytest_count"] == 0
            record = {
                "at_utc": utc(),
                **load,
                "waited_seconds": elapsed,
                "threshold_met": allowed,
            }
            print(json.dumps(record), file=log, flush=True)
            if allowed or elapsed >= CPU_WAIT_SECONDS:
                return record
            if status != (load["pytest_count"], load["load_1m"] < 2):
                print(
                    f"Condition {directory.name} waiting for CPU: {record}", flush=True
                )
                status = (load["pytest_count"], load["load_1m"] < 2)
            time.sleep(min(CPU_SAMPLE_SECONDS, CPU_WAIT_SECONDS - elapsed))


def host_resources(directory):
    import psutil

    return {
        "host": socket.gethostname(),
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(directory).free,
    }


def heartbeat_at(directory):
    path = directory / "heartbeat.json"
    return read(path) if path.is_file() else None


def run_condition(root, condition, commit):
    directory = root / condition
    directory.mkdir(exist_ok=False)
    load = wait_for_cpu(directory)
    start, started_at = time.monotonic(), utc()
    write(
        directory / "started.json",
        {
            "condition": condition,
            "started_at": started_at,
            "cpu": load,
            "host_resources": host_resources(directory),
            "heartbeat_timeout_seconds": HEARTBEAT_SECONDS,
            "external_time_limit_seconds": RESOURCES["probe_limit_seconds"],
            "gpu_query_interval_seconds": SAMPLE_INTERVAL_SECONDS
            if condition == "B"
            else None,
            "gpu_query_timeout_seconds": QUERY_TIMEOUT_SECONDS
            if condition == "B"
            else None,
        },
    )
    print(f"Condition {condition} started: {directory}", flush=True)

    command = [
        sys.executable,
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(root),
        "--condition",
        condition,
        "--expect-commit",
        commit,
    ]
    environment = {
        **os.environ,
        "PYTHONHASHSEED": str(SPEC["seed"]),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    query_samples, error, termination_pending = 0, None, False
    with (
        (directory / "worker.log").open("x") as log,
        (directory / "gpu-queries.jsonl").open("x") as query_log,
    ):
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        next_query = start
        try:
            while process.poll() is None:
                now = time.monotonic()
                stamp = heartbeat_at(directory)
                if now - (stamp["monotonic"] if stamp else start) >= HEARTBEAT_SECONDS:
                    error = "heartbeat_timeout"
                    break
                if now - start >= RESOURCES["probe_limit_seconds"]:
                    error = "external_time_limit"
                    break
                if condition == "B" and now >= next_query:
                    sample_time = utc()
                    try:
                        used, driver = query_health()
                    except (HealthError, OSError) as exc:
                        error = (
                            exc.kind
                            if isinstance(exc, HealthError)
                            else "gpu_query_failed_to_start"
                        )
                        print(
                            json.dumps(
                                {
                                    "at_utc": sample_time,
                                    "error": error,
                                    "detail": str(exc),
                                }
                            ),
                            file=query_log,
                            flush=True,
                        )
                        break
                    print(
                        json.dumps(
                            {"at_utc": sample_time, "used_mib": used, "driver": driver}
                        ),
                        file=query_log,
                        flush=True,
                    )
                    query_samples += 1
                    next_query = now + SAMPLE_INTERVAL_SECONDS
                wait = (
                    min(0.25, max(0, next_query - time.monotonic()))
                    if condition == "B"
                    else 0.25
                )
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=wait)
            stopped_at, elapsed = utc(), time.monotonic() - start
            last = heartbeat_at(directory)
            if error:
                write(
                    directory / "stop.json",
                    {"stopped_at": stopped_at, "error": error, "last_heartbeat": last},
                )
            elif process.returncode != 0:
                error = "worker_failed"
        finally:
            if process.poll() is None:
                termination_pending = terminate_variant(process)

    completed_path = directory / "completed.json"
    complete = read(completed_path) if completed_path.exists() else None
    expected_frames = read(root / "pins.json")["clip"]["num_frames"]
    if not error and (
        not complete
        or (complete["frames_completed"], complete["forwards_completed"])
        != (expected_frames, expected_frames)
    ):
        error = "missing_complete_timeline"
    result = {
        "condition": condition,
        "status": "failed" if error else "complete",
        "error": error,
        "started_at": started_at,
        "stopped_at": stopped_at,
        "elapsed_seconds": elapsed,
        "returncode": process.poll(),
        "termination_pending": termination_pending,
        "cpu": load,
        "last_heartbeat": last,
        "gpu_query_samples": query_samples,
    }
    write(directory / "terminal.json", result)
    return result


def collect_log(directory, name, command):
    try:
        result = subprocess.run(command, capture_output=True, timeout=15, check=False)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = exc.output or b"", exc.stderr or b"", "timeout"
    except OSError as exc:
        stdout, stderr, code = b"", str(exc).encode(), "failed_to_start"
    (directory / f"{name}.log").write_bytes(stdout)
    (directory / f"{name}.stderr.log").write_bytes(stderr)
    return code


def collect_driver_logs(directory, result):
    start = datetime.fromisoformat(result["started_at"]) - timedelta(seconds=5)
    end = datetime.fromisoformat(utc())
    journal = [
        "journalctl",
        "-k",
        "--no-pager",
        "--output=short-iso",
        "--grep=dxg",
        "--since",
        start.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "--until",
        end.strftime("%Y-%m-%d %H:%M:%S UTC"),
    ]
    windows = f"""
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$start = [DateTimeOffset]::Parse('{start.isoformat()}').LocalDateTime
$end = [DateTimeOffset]::Parse('{end.isoformat()}').LocalDateTime
try {{
  $events = @(Get-WinEvent -FilterHashtable @{{
    LogName='System'; ProviderName='nvlddmkm'; StartTime=$start; EndTime=$end
  }} -ErrorAction Stop | ForEach-Object {{
    [pscustomobject]@{{
      utc=$_.TimeCreated.ToUniversalTime().ToString('o'); id=$_.Id
      record_id=$_.RecordId; provider=$_.ProviderName; message=$_.Message
    }}
  }})
  ConvertTo-Json -InputObject $events -Compress
}} catch {{
  if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {{ '[]' }}
  else {{ [Console]::Error.WriteLine($_.ToString()); exit 1 }}
}}
"""
    codes = {
        "journalctl": collect_log(directory, "kernel-dxg", journal),
        "Get-WinEvent": collect_log(
            directory,
            "windows-nvlddmkm",
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                windows,
            ],
        ),
    }
    write(
        directory / "driver-log-status.json",
        {"from_utc": start.isoformat(), "until_utc": end.isoformat(), "codes": codes},
    )


def run(root, commit):
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("Run GPU diagnostics through the shared training queue")
    if read(root / "pins.json")["commit"] != commit:
        raise ValueError("Prepared pins belong to a different checkout")
    results = {}
    for condition in ("A", "B"):
        result = run_condition(root, condition, commit)
        results[condition] = result
        collect_driver_logs(root / condition, result)
        write(
            root / "results.json",
            {
                "job_id": os.environ["TENNIS_RUN_ID"],
                "commit": commit,
                "conditions": results,
            },
        )
        print(json.dumps(result), flush=True)
        if result["status"] != "complete":
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--condition", choices=("A", "B"))
    parser.add_argument("--expect-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (
        not root.is_relative_to((REPO / "outputs").resolve())
        or root == (REPO / "outputs").resolve()
    ):
        parser.error("--root must be below the current worktree outputs")
    verify_checkout(args.expect_commit)
    if args.action == "prepare":
        if not args.previous:
            parser.error("prepare requires the previous probe run directory")
        prepare(root, args.previous.resolve(strict=True), args.expect_commit)
    elif args.action == "worker":
        if (
            not args.condition
            or not os.environ.get("TENNIS_RUN_ID")
            or not os.environ.get("TENNIS_REPRO_DIR")
        ):
            parser.error("worker is launched by the training queue supervisor")
        worker(root, args.condition)
    else:
        return run(root, args.expect_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
