"""Check real-frame FP32 parity and exercise each precision before the clip run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gpu_guard import heartbeat
from hydra import compose, initialize_config_dir

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    Pose2DFrameSequenceRequest,
    Pose2DRequest,
    ViTPosePose2D,
)
from src.submodules.models.dino.extension import validate_dino_extension
from src.tasks.person_tracking.features import infer_poses
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.video import read_video_frame

CLIPS = ("clip_037", "clip_000")
BASE = "68b3bb4a491f6a7cca014c72d790a230a6a87894"


def samples(dataset):
    for clip in CLIPS:
        manifest = ClipManifest.load(dataset / "videos/video_000/clips" / clip)
        for camera in manifest.camera_ids:
            path = manifest.media_path(camera)
            index = manifest.num_frames // 2
            yield (clip, camera, index), path


def compose_runtime():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path.cwd() / "src/tennis_scene/configs")):
        return PipelineRuntimeConfig.from_config(compose(config_name="pipeline"), bind_inputs=False)


def pin_inputs(dataset, assets, binary):
    return {
        "samples": [{"clip": key[0], "camera_id": key[1], "frame": key[2],
                     "video_sha256": dual_sha256(path)} for key, path in samples(dataset)],
        "checkpoints": {path.name: dual_sha256(path) for path in assets},
        "dino_binary_sha256": dual_sha256(binary),
        "dino_submodule_commit": subprocess.check_output(
            ["git", "-C", "third_party/DINO", "rev-parse", "HEAD"], text=True,
        ).strip(),
    }


def previous_class(path, name, **request_types):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"])
    module = types.ModuleType(f"_legacy_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    # Feed the same typed requests to the historical model implementation.
    module.__dict__.update(request_types)
    return getattr(module, name)


def measure(model, frames, infer):
    heartbeat(f"load {type(model).__name__}")
    model.load()
    heartbeat(f"warmup {type(model).__name__}")
    infer(model, 0, frames[0])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = []
    elapsed = 0.0
    try:
        for i, frame in enumerate(frames):
            heartbeat(f"inference {type(model).__name__} sample {i}")
            started = perf_counter()
            results.append(infer(model, i, frame))
            torch.cuda.synchronize()
            elapsed += perf_counter() - started
        return results, {"seconds": elapsed, "calls": len(results),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    finally:
        heartbeat(f"unload {type(model).__name__}")
        model.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    cfg = compose_runtime()
    people = cfg.people
    assets = people.dino_checkpoint, people.vitpose_checkpoint
    binary = Path(os.environ["TENNIS_LAB_DINO_EXTENSION"])
    if args.freeze:
        write_json_atomic(args.pin, pin_inputs(args.dataset, assets, binary))
        return
    heartbeat("verify pinned input and DINO extension")
    expected = json.loads(args.pin.read_text())
    if expected != pin_inputs(args.dataset, assets, binary):
        raise ValueError("Smoke inputs changed")
    validate_dino_extension()
    torch.manual_seed(42)
    frames = [read_video_frame(path, key[2]).frame for key, path in samples(args.dataset)]
    summary = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "baseline_source_commit": BASE, "gpu": torch.cuda.get_device_name(),
               "torch": str(torch.__version__), "cuda": torch.version.cuda, "measurements": {}}
    report = summary["measurements"]

    def save():
        write_json_atomic(args.output, summary)

    dino_cfg = people.runtime.dino_detector
    detector_options = dict(device="cuda", confidence=dino_cfg.confidence,
                            short_side=dino_cfg.short_side, max_long_side=dino_cfg.max_long_side)

    old_detector = previous_class("src/submodules/models/dino/person_detector.py", "DinoPersonDetector")
    for name, detector in (
        ("old_dino_fp32", old_detector(people.dino_checkpoint, people.dino_repository, **detector_options)),
        ("dino_fp32", DinoPersonDetector(people.dino_checkpoint, people.dino_repository, **detector_options)),
    ):
        detections, report[name] = measure(detector, frames, lambda model, i, frame: model.predict(PersonDetectionRequest(frame)))
        save()
        if name == "old_dino_fp32":
            previous_detections = detections
    summary["dino_fp32_exact"] = all(
        np.array_equal(before.boxes_xyxy, after.boxes_xyxy) and np.array_equal(before.scores, after.scores)
        for before, after in zip(previous_detections, detections, strict=True)
    )
    summary["dino_rows_by_sample"] = [len(result.scores) for result in detections]
    save()

    vitpose_cfg = people.runtime.vitpose
    pose_options = dict(device="cuda", batch_size=vitpose_cfg.batch_size, head_config=vitpose_cfg.head, show_progress=False)
    old_pose = previous_class(
        "src/submodules/models/vitpose/pose2d.py", "ViTPosePose2D",
        Pose2DRequest=Pose2DRequest, Pose2DFrameSequenceRequest=Pose2DFrameSequenceRequest,
    )
    for name, cls, options in (
        ("old_vitpose_fp32", old_pose, {"flip_test": True}),
        ("vitpose_fp32", ViTPosePose2D, {"flip_test": True}),
        ("vitpose_bf16", ViTPosePose2D, {"flip_test": True, "precision": "bf16"}),
        ("vitpose_fp16", ViTPosePose2D, {"flip_test": True, "precision": "fp16"}),
    ):
        pose = cls(people.vitpose_checkpoint, **pose_options, **options)
        poses, report[name] = measure(pose, frames, lambda model, i, frame: infer_poses(
            i, frame, detections[i].boxes_xyxy, model, cfg.tracking.features,
        ))
        if name == "old_vitpose_fp32":
            old_poses = np.concatenate(poses)
        else:
            values = np.concatenate(poses)
            report[name]["coordinates_max_abs_px"] = float(np.abs(values[..., :2] - old_poses[..., :2]).max())
            report[name]["confidence_max_abs"] = float(np.abs(values[..., 2] - old_poses[..., 2]).max())
            if name == "vitpose_fp32":
                summary["vitpose_fp32_exact"] = bool(np.array_equal(values, old_poses))
        save()

    for precision in ("bf16", "fp16"):
        detector = DinoPersonDetector(people.dino_checkpoint, people.dino_repository, precision=precision, **detector_options)
        results, report[f"dino_{precision}"] = measure(detector, frames, lambda model, i, frame: model.predict(PersonDetectionRequest(frame)))
        if any(not np.isfinite(det.boxes_xyxy).all() or not np.isfinite(det.scores).all() for det in results):
            raise ValueError(f"Nonfinite DINO detections under {precision}")
        report[f"dino_{precision}"]["rows_by_sample"] = [len(det.scores) for det in results]
        save()

    pose = ViTPosePose2D(people.vitpose_checkpoint, flip_test=False, **pose_options)
    poses, report["vitpose_no_flip"] = measure(pose, frames, lambda model, i, frame: infer_poses(
        i, frame, detections[i].boxes_xyxy, model, cfg.tracking.features,
    ))
    values = np.concatenate(poses)
    report["vitpose_no_flip"]["coordinates_max_abs_px"] = float(np.abs(values[..., :2] - old_poses[..., :2]).max())
    report["vitpose_no_flip"]["confidence_max_abs"] = float(np.abs(values[..., 2] - old_poses[..., 2]).max())
    summary["status"] = "measured"
    save()
    if not summary["dino_fp32_exact"] or not summary["vitpose_fp32_exact"]:
        raise ValueError("The default GPU outputs changed")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
