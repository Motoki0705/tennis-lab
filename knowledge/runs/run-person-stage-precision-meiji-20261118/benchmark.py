"""Run the declared pipeline through COCO17 triangulation on two pinned clips."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from gpu_guard import heartbeat
from hydra import compose, initialize_config_dir

from src.submodules.models import DinoPersonDetector, ViTPosePose2D
from src.submodules.models.dino.extension import validate_dino_extension
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.clip_inputs import (
    prepare_clip_source,
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    file_identity,
    standard_definition,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.checksum import dual_sha256

HERE = Path(__file__).resolve().parent
CLIP_IDS = ("video_000/clip_037", "video_000/clip_000")
PREFIX = "people_models.runtime."
CONDITIONS = {
    "fp32": (),
    "vitpose_bf16": (PREFIX + "vitpose.precision=bf16",),
    "vitpose_fp16": (PREFIX + "vitpose.precision=fp16",),
    "dino_bf16": (PREFIX + "dino_detector.precision=bf16",),
    "dino_fp16": (PREFIX + "dino_detector.precision=fp16",),
    "vitpose_no_flip": (PREFIX + "vitpose.flip_test=false",),
}
DIAGNOSTIC_MARGIN = 0.12


class StageMeasurements:
    """Collect synchronous call timing and raw pose rows during standard execution."""

    def __init__(self, stage, directory, keep_appearance):
        self.stage = stage
        self.directory = directory
        self.counts = {}
        self.seconds = {}
        self.pose_rows = []
        self.appearance = [] if keep_appearance else None

    def timer(self, label, original):
        @wraps(original)
        def timed(*args, **kwargs):
            count = self.counts.get(label, 0) + 1
            heartbeat(f"{self.stage} {label} call {count}")
            started = perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.counts[label] = count
            self.seconds[label] = self.seconds.get(label, 0.0) + (perf_counter() - started)
            return result

        return timed

    def instrument(self, stack):
        for cls, method, label in (
            (DinoPersonDetector, "_predict_compatible_batch", "dino"),
            (ViTPosePose2D, "_predict_impl", "vitpose"),
            (UnpromptedEncoder, "embed", "clip_reid"),
        ):
            stack.enter_context(patch.object(cls, method, self.timer(label, getattr(cls, method))))
        if self.stage.startswith("person_tracking/"):
            original = FeatureExtractor.extract

            @wraps(original)
            def extract(*args, **kwargs):
                result = original(*args, **kwargs)
                self.pose_rows.append((result.rows, result.require_poses()))
                if self.appearance is not None:
                    self.appearance.append((result.embeddings, result.appearance_valid))
                return result

            stack.enter_context(patch.object(FeatureExtractor, "extract", extract))

    def save_poses(self, output):
        if self.pose_rows:
            camera = self.stage.split("/")[1]
            rows, poses = zip(*self.pose_rows, strict=True)
            np.savez(self.directory / f"pose-{camera}.npz", rows=np.concatenate(rows), keypoints=np.concatenate(poses))
            if self.appearance is not None:
                embeddings, valid = zip(*self.appearance, strict=True)
                np.savez(self.directory / f"appearance-{camera}.npz", rows=np.concatenate(rows),
                         embeddings=np.concatenate(embeddings), valid=np.concatenate(valid))
            pose_count = sum(len(row) for row in rows)
            if pose_count < int(output.observed.sum()):
                raise ValueError("Pose instrumentation lost tracked rows")
            return pose_count
        return 0


def config(args, condition):
    court = json.loads((HERE / "court-settings.json").read_text())
    overrides = [
        f"paths.artifact_root={args.shared_root}",
        f"paths.output_root={args.output}",
        "paths.project_root=.",
        "player_association.players_per_side=2",
        "gvhmr.enabled=false",
        "cache.overwrite=false",
        *court["overrides"],
        *CONDITIONS[condition],
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(Path.cwd() / "src/tennis_scene/configs")):
        composed = compose(config_name="pipeline_x3d_dpt", overrides=overrides)
    return PipelineRuntimeConfig.from_config(composed, bind_inputs=False)


def prepare_cases(args):
    cfg = config(args, "fp32")
    orchestrator = TennisSceneOrchestrator(cfg)
    shared = orchestrator.prepare_video_court()
    if shared is None:
        raise ValueError("The comparison requires fixed shared court geometry")
    clips = {}
    for record in select_dataset_clips(args.dataset, CLIP_IDS):
        clip = resolve_dataset_clip(args.dataset, record)
        inputs = prepare_clip_source(clip.video_paths, clip.camera_ids, clip_id=record.clip_id)
        source = inputs.source
        if len(source.camera_ids) != 4 or source.num_frames != record.num_frames:
            raise ValueError(f"{record.clip_id}: expected the full four-camera timeline")
        shared.bind(source)
        clips[record.clip_id] = (clip, inputs)
    pin = {
        "clips": {clip_id: {
            "manifest_sha256": clip.manifest_sha256, "media_sha256": dict(clip.media_sha256),
            "frames": inputs.source.num_frames, "camera_ids": list(inputs.source.camera_ids), "fps": inputs.source.fps,
        } for clip_id, (clip, inputs) in clips.items()},
        "weights": {key: file_identity(path)["sha256"] for key, path in enabled_model_assets(cfg).items()},
        "dino_binary_sha256": dual_sha256(Path(os.environ["TENNIS_LAB_DINO_EXTENSION"])),
        "dino_submodule_commit": subprocess.check_output(["git", "-C", "third_party/DINO", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": dual_sha256(Path("src/tennis_scene/configs/pipeline.yaml")),
        "ball_profile_sha256": dual_sha256(Path("src/tennis_scene/configs/pipeline_x3d_dpt.yaml")),
        "court_settings_sha256": dual_sha256(HERE / "court-settings.json"),
        "conditions": {name: list(overrides) for name, overrides in CONDITIONS.items()},
        "diagnostic_court_side_min_margin": DIAGNOSTIC_MARGIN,
        "nominal_court_side_min_margin": cfg.court_side.min_margin,
        "joint_confidence": cfg.joint_confidence,
        "human_visibility_threshold": cfg.human_vis_threshold,
        "seed": 42,
    }
    return pin, clips, orchestrator.code_identity, shared


def fork_baseline(baseline, destination):
    shutil.copytree(baseline, destination, copy_function=os.link,
                    ignore=shutil.ignore_patterns("scene.json", ".scene.lock"))
    shutil.copyfile(baseline / "scene.json", destination / "scene.json")


def run_nodes(runner, targets, run_dir, phase, case, keep_appearance):
    plan = runner.plan
    clip_id = runner.clip_id
    nodes = plan.target_nodes([(clip_id, name) for name in targets])
    result = {}
    for key in plan.order:
        if key not in nodes:
            continue
        _, stage = key
        missing = [name for parent_id, name in plan.dependencies[key].values() if (parent_id, name) not in plan.references]
        if missing:
            result[stage] = {"status": "blocked", "missing_dependencies": missing}
            continue
        heartbeat(f"{case} {phase} {stage} start")
        print(f"{case} {phase} {stage}", flush=True)
        measurement = StageMeasurements(stage, run_dir, keep_appearance)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        with ExitStack() as stack:
            measurement.instrument(stack)
            try:
                reference = runner.executor.execute(key)
            except ReconstructionUnavailable as error:
                result[stage] = {"status": "unavailable", "reason": error.reason, "message": str(error),
                                 "seconds": perf_counter() - started}
            else:
                torch.cuda.synchronize()
                status = plan.statuses[key]
                result[stage] = {
                    "status": status, "reference": json_value(reference), "seconds": perf_counter() - started,
                    "calls": measurement.counts, "inference_seconds": measurement.seconds,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                if measurement.pose_rows:
                    result[stage]["pose_rows"] = measurement.save_poses(runner.output(stage))
        write_json_atomic(run_dir / f"{phase}.json", result)
    return result


def run_case(cfg, clip_input, code_identity, shared, case_dir, baseline, clip_id, condition):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    case_dir.mkdir(parents=True, exist_ok=False)
    source = clip_input.source
    store_root = case_dir / "store"
    if condition != "fp32":
        fork_baseline(baseline, store_root)
    store = ClipStore(store_root, json_value(source), memory_entries=2)
    case = f"{condition}/{clip_id}"

    def run(cfg, targets, phase):
        runner = ComponentRunner(standard_definition(cfg, source, code_identity=code_identity, video_court=shared), store)
        return run_nodes(runner, targets, case_dir, phase, case, condition == "fp32" and phase == "nominal")

    targets = [f"pose_estimation/{camera}" for camera in source.camera_ids] + ["player_triangulation"]
    nominal = run(cfg, targets, "nominal")
    if clip_id == CLIP_IDS[0] and nominal["court_side"].get("reason") == "court_side_ambiguous_margin":
        diagnostic = replace(cfg, court_side=replace(cfg.court_side, min_margin=DIAGNOSTIC_MARGIN))
        print(f"{case} court_side_ambiguous_margin, diagnostic replay min_margin={DIAGNOSTIC_MARGIN}", flush=True)
        run(diagnostic, ["player_triangulation"], "diagnostic")
    clip_input.snapshot.assert_unchanged()
    write_json_atomic(case_dir / "complete.json", {"condition": condition, "clip_id": clip_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    args.dataset, args.shared_root, args.output = args.dataset.resolve(), args.shared_root.resolve(), args.output.resolve()
    if not args.freeze:
        heartbeat("verify smoke receipt and pinned full-clip inputs")
        if args.smoke is None or json.loads(args.smoke.read_text()) != {"status": "complete"}:
            raise ValueError("Require the completed GPU smoke receipt")
    pin, clips, code_identity, shared = prepare_cases(args)
    if args.freeze:
        write_json_atomic(args.pin, pin)
        return
    if json.loads(args.pin.read_text()) != pin:
        raise ValueError("Benchmark inputs changed")
    validate_dino_extension()
    write_json_atomic(args.output / "environment.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "run_id": os.environ["TENNIS_RUN_ID"], "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "code_identity": code_identity,
    })
    for condition in CONDITIONS:
        cfg = config(args, condition)
        for clip_id, (clip, inputs) in clips.items():
            heartbeat(f"{condition}/{clip_id} preparing isolated store")
            name = clip_id.split("/")[-1]
            run_case(cfg, inputs, code_identity, shared, args.output / condition / name,
                     args.output / "fp32" / name / "store", clip_id, condition)
            clip.snapshot.assert_unchanged()
    write_json_atomic(args.output / "complete.json", {"conditions": list(CONDITIONS), "clips": list(CLIP_IDS)})


if __name__ == "__main__":
    main()
