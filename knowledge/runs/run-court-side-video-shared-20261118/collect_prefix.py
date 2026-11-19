"""Restore four measured ball streams and collect the missing X3D prefix for #2241."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source, resolve_dataset_clip, select_dataset_clips
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.paths import PROJECT_ROOT

COURT_DIGEST = "4ac8e386763687fcb6c81f46abf0774e38fc118af4f79b31cdb6e3c2283e3f75"
DRIFT_DIGEST = "7a4be8bf20397552a7b287131c3cb0e5fa57ba0dedc16e4a05fa23287c81a7f6"
THREE = {"clip_000", "clip_027", "clip_037"}


def collect(args):
    work = Path.cwd()
    root = args.outputs.resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        cfg = compose(config_name="pipeline_x3d_dpt", overrides=[
            f"paths.project_root={work}", f"paths.data_root={work / 'data'}", f"paths.checkpoint_root={work / 'ckpt'}",
            f"paths.artifact_root={root}", f"paths.external_asset_root={work / 'third_party'}",
            "person_observations.enabled=false", "player_reconstruction.enabled=false", "gvhmr.enabled=false",
            "court_kp.mode=video_shared", "court_kp.video_shared.action=load", "court_kp.video_shared.dataset_id=meiji_4cam",
            "court_kp.video_shared.video_id=video_000", f"court_kp.video_shared.artifact_digest={COURT_DIGEST}",
            "court_kp.video_shared.directory=court-video-shared/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store",
            "court_kp.video_shared.drift_directory=court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117/drift-store",
            f"court_kp.video_shared.drift_digest={DRIFT_DIGEST}",
        ])
    runtime = PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)
    pipeline = TennisSceneOrchestrator(runtime)
    court = pipeline.prepare_video_court()
    dataset = (work / "data/tennis_multiview/processed/meiji_4cam/dataset").resolve()
    stats = {}
    for record in select_dataset_clips(dataset):
        clip = Path(record.clip_id).name
        bound = resolve_dataset_clip(dataset, record)
        inputs = prepare_clip_source(bound.video_paths, bound.camera_ids, clip_id=record.clip_id)
        source = inputs.source
        out = args.run / "stores" / clip
        if clip in THREE:
            out = root / "video-batch-cli/tennis_scene/evaluate/video-batch-doubles-20261118/datasets/batch" / record.path / "annotations/tennis_scene"
        elif clip == "clip_018" and not args.preflight and not out.exists():
            shutil.copytree(args.old_x3d, out)
        if clip in THREE or clip == "clip_018":
            saved = (args.old_x3d if clip == "clip_018" and args.preflight else out) / "scene.json"
            original = restore_type(json.loads(saved.read_text())["source"], ClipSource)
            if original.clip_id != source.clip_id or [(v.camera_id, v.sha256, v.width, v.height, v.fps, v.num_frames) for v in original.videos] != [
                (v.camera_id, v.sha256, v.width, v.height, v.fps, v.num_frames) for v in source.videos
            ]:
                raise ValueError(f"Archived source mismatch: {clip}")
            source = original
        if clip in THREE:
            targets = []
        elif clip == "clip_018":
            targets = ["court_calibration"]
        else:
            targets = ["court_calibration", *(f"ball_points/{camera}" for camera in source.camera_ids)]
        if args.preflight:
            if clip == "clip_001":
                standard_definition(runtime, source, video_court=court, code_identity=pipeline.code_identity)
            continue
        times = {}
        if targets:
            nodes = standard_definition(runtime, source, video_court=court, code_identity=pipeline.code_identity)
            runner = ComponentRunner(nodes, ClipStore(out, json_value(source), memory_entries=1))
            for target in targets:
                write_json_atomic(args.run / "prefix/heartbeat.json", {"monotonic": time.monotonic(), "clip": clip, "target": target})
                runner.run(targets=[target])
                print(clip, target, runner.statuses[target], flush=True)
            times = runner.seconds
        stats[record.clip_id] = {"directory": str(out.relative_to(root)), "targets": targets, "seconds": times}
        inputs.snapshot.assert_unchanged()
        write_json_atomic(args.run / "prefix-results.json", stats)
    print("validated dataset", len(select_dataset_clips(dataset)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--old-x3d", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        collect(args)
    else:
        with Path(os.environ["TENNIS_STACK_FILE"]).open("a") as stack:
            faulthandler.enable(file=stack, all_threads=True)
            faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
            try:
                collect(args)
            except BaseException:
                faulthandler.dump_traceback(file=stack, all_threads=True)
                with (args.run / "gpu-at-exception.log").open("w") as report:
                    subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "-q"], stdout=report, stderr=report, timeout=10, check=False)
                raise
