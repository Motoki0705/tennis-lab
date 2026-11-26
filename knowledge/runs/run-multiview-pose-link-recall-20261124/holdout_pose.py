"""Prepare and publish all-row ViTPose for the reserved three-clip evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import torch
from holdout_inputs import inspect
from hydra import compose, initialize_config_dir
from probe import read_json, read_output

from src.tasks.person_tracking.features import FeatureConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonFeaturesInput,
    PersonPoseFeaturesModule,
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

_RUN = Path(__file__).resolve().parent
_ROOT = _RUN.parents[2]
_DEV_POSE = _ROOT / "knowledge/runs/run-i964-coco-person-features-r7-20260930/plan.json"
_SOLVER_COMMIT = "466ba35cfacc496a9b8b895de2f85fad6a8dd849"
_DETECTION = ArtifactCodec(PersonDetectionOutput)
_POSE = ArtifactCodec(PersonPoseFeaturesOutput)
_SCHEMA = "meiji_3cam_unseen_vitpose_plan_v1"


def _solver_config() -> dict[str, Any]:
    subprocess.run(
        ["git", "diff", "--exit-code", _SOLVER_COMMIT, "--", "src"],
        cwd=_ROOT,
        check=True,
    )
    config: dict[str, Any] = read_json(_RUN / "meiji3.json")["config"]
    if config != json_value(DEFAULT_CONFIG):
        raise ValueError("The frozen solver configuration changed")
    return config


def _pose_module() -> tuple[PersonPoseFeaturesModule, dict[str, Any]]:
    original = read_json(_DEV_POSE)
    expected = original["models"]["pose"]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(_ROOT / "src/tennis_scene/configs"),
    ):
        composed = compose(
            config_name="pipeline",
            overrides=[
                f"paths.project_root={_ROOT}",
                "device=cuda",
                "output_directory=holdout_vitpose",
                "people_models.runtime.vitpose.batch_size=4",
            ],
        )
    config = PipelineRuntimeConfig.from_config(composed, bind_inputs=False).people
    runtime = json_value(config.runtime.vitpose)
    for field, value in expected["runtime"].items():
        if runtime[field] != value:
            raise ValueError(f"The development ViTPose setting changed: {field}")
    if runtime["precision"] != "fp32" or runtime["engines"]:
        raise ValueError("Expected the development ViTPose PyTorch FP32 profile")
    checkpoint = config.vitpose_checkpoint
    if dual_sha256(checkpoint) != expected["sha256"]:
        raise ValueError(
            "The ViTPose checkpoint differs from the development checkpoint"
        )
    features = FeatureConfig(**original["feature_config"])
    module = PersonPoseFeaturesModule(config, features)
    identity = {
        "checkpoint": {"path": str(checkpoint), "sha256": expected["sha256"]},
        "runtime": runtime,
        "preprocessing": module.cache_identity(),
    }
    return module, identity


def prepare(dataset: Path, archive: Path, output: Path) -> dict[str, Any]:
    config = _solver_config()
    attestation = inspect(dataset, archive)
    if attestation != read_json(_RUN / "holdout-inputs.json"):
        raise ValueError(
            "The reserved raw inputs differ from their checked-in evidence"
        )
    _, model = _pose_module()
    clips = {}
    for clip, facts in attestation["clips"].items():
        directory = archive / clip / "store"
        index = read_json(directory / "scene.json")
        source = restore_type(index["source"], ClipSource)
        refs = {}
        for video in source.videos:
            name = f"person_detection/{video.camera_id}"
            ref = index["artifacts"][name]
            if dual_sha256(directory / ref["path"]) != ref["sha256"]:
                raise ValueError(
                    f"Archived raw detection changed: {clip}/{video.camera_id}"
                )
            refs[video.camera_id] = ref
        clips[clip] = {"source": index["source"], **facts, "detections": refs}
    plan = {
        "schema": _SCHEMA,
        "solver_commit": _SOLVER_COMMIT,
        "solver_config": config,
        "r18_plan_sha256": attestation["r18_plan_sha256"],
        "dataset": str(dataset),
        "archive": str(archive),
        "pose_model": model,
        "clips": clips,
    }
    if output.exists():
        if read_json(output) != plan:
            raise ValueError("A different plan already exists at the requested path")
    else:
        write_json_atomic(output, plan)
    count = sum(sum(clip["raw_dino_rows"].values()) for clip in clips.values())
    print(
        f"Pinned {len(clips)} clips, {sum(len(c['detections']) for c in clips.values())} views and {count} DINO rows",
        flush=True,
    )
    print(f"plan_sha256={dual_sha256(output)}", flush=True)
    return plan


def infer(plan_path: Path, expected_sha: str, output: Path) -> None:
    if dual_sha256(plan_path) != expected_sha:
        raise ValueError("The pose plan does not match its pinned digest")
    plan = read_json(plan_path)
    if (
        plan["schema"] != _SCHEMA
        or plan["solver_commit"] != _SOLVER_COMMIT
        or plan["solver_config"] != _solver_config()
    ):
        raise ValueError("Expected the frozen holdout pose plan")
    completion = output / "complete.json"
    if completion.exists():
        raise FileExistsError(
            f"A completed holdout ViTPose run already exists: {completion}"
        )
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_GPU_SLOT"):
        raise RuntimeError("ViTPose requires an admitted training-queue allocation")

    module, model = _pose_module()
    if model != plan["pose_model"]:
        raise ValueError("The ViTPose identity changed since preparation")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    dataset, archive = Path(plan["dataset"]), Path(plan["archive"])
    keys = tuple(
        (clip, name)
        for clip, entry in plan["clips"].items()
        for name in entry["detections"]
    )
    specs = module.model_specs()
    checkpoint = Path(model["checkpoint"]["path"])
    results: dict[str, dict[str, Any]] = {}
    with InferenceRuntime(
        {key: specs for key in keys},
        expected_assets={
            key: {checkpoint: model["checkpoint"]["sha256"]} for key in keys
        },
    ) as inference:
        for clip, entry in plan["clips"].items():
            source = restore_type(entry["source"], ClipSource)
            directory = archive / clip / "store"
            index = read_json(directory / "scene.json")
            if document_digest(source) != index["source_sha256"]:
                raise ValueError(f"Source identity changed: {clip}")
            target = ClipStore(output / "stores" / clip, entry["source"])
            video_id, clip_id = clip.split("/")
            views: dict[str, Any] = {}
            for raw_video in source.videos:
                camera = raw_video.camera_id
                media = (
                    dataset
                    / "videos"
                    / video_id
                    / "clips"
                    / clip_id
                    / "media"
                    / raw_video.path.name
                )
                if dual_sha256(media) != raw_video.sha256:
                    raise ValueError(f"Video bytes changed: {clip}/{camera}")
                video = replace(raw_video, path=media)
                node = f"person_detection/{camera}"
                pinned = entry["detections"][camera]
                if index["artifacts"][node] != pinned:
                    raise ValueError(f"Archived raw detection changed: {clip}/{camera}")
                detections, dependencies = read_output(
                    directory,
                    index,
                    node,
                    "person_detections",
                    2,
                    PersonDetectionOutput,
                )
                rows = require_alignment(video, detections)
                if dependencies or len(rows) != entry["raw_dino_rows"][camera]:
                    raise ValueError(f"Incorrect raw detection rows: {clip}/{camera}")
                identity = {
                    "holdout_plan_sha256": expected_sha,
                    "raw_detection": pinned,
                }
                det = target.publish(
                    node,
                    detections,
                    _DETECTION,
                    schema="person_detections",
                    version=2,
                    identity=identity,
                    dependencies={},
                    provenance={"archive_plan_sha256": plan["r18_plan_sha256"]},
                )
                pose_node = f"person_pose_features/{camera}"
                pose_ref = target.active(pose_node)
                key = (clip, camera)
                links = {"detections": det}
                if pose_ref is None:
                    with inference.node(key) as models:
                        pose = module.process_with_models(
                            PersonFeaturesInput(video, detections), models
                        )
                    require_alignment(video, detections, pose)
                    pose_ref = target.publish(
                        pose_node,
                        pose,
                        _POSE,
                        schema="person_pose_features",
                        version=2,
                        identity=identity,
                        dependencies=links,
                        provenance={
                            "pose_model": model,
                            "queue_job": os.environ["TENNIS_RUN_ID"],
                        },
                    )
                else:
                    descriptor = target.descriptor(pose_ref)
                    if (
                        (pose_ref.schema, pose_ref.version)
                        != ("person_pose_features", 2)
                        or descriptor["identity"] != identity
                        or descriptor["dependencies"] != json_value(links)
                    ):
                        raise ValueError(
                            f"The existing pose belongs to different inputs: {clip}/{camera}"
                        )
                    inference.discard((key,))
                restored = target.load(pose_ref, _POSE)
                require_alignment(video, detections, restored)
                views[camera] = {"rows": len(rows), "pose": json_value(pose_ref)}
                print(f"{clip}/{camera}: {len(rows)} verified ViTPose rows", flush=True)
            results[clip] = views
    write_json_atomic(
        completion, {"plan_sha256": expected_sha, "pose_model": model, "clips": results}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--archive", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("infer")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        prepare(args.dataset.resolve(), args.archive.resolve(), args.output.resolve())
    else:
        infer(args.plan.resolve(), args.plan_sha256, args.output.resolve())


if __name__ == "__main__":
    main()
