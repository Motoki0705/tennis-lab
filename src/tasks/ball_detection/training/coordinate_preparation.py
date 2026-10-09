"""Freeze one data preparation and enumerate 36 models without starting training."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.tasks.ball_detection.data.coordinate_manifest import (
    build_coordinate_manifest,
    shared_pose_reference,
)
from src.tasks.ball_detection.data.coordinate_snapshot import snapshot_ball_store
from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.data.temporal_sampling import TemporalSamplingConfig
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig
from src.tasks.ball_detection.models.mdd_pose.variants import coordinate_variants
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore
from src.utils.checksum import dual_sha256

from .coordinate_provenance import coordinate_source_identity


def prepare_coordinate_training(
    ball_directory: Path, pose_directory: Path, model_config: Path, output: Path, *,
    play_config: PlayIntervalConfig,
    sampling: TemporalSamplingConfig,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Hardlink RGB, copy small pose artifacts, freeze labels and evaluation scopes."""
    code = coordinate_source_identity()
    base = MDDPoseConfig.load(model_config)
    reference, approved = shared_pose_reference(BallFrameStore(ball_directory), pose_directory)
    pose_reader = PlayerPoseStore(pose_directory)
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "PREPARATION_INCOMPLETE"
    incomplete.write_text("No training plan is available until experiments.json is published.\n")
    frozen, shard_hashes = snapshot_ball_store(
        ball_directory, output / "ball_snapshot",
        progress=(lambda done, total: progress("jpeg_hashes", done, total)) if progress is not None else None,
    )
    frozen_poses = output / "poses"
    frozen_poses.mkdir()
    for index, (clip_id, entry) in enumerate(approved.items()):
        # This validates approval decision, artifact/review hashes, all pose
        # arrays, observation masks and their exact original frame/PTS sequence.
        if pose_reader.read_clip(clip_id) is None:
            raise ValueError(f"Approved pose was not readable: {clip_id}")
        reference_clip = pose_reader.ball_store.clip_by_id(clip_id)
        current_clip = frozen.clip_by_id(clip_id)
        old_rgb = pose_reader.ball_store.directory / "shards" / shard_name(reference_clip.index)
        new_rgb = frozen.directory / "shards" / shard_name(current_clip.index)
        if not old_rgb.samefile(new_rgb) and dual_sha256(old_rgb) != shard_hashes[clip_id]:
            raise ValueError(f"Shared evaluation RGB differs from the pose input snapshot: {clip_id}")
        for field, checksum in (("file", "sha256"), ("review_file", "review_sha256")):
            fragment = Path(entry[field])
            source, target = pose_directory / fragment, frozen_poses / fragment
            if (fragment.is_absolute() or not source.resolve().is_relative_to(pose_directory.resolve())
                    or not target.resolve().is_relative_to(frozen_poses.resolve())):
                raise ValueError("Pose artifact path leaves its declared dataset")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if dual_sha256(target) != entry[checksum]:
                raise ValueError("Pose artifact changed while freezing training data")
        if progress is not None:
            progress("approved_poses", index + 1, len(approved))
    if dual_sha256(pose_directory / "manifest.json") != reference["pose_manifest_sha256"]:
        raise ValueError("Pose approval membership changed during preparation")
    shutil.copyfile(pose_directory / "manifest.json", output / "pose_source_manifest.json")
    manifests = {}
    for kind in ("mdd_pose", "mdd_only"):
        manifest = build_coordinate_manifest(
            frozen, play_config, sampling, input_kind=kind,
            pose_directory=frozen_poses if kind == "mdd_pose" else None,
            approved_poses=approved if kind == "mdd_pose" else None,
            common_reference=reference, shard_hashes=shard_hashes,
        )
        for split in ("train", "val", "test"):
            selected = [r for r in manifest["clips"] if r["clip"]["split"] == split]
            for step in sampling.frame_steps:
                if not any(w["frame_step"] == step for r in selected for w in r["windows"]):
                    raise ValueError(f"No {kind}/{split} windows at frame step {step}")
                if split != "train" and not any(
                    w["frame_step"] == step for r in selected if r["common_evaluation"] for w in r["windows"]
                ):
                    raise ValueError(f"No identity-matched common {split} windows at frame step {step}")
        path = output / f"{kind}_windows.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifests[kind] = dict(path=str(path), sha256=dual_sha256(path), counts=manifest["counts"],
                               clips=len(manifest["clips"]), skipped_clips=len(manifest["skipped"]))
    models = output / "models"
    models.mkdir()
    experiments = []
    for variant in coordinate_variants(base):
        kind = "mdd_pose" if variant.config.requires_pose else "mdd_only"
        path = models / f"{variant.name}.yaml"
        path.write_text(yaml.safe_dump(dict(name="mdd_pose", **asdict(variant.config)), sort_keys=False))
        experiments.append(dict(id=variant.name, model_config=str(path), model_config_sha256=dual_sha256(path),
                                input_kind=kind, manifest=manifests[kind]["path"],
                                command_template=[".venv/bin/python", "-m", "src.tasks.ball_detection.scripts.train_mdd_pose",
                                                  "--manifest", manifests[kind]["path"], "--model-config", str(path),
                                                  "--output", "<new-absolute-training-run>", "--epochs", "<epochs>",
                                                  "--learning-rate", "<learning-rate>", "--seed", "<seed>",
                                                  "--device", "cuda", "--precision", "bf16",
                                                  "--compile-mode", "default", "--selection-scope", "common",
                                                  "--jpeg-decoder", "nvjpeg", "--image-prefetch",
                                                  "--input-verification", "upfront", "--num-workers", "8",
                                                  "--prefetch-factor", "4", "--pin-memory"]))
    if coordinate_source_identity()["source_sha256"] != code["source_sha256"]:
        raise ValueError("Coordinate implementation changed during data preparation")
    plan = dict(schema="mdd_coordinate_experiments.v1", models=experiments, datasets=manifests,
                code=code,
                play_config=asdict(play_config), sampling=asdict(sampling),
                training_policy=dict(fps_mix="equal frame-step quotas, shuffled within each FPS",
                                     epoch_windows="native-FPS training window count unless explicitly overridden",
                                     initialization="random", gpu_launch="repository shared training queue only"),
                evaluation_policy=dict(scopes=["full", "common"], by_frame_step=True, by_source=True,
                                       checkpoint_selection="common val; equal mean of per-FPS means",
                                       test="explicit post-selection evaluator; excluded from training"),
                reference=reference, status="prepared; no training started")
    (output / "experiments.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    incomplete.unlink()
    return plan
