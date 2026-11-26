"""Make a real update from serialized SceneResult labels and a v3 checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.model_io.checkpoint import CHECKPOINT_SCHEMA
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner
from src.tasks.ball_refiner_3d.post_training.dataset import (
    PREPARED_SCHEMA,
    prepare_clip,
    save_clip,
    sha256_file,
)
from src.tasks.ball_refiner_3d.post_training.labels import Annotations, Event
from src.tasks.ball_refiner_3d.post_training.motion import BodySkeleton
from src.tasks.ball_refiner_3d.scripts.post_train import main
from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.ball_event_schema import ball_event_metadata
from src.tennis_scene.schema import SceneResult
from src.utils.io import write_json_atomic

pytestmark = pytest.mark.integration


def _archive(path: Path) -> SceneResult:
    frames = 12
    frame_valid: NDArray[np.bool_] = np.ones(frames, dtype=np.bool_)
    root = np.tile(np.array([[[1, -4, 1]]], np.float32), (1, frames, 1))
    positions = np.tile(np.array([[1, -3, 2]], np.float32), (frames, 1))
    positions[4] = 0
    frame_valid[4] = False
    joints_valid: NDArray[np.bool_] = np.zeros((1, frames, 17), dtype=np.bool_)
    scene = SceneResult(
        frames,
        60000 / 1001,
        640,
        360,
        np.zeros((2, frames, 14, 2), np.float32),
        np.ones((2, frames, 14), np.float32),
        root,
        np.zeros((1, frames), np.float32),
        ball_uv=np.zeros((2, frames, 2), np.float32),
        ball_vis=np.repeat(frame_valid[None], 2, axis=0),
        ball_3d=positions,
        ball_3d_valid=frame_valid,
        ball_rejection_code=(~frame_valid).astype(np.uint8),
        player_track_ids=np.array([5], np.int32),
        human_kp_2d=np.zeros((1, 2, frames, 17, 2), np.float32),
        human_kp_vis=np.ones((1, 2, frames, 17), np.float32),
        player_kp_3d=np.zeros((1, frames, 17, 3), np.float32),
        player_kp_3d_vis=joints_valid,
        player_kp_3d_rejection_code=(~joints_valid).astype(np.uint8),
        player_observed=np.ones((1, frames), dtype=np.bool_),
        player_valid=np.ones((1, frames), dtype=np.bool_),
        player_rejection_code=np.zeros((1, frames), dtype=np.uint8),
        player_heading_valid=np.zeros((1, frames), dtype=np.bool_),
        player_smpl_valid=np.zeros((1, frames), dtype=np.bool_),
        metadata={
            "scene_schema_version": 3,
            "player_sides": [-1],
            "court_reference": {},
            "ball_events": ball_event_metadata(
                "annotation",
                {
                    "code_commit": "a" * 40,
                    "annotation_sha256": "b" * 64,
                    "labeler": "fixture",
                    "labeler_original": "Fixture",
                },
            ),
        },
        ball_event_frame=np.array([2, 5, 8], np.int32),
        ball_event_kind=np.array([0, 1, 2], np.uint8),
        ball_event_status=np.array([0, 1, 0], np.uint8),
        ball_event_coverage=np.ones(frames, dtype=np.bool_),
        ball_event_in_point=np.array([True, True, False]),
        ball_event_frame_uncertainty=np.array([8, 5, 3], np.uint8),
        ball_event_subtype=np.array([0, 7, 4], np.uint8),
        ball_event_side=np.array([-1, 0, 0], np.int8),
    )
    save_scene_result(scene, path)
    return load_scene_result(path)


def test_motion_runner_uses_worktree_links_and_scores_two_label_sources(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    worktree_link = tmp_path / "worktree-shared"
    worktree_link.symlink_to(shared, target_is_directory=True)
    prepared = shared / "prepared"
    scene = _archive(tmp_path / "scene.npz")
    skeleton = BodySkeleton(
        np.zeros((22, 3), np.float32), np.arange(22, dtype=np.int32) - 1
    )
    reference = Annotations(
        (Event(6, "bounce", "certain", True, None),),
        np.ones(scene.num_frames, dtype=np.bool_),
    )
    clips = {}
    for role in ("train", "val", "test"):
        clip = prepare_clip(
            scene, role, skeleton, reference if role == "test" else None
        )
        clips[role] = save_clip(prepared / "clips" / f"{role}.npz", clip)
    manifest_sha256 = "f" * 64
    write_json_atomic(
        prepared / "inputs.json",
        {
            "manifest_sha256": manifest_sha256,
            "schema": PREPARED_SCHEMA,
            "splits": {role: [role] for role in clips},
            "clips": clips,
        },
    )
    config = ModelConfig(
        3, "regression", 16, 1, 2, 0.05, 2, 32, 8, 10000, "swiglu", False
    )
    checkpoint = shared / "ball.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "fps": 60,
            "event_classes": list(EVENT_CLASSES),
            "event_sigma_frames": 2.0,
            "model_config": asdict(config),
            "model": RegressionRefiner(config).state_dict(),
        },
        checkpoint,
    )
    output = shared / "fit"
    command = [
        "run",
        "--dataset",
        str(worktree_link / prepared.name),
        "--manifest-sha256",
        manifest_sha256,
        "--prepared-sha256",
        sha256_file(prepared / "inputs.json"),
        "--checkpoint",
        str(worktree_link / checkpoint.name),
        "--checkpoint-sha256",
        sha256_file(checkpoint),
        "--output",
        str(worktree_link / output.name),
        "--mode",
        "set-attention",
        "--seed",
        "42",
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    try:
        main(command)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    run = json.loads((output / "run.json").read_text())
    assert run["state"] == "complete" and run["best_epoch"] == 1
    weights = torch.load(output / "best.pt", weights_only=True)
    assert weights["fusion"]["interaction.output.weight"].count_nonzero() > 0
    assert weights["identity"]["checkpoint_sha256"] == sha256_file(checkpoint)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["annotation"]["all"]["by_class"]["net"]["events"] == 1
    assert metrics["annotation"]["in_point"]["by_class"]["net"]["events"] == 0
    assert metrics["independent"]["all"]["by_class"]["bounce"]["events"] == 1
    with np.load(output / "predictions" / "test.npz") as prediction:
        assert prediction["manifest_sha256"].item() == manifest_sha256
        assert prediction["fps"].item() == scene.fps
        np.testing.assert_array_equal(prediction["frame"], np.arange(scene.num_frames))
        assert not prediction["ball_valid"][4]
        assert prediction["event_logits"].shape == (scene.num_frames, 3)
