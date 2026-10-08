"""Frozen inputs, mixed FPS, 36 plans, and a CPU-only CLI roundtrip."""

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.coordinate_manifest import (
    build_coordinate_manifest,
    shared_pose_reference,
)
from src.tasks.ball_detection.data.coordinate_snapshot import snapshot_ball_store
from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.data.temporal_sampling import (
    TemporalSamplingConfig,
    select_sampled_windows,
)
from src.tasks.ball_detection.model_io.mdd import luminance_to_mdd, mdd_coefficients
from src.tasks.ball_detection.training.coordinate_preparation import (
    prepare_coordinate_training,
)
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def pose_manifest(root: Path, directory: Path) -> Path:
    store = BallFrameStore(root)
    directory.mkdir()
    records = []
    for clip in store.clips:
        if not clip.clip_id.endswith("pose"):
            records.append(dict(clip_id=clip.clip_id, pose_status="skipped"))
            continue
        length = clip.frame_count
        file = directory / f"{clip.index}.npz"
        points = np.full((length, 1, 17, 3), .9, np.float32)
        points[..., :2] = 12
        rows = store.clip_rows(clip)
        np.savez(file, keypoints=points, observed=np.ones((length, 1), bool),
                 player_ids=np.array(["p1"]), frame_index=store.frames["frame_index"][rows], pts=store.frames["pts"][rows],
                 boxes_xyxy=np.tile([0., 0., 30., 30.], (length, 1, 1)).astype(np.float32),
                 raw_track_ids=np.ones((length, 1), np.int64), detection_rows=np.arange(length, dtype=np.int64)[:, None])
        review = directory / f"{clip.index}.json"
        review.write_text(json.dumps(dict(status="approved", clip_id=clip.clip_id, raw_tracks_sha256="raw-fixture")))
        records.append(dict(clip_id=clip.clip_id, pose_status="approved", file=file.name, sha256=dual_sha256(file),
                            review_file=review.name, review_sha256=dual_sha256(review), raw_tracks_sha256="raw-fixture"))
    manifest = dict(schema="ball_detection_player_poses.v1", coordinate_system="stored_jpeg_pixels",
                    ball_store=dict(directory=str(root), hashes={name: dual_sha256(root / name)
                                                               for name in ("metadata.json", "index.npz")}), clips=records)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_low_clip_presence_needs_no_pose_and_mdd_follows_sampled_rgb(tmp_path: Path) -> None:
    root = write_store_clip(tmp_path / "ball", "train/plain", [
        frame(i, ball(xy=(8., 9.))) if i < 128 else frame(i) for i in range(400)
    ], size=(32, 32))
    store = BallFrameStore(root)
    data = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig(), input_kind="mdd_only")
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(data))
    dataset = CoordinateWindowDataset(path, split="train", requires_pose=False, mdd_a=.2, mdd_b=.15)
    assert data["semantics"]["clip_presence_gate"] is None
    assert {window.frame_step for _, window in dataset.windows} == {1, 2, 4}
    index = next(i for i, (_, w) in enumerate(dataset.windows) if w.start == 0 and w.frame_step == 4)
    sample = dataset[index]
    assert "pose" not in sample and "pose_valid" not in sample
    torch.testing.assert_close(sample["frame_indices"], torch.arange(32) * 4)
    torch.testing.assert_close(sample["timestamps"], torch.arange(32).float() * 4 / 30)
    gray = []
    for frame_index in range(128):
        image = store.read_bgr(store.row_of(store.clips[0], frame_index)).astype(np.float32) / 255
        gray.append(.114 * image[..., 0] + .587 * image[..., 1] + .299 * image[..., 2])
    gray_tensor = torch.from_numpy(np.stack(gray))[None]
    gain, offset = mdd_coefficients(.2, .15)
    expected = luminance_to_mdd(gray_tensor[:, :125:4], gain=gain, offset=offset)[0]
    wrong_order = luminance_to_mdd(gray_tensor, gain=gain, offset=offset)[0, :, :125:4]
    torch.testing.assert_close(sample["mdd"], expected)
    assert not torch.allclose(sample["mdd"], wrong_order)
    batch = collate_coordinate_windows([sample])
    assert "pose" not in batch and batch["mdd"].shape == (1, 2, 32, 32, 32)
    sampler = FPSMixSampler(dataset, windows_per_epoch=30, seed=19)
    first = list(sampler)
    assert Counter(dataset.windows[i][1].frame_step for i in first) == {1: 10, 2: 10, 4: 10}
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_subsampled_windows_do_not_cross_timestamp_or_reference_barriers(tmp_path: Path) -> None:
    root = write_store_clip(tmp_path / "ball", "clip", [frame(i, ball()) for i in range(160)])
    store = BallFrameStore(root)
    states = annotation_states(store, store.clips[0])
    times = np.r_[np.arange(64) / 30, 10 + np.arange(96) / 30]
    spans, windows = select_sampled_windows(replace(states, times=times), PlayIntervalConfig(), TemporalSamplingConfig())
    assert spans == ((0, 64), (64, 160))
    assert all(not (w.indices()[0] < 64 <= w.indices()[-1]) for w in windows)
    assert not any(w.frame_step == 4 for w in windows)
    target = states.target.copy()
    target[64] = False
    _, windows = select_sampled_windows(replace(states, target=target), PlayIntervalConfig(), TemporalSamplingConfig())
    assert all(not (w.indices()[0] <= 64 <= w.indices()[-1]) for w in windows)


def test_common_identity_checks_gt_not_only_clip_id_or_declared_annotation_hash(tmp_path: Path) -> None:
    old = write_store_clip(tmp_path / "old", "train/pose", [frame(i, ball(xy=(8., 8.))) for i in range(128)])
    poses = pose_manifest(old, tmp_path / "poses")
    live = write_store_clip(tmp_path / "live", "train/pose", [frame(i, ball(xy=(9., 8.))) for i in range(128)])
    with pytest.raises(ValueError, match="GT/split/timeline"):
        shared_pose_reference(BallFrameStore(live), poses)


def test_snapshot_hardlinks_rgb_but_freezes_indexes_and_detects_modified_bytes(tmp_path: Path) -> None:
    source = write_store_clip(tmp_path / "ball", "clip", [frame(i, ball()) for i in range(128)])
    store, hashes = snapshot_ball_store(source, tmp_path / "snapshot")
    clip = store.clips[0]
    first = source / "shards" / shard_name(clip.index)
    frozen = store.directory / "shards" / shard_name(clip.index)
    assert first.samefile(frozen)
    assert not (source / "metadata.json").samefile(store.directory / "metadata.json")
    data = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig(), input_kind="mdd_only", shard_hashes=hashes)
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(data))
    with first.open("r+b") as stream:
        stream.seek(10)
        byte = stream.read(1)[0]
        stream.seek(10)
        stream.write(bytes([byte ^ 1]))
    dataset = CoordinateWindowDataset(path, split="train", requires_pose=False, mdd_a=.2, mdd_b=.15)
    with pytest.raises(ValueError, match="checksum"):
        dataset[0]


@pytest.fixture
def preparation(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "ball"
    for split in ("train", "val", "test"):
        for kind in ("pose", "plain"):
            length = 128 if kind == "pose" else 400
            write_store_clip(root, f"{split}/{kind}", [frame(i, ball(xy=(8., 9.))) if i < 128 else frame(i)
                                                       for i in range(length)], split=split, size=(32, 32))
    poses = pose_manifest(root, tmp_path / "poses")
    config = tmp_path / "model.yaml"
    config.write_text("name: mdd_pose\ncompression: conv2d\npose_pooling: attention\nreadout: query\nframes: 32\n"
                      "stem_channels: [4, 4, 8, 8]\nmixed_channels: [8, 8]\ndim: 16\nheads: 2\nlayers: 1\ndropout: 0.0\nrope_base: 10000.0\n")
    output = tmp_path / "prepared"
    plan = prepare_coordinate_training(root, poses, config, output,
                                       play_config=PlayIntervalConfig(), sampling=TemporalSamplingConfig())
    return output, plan


def test_preparation_freezes_36_models_and_pose_free_loading_survives_missing_poses(preparation: tuple[Path, dict]) -> None:
    output, plan = preparation
    assert len(plan["models"]) == 36 and len({m["id"] for m in plan["models"]}) == 36
    assert Counter(m["input_kind"] for m in plan["models"]) == {"mdd_pose": 32, "mdd_only": 4}
    assert plan["datasets"]["mdd_pose"]["clips"] == 3
    assert plan["datasets"]["mdd_only"]["clips"] == 6
    assert not (output / "PREPARATION_INCOMPLETE").exists()
    (output / "poses").rename(output / "hidden_poses")
    data = CoordinateWindowDataset(Path(plan["datasets"]["mdd_only"]["path"]), split="val", requires_pose=False, mdd_a=.2, mdd_b=.15)
    assert {data[i]["common_evaluation"] for i in range(len(data))} == {True, False}
    assert all("pose" not in data[i] for i in range(len(data)))


def test_cpu_training_and_explicit_test_evaluation_report_both_scopes(preparation: tuple[Path, dict], tmp_path: Path) -> None:
    _, plan = preparation
    variant = next(m for m in plan["models"] if m["id"] == "conv2d-query_only")
    output = tmp_path / "training"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train_mdd_pose",
                             "--manifest", variant["manifest"], "--model-config", variant["model_config"],
                             "--output", str(output), "--epochs", "1", "--learning-rate", ".001", "--seed", "5",
                             "--device", "cpu", "--windows-per-epoch", "3", "--batch-size", "2",
                             "--num-workers", "2"], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    progress = json.loads((output / "train.jsonl").read_text().splitlines()[-1])
    assert progress["global_step"] == 2 and progress["windows"] == 3
    assert np.isfinite(progress["train_loss"])
    report_path = tmp_path / "evaluation.json"
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate_mdd_coordinates",
                             "--checkpoint", str(output / "epoch-000.pt"), "--manifest", variant["manifest"],
                             "--output", str(report_path), "--artifact-root", str(tmp_path), "--output-root", str(tmp_path),
                             "--split", "test", "--device", "cpu"], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["split"] == "test"
    for step in ("1", "2", "4"):
        full = report["scopes"]["full"]["by_frame_step"][step]
        common = report["scopes"]["common"]["by_frame_step"][step]
        assert full["observed_frames"] == 2 * common["observed_frames"] > 0
        assert np.isfinite(full["mean_error_px"]) and np.isfinite(common["mean_error_px"])


def test_epoch_resume_matches_uninterrupted_training_and_rejects_changed_recipe(
    preparation: tuple[Path, dict], tmp_path: Path,
) -> None:
    _, plan = preparation
    variant = next(m for m in plan["models"] if m["id"] == "conv2d-query_only")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.train_mdd_pose",
               "--manifest", variant["manifest"], "--model-config", variant["model_config"],
               "--learning-rate", ".001", "--seed", "5", "--device", "cpu",
               "--windows-per-epoch", "6", "--batch-size", "2", "--num-workers", "2"]
    resumed, full = tmp_path / "resumed", tmp_path / "uninterrupted"
    for arguments in (["--output", str(resumed), "--epochs", "1"],
                      ["--output", str(resumed), "--epochs", "2", "--resume", str(resumed / "epoch-000.pt")],
                      ["--output", str(full), "--epochs", "2"]):
        result = subprocess.run(command + arguments, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    continued = torch.load(resumed / "epoch-001.pt", weights_only=True)
    reference = torch.load(full / "epoch-001.pt", weights_only=True)
    assert continued["training_state"]["global_step"] == 6
    assert continued["validation"] == reference["validation"]
    for name, tensor in reference["state_dict"].items():
        torch.testing.assert_close(continued["state_dict"][name], tensor, rtol=0, atol=0)
    for extra, expected in ((["--learning-rate", ".002"], "training recipe changed"),
                            ([], "latest completed epoch")):
        result = subprocess.run(command + ["--output", str(resumed), "--epochs", "3", "--resume",
                                           str(resumed / "epoch-000.pt"), *extra],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode != 0 and expected in result.stderr
