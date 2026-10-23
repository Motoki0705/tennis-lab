"""Snapshot/annotation freeze, pack loading, JPEG alignment and once-only scoring."""

import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    ALL_FRAME_MANIFEST_SCHEMA,
    build_all_frame_manifests,
)
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    verified_label_kinds,
)
from src.tasks.ball_detection.data.windows.dataset import (
    FrozenBallPackDataset,
    FrozenBallWindowDataset,
    collate_pack_samples,
    collate_window_samples,
)
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import build_coordinate_manifest
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    for name, count, split in (("sparse", 145, "train"), ("barriers", 300, "train"),
                               ("masked", 160, "train"), ("negative", 145, "train"),
                               ("short", 127, "train"), ("validation", 145, "val"), ("test", 145, "test")):
        if name == "sparse":
            frames = [frame(i, ball() if i == 2 else ball("out_of_frame", None) if i == 140 else ball("unresolved", None)) for i in range(count)]
        elif name == "masked":
            frames = [frame(i, ball("unresolved", None)) for i in range(count)]
        elif name == "negative":
            frames = [frame(i) for i in range(count)]
        else:
            examples = [frame(0, ball()), frame(0), frame(0, ball("out_of_frame", None)),
                        frame(0, ball("interpolated")), frame(0, ball("occlusion_estimated")),
                        frame(0, ball("unresolved", None)), frame(0, ball(), ball("out_of_frame", None, track="b002")),
                        frame(0, ball(), annotated=False)]
            frames = [replace(examples[i % len(examples)], pts=i) for i in range(count)]
        if name == "barriers":
            frames[128] = replace(frames[128], is_target=False)
        write_store_clip(root, f"tracknet/{name}/clip", frames, split=split, size=(32, 32))
    frames = [frame(i) if i in (10, 80) or i >= 64 else frame(i, ball()) for i in range(145)]
    write_store_clip(root, "chat_annotation/example", frames, source="chat_annotation", size=(32, 32))
    store = BallFrameStore(root)
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for clip in store.clips:
        rows = store.clip_rows(clip)
        indices: NDArray[np.int64] = np.arange(clip.frame_count, dtype=np.int64)
        pts = indices * 3003 + indices % 3
        if clip.clip_id == "tracknet/barriers/clip":
            pts[148:] += 90000
        arrays["pts"][rows] = pts
    np.savez(root / "index.npz", **arrays)
    metadata = json.loads((root / "metadata.json").read_text())
    annotation = tmp_path / "chat.json"
    annotation.write_text(json.dumps(dict(
        schema_version="tennis_chat_ball_annotation.v1", clip_id="example", frame_count=len(frames),
        frames=[dict(frame_index=i, reviewed=True, balls=[{}] * len(f.balls),
                     notes="unknown position" if i in (10, 80) else "") for i, f in enumerate(frames)],
    )))
    for entry in metadata["clips"]:
        entry.update(time_base="1/90000", fps="30000/1001")
        if entry["source"] == "chat_annotation":
            entry.update(annotation_path=str(annotation), annotation_sha256=dual_sha256(annotation))
    (root / "metadata.json").write_text(json.dumps(metadata))
    BallFrameStore(root)
    return root


@pytest.fixture
def prepared(snapshot: Path, tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "prepared"
    return output, prepare_all_frame_training(snapshot, output, configs=[AllFrameWindowConfig(t) for t in (32, 64, 128)])


def manifest_path(plan: dict, length: int, mode: str) -> Path:
    return Path(plan["datasets"][str(length)][mode]["path"])


def test_preparation_references_snapshot_without_changing_files_or_links(snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [*snapshot.rglob("*"), tmp_path / "chat.json"]
    before = {p: file_version(p) for p in paths if p.is_file()}

    def forbidden_link(*args: object, **kwargs: object) -> None:
        raise AssertionError("All-frame preparation must not create hardlinks")

    monkeypatch.setattr(os, "link", forbidden_link)
    output = tmp_path / "prepared"
    plan = prepare_all_frame_training(snapshot, output, configs=[AllFrameWindowConfig(t) for t in (32, 64, 128)])
    assert not (output / "PREPARATION_INCOMPLETE").exists()
    assert (output / "experiments.json").is_file() and not (output / "ball_snapshot").exists()
    assert {p: file_version(p) for p in before} == before
    for length in (32, 64, 128):
        for mode in ("train", "evaluation"):
            path = manifest_path(plan, length, mode)
            raw = json.loads(path.read_text())
            assert raw["schema"] == ALL_FRAME_MANIFEST_SCHEMA
            assert raw["config"] == dict(window_length=length, max_window_length=128, pack_length=128,
                                         pack_stride=16, frame_step=1, timestamp_gap_factor=2.5)
            assert dual_sha256(path) == plan["datasets"][str(length)][mode]["sha256"]
            assert Path(raw["ball_store"]["directory"]) == snapshot
            barrier = next(r for r in raw["clips"] if r["clip"]["clip_id"] == "tracknet/barriers/clip")
            assert barrier["common_intervals"] == [[0, 128], [148, 300]]
    with pytest.raises(FileExistsError):
        prepare_all_frame_training(snapshot, output, configs=[AllFrameWindowConfig(32)])


def test_all_lengths_score_the_identical_known_frame_set_exactly_once(prepared: tuple[Path, dict]) -> None:
    _, plan = prepared
    scored, summaries = [], []
    for length in (32, 64, 128):
        path = manifest_path(plan, length, "evaluation")
        dataset = FrozenBallWindowDataset(path, split="train", requires_pose=False)
        ownership: Counter = Counter()
        selected: Counter = Counter()
        for sample in dataset:
            indices = sample["frame_indices"]
            expected_edge = torch.minimum(indices - sample["interval_start"], sample["interval_stop"] - 1 - indices)
            assert torch.equal(sample["edge_distance"], expected_edge)
            assert torch.equal(sample["edge_bin"], torch.bucketize(expected_edge, torch.tensor([1, 8, 16, 32, 64]), right=True))
            assert torch.equal(sample["score_mask"], sample["owner_mask"] & sample["heatmap_valid"])
            assert sample["pts"].shape == sample["position_valid"].shape == (length,)
            for i, f in enumerate(indices.tolist()):
                key = (sample["clip_id"], f, 1, sample["native_fps"])
                ownership[key] += int(sample["owner_mask"][i])
                if sample["score_mask"][i]:
                    selected[key] += 1
        assert set(ownership.values()) == {1} and set(selected.values()) == {1}
        expected: set[tuple[str, int, int, str]] = set()
        for record in dataset.records:
            clip = dataset.store.clip_by_id(record["clip"]["clip_id"])
            states = annotation_states(dataset.store, clip)
            kinds, _ = verified_label_kinds(dataset.store, clip, states)
            expected.update((clip.clip_id, f, 1, str(Fraction(clip.fps))) for a, b in record["common_intervals"]
                            for f in range(a, b) if kinds[f])
        assert set(selected) == expected
        scored.append(selected)
        summaries.append(dataset.manifest["summaries"])
    assert scored[0] == scored[1] == scored[2]
    for split in ("train", "val", "test"):
        for digest_name in ("common_intervals_sha256", "target_identity_sha256", "teacher_identity_sha256",
                    "scored_frame_sha256", "training_packs_sha256"):
            assert len({s[split][digest_name] for s in summaries}) == 1


def test_pack_partition_and_all_frame_masks_pts_rgb_match_annotations(prepared: tuple[Path, dict]) -> None:
    _, plan = prepared
    pack_keys = []
    for length in (32, 64, 128):
        packs = FrozenBallPackDataset(manifest_path(plan, length, "train"), split="train")
        dataset = packs.window_dataset
        index = next(i for i, members in enumerate(dataset.pack_windows)
                     if dataset.records[dataset.windows[members[0]][0]]["clip"]["clip_id"] == "tracknet/sparse/clip")
        children = packs[index]
        assert len(children) == 128 // length
        batch = collate_pack_samples([children, children])
        assert batch["pack_offsets"] == (0, len(children), 2 * len(children))
        assert batch["rgb"].shape[:3] == (2 * len(children), length, 3)
        frames = torch.cat([s["frame_indices"] for s in children])
        assert torch.equal(frames, torch.arange(128))
        assert sum(int(s["position_valid"].sum()) for s in children) == 1
        if length < 128:
            assert any(not s["heatmap_valid"].any() for s in children)
        for sample in children:
            clip = dataset.store.clip_by_id(sample["clip_id"])
            rows = dataset.store.clip_rows(clip)[sample["frame_indices"].numpy()]
            pts = dataset.store.frames["pts"][rows]
            np.testing.assert_array_equal(sample["pts"].numpy(), pts)
            expected = ((pts - pts[0]) * float(Fraction(clip.time_base))).astype(np.float32)
            np.testing.assert_array_equal(sample["timestamps"].numpy(), expected)
            assert torch.equal(sample["position_valid"], sample["label_kind"] == 1)
            assert torch.equal(sample["heatmap_valid"], sample["label_kind"] != 0)
            reference = dataset.store.read_bgr(int(rows[0]))[..., ::-1].transpose(2, 0, 1).copy()
            np.testing.assert_array_equal(sample["rgb"][0].numpy(), reference)
        pack_keys.append([(r["clip"]["clip_id"], p["start"]) for r in dataset.records for p in r["packs"]])
    assert pack_keys[0] == pack_keys[1] == pack_keys[2]


@pytest.mark.parametrize("length", [32, 64, 128])
def test_packed_jpeg_retains_every_frame_and_collates_dynamic_t(prepared: tuple[Path, dict], length: int) -> None:
    _, plan = prepared
    path = manifest_path(plan, length, "evaluation")
    encoded = FrozenBallWindowDataset(path, split="val", requires_pose=False, jpeg_decoder="nvjpeg")
    decoded = FrozenBallWindowDataset(path, split="val", requires_pose=False)
    samples = []
    for i in (0, 1):
        sample, reference = encoded[i], decoded[i]
        assert sample["image_shape"] == (length, 3, 32, 32)
        for key in ("pts", "timestamps", "position_valid", "heatmap_valid", "label_kind", "score_mask"):
            torch.testing.assert_close(sample[key], reference[key], rtol=0, atol=0)
        for t, jpeg in enumerate(sample["jpeg"].split(sample["jpeg_lengths"])):
            image = cv2.imdecode(jpeg.numpy(), cv2.IMREAD_COLOR)
            assert image is not None
            pixels = image[..., ::-1].transpose(2, 0, 1).copy()
            np.testing.assert_array_equal(pixels, reference["rgb"][t].numpy())
        samples.append(sample)
    batch = collate_window_samples(samples)
    assert batch["image_shape"] == (2, length, 3, 32, 32)
    assert len(batch["jpeg_lengths"]) == 2 * length
    assert batch["jpeg"].numel() == sum(batch["jpeg_lengths"])


def test_chat_notes_are_checked_then_frozen_without_reopening_original(prepared: tuple[Path, dict], tmp_path: Path) -> None:
    _, plan = prepared
    (tmp_path / "chat.json").unlink()
    dataset = FrozenBallWindowDataset(manifest_path(plan, 128, "evaluation"), split="train", requires_pose=False)
    sample = next(s for s in dataset if s["clip_id"] == "chat_annotation/example")
    assert not sample["heatmap_valid"][10] and not sample["heatmap_valid"][80]
    assert sample["label_kind"][81] == 2 and sample["heatmap_valid"][81]
    with pytest.raises(ValueError, match="missing or checksum"):
        build_all_frame_manifests(dataset.store, [AllFrameWindowConfig(32)])


def test_legacy_reader_and_schema_mixing_are_explicit(prepared: tuple[Path, dict], tmp_path: Path) -> None:
    _, plan = prepared
    new_path = manifest_path(plan, 32, "train")
    new = FrozenBallWindowDataset(new_path, split="train", requires_pose=False)
    old = build_coordinate_manifest(new.store, PlayIntervalConfig(), TemporalSamplingConfig((1,)), input_kind="mdd_only")
    old_path = tmp_path / "legacy.json"
    old_path.write_text(json.dumps(old))
    legacy = FrozenBallWindowDataset(old_path, split="train", requires_pose=False)
    assert legacy[0]["rgb"].shape[0] == 32
    with pytest.raises(ValueError, match="schemas"):
        collate_window_samples([new[0], legacy[0]])
    for raw, schema in ((new.manifest, old["schema"]), (old, ALL_FRAME_MANIFEST_SCHEMA)):
        changed = copy.deepcopy(raw)
        changed["schema"] = schema
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="schema|semantics"):
            FrozenBallWindowDataset(path, split="train", requires_pose=False)


@pytest.mark.parametrize("change", ["owner", "hash", "interval", "kind", "incomplete", "pack"])
def test_corrupt_or_unpublished_all_frame_manifests_fail_before_images(prepared: tuple[Path, dict], tmp_path: Path, change: str) -> None:
    _, plan = prepared
    mode = "train" if change == "pack" else "evaluation"
    raw = json.loads(manifest_path(plan, 32, mode).read_text())
    if change == "owner":
        raw["clips"][0]["windows"][0]["owner_range"][1] -= 1
    elif change == "hash":
        raw["summaries"]["train"]["scored_frame_sha256"] = "changed"
    elif change == "interval":
        raw["clips"][0]["common_intervals"][0][1] -= 1
    elif change == "kind":
        raw["clips"][0]["label_runs"][0][2] = 3
    elif change == "pack":
        raw["clips"][0]["packs"].pop()
    else:
        (tmp_path / "PREPARATION_INCOMPLETE").touch()
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        FrozenBallWindowDataset(path, split="train", requires_pose=False)


def test_source_mutation_leaves_preparation_incomplete(snapshot: Path, tmp_path: Path) -> None:
    shard = BallFrameStore(snapshot).image_path(BallFrameStore(snapshot).clips[0])

    def mutate(stage: str, completed: int, total: int) -> None:
        if stage == "annotations" and completed == 1:
            shard.write_bytes(shard.read_bytes())

    output = tmp_path / "interrupted"
    with pytest.raises(ValueError, match="Source files changed"):
        prepare_all_frame_training(snapshot, output, configs=[AllFrameWindowConfig(32)], progress=mutate)
    assert (output / "PREPARATION_INCOMPLETE").exists()
    assert not (output / "experiments.json").exists()


def test_all_frame_prepare_cli_and_incompatible_options(snapshot: Path, tmp_path: Path) -> None:
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.prepare_data", "--mode", "all_frames",
               "--ball-store", str(snapshot), "--output", str(tmp_path / "cli"), "--window-lengths", "32", "64", "128"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "cli/experiments.json").is_file()
    rejected = subprocess.run(command + ["--frame-steps", "1"], env=env, capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0 and "incompatible" in rejected.stderr


def test_chat_without_empty_labels_does_not_require_unused_original(tmp_path: Path) -> None:
    root = write_store_clip(tmp_path / "ball", "chat_annotation/observed", [frame(i, ball()) for i in range(128)],
                            source="chat_annotation", size=(32, 32))
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["clips"][0]["annotation_path"] = str(tmp_path / "missing-unused-original.json")
    (root / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "prepared"
    plan = prepare_all_frame_training(root, output, configs=[AllFrameWindowConfig(32)])
    dataset = FrozenBallPackDataset(manifest_path(plan, 32, "train"), split="train")
    assert len(dataset) == 1 and all(s["position_valid"].all() for s in dataset[0])
    with pytest.raises(ValueError, match="every declared child"):
        collate_pack_samples([dataset[0][:-1]])
