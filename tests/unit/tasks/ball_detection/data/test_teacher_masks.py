"""Adoption masks reach teachers without erasing the reviewed source labels."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.store_dataset import BallStoreDataset, StoreWindow
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    FrameSupervisionPolicy,
    resolve_frame_supervision,
)
from src.tasks.ball_detection.data.teacher_masks import EXPLICIT_MASK_SCHEMA, LEGACY
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import (
    build_coordinate_manifest,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.tasks.ball_detection.visualization.io.store_frames import StoreSceneFrames
from tests.support.tasks.ball_detection.store import (
    ball,
    frame,
    store_config,
    write_store_clip,
)

ESTIMATES = FrameSupervisionPolicy(
    positive=frozenset({"observed", "interpolated", "occlusion_estimated"}),
    absent=frozenset({"out_of_frame"}),
    ignore=frozenset({"unresolved"}),
)


def upgrade(root: Path, heatmap: list[int], coordinate: list[int]) -> None:
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        columns = dict(archive)
    columns["teacher_heatmap"] = np.asarray(heatmap, np.uint8)
    columns["teacher_coordinate"] = np.asarray(coordinate, np.uint8)
    np.savez(root / "index.npz", **columns)
    path = root / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = EXPLICIT_MASK_SCHEMA
    path.write_text(json.dumps(metadata))


@pytest.fixture
def adopted(tmp_path: Path) -> Path:
    frames = [
        frame(0),  # reviewed, still unknown; must not become a negative
        frame(1),  # confirmed absence
        frame(2, ball()),  # visible point retained, but a blocking review note
        frame(3, ball()),
        frame(4, ball(), ball(track="b2", xy=(30.0, 20.0))),
        replace(frame(5, ball()), is_target=False),
        frame(6, ball("occlusion_estimated")),
        frame(7, ball("interpolated")),
        frame(8, ball("unresolved", None)),
        frame(9, ball("out_of_frame", None)),
        frame(10, ball(), annotated=False),
        frame(11, ball(), ball("out_of_frame", None, "b2")),
    ]
    root: Path = write_store_clip(tmp_path / "store", "train/adopted", frames)
    upgrade(root, [0, 1, 0, 2, 2, 0, 0, 0, 0, 1, 0, 2], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    return root


def test_masks_preserve_review_facts_and_limit_both_teacher_heads(adopted: Path) -> None:
    store = BallFrameStore(adopted)
    states = annotation_states(store, store.clips[0])
    assert states.reviewed[:3].tolist() == [True, True, True]
    assert states.visible_observation[2] and states.evidence[2]
    np.testing.assert_array_equal(states.xy[2], [10.0, 20.0])
    assert states.supervision.tolist() == [False] * 3 + [True] + [False] * 8
    assert states.kinds[4].sum() == 2
    expected = [False, True, False, True, True, False, False, False, False, True, False, True]
    for policy in (OBSERVED_ONLY, ESTIMATES):
        supervision = resolve_frame_supervision(store, policy)
        assert supervision.supervised.tolist() == expected
        owner: NDArray[np.int64] = np.repeat(np.arange(len(store), dtype=np.int64), store.frames["inst_count"])
        assert owner[supervision.positive].tolist() == [3, 4, 4, 11]
    scene = StoreSceneFrames(store, store.clips[0].clip_id)
    assert scene.annotated(0) and scene.annotated(2)
    assert not scene.supervised(0) and not scene.supervised(2)
    assert scene.supervised(1)
    assert scene.labels(2)[0].x == 10.0
    assert scene.labels(2)[0].state == "observed"


def test_candidate_references_obey_masks_without_hiding_raw_labels(adopted: Path, tmp_path: Path) -> None:
    store = BallFrameStore(adopted)
    dataset = BallStoreDataset(
        store=store, supervision=resolve_frame_supervision(store, OBSERVED_ONLY),
        windows=[StoreWindow(0, 2)], config=store_config(tmp_path),
    )
    window = dataset.read_window(0)
    assert window.frames[0].observed_xy is None
    assert not window.frames[0].supervised
    assert window.frames[1].observed_xy == (10.0, 20.0)
    assert window.frames[1].supervised
    assert store.instances_of(2).xy.tolist() == [[10.0, 20.0]]


@pytest.mark.parametrize("policy", [OBSERVED_ONLY, ESTIMATES])
def test_legacy_clips_keep_policy_labels_and_semantic_identity(tmp_path: Path, policy: FrameSupervisionPolicy) -> None:
    root = write_store_clip(tmp_path / "store", "old/clip", [
        frame(0), frame(1, ball()), frame(2, ball("interpolated")),
        frame(3, ball("occlusion_estimated")), frame(4, ball("unresolved", None)),
        frame(5, ball(), annotated=False),
    ])
    before = BallFrameStore(root)
    expected = resolve_frame_supervision(before, policy)
    digest = clip_semantic_digest(before, before.clips[0])
    upgrade(root, [LEGACY] * 6, [LEGACY] * 6)
    after = BallFrameStore(root)
    actual = resolve_frame_supervision(after, policy)
    np.testing.assert_array_equal(actual.supervised, expected.supervised)
    np.testing.assert_array_equal(actual.positive, expected.positive)
    for name, column in before.frames.items():
        np.testing.assert_array_equal(after.frames[name], column)
    for name, column in before.instances.items():
        np.testing.assert_array_equal(after.instances[name], column)
    assert clip_semantic_digest(after, after.clips[0]) == digest
    np.testing.assert_array_equal(annotation_states(after, after.clips[0]).supervision,
                                  annotation_states(before, before.clips[0]).supervision)


def test_mixed_store_declares_legacy_mode_per_clip(tmp_path: Path) -> None:
    root = write_store_clip(tmp_path / "store", "old/clip", [frame(0, ball("interpolated"))])
    write_store_clip(root, "new/clip", [frame(0, ball("interpolated"))])
    upgrade(root, [LEGACY, 0], [LEGACY, 0])
    supervision = resolve_frame_supervision(BallFrameStore(root), ESTIMATES)
    assert supervision.supervised.tolist() == [True, False]
    assert supervision.positive.tolist() == [True, False]


@pytest.mark.parametrize(("mutation", "error"), [
    ("missing", "require column"), ("dtype", "uint8"), ("length", "one value per frame"),
    ("heatmap_code", "unknown teacher mask code"), ("coordinate_code", "unknown teacher mask code"),
    ("one_legacy", "both heads"), ("mixed_clip", "mix legacy and adopted"),
    ("v1", "explicit v2"),
])
def test_invalid_mask_storage_is_rejected(adopted: Path, mutation: str, error: str) -> None:
    path = adopted / "index.npz"
    with np.load(path, allow_pickle=False) as archive:
        columns = dict(archive)
    if mutation == "missing":
        del columns["teacher_coordinate"]
    elif mutation == "dtype":
        columns["teacher_heatmap"] = columns["teacher_heatmap"].astype(np.int32)
    elif mutation == "length":
        columns["teacher_coordinate"] = columns["teacher_coordinate"][:-1]
    elif mutation == "heatmap_code":
        columns["teacher_heatmap"][0] = 3
    elif mutation == "coordinate_code":
        columns["teacher_coordinate"][0] = 2
    elif mutation == "one_legacy":
        columns["teacher_heatmap"][0] = LEGACY
    elif mutation == "mixed_clip":
        columns["teacher_heatmap"][0] = columns["teacher_coordinate"][0] = LEGACY
    elif mutation == "v1":
        metadata_path = adopted / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["schema_version"] = "ball_detection_frames.v1"
        metadata_path.write_text(json.dumps(metadata))
    np.savez(path, **columns)
    with pytest.raises(ValueError, match=error):
        BallFrameStore(adopted)


@pytest.mark.parametrize(("row", "heatmap", "coordinate", "error"), [
    (0, 2, 0, "positive ball frame"), (2, 1, 0, "negative ball frame"),
    (3, 2, 0, "coordinate mask"), (4, 2, 1, "coordinate mask"),
    (5, 2, 1, "reviewed target"), (6, 2, 0, "Estimated or unresolved"),
    (7, 2, 0, "Estimated or unresolved"), (8, 1, 0, "Estimated or unresolved"),
    (10, 2, 1, "reviewed target"), (11, 2, 1, "coordinate mask"),
    (2, 0, 1, "coordinate mask"),
])
def test_masks_cannot_invent_teacher_evidence(adopted: Path, row: int, heatmap: int, coordinate: int, error: str) -> None:
    path = adopted / "index.npz"
    with np.load(path, allow_pickle=False) as archive:
        columns = dict(archive)
    columns["teacher_heatmap"][row] = heatmap
    columns["teacher_coordinate"][row] = coordinate
    np.savez(path, **columns)
    with pytest.raises(ValueError, match=error):
        BallFrameStore(adopted)


def test_changing_only_adopted_mask_changes_clip_identity(adopted: Path) -> None:
    before = BallFrameStore(adopted)
    digest = clip_semantic_digest(before, before.clips[0])
    assert before.teacher_masks is not None
    heatmap = before.teacher_masks.heatmap.tolist()
    heatmap[0] = 1
    upgrade(adopted, heatmap, before.teacher_masks.coordinate.tolist())
    after = BallFrameStore(adopted)
    assert clip_semantic_digest(after, after.clips[0]) != digest
    np.testing.assert_array_equal(after.frames["annotated"], before.frames["annotated"])


def test_frozen_windows_apply_masks_to_coordinate_and_heatmap_training(tmp_path: Path) -> None:
    frames = [frame(i, ball(xy=(8.0, 9.0))) for i in range(64)]
    frames[4] = frame(4)
    frames[5] = frame(5)
    root = write_store_clip(tmp_path / "store", "train/clip", frames, size=(32, 32))
    heatmap, coordinate = [2] * 64, [1] * 64
    heatmap[3:6], coordinate[3:6] = [0, 0, 1], [0, 0, 0]
    upgrade(root, heatmap, coordinate)
    store = BallFrameStore(root)
    manifest = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig((1,)), input_kind="mdd_only")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = FrozenBallWindowDataset(path, split="train", requires_pose=False)
    sample = dataset[0]
    assert sample["frame_indices"].tolist() == list(range(32))
    assert sample["position_valid"][2:7].tolist() == [True, False, False, False, True]
    assert sample["heatmap_valid"][2:7].tolist() == [True, False, False, True, True]
    assert sample["uv"][3].tolist() == [0.0, 0.0]
    assert sample["timestamps"][1] == pytest.approx(1 / 30)
    assert sample["rgb"].shape == (32, 3, 32, 32)
