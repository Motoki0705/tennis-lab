"""Ball frame store: source mappings, resizing, splits and read-side validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tasks.ball_detection.data.store import (
    EVENT_CODES,
    INDEX_FILE,
    METADATA_FILE,
    POINT_KIND_CODES,
    BallFrameStore,
)
from src.tasks.ball_detection.generate_dataset.frame_store.builder import build_dataset
from tests.support.tasks.ball_detection.frame_sources import (
    CHAT_SOURCE,
    MEIJI_SOURCE,
    TRACKNET_ROWS,
    TRACKNET_SOURCE,
    SourceRoots,
    meiji_pixels,
    write_chat,
    write_meiji,
    write_tracknet,
)

K = POINT_KIND_CODES


@pytest.fixture
def roots(tmp_path: Path) -> SourceRoots:
    return SourceRoots(tmp_path)


def test_tracknet_frames_are_stored_byte_for_byte_with_events(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    store = BallFrameStore(build_dataset(roots.config({"tracknet": TRACKNET_SOURCE})))

    assert [(c.clip_id, c.split, c.fps, c.time_base) for c in store.clips] == [
        ("tracknet/game1/Clip1", "train", "30", "1/30"),
        ("tracknet/game2/Clip1", "val", "30", "1/30"),
        ("tracknet/game3/Clip1", "test", "30", "1/30"),
    ]
    clip = store.clips[0]
    rows = store.clip_rows(clip)
    assert store.frames["pts"][rows].tolist() == [0, 1, 2, 3]
    assert store.frames["event"][rows].tolist() == [
        EVENT_CODES["none"], EVENT_CODES["hit"], EVENT_CODES["unlabeled"], EVENT_CODES["bounce"]
    ]
    assert store.frames["inst_count"][rows].tolist() == [1, 1, 0, 1]  # visibility 0 = negative frame
    assert store.frames["annotated"][rows].all()
    kinds = [store.instances_of(int(row)).point_kind.tolist() for row in rows]
    assert kinds == [[K["observed"]], [K["observed"]], [], [K["occlusion_estimated"]]]
    assert store.instances_of(int(rows[3])).occluded.tolist() == [True]
    np.testing.assert_array_equal(store.instances_of(int(rows[1])).xy, [[20, 14]])
    source = roots.data / "tennis" / "tracknet" / "game1" / "Clip1" / "0001.jpg"
    assert store.read_jpeg(int(rows[1])).tobytes() == source.read_bytes()
    counts = json.loads((store.directory / METADATA_FILE).read_text())["counts"]["by_source"]["tracknet"]
    assert counts["negative_frames"] == 3 and counts["frames"] == 3 * len(TRACKNET_ROWS)


def test_meiji_cameras_are_downscaled_with_their_labels(roots: SourceRoots) -> None:
    write_meiji(roots.data / "meiji")
    # 96x64 -> 48x32: labels are scaled by exactly 0.5.
    store = BallFrameStore(build_dataset(roots.config({"meiji": MEIJI_SOURCE}, max_height=32)))

    assert [(c.clip_id, c.camera_id, c.group_id, c.split) for c in store.clips] == [
        ("meiji/video_000/clip_000/cam0", "cam0", "video_000", "train"),
        ("meiji/video_000/clip_000/cam1", "cam1", "video_000", "train"),
        ("meiji/video_001/clip_000/cam0", "cam0", "video_001", "test"),
        ("meiji/video_001/clip_000/cam1", "cam1", "video_001", "test"),
    ]
    clip = store.clips[0]
    assert (clip.width, clip.height, clip.source_width, clip.source_height, clip.scale) == (48, 32, 96, 64, 0.5)
    assert clip.fps == "2997003/50000" and not clip.has_events
    rows = store.clip_rows(clip)
    assert store.frames["pts"][rows].tolist() == [0, 50000, 100000, 150000, 200000]
    assert store.frames["segment_break"][rows].tolist() == [False, False, True, False, False]
    assert (store.frames["event"][rows] == EVENT_CODES["unlabeled"]).all()
    instances = [store.instances_of(int(row)) for row in rows]
    assert [i.point_kind.tolist() for i in instances] == [
        [K["observed"]], [K["interpolated"]], [K["occlusion_estimated"]], [K["unresolved"]], [K["observed"]]
    ]
    np.testing.assert_allclose(instances[0].xy, [[6.0, 15.0]])
    assert np.isnan(instances[3].xy).all()
    assert [bool(i.occluded[0]) for i in instances] == [False, False, True, False, False]
    expected = cv2.resize(cv2.cvtColor(meiji_pixels(2), cv2.COLOR_RGB2BGR), (48, 32), interpolation=cv2.INTER_AREA)
    assert np.abs(store.read_bgr(int(rows[2])).astype(np.int16) - expected.astype(np.int16)).mean() < 6.0


def test_chat_statuses_map_to_point_kinds_and_unreviewed_frames(roots: SourceRoots) -> None:
    manifest = write_chat(roots.outputs / "chat_annotation")
    store = BallFrameStore(build_dataset(roots.config({"chat_annotation": CHAT_SOURCE})))

    (clip,) = store.clips
    assert clip.clip_id == "chat_annotation/src_a__run__f000-006"
    assert clip.group_id == "src_a" and clip.track_ids == ("ball_001", "ball_002")
    rows = store.clip_rows(clip)
    assert store.frames["pts"][rows].tolist() == [frame.clip_pts for frame in manifest.frames]
    assert store.frames["annotated"][rows].tolist() == [True] * 5 + [False]
    assert store.frames["segment_break"][rows].tolist() == [False, True, False, False, False, False]
    instances = [store.instances_of(int(row)) for row in rows]
    assert [i.point_kind.tolist() for i in instances] == [
        [K["observed"]],
        [K["occlusion_estimated"]],
        [K["unresolved"]],  # occluded without a centre
        [K["out_of_frame"]],
        [K["unresolved"], K["observed"]],
        [],
    ]
    assert [i.occluded.tolist() for i in instances[:3]] == [[False], [True], [True]]
    assert instances[4].track_index.tolist() == [0, 1]


def test_all_sources_share_one_store(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    write_meiji(roots.data / "meiji")
    write_chat(roots.outputs / "chat_annotation")
    store = BallFrameStore(
        build_dataset(
            roots.config({"tracknet": TRACKNET_SOURCE, "meiji": MEIJI_SOURCE, "chat_annotation": CHAT_SOURCE})
        )
    )
    assert [clip.source for clip in store.clips] == ["tracknet"] * 3 + ["meiji"] * 4 + ["chat_annotation"]
    assert [c.clip_id for c in store.split_clips("test", sources=["meiji"])] == [
        "meiji/video_001/clip_000/cam0",
        "meiji/video_001/clip_000/cam1",
    ]
    last = store.clips[-1]
    assert store.frame_key(store.row_of(last, 4)) == f"{last.clip_id}:4"
    readme = (store.directory / "README.md").read_text()
    assert "| meiji | 4 | 20 |" in readme


def test_explicit_split_must_name_exactly_the_data_groups(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    missing = {**TRACKNET_SOURCE, "split": {"train": ["game1"], "val": ["game2"], "test": []}}
    with pytest.raises(ValueError, match="unassigned"):
        build_dataset(roots.config({"tracknet": missing}))
    stale = {**TRACKNET_SOURCE, "split": {"train": ["game1", "game9"], "val": ["game2"], "test": ["game3"]}}
    with pytest.raises(ValueError, match="absent"):
        build_dataset(roots.config({"tracknet": stale}))
    assert not (roots.data / "ball_detection").exists() or not any((roots.data / "ball_detection").iterdir())


def test_build_refuses_existing_version_and_inconsistent_sources(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    config = roots.config({"tracknet": TRACKNET_SOURCE})
    build_dataset(config)
    with pytest.raises(FileExistsError):
        build_dataset(config)

    label = roots.data / "tennis" / "tracknet" / "game1" / "Clip1" / "Label.csv"
    label.write_text(label.read_text().replace("0002.jpg,0,0,0,", "0002.jpg,0,0,0,1"))
    with pytest.raises(ValueError, match="invisible frame"):
        build_dataset(roots.config({"tracknet": TRACKNET_SOURCE}, version="test-v2"))

    write_meiji(roots.data / "meiji")
    video = roots.data / "meiji" / "videos" / "video_000" / "clips" / "clip_000" / "media" / "cam1.mp4"
    video.write_bytes(video.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        build_dataset(roots.config({"meiji": MEIJI_SOURCE}, version="test-v3"))


def test_non_integral_downscale_is_rejected(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    with pytest.raises(ValueError, match="aspect ratio"):
        build_dataset(roots.config({"tracknet": TRACKNET_SOURCE}, max_height=35))


def test_reader_rejects_a_tampered_index(roots: SourceRoots, tmp_path: Path) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    directory = build_dataset(roots.config({"tracknet": TRACKNET_SOURCE}))
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    with np.load(copy / INDEX_FILE) as data:
        columns = {name: data[name] for name in data.files}
    kinds = columns["point_kind"].copy()
    kinds[0] = K["unresolved"]  # an unresolved ball with a position
    np.savez(copy / INDEX_FILE, **{**columns, "point_kind": kinds})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NaN positions"):
        BallFrameStore(copy)


def test_tracknet_labels_on_the_far_border_are_clamped_and_recorded(roots: SourceRoots) -> None:
    write_tracknet(roots.data / "tennis" / "tracknet")
    label = roots.data / "tennis" / "tracknet" / "game2" / "Clip1" / "Label.csv"
    label.write_text(label.read_text().replace("0000.jpg,1,10,12,0", "0000.jpg,1,10,48,0"))
    store = BallFrameStore(build_dataset(roots.config({"tracknet": TRACKNET_SOURCE})))
    clip = store.clip_by_id("tracknet/game2/Clip1")
    np.testing.assert_array_equal(store.instances_of(store.row_of(clip, 0)).xy, [[10, 47]])
    metadata = json.loads((store.directory / METADATA_FILE).read_text())
    assert metadata["clips"][clip.index]["provenance"]["border_clamped"] == [{"frame": "0000.jpg", "from": [10.0, 48.0]}]

    label.write_text(label.read_text().replace("0000.jpg,1,10,48,0", "0000.jpg,1,10,49,0"))
    with pytest.raises(ValueError, match="outside"):
        build_dataset(roots.config({"tracknet": TRACKNET_SOURCE}, version="test-v2"))


def test_shipped_config_composes_into_a_strict_build_config() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import src.utils.hydra  # noqa: F401  (registers the tennis_* resolvers)
    from src.tasks.ball_detection.generate_dataset.frame_store.config import (
        FrameStoreBuildConfig,
    )

    configs = Path(__file__).resolve().parents[5] / "src" / "tasks" / "ball_detection" / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        raw = compose("generate_dataset")
    container = OmegaConf.to_container(raw, resolve=False)
    assert isinstance(container, dict)
    container.pop("hydra", None)
    config = FrameStoreBuildConfig.from_config(OmegaConf.create(container))
    assert config.dataset_dir.parts[-2:] == ("ball_detection", "ball-mix-v1")
    assert config.meiji is not None and dict(config.meiji[1].groups) == {
        "video_000": "val",
        "video_001": "test",
        "video_002": "train",
    }
    assert config.tracknet is not None and config.tracknet[1].groups["game10"] == "test"
