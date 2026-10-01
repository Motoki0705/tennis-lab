"""Unit tests for the ball-detection dataset catalog and frame access."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.tasks.ball_detection.visualization.review.datasets import (
    BallDatasetCatalog,
    BallDatasetCatalogError,
    split_scene_id,
)


def catalog_for(tmp_path: Path) -> BallDatasetCatalog:
    return BallDatasetCatalog(tmp_path)


def tracknet_root(tmp_path: Path) -> Path:
    """Return the ``ball_detection/test-v1/tracknet`` fixture prefix below a synthetic data root."""
    return tmp_path / "ball_detection" / "test-v1" / "tracknet"


def test_clip_discovery_and_multi_instance_labels(
    tmp_path: Path, make_clip_dataset: Callable[..., Path]
) -> None:
    make_clip_dataset(tracknet_root(tmp_path))
    # Unpublished versions without metadata are not discoverable.
    (tmp_path / "ball_detection" / "building").mkdir(parents=True)

    catalog = catalog_for(tmp_path)
    entries = {entry.spec.id: entry for entry in catalog.entries()}
    tracknet = entries["store/test-v1"]
    assert tracknet.available
    assert tracknet.count == 1
    ref = catalog.refs("store/test-v1")[0]
    assert ref.id == "store/test-v1::tracknet/game1/Clip1"
    assert ref.frames == 6

    scene = catalog.resolve("store/test-v1", "tracknet/game1/Clip1")
    assert scene.mode == "temporal"
    assert scene.original_size(0) == (64, 48)
    labels = scene.labels(0)
    assert len(labels) == 1
    assert labels[0].x == 10.0 and labels[0].y == 20.0
    assert scene.read_rgb(3).shape == (48, 64, 3)


def test_multi_instance_and_negative_rows_are_preserved(
    tmp_path: Path, make_clip_dataset: Callable[..., Path]
) -> None:
    # Two visible balls plus an explicitly annotated negative row: the parser
    # must keep both instances and must not invent a row for the negative.
    make_clip_dataset(
        tracknet_root(tmp_path),
        extra_rows=[
            {
                "file name": "0000.jpg",
                "instance id": "b002",
                "visibility": 1,
                "x-coordinate": 30,
                "y-coordinate": 40,
            },
            {
                "file name": "0001.jpg",
                "instance id": "",
                "visibility": 0,
                "x-coordinate": "",
                "y-coordinate": "",
            },
        ],
    )
    scene = catalog_for(tmp_path).resolve("store/test-v1", "tracknet/game1/Clip1")
    first = scene.labels(0)
    assert {label.instance_id for label in first} == {"b001", "b002"}
    # The negative row carries an empty instance id and no coordinates, so it
    # must not be turned into a phantom second instance.
    second = scene.labels(1)
    assert [label.instance_id for label in second] == ["b001"]
    assert all(label.visibility > 0 for label in second)


def test_invalid_store_stays_visible_with_reason(tmp_path: Path) -> None:
    root = tmp_path / "ball_detection" / "test-v1"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text('{"schema_version": "broken"}')
    entry = {entry.spec.id: entry for entry in catalog_for(tmp_path).entries()}["store/test-v1"]
    assert not entry.available
    assert entry.reason is not None and "schema" in entry.reason


def test_path_traversal_and_unknown_scene_are_rejected(
    tmp_path: Path, make_clip_dataset: Callable[..., Path]
) -> None:
    make_clip_dataset(tracknet_root(tmp_path))
    catalog = catalog_for(tmp_path)
    for candidate in (
        "../../etc/passwd",
        "game1/../../escape",
        "/etc/passwd",
        "tracknet/game1/Clip1/../../..",
    ):
        with pytest.raises(BallDatasetCatalogError):
            catalog.resolve("store/test-v1", candidate)
    with pytest.raises(BallDatasetCatalogError, match="Unknown dataset"):
        catalog.resolve("nope", "tracknet/game1/Clip1")


def test_scene_id_requires_separator() -> None:
    assert split_scene_id("store/test-v1::tracknet/game1/Clip1") == ("store/test-v1", "tracknet/game1/Clip1")
    for bad in ("", "store/test-v1", "::scene", "scene::"):
        with pytest.raises(BallDatasetCatalogError):
            split_scene_id(bad)


def test_unified_web_store_static_and_temporal(
    tmp_path: Path, make_web_store: Callable[[Path], Path]
) -> None:
    make_web_store(tmp_path)
    catalog = catalog_for(tmp_path)
    entries = {entry.spec.id: entry for entry in catalog.entries()}
    assert entries["web_static"].available
    assert entries["web_static"].count == 2
    assert entries["web_static"].max_scene_frames == 1
    assert entries["web_temporal"].available
    assert entries["web_temporal"].count == 1
    assert entries["web_temporal"].max_scene_frames == 2

    static_ref = catalog.refs("web_static")[0]
    assert static_ref.id == "web_static::0"
    assert static_ref.frames == 1
    static = catalog.resolve("web_static", "0")
    assert static.mode == "static"
    assert static.original_size(0) == (64, 48)
    assert static.name(0) == "sample_000000.jpg"
    assert [label.x for label in static.labels(0)] == [12.5]

    negative = catalog.resolve("web_static", "1")
    assert negative.labels(0) == ()

    temporal_ref = catalog.refs("web_temporal")[0]
    assert temporal_ref.id == "web_temporal::video-0001"
    assert temporal_ref.frames == 2
    temporal = catalog.resolve("web_temporal", "video-0001")
    assert temporal.mode == "temporal"
    assert [temporal.name(index) for index in range(2)] == [
        "000000.jpg",
        "000001.jpg",
    ]
    assert temporal.labels(0)[0].x == 7.0
    assert temporal.labels(1)[0].x == 30.0
    # Sequence order follows the stored frame index, not the image content.
    assert int(temporal.read_rgb(0)[0, 0, 0]) == 10
    assert int(temporal.read_rgb(1)[0, 0, 0]) == 200


def test_missing_unified_store_reports_reason(tmp_path: Path) -> None:
    entries = {entry.spec.id: entry for entry in catalog_for(tmp_path).entries()}
    for dataset_id in ("web_static", "web_temporal"):
        entry = entries[dataset_id]
        assert not entry.available
        assert entry.count == 0
        assert entry.reason is not None and "index.npz" in entry.reason


def test_frame_index_bounds_and_type(
    tmp_path: Path, make_clip_dataset: Callable[..., Path]
) -> None:
    make_clip_dataset(tracknet_root(tmp_path))
    scene = catalog_for(tmp_path).resolve("store/test-v1", "tracknet/game1/Clip1")
    with pytest.raises(IndexError, match="has no frame"):
        scene.name(6)
    with pytest.raises(ValueError, match="must be an int"):
        scene.labels(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an int"):
        scene.original_size(True)  # type: ignore[arg-type]


def test_search_and_paging(
    tmp_path: Path, make_clip_dataset: Callable[..., Path]
) -> None:
    make_clip_dataset(tracknet_root(tmp_path), clips=3, frames=4)
    refs = catalog_for(tmp_path).refs("store/test-v1")
    assert [ref.local_id for ref in refs] == [
        "tracknet/game1/Clip1",
        "tracknet/game1/Clip2",
        "tracknet/game1/Clip3",
    ]
    assert [ref.frames for ref in refs] == [4, 4, 4]


def test_scene_order_is_natural_not_lexicographic(tmp_path: Path) -> None:
    from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

    for name in ("Clip1", "Clip2", "Clip10"):
        write_store_clip(tmp_path / "ball_detection" / "test-v1", f"tracknet/game1/{name}", [frame(0, ball())])
    refs = catalog_for(tmp_path).refs("store/test-v1")
    assert [ref.local_id for ref in refs] == ["tracknet/game1/Clip1", "tracknet/game1/Clip2", "tracknet/game1/Clip10"]


def test_frame_read_rechecks_the_root_boundary(tmp_path: Path, make_clip_dataset: Callable[..., Path]) -> None:
    make_clip_dataset(tracknet_root(tmp_path))
    scene = catalog_for(tmp_path).resolve("store/test-v1", "tracknet/game1/Clip1")
    shard = tmp_path / "ball_detection" / "test-v1" / "shards" / "clip-00000.bin"
    secret = tmp_path / "outside.bin"
    secret.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(secret)
    with pytest.raises(ValueError, match="resolves outside"):
        scene.read_rgb(0)
