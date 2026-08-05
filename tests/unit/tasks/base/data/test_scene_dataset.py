"""Unit tests for the base scene dataset and its dataclasses/helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
    TemporalWindow,
)
from src.utils.configuration import MissingConfigurationKeyError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_temporal_window_slice() -> None:
    w = TemporalWindow(start=2, end=5, seq_len=3, full_len=10)
    assert w.sl == slice(2, 5)
    arr = np.arange(10)
    assert arr[w.sl].tolist() == [2, 3, 4]


def test_camera_selection_primary() -> None:
    assert CameraSelection(indices=(2, 0, 1)).primary == 2


def test_camera_selection_empty_primary_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = CameraSelection(indices=()).primary


# ---------------------------------------------------------------------------
# Scene accessors
# ---------------------------------------------------------------------------


def test_scene_has_and_require_key(make_scene) -> None:
    scene = make_scene(data={"position": np.zeros((4, 3))})
    assert scene.has_key("position")
    assert not scene.has_key("missing")
    scene.require_key("position")  # no raise
    with pytest.raises(KeyError, match="Missing key 'missing'"):
        scene.require_key("missing")


def test_scene_metadata_properties() -> None:
    scene = Scene(
        path=Path("/x"),
        data={},
        meta={"scene_id": 42, "rally_length": "7", "shots": [{"t": 1}, "bad", {"t": 2}]},
        num_frames=10,
        num_cameras=1,
    )
    assert scene.scene_id == "42"
    assert scene.rally_length == 7
    assert scene.shots == [{"t": 1}, {"t": 2}]  # non-dicts filtered out


def test_scene_metadata_missing_or_invalid() -> None:
    scene = Scene(path=Path("/x"), data={}, meta={"rally_length": "nope"}, num_frames=3, num_cameras=1)
    assert scene.scene_id is None
    assert scene.rally_length is None  # invalid int -> None
    assert scene.shots == []


def test_effective_num_frames_takes_min_positive(make_scene) -> None:
    scene = make_scene(num_frames=10)
    assert scene.effective_num_frames(6, 8) == 6
    assert scene.effective_num_frames(20) == 10  # candidates larger -> num_frames
    assert scene.effective_num_frames(0, -3) == 10  # non-positive candidates ignored


def test_get_camera_array_copies_and_windows() -> None:
    data: dict[str, np.ndarray] = {
        "cam_1_ball_uv": np.arange(8 * 2).reshape(8, 2).astype(np.float32)
    }
    scene = Scene(path=Path("/x"), data=data, meta={}, num_frames=8, num_cameras=2)
    full = scene.get_camera_array(1, "ball_uv")
    assert full.shape == (8, 2)
    # mutating the returned array must not touch the payload (copy semantics)
    full[0, 0] = -999
    assert scene.data["cam_1_ball_uv"][0, 0] == 0

    w = TemporalWindow(start=1, end=4, seq_len=3, full_len=8)
    windowed = scene.get_camera_array(1, "ball_uv", window=w)
    assert windowed.shape == (3, 2)


def test_get_camera_array_out_of_range() -> None:
    scene = Scene(path=Path("/x"), data={"cam_0_x": np.zeros((4, 2))}, meta={}, num_frames=4, num_cameras=1)
    with pytest.raises(ValueError, match="out of range"):
        scene.get_camera_array(1, "x")


def test_get_array_scalar_window_raises() -> None:
    scene = Scene(path=Path("/x"), data={"scalar": np.asarray(3.0)}, meta={}, num_frames=1, num_cameras=1)
    w = TemporalWindow(start=0, end=1, seq_len=1, full_len=1)
    with pytest.raises(ValueError, match="scalar and cannot be temporally sliced"):
        scene.get_array("scalar", window=w)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, msg",
    [
        ((1, 2, 3), "length 2"),
        ((0, 5), "positive integers"),
        ((-1, 5), "positive integers"),
        ((5, 2), "min must be <= max"),
    ],
)
def test_validate_range_errors(value, msg) -> None:
    with pytest.raises(ValueError, match=msg):
        SceneDatasetBase._validate_range(value, name="r")


def test_validate_range_ok() -> None:
    SceneDatasetBase._validate_range((1, 1024), name="r")  # no raise


def test_parse_int_range_present() -> None:
    cfg = {"seq_len_range": [3, 9]}
    assert SceneDatasetBase._parse_int_range(cfg, "seq_len_range") == (3, 9)


def test_parse_int_range_missing_no_default_raises() -> None:
    with pytest.raises(KeyError, match="seq_len_range"):
        SceneDatasetBase._parse_int_range({}, "seq_len_range")


def test_parse_camera_mode_lowercases_strings() -> None:
    assert SceneDatasetBase._parse_camera_mode({"camera_mode": "RANDOM"}) == "random"
    assert SceneDatasetBase._parse_camera_mode({"camera_mode": 2}) == 2
    with pytest.raises(KeyError, match="data.camera_mode"):
        SceneDatasetBase._parse_camera_mode({})


def test_resolve_data_cfg_requires_explicit_data_mapping() -> None:
    assert SceneDatasetBase._resolve_data_cfg({"data": {"a": 1}}) == {"a": 1}
    assert SceneDatasetBase._resolve_data_cfg({"data": {}}) == {}
    with pytest.raises(TypeError, match="configuration: expected mapping"):
        SceneDatasetBase._resolve_data_cfg(None)
    with pytest.raises(MissingConfigurationKeyError, match="configuration.data"):
        SceneDatasetBase._resolve_data_cfg({})


# ---------------------------------------------------------------------------
# Window sampling (deterministic with a seeded rng)
# ---------------------------------------------------------------------------


def test_select_window_center_mode(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_frames=10)
    scene = make_scene(num_frames=10)
    w = ds.select_window(scene, seq_len_range=(4, 4), crop_mode="center")
    assert w.seq_len == 4
    # center crop: max_start=6 -> start=3
    assert w.start == 3
    assert w.end == 7
    assert w.full_len == 10


def test_select_window_full_when_seq_ge_full(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_frames=5)
    scene = make_scene(num_frames=5)
    w = ds.select_window(scene, seq_len_range=(5, 5), crop_mode="random")
    assert w.start == 0
    assert w.end == 5


def test_select_window_random_in_bounds(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_frames=12)
    scene = make_scene(num_frames=12)
    for _ in range(20):
        w = ds.select_window(scene, seq_len_range=(2, 6), crop_mode="random")
        assert 2 <= w.seq_len <= 6
        assert w.start >= 0
        assert w.end <= 12
        assert w.end - w.start == w.seq_len


def test_select_window_too_short_raises(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_frames=8)
    short = make_scene(num_frames=2)
    with pytest.raises(ValueError, match="too short"):
        ds.select_window(short, seq_len_range=(5, 5))


def test_select_window_bad_crop_mode_raises(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_frames=8)
    scene = make_scene(num_frames=8)
    with pytest.raises(ValueError, match="Unsupported crop_mode"):
        ds.select_window(scene, crop_mode="weird")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------


def test_select_cameras_first_mode(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_cameras=4)
    scene = make_scene(num_cameras=4)
    sel = ds.select_cameras(scene, num_views_range=(3, 3), camera_mode="first")
    assert sel.indices == (0, 1, 2)


def test_select_cameras_random_unique(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_cameras=5)
    scene = make_scene(num_cameras=5)
    for _ in range(15):
        sel = ds.select_cameras(scene, num_views_range=(2, 4), camera_mode="random")
        assert 2 <= len(sel.indices) <= 4
        assert len(set(sel.indices)) == len(sel.indices)  # unique
        assert all(0 <= i < 5 for i in sel.indices)


def test_select_cameras_int_mode_primary_first(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_cameras=5)
    scene = make_scene(num_cameras=5)
    sel = ds.select_cameras(scene, num_views_range=(3, 3), camera_mode=2)
    assert sel.primary == 2
    assert len(sel.indices) == 3
    assert len(set(sel.indices)) == 3


def test_select_cameras_not_enough_raises(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_cameras=2)
    scene = make_scene(num_cameras=1)
    with pytest.raises(ValueError, match="min_views"):
        ds.select_cameras(scene, num_views_range=(2, 2))


def test_select_camera_returns_primary(make_scene_dataset, make_scene) -> None:
    ds = make_scene_dataset(num_cameras=3)
    scene = make_scene(num_cameras=3)
    cam = ds.select_camera(scene)
    assert isinstance(cam, int)
    assert 0 <= cam < 3


# ---------------------------------------------------------------------------
# Disk-backed construction, indexing, filtering, __getitem__
# ---------------------------------------------------------------------------


def test_dataset_construction_indexes_scenes(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=3, num_frames=8)
    assert len(ds) == 3
    assert len(ds.scenes) == 3
    # headers carry the parsed num_frames / num_cameras
    header = ds.get_scene_header(ds.scenes[0])
    assert header.num_frames == 8
    assert header.num_cameras == 1


def test_dataset_getitem_builds_sample(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=2, num_frames=6)
    sample = ds[0]
    assert sample["num_frames"] == 6
    assert "path" in sample


def test_dataset_filters_short_scenes(make_scene_dataset, scene_writer, tmp_path: Path) -> None:
    # seq_len_range min=5 filters out 3-frame scenes.
    root = tmp_path / "mixed"
    scenes_dir = root / "scenes"

    scene_writer(scenes_dir / "long", num_frames=10)
    scene_writer(scenes_dir / "short", num_frames=3)
    (root / "train.txt").write_text("long\nshort\n", encoding="utf-8")

    cfg = SceneDatasetConfig(
        scene_dir=root,
        split_file=root / "train.txt",
        seq_len_range=(5, 10),
        num_views_range=(1, 1),
        camera_mode="random",
        crop_mode="random",
        min_num_frames=1,
        min_num_cameras=1,
    )
    ds = make_scene_dataset(config=cfg, root=root, n_scenes=0)
    assert len(ds) == 1
    assert ds.scenes[0].name == "long"


def test_dataset_missing_split_file_raises(make_scene_dataset, tmp_path: Path) -> None:
    scene_dir = tmp_path / "nope"
    scene_dir.mkdir()
    cfg = SceneDatasetConfig(
        scene_dir=scene_dir,
        split_file=scene_dir / "train.txt",
        seq_len_range=(1, 4),
        num_views_range=(1, 1),
        camera_mode="random",
        crop_mode="random",
        min_num_frames=1,
        min_num_cameras=1,
    )
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        make_scene_dataset(config=cfg, root=scene_dir, n_scenes=0)


def test_get_scene_header_unknown_path_raises(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=1)
    with pytest.raises(KeyError, match="Scene header not found"):
        ds.get_scene_header(Path("/not/a/scene"))


def test_decode_meta_variants(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=1)
    assert ds._decode_meta({"a": 1}) == {"a": 1}
    assert ds._decode_meta('{"a": 2}') == {"a": 2}
    assert ds._decode_meta(b'{"a": 3}') == {"a": 3}
    assert ds._decode_meta("not json") == {}
    assert ds._decode_meta(123) == {}


def test_resolve_num_frames_warns_on_overflow(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=1)
    payload = {"position": np.zeros((5, 3))}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = ds._resolve_num_frames(path=Path("/x"), meta={"num_frames": 99}, payload=payload)
    assert n == 5  # clamped to available length
    assert any("exceeds available length" in str(w.message) for w in caught)


def test_resolve_num_frames_uses_fallback_when_invalid(make_scene_dataset) -> None:
    ds = make_scene_dataset(n_scenes=1)
    payload = {"position": np.zeros((7, 3))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = ds._resolve_num_frames(path=Path("/x"), meta={"num_frames": 0}, payload=payload)
    assert n == 7


def test_scene_dataset_config_preserves_explicit_contract() -> None:
    cfg = SceneDatasetConfig(
        scene_dir=Path("/a"),
        split_file=Path("/a/train.txt"),
        seq_len_range=(1, 1024),
        num_views_range=(1, 1),
        camera_mode="random",
        crop_mode="random",
        min_num_frames=1,
        min_num_cameras=1,
    )
    assert cfg.seq_len_range == (1, 1024)
    assert cfg.num_views_range == (1, 1)
    assert cfg.camera_mode == "random"
    assert cfg.crop_mode == "random"
