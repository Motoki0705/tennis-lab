"""CPU tests for the placement shot timeline and its synthetic bundle fixture.

The Blender-only modules under ``shots/`` are excluded from ruff/mypy and are
verified by running Blender.  This module keeps the numeric contract testable on
the CPU: stage boundaries, input-fps time mapping, intro shift, invalid masks and
input rejection.  ``write_synthetic_bundle`` writes a complete
``tennis_scene_blender_placement_v1`` bundle; the parent can load this module with
``runpy`` and use it for the real Blender build/validation run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.visualization.blender.export.bundle import (
    PLATE_PATTERN,
    court_geometry,
)
from src.tennis_scene.visualization.blender.export.camera import (
    PinholeCamera as ExportCamera,
)
from src.tennis_scene.visualization.blender.export.camera import to_blender_camera
from src.tennis_scene.visualization.blender.export.pointcache import (
    read_pc2,
    write_pc2,
)
from src.tennis_scene.visualization.blender.shots.player_placement import (
    OUTPUT_FPS,
    PlacementShot,
    PlacementTimeline,
    _camera_direction,
    _rotation_z,
    body_forward,
    body_transform_at,
    body_visible_at,
    build_timeline,
    check_output_available,
    load_manifest,
    replay_source_indices,
    require_plate_frames,
    validate_manifest,
    validate_t0,
)
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.encoding import encode_jpeg

VERTICES = 24
RINGS = 6
PER_RING = 4
IDENTITY = 7
DEFAULT_START = 12
DEFAULT_FRAMES = 200
DEFAULT_SIZE = (320, 180)
DEFAULT_SCALE = 1.07

# COCO17 local coordinates, hip-centred (nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles).
_JOINT_LOCAL = np.asarray(
    [
        [0.00, 0.05, 0.72],
        [-0.04, 0.08, 0.76],
        [0.04, 0.08, 0.76],
        [-0.08, 0.04, 0.74],
        [0.08, 0.04, 0.74],
        [-0.18, 0.00, 0.50],
        [0.18, 0.00, 0.50],
        [-0.28, 0.00, 0.22],
        [0.28, 0.00, 0.22],
        [-0.34, 0.00, -0.05],
        [0.34, 0.00, -0.05],
        [-0.10, 0.00, -0.05],
        [0.10, 0.00, -0.05],
        [-0.11, 0.00, -0.50],
        [0.11, 0.00, -0.50],
        [-0.12, 0.00, -0.90],
        [0.12, 0.00, -0.90],
    ],
    np.float64,
)


def _body_mesh() -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    vertices: NDArray[np.float64] = np.empty((VERTICES, 3), np.float64)
    for ring in range(RINGS):
        z = -0.9 + 1.8 * ring / (RINGS - 1)
        for index in range(PER_RING):
            angle = 2.0 * math.pi * index / PER_RING + 0.25 * ring
            vertices[ring * PER_RING + index] = (
                0.16 * math.cos(angle),
                0.12 * math.sin(angle),
                z,
            )
    faces: list[tuple[int, int, int]] = []
    for ring in range(RINGS - 1):
        for index in range(PER_RING):
            nxt = (index + 1) % PER_RING
            a = ring * PER_RING + index
            b = ring * PER_RING + nxt
            c = (ring + 1) * PER_RING + index
            d = (ring + 1) * PER_RING + nxt
            faces.append((a, b, d))
            faces.append((a, d, c))
    return vertices.astype(np.float32), np.asarray(faces, np.int64)


def _target_valid(
    valid: NDArray[np.bool_], start: int, missing: set[tuple[int, int]]
) -> NDArray[np.bool_]:
    result = np.broadcast_to(valid[:, None], (len(valid), 17)).copy()
    for frame, joint in missing:
        local = frame - start
        if 0 <= local < len(result):
            result[local, joint] = False
    return result


def _look_at_camera(
    target: NDArray[np.float64],
    offset: NDArray[np.float64],
    *,
    width: int,
    height: int,
) -> ExportCamera:
    """A plausible calibrated plate camera looking at the representative hip."""
    position = target + offset
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    translation = -rotation @ position
    intrinsic = np.asarray(
        [[260.0, 0.0, width / 2.0], [0.0, 260.0, height / 2.0], [0.0, 0.0, 1.0]]
    )
    return ExportCamera(
        K=intrinsic,
        R=rotation,
        t=translation,
        width=width,
        height=height,
    )


def write_synthetic_bundle(
    output: Path,
    *,
    fps: float = 30.0,
    num_frames: int = DEFAULT_FRAMES,
    start: int = DEFAULT_START,
    t0: int | None = None,
    identity: int = IDENTITY,
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
) -> Path:
    """Write a tiny placement bundle for CPU tests and real Blender checks.

    ``num_frames`` must cover three seconds of the plate (``>= 3 * fps``) when the
    caller wants to verify ``intro_plate=True``; the default 200 frames is enough
    for both 30 and 60 fps.  ``t0`` defaults to ``start + num_frames // 5`` and is
    required to differ from ``start``.  The fixture contains non-trivial scale,
    yaw and hip motion, validity holes, missing target joints and a non-zero
    source start.  Returns the manifest path.
    """
    fps_value = float(fps)
    if not math.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError("fps must be finite and positive.")
    if type(num_frames) is not int or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer.")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer.")
    stop = start + num_frames
    representative = start + max(1, num_frames // 5) if t0 is None else int(t0)
    if not start < representative < stop:
        raise ValueError("t0 must lie inside frames and differ from start.")
    if output.exists():
        raise FileExistsError(f"Bundle output already exists: {output}")

    rng = np.random.default_rng(20261021)
    frame_index = np.arange(num_frames)
    valid: NDArray[np.bool_] = np.ones(num_frames, bool)
    for hole in (3, 4, 20, 55, 56, 57, 120):
        if hole < num_frames:
            valid[hole] = False
    # Keep the representative frame valid and near a full triangulation.
    valid[representative - start] = True
    missing = {
        (start + 30, 7),
        (start + 31, 7),
        (start + 70, 10),
        (start + 90, 3),
        (representative, 4),
    }
    target_valid = _target_valid(valid, start, missing)

    body_base, faces = _body_mesh()
    sway = np.stack(
        [
            0.02 * np.sin(0.05 * frame_index),
            0.012 * np.cos(0.04 * frame_index),
            0.006 * np.sin(0.03 * frame_index),
        ],
        axis=1,
    )
    body_local = (body_base[None] + sway[:, None, :]).astype(np.float32)
    body_local[~valid] = 0.0

    pose = 0.01 * np.stack(
        [
            np.sin(0.05 * frame_index[:, None] + np.arange(17)[None, :]),
            np.cos(0.04 * frame_index[:, None] + np.arange(17)[None, :]),
            0.5 * np.sin(0.03 * frame_index[:, None] + np.arange(17)[None, :]),
        ],
        axis=-1,
    )
    joints_local = _JOINT_LOCAL[None] + pose
    hip_position = np.stack(
        [
            2.0 + 0.6 * np.sin(0.04 * frame_index),
            -1.2 + 0.5 * np.cos(0.03 * frame_index),
            0.95 + 0.03 * np.sin(0.07 * frame_index),
        ],
        axis=1,
    )
    yaw_correction = 0.35 + 0.55 * np.sin(0.02 * frame_index)
    target: NDArray[np.float64] = np.empty((num_frames, 17, 3), np.float64)
    for frame in range(num_frames):
        target[frame] = hip_position[frame] + DEFAULT_SCALE * (
            _rotation_z(float(yaw_correction[frame])) @ joints_local[frame].T
        ).T
    target += rng.normal(0.0, 0.012, size=target.shape)
    target[~target_valid] = np.nan
    target_weights = np.where(
        target_valid,
        0.55
        + 0.40
        * np.abs(
            np.sin(0.05 * frame_index[:, None] + np.arange(17)[None, :])
        ),
        0.0,
    )
    representative_index = representative - start
    placed_vertices_t0 = (
        hip_position[representative_index]
        + DEFAULT_SCALE
        * (
            _rotation_z(float(yaw_correction[representative_index]))
            @ body_local[representative_index].T
        ).T
    ).astype(np.float32)

    output.mkdir(parents=True)
    np.save(output / "smpl_faces.npy", faces)
    write_pc2(output / "body_local.pc2", body_local, start_frame=0.0)
    np.save(output / "joints_local.npy", joints_local)
    np.save(output / "target_joints.npy", target.astype(np.float32))
    np.save(output / "target_weights.npy", target_weights)
    np.save(output / "hip_position.npy", hip_position)
    np.save(output / "yaw_correction.npy", yaw_correction)
    np.save(output / "valid.npy", valid)
    np.save(output / "placed_vertices_t0.npy", placed_vertices_t0)

    plate_directory = output / "plate"
    plate_directory.mkdir()
    for source_frame in range(start, stop):
        image: NDArray[np.uint8] = np.zeros((height, width, 3), np.uint8)
        image[..., 0] = (source_frame * 7) % 256
        image[..., 1] = (source_frame * 13) % 256
        image[..., 2] = (source_frame * 3) % 256
        image[: max(2, height // 20), :, :] = 255
        (plate_directory / f"{source_frame:05d}.jpg").write_bytes(
            encode_jpeg(image, quality=90)
        )

    camera = _look_at_camera(
        hip_position[representative_index],
        np.asarray([-1.0, -10.0, 7.0]),
        width=width,
        height=height,
    )
    pose = to_blender_camera(camera, sensor_width_mm=36.0)
    source_frames = list(range(start, stop))
    manifest = {
        "schema": "tennis_scene_blender_placement_v1",
        "fps": fps_value,
        "width": width,
        "height": height,
        "camera": {
            "opencv": {
                "K": camera.K.tolist(),
                "R": camera.R.tolist(),
                "t": camera.t.tolist(),
            },
            "blender": pose.to_dict(),
        },
        "frames": {
            "start": start,
            "stop": stop,
            "source_frames": source_frames,
        },
        "t0": representative,
        "identity": identity,
        "gvhmr_camera_id": "cam0",
        "scale": DEFAULT_SCALE,
        "coco17_skeleton": [list(edge) for edge in COCO17_SKELETON],
        "court": court_geometry(),
        "plate": {
            "camera_id": "cam0",
            "video": str(output.parent / "synthetic_plate.mp4"),
            "video_sha256": "0" * 64,
            "pattern": PLATE_PATTERN,
            "jpeg_quality": 90,
            "frames": source_frames,
        },
        "source": {
            "clip_id": "synthetic",
            "camera_ids": ["cam0"],
            "pixel_threshold_scale": 1.0,
        },
        "files": {
            "smpl_faces": "smpl_faces.npy",
            "body_local": "body_local.pc2",
            "joints_local": "joints_local.npy",
            "target_joints": "target_joints.npy",
            "target_weights": "target_weights.npy",
            "hip_position": "hip_position.npy",
            "yaw_correction": "yaw_correction.npy",
            "valid": "valid.npy",
            "placed_vertices_t0": "placed_vertices_t0.npy",
        },
        "provenance": {
            "data_origin": "measured",
            "elements": {"body_local": "measured"},
            "git_commit": "0" * 40,
            "resolved_config": {"synthetic": True},
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _bundle(
    tmp_path: Path, *, fps: float = 30.0, num_frames: int = DEFAULT_FRAMES
) -> tuple[Path, dict[str, Any]]:
    bundle = tmp_path / f"bundle-{fps:g}-{num_frames}"
    write_synthetic_bundle(bundle, fps=fps, num_frames=num_frames)
    return bundle, load_manifest(bundle)


def _timeline(
    tmp_path: Path, *, fps: float = 30.0, num_frames: int = DEFAULT_FRAMES, intro: bool = False
) -> tuple[Path, dict[str, Any], PlacementTimeline]:
    bundle, manifest = _bundle(tmp_path, fps=fps, num_frames=num_frames)
    return bundle, manifest, build_timeline(manifest, PlacementShot(intro_plate=intro))


def test_default_stage_boundaries_and_t0_hold(tmp_path: Path) -> None:
    _, manifest, timeline = _timeline(tmp_path)
    assert timeline.stages == (
        ("court", 0, 60),
        ("appear", 60, 120),
        ("scale", 120, 180),
        ("yaw", 180, 240),
        ("position", 240, 300),
        ("final", 300, 330),
        ("replay", 330, 530),
    )
    assert timeline.frame_start == 0
    assert timeline.frame_end == 529
    assert len(timeline.replay_source_frames) == DEFAULT_FRAMES
    t0_index = manifest["t0"] - manifest["frames"]["start"]
    for frame in (0, 59, 60, 119, 120, 179, 180, 239, 240, 299, 300, 329):
        assert timeline.pc2_frame_at(frame) == t0_index
        assert timeline.source_frame_at(frame) == manifest["t0"]
    assert timeline.pc2_frame_at(330) == 0
    assert timeline.source_frame_at(330) == manifest["frames"]["start"]
    assert timeline.source_frame_at(529) == manifest["frames"]["stop"] - 1


def test_intro_shifts_every_stage_by_three_seconds(tmp_path: Path) -> None:
    bundle, manifest, timeline = _timeline(tmp_path, intro=True)
    assert timeline.stages == (
        ("intro", 0, 90),
        ("court", 90, 150),
        ("appear", 150, 210),
        ("scale", 210, 270),
        ("yaw", 270, 330),
        ("position", 330, 390),
        ("final", 390, 420),
        ("replay", 420, 620),
    )
    valid = np.load(bundle / "valid.npy")
    assert len(timeline.intro_source_frames) == 90
    assert timeline.intro_source_frames[0] == manifest["frames"]["start"]
    assert timeline.intro_source_frames[-1] == manifest["frames"]["start"] + 89
    assert not body_visible_at(timeline, 0, valid)
    assert not body_visible_at(timeline, 89, valid)
    assert not body_visible_at(timeline, 149, valid)
    assert body_visible_at(timeline, 150, valid)
    # Intro frames select explicit source frames, never a loop or frozen image.
    assert len(set(timeline.intro_source_frames)) == 90


def test_60fps_intro_samples_three_seconds_at_30fps(tmp_path: Path) -> None:
    _, manifest, timeline = _timeline(tmp_path, fps=60.0, intro=True)
    assert len(timeline.intro_source_frames) == 90
    assert timeline.intro_source_frames[0] == manifest["frames"]["start"]
    assert timeline.intro_source_frames[1] == manifest["frames"]["start"] + 2
    assert timeline.intro_source_frames[-1] == manifest["frames"]["start"] + 178
    # The fixture has at least 3 seconds of plate for the intro.
    assert manifest["frames"]["stop"] - manifest["frames"]["start"] >= 3 * 60


def test_replay_source_frames_follow_real_time() -> None:
    # Output frame k is k/30 s after start, so the source frame is
    # start + round(k * input_fps / 30).  The last source frame is not forced.
    assert replay_source_indices(100, 110, 60.0) == (100, 102, 104, 106, 108)
    assert replay_source_indices(100, 110, 30.0) == tuple(range(100, 110))
    assert replay_source_indices(100, 110, 59.94006) == (100, 102, 104, 106, 108)
    assert replay_source_indices(100, 110, 29.97) == tuple(range(100, 110))
    assert replay_source_indices(100, 110, 24.0) == (
        100,
        101,
        102,
        102,
        103,
        104,
        105,
        106,
        106,
        107,
        108,
        109,
        109,
    )


def test_replay_source_frames_match_fixture_duration(tmp_path: Path) -> None:
    for fps, count in ((30.0, 200), (60.0, 100), (24.0, 250)):
        _, manifest = _bundle(tmp_path, fps=fps)
        timeline = build_timeline(manifest, PlacementShot())
        replay = timeline.replay_source_frames
        assert len(replay) == count, fps
        assert replay[0] == manifest["frames"]["start"]
        # The replay duration stays within half a 30 fps frame of the original.
        assert abs(len(replay) / OUTPUT_FPS - DEFAULT_FRAMES / fps) <= 0.5 / OUTPUT_FPS + 1e-12
        differences = np.diff(np.asarray(replay))
        assert (differences >= 0).all()
        if fps == 30.0:
            np.testing.assert_array_equal(
                replay, np.arange(manifest["frames"]["start"], manifest["frames"]["stop"])
            )
        if fps == 60.0:
            assert replay[:5] == (
                manifest["frames"]["start"],
                manifest["frames"]["start"] + 2,
                manifest["frames"]["start"] + 4,
                manifest["frames"]["start"] + 6,
                manifest["frames"]["start"] + 8,
            )
            # The 60 fps replay is sampled every second source frame; it is not
            # stretched to include the final source frame.
            assert replay[-1] == manifest["frames"]["start"] + 198


def test_body_transform_matches_stage_arithmetic(tmp_path: Path) -> None:
    bundle, manifest, timeline = _timeline(tmp_path)
    scale = float(manifest["scale"])
    start = manifest["frames"]["start"]
    t0_index = manifest["t0"] - start
    hip = np.load(bundle / "hip_position.npy")
    yaw = np.load(bundle / "yaw_correction.npy")
    anchor = hip[t0_index] + np.asarray(PlacementShot().display_offset_m)
    def transform(frame: int) -> tuple[float, float, NDArray[np.float64]]:
        frame_scale, frame_yaw, frame_location = body_transform_at(
            timeline,
            frame,
            scale=scale,
            t0_yaw=float(yaw[t0_index]),
            t0_hip=hip[t0_index],
            anchor=anchor,
            hip_position=hip,
            yaw_correction=yaw,
        )
        return float(frame_scale), float(frame_yaw), np.asarray(frame_location, np.float64)

    assert transform(60)[:2] == pytest.approx((1.0, 0.0))
    np.testing.assert_allclose(transform(60)[2], anchor)
    assert transform(120)[0] == pytest.approx(1.0)
    assert transform(150)[0] == pytest.approx((1.0 + scale) / 2.0)
    assert transform(180)[:2] == pytest.approx((scale, 0.0))
    assert transform(210)[1] == pytest.approx(float(yaw[t0_index]) / 2.0)
    np.testing.assert_allclose(transform(240)[2], anchor)
    np.testing.assert_allclose(
        transform(270)[2], (anchor + hip[t0_index]) / 2.0
    )
    final_scale, final_yaw, final_location = transform(300)
    assert final_scale == pytest.approx(scale)
    assert final_yaw == pytest.approx(float(yaw[t0_index]))
    np.testing.assert_allclose(final_location, hip[t0_index])
    np.testing.assert_allclose(transform(329)[2], hip[t0_index])
    assert transform(330)[1] == pytest.approx(float(yaw[0]))
    np.testing.assert_allclose(transform(330)[2], hip[0])


def test_invalid_frames_hide_objects_without_interpolation(tmp_path: Path) -> None:
    bundle, manifest, timeline = _timeline(tmp_path)
    valid = np.load(bundle / "valid.npy")
    replay_start = timeline.stage_bounds()["replay"][0]
    holes = np.flatnonzero(~valid)
    assert len(holes) > 0
    for source_index in holes:
        frame = replay_start + int(source_index)
        assert timeline.source_frame_at(frame) == manifest["frames"]["start"] + int(
            source_index
        )
        assert not body_visible_at(timeline, frame, valid)
    for source_index in (2, 5, 21):
        frame = replay_start + source_index
        assert body_visible_at(timeline, frame, valid)
    # Replay does not extend validity across a hole: every output frame has its
    # own explicit source index (30 fps here), including the invalid ones.
    np.testing.assert_array_equal(
        np.asarray(timeline.replay_source_frames),
        np.arange(manifest["frames"]["start"], manifest["frames"]["stop"]),
    )


def test_input_rejections_are_explicit(tmp_path: Path) -> None:
    bundle, manifest = _bundle(tmp_path)
    copied = json.loads(json.dumps(manifest))
    copied["schema"] = "wrong"
    with pytest.raises(ValueError, match="Unsupported bundle schema"):
        validate_manifest(copied)

    copied = json.loads(json.dumps(manifest))
    copied["fps"] = 0.0
    with pytest.raises(ValueError, match="fps"):
        validate_manifest(copied)

    copied = json.loads(json.dumps(manifest))
    copied["frames"] = {"start": 20, "stop": 20, "source_frames": []}
    with pytest.raises(ValueError, match="frames"):
        validate_manifest(copied)

    copied = json.loads(json.dumps(manifest))
    copied["t0"] = copied["frames"]["stop"]
    with pytest.raises(ValueError, match="t0"):
        validate_manifest(copied)

    valid = np.load(bundle / "valid.npy")
    invalid = valid.copy()
    invalid[manifest["t0"] - manifest["frames"]["start"]] = False
    with pytest.raises(ValueError, match="not a valid placement frame"):
        validate_t0(
            invalid,
            manifest["t0"],
            manifest["frames"]["start"],
            manifest["frames"]["stop"],
        )

    _, short = _bundle(tmp_path, fps=60.0, num_frames=120)
    with pytest.raises(ValueError, match="intro_plate needs"):
        build_timeline(short, PlacementShot(intro_plate=True))

    timeline = build_timeline(manifest, PlacementShot(intro_plate=True))
    require_plate_frames(bundle, manifest, timeline)
    required = timeline.intro_source_frames[5]
    (bundle / manifest["plate"]["pattern"].format(frame=required)).unlink()
    with pytest.raises(ValueError, match="not present"):
        require_plate_frames(bundle, manifest, timeline)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        check_output_available(existing)


def test_synthetic_bundle_is_a_complete_fixture(tmp_path: Path) -> None:
    bundle, manifest = _bundle(tmp_path)
    start = manifest["frames"]["start"]
    stop = manifest["frames"]["stop"]
    assert manifest["schema"] == "tennis_scene_blender_placement_v1"
    assert manifest["fps"] == 30.0
    assert start == DEFAULT_START
    assert stop == start + DEFAULT_FRAMES
    assert manifest["t0"] != start
    assert manifest["t0"] == start + DEFAULT_FRAMES // 5
    assert len(list((bundle / "plate").iterdir())) == DEFAULT_FRAMES
    assert 3 * manifest["fps"] <= DEFAULT_FRAMES

    body, pc2_start, pc2_rate = read_pc2(bundle / "body_local.pc2")
    assert body.shape == (DEFAULT_FRAMES, VERTICES, 3)
    assert body.dtype == np.float32
    assert pc2_start == 0.0 and pc2_rate == 1.0
    valid = np.load(bundle / "valid.npy")
    assert valid.any() and not valid.all()
    np.testing.assert_array_equal(body[~valid], 0.0)
    assert np.isfinite(body[valid]).all()

    target = np.load(bundle / "target_joints.npy")
    weights = np.load(bundle / "target_weights.npy")
    assert target.shape == (DEFAULT_FRAMES, 17, 3)
    assert target.dtype == np.float32
    assert weights.shape == (DEFAULT_FRAMES, 17)
    has_target = np.isfinite(target).all(-1) & (weights > 0.0)
    assert np.isnan(target[~has_target]).all()
    assert (weights[~has_target] == 0.0).all()
    assert int(has_target[manifest["t0"] - start].sum()) >= 15

    hip = np.load(bundle / "hip_position.npy")
    yaw = np.load(bundle / "yaw_correction.npy")
    assert hip.shape == (DEFAULT_FRAMES, 3)
    assert float(np.ptp(hip, axis=0).max()) > 0.1
    assert float(np.ptp(yaw)) > 0.1
    assert manifest["scale"] == float(DEFAULT_SCALE)
    planned = (
        hip[manifest["t0"] - start]
        + DEFAULT_SCALE
        * (
            _rotation_z(float(yaw[manifest["t0"] - start]))
            @ body[manifest["t0"] - start].T
        ).T
    )
    np.testing.assert_allclose(
        np.load(bundle / "placed_vertices_t0.npy"), planned, atol=1e-6
    )


def test_replay_source_indices_short_and_low_fps_edges() -> None:
    indices = replay_source_indices(5, 17, 24.0)
    assert len(indices) == round(12 * OUTPUT_FPS / 24.0)
    assert indices[0] == 5
    assert max(indices) < 17
    assert (np.diff(np.asarray(indices)) >= 0).all()
    assert set(np.diff(np.asarray(indices))).issubset({0, 1})
    one_frame = replay_source_indices(5, 6, 1.0)
    assert len(one_frame) == 30
    assert one_frame == (5,) * 30
    two_frames = replay_source_indices(5, 7, 1.0)
    assert len(two_frames) == 60
    assert set(two_frames) == {5, 6}
    assert max(two_frames) < 7
    with pytest.raises(ValueError):
        replay_source_indices(5, 5, 30.0)
    with pytest.raises(ValueError):
        replay_source_indices(5, 17, 0.0)


def test_body_forward_and_camera_angle() -> None:
    shot = PlacementShot()
    joints = _JOINT_LOCAL.copy()
    yaw = 0.42
    forward = body_forward(yaw, joints)
    assert forward.shape == (3,)
    assert forward[2] == 0.0
    assert float(np.linalg.norm(forward)) == pytest.approx(1.0)
    direction = _camera_direction(shot, forward, 0.0)
    assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
    assert float(np.dot(forward, direction)) == pytest.approx(
        math.cos(math.radians(shot.camera_azimuth_offset_deg)), abs=1e-12
    )
    orbit = _camera_direction(shot, forward, 1.0)
    assert float(np.dot(forward, orbit)) == pytest.approx(
        math.cos(
            math.radians(
                shot.camera_azimuth_offset_deg + shot.camera_orbit_deg
            )
        ),
        abs=1e-12,
    )
    flipped = body_forward(yaw + math.pi, joints)
    np.testing.assert_allclose(flipped, -forward, atol=1e-12)
    flipped_direction = _camera_direction(shot, flipped, 0.0)
    np.testing.assert_allclose(flipped_direction, -direction, atol=1e-12)
    relative = direction * shot.camera_distance_m
    flipped_relative = flipped_direction * shot.camera_distance_m
    np.testing.assert_allclose(flipped_relative, -relative, atol=1e-12)
    degenerate = joints.copy()
    degenerate[12] = degenerate[11]
    with pytest.raises(ValueError, match="hip"):
        body_forward(0.0, degenerate)
    non_finite = joints.copy()
    non_finite[11, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        body_forward(0.0, non_finite)


def test_replay_orbit_is_anchored_to_t0_forward() -> None:
    shot = PlacementShot()
    joints = _JOINT_LOCAL.copy()
    t0_yaw = 0.2
    t0_forward = body_forward(t0_yaw, joints)
    count = 100
    directions = [
        _camera_direction(shot, t0_forward, index / (count - 1))
        for index in range(count)
    ]
    azimuths = np.unwrap(
        np.asarray([math.atan2(item[1], item[0]) for item in directions])
    )
    steps = np.diff(azimuths)
    expected_total = math.radians(shot.camera_orbit_deg)
    assert float(azimuths[-1] - azimuths[0]) == pytest.approx(expected_total)
    np.testing.assert_allclose(steps, expected_total / (count - 1), atol=1e-12)
    assert bool((steps > 0).all())
    # The body orientation changes strongly across the replay; the fixed
    # t0-forward orbit must not use it.  The scene-level check is performed by
    # validate_placement.py on the actual camera positions.
    varying = [
        body_forward(t0_yaw + math.pi * math.sin(index / 10.0), joints)
        for index in range(count)
    ]
    assert not np.allclose(varying[0], varying[-1])


def test_fixture_t0_camera_is_in_front_of_the_body(tmp_path: Path) -> None:
    bundle, manifest = _bundle(tmp_path)
    joints = np.load(bundle / "joints_local.npy")
    yaw = np.load(bundle / "yaw_correction.npy")
    t0_index = manifest["t0"] - manifest["frames"]["start"]
    forward = body_forward(float(yaw[t0_index]), joints[t0_index])
    direction = _camera_direction(PlacementShot(), forward, 0.0)
    assert float(np.dot(forward, direction)) == pytest.approx(
        math.cos(math.radians(PlacementShot().camera_azimuth_offset_deg)),
        abs=1e-12,
    )
