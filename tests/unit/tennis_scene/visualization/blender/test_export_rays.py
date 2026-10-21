"""Analytic geometry and saved-data contract tests for the rays bundle."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.visualization.blender.export.rays import (
    RayLengthRule,
    RaysConfig,
    RayState,
    export_ranking,
    export_rays_bundle,
    load_rays_input,
    rank_windows,
    ray_arrays,
    validate_scene_ball,
)
from src.tennis_scene.visualization.blender.export.rays_configuration import (
    parse_rays_config,
)
from src.utils.checksum import dual_sha256
from src.utils.configuration import SemanticConfigurationError
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints
from src.utils.paths import PROJECT_ROOT
from src.utils.video.writer import save_video_rgb


def cameras() -> tuple[PinholeCamera, ...]:
    result = []
    for i, center in enumerate(
        ([8.0, -15.0, 6.0], [-12.0, 10.0, 7.0], [13.0, 14.0, 5.0])
    ):
        c = np.array(center)
        forward = np.array([0.0, 0.0, 1.0]) - c
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        result.append(
            PinholeCamera(
                f"cam{i}",
                np.array([[900.0, 0.0, 310.0], [0.0, 820.0, 245.0], [0.0, 0.0, 1.0]]),
                rotation,
                -rotation @ c,
            )
        )
    return tuple(result)


def ball_fixture(frames: int = 6) -> BallReconstructionResult:
    points = np.tile(np.array([[1.0, 2.0, 1.5]], np.float32), (frames, 1))
    uv = np.stack([cam.project(points)[0] for cam in cameras()]).astype(np.float32)
    return BallReconstructionResult(
        uv,
        np.ones((3, frames), bool),
        TriangulatedPoints(
            points,
            np.ones(frames, bool),
            np.zeros(frames, np.uint8),
            np.ones((3, frames), bool),
            np.zeros((3, frames), np.float32),
        ),
        "ok",
    )


@pytest.mark.parametrize("noise_px", [0.0, 12.0])
def test_rays_and_closest_points_match_analytic_solution(noise_px: float) -> None:
    ball = ball_fixture()
    ball.uv_px[1, :, 0] += noise_px
    arrays = ray_arrays(cameras(), ball, fallback_ray_length_m=60.0)
    for view, cam in enumerate(cameras()):
        pixels = np.r_[ball.uv_px[view, 0], 1.0]
        expected = cam.rotation.T @ np.linalg.solve(cam.intrinsic, pixels)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            arrays["ray_direction"][view], np.tile(expected, (6, 1)), atol=1e-12
        )
        point = ball.trajectory.positions[0]
        closest = cam.center + np.dot(point - cam.center, expected) * expected
        np.testing.assert_allclose(
            arrays["closest_point"][view], np.tile(closest, (6, 1)), atol=1e-12
        )
        if noise_px == 0:
            np.testing.assert_allclose(
                arrays["closest_point"][view], ball.trajectory.positions, atol=2e-6
            )


def test_five_states_mask_invalid_coordinates() -> None:
    ball = ball_fixture()
    ball.visibility[:, 2:] = False
    ball.visibility[0, 2] = True
    ball.visibility[:2, 3] = True
    ball.trajectory.valid[2:] = False
    ball.trajectory.reasons[2:] = 1
    ball.trajectory.inliers[:, 2:] = False
    ball.trajectory.inliers[2, 1] = False
    arrays = ray_arrays(cameras(), ball, fallback_ray_length_m=60.0)
    assert arrays["ray_state"].dtype == np.uint8
    np.testing.assert_array_equal(
        arrays["ray_state"][:, :5], [[1, 1, 3, 4, 0], [1, 1, 0, 4, 0], [1, 2, 0, 0, 0]]
    )
    assert set(np.unique(arrays["ray_state"])) == {int(state) for state in RayState}
    assert np.isnan(arrays["ray_direction"][~ball.visibility]).all()
    assert np.isnan(arrays["ball_position"][2:]).all()
    assert np.isnan(arrays["closest_point"][:, 2:]).all()
    assert np.isnan(arrays["detection_uv"][~ball.visibility]).all()


def test_three_length_rules_are_explicit() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    camera = PinholeCamera("c", intrinsic, rotation, -rotation @ [0.0, 0.0, 5.0])
    uv = np.array(
        [[[50.0, 50.0], [50.0, 150.0], [50.0, -50.0], [50.0, 50.0]]], np.float32
    ).repeat(2, axis=0)
    visible = np.array([[True, True, True, False]]).repeat(2, axis=0)
    valid = np.array([True, False, False, False])
    trajectory = TriangulatedPoints(
        np.array([[10.0, 0.0, 5.0]] * 4, np.float32),
        valid,
        np.array([0, 1, 1, 1], np.uint8),
        visible & valid,
        np.zeros((2, 4), np.float32),
    )
    ball = BallReconstructionResult(uv, visible, trajectory, "ok")
    arrays = ray_arrays((camera, camera), ball, fallback_ray_length_m=42.0)
    np.testing.assert_allclose(
        arrays["ray_length"][0, :3], [11.5, 5 * np.sqrt(2), 42.0]
    )
    np.testing.assert_array_equal(
        arrays["ray_length_rule"][0],
        [
            RayLengthRule.BALL,
            RayLengthRule.GROUND,
            RayLengthRule.FALLBACK,
            RayLengthRule.UNOBSERVED,
        ],
    )
    assert np.isnan(arrays["ray_length"][0, 3])


def rank_fixture() -> BallReconstructionResult:
    ball = ball_fixture(20)
    ball.trajectory.inliers[2, 2] = False
    ball.visibility[:2, 8] = False
    ball.trajectory.valid[8] = False
    ball.trajectory.reasons[8] = 1
    ball.trajectory.inliers[:, 8] = False
    return ball


def test_rank_requires_outlier_and_single_view_and_sorts_by_valid_ratio() -> None:
    ball = rank_fixture()
    ball.trajectory.valid[0] = False
    ball.trajectory.reasons[0] = 1
    ranked = rank_windows(ball, window_frames=12)
    assert ranked
    # Lower-validity and adjacent windows cannot displace the best candidate
    # or fill the result list with the same event.
    assert [item["start"] for item in ranked] == [1]
    for item in ranked:
        start, stop, t0 = (item[key] for key in ("start", "stop", "t0"))
        assert stop - start == 12
        assert t0 == (start + stop - 1) // 2
        assert item["outlier_frames"] and item["single_view_frames"]
    assert [r["valid_ratio"] for r in ranked] == sorted(
        (r["valid_ratio"] for r in ranked), reverse=True
    )
    ball.trajectory.inliers[2, 2] = True
    assert rank_windows(ball, window_frames=12) == []
    ball = rank_fixture()
    ball.visibility[:2, 8] = True
    assert rank_windows(ball, window_frames=12) == []
    ball = rank_fixture()
    ball.trajectory.inliers[2, :] = False
    ranked = rank_windows(ball, window_frames=12)
    assert ranked and ranked[0]["t0_inlier_views"] == 2


@pytest.mark.parametrize("suppression", ["non_overlapping", "t0_distance"])
def test_rank_suppresses_neighbours_before_limiting_to_ten(suppression: str) -> None:
    ball = ball_fixture(144)
    for base in range(0, 144, 12):
        ball.trajectory.inliers[2, base + 2] = False
        ball.visibility[:2, base + 8] = False
        ball.trajectory.valid[base + 8] = False
        ball.trajectory.reasons[base + 8] = 1
        ball.trajectory.inliers[:, base + 8] = False
    ranked = rank_windows(ball, window_frames=12, suppression=suppression)
    assert len(ranked) == 10
    assert [item["start"] for item in ranked] == list(range(0, 120, 12))
    for index, left in enumerate(ranked):
        for right in ranked[index + 1 :]:
            if suppression == "non_overlapping":
                assert left["stop"] <= right["start"] or right["stop"] <= left["start"]
            else:
                assert abs(left["t0"] - right["t0"]) >= 12
    # Windows meeting at the half-open boundary are distinct.
    assert ranked[0]["stop"] == ranked[1]["start"]


def test_rank_records_event_counts_and_reports_only_available_candidates() -> None:
    ball = rank_fixture()
    ball.trajectory.valid[3] = False
    ball.trajectory.reasons[3] = 4
    ball.trajectory.inliers[:, 3] = False
    ranked = rank_windows(ball, window_frames=12)
    assert len(ranked) == 1
    assert ranked[0]["event_counts"] == {"outlier": 1, "single_view": 1, "rejected": 1}
    assert ranked[0]["rejected_frames"] == [3]


def test_midpoint_metadata_makes_distance_suppression_match_non_overlap() -> None:
    ball = ball_fixture(24)
    ball.trajectory.inliers[2, :] = False
    ball.trajectory.inliers[2, [4, 16]] = True
    for frame in (8, 20):
        ball.visibility[:2, frame] = False
        ball.trajectory.valid[frame] = False
        ball.trajectory.reasons[frame] = 1
        ball.trajectory.inliers[:, frame] = False
    non_overlapping = rank_windows(
        ball, window_frames=12, suppression="non_overlapping"
    )
    separated_t0 = rank_windows(ball, window_frames=12, suppression="t0_distance")
    assert [item["start"] for item in non_overlapping] == [0, 12]
    assert separated_t0 == non_overlapping
    assert [item["t0"] for item in separated_t0] == [5, 17]


def test_reject_bad_status_and_scene_disagreement() -> None:
    ball = ball_fixture()
    with pytest.raises(ValueError, match="status"):
        ray_arrays(
            cameras(),
            replace(ball, status="ball_insufficient_support"),
            fallback_ray_length_m=60.0,
        )
    scene = {
        "ball_3d": ball.trajectory.positions.copy(),
        "ball_3d_valid": ball.trajectory.valid.copy(),
    }
    validate_scene_ball(ball, scene)
    scene["ball_3d"][0, 0] += 1
    with pytest.raises(ValueError, match="ball_3d disagrees"):
        validate_scene_ball(ball, scene)
    scene["ball_3d_valid"][0] = False
    with pytest.raises(ValueError, match="ball_3d_valid"):
        validate_scene_ball(ball, scene)


def config(tmp_path: Path, *overrides: str) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        return compose(
            config_name="export_blender_rays",
            overrides=[
                f"paths.project_root={tmp_path}",
                "input=clip/store",
                *overrides,
            ],
        )


def test_config_has_explicit_roles_and_export_selection(tmp_path: Path) -> None:
    runtime = parse_rays_config(config(tmp_path))
    assert runtime.input_path == tmp_path / "outputs/clip/store"
    assert runtime.output_path.is_relative_to(
        tmp_path / "outputs/tennis_scene/visualize/export_blender_rays"
    )
    assert runtime.config.fallback_ray_length_m == 60
    assert runtime.config.ranking_suppression == "non_overlapping"
    assert (
        parse_rays_config(
            config(tmp_path, "ranking_suppression=t0_distance")
        ).config.ranking_suppression
        == "t0_distance"
    )
    with pytest.raises(SemanticConfigurationError, match="requires"):
        parse_rays_config(config(tmp_path, "mode=export"))
    with pytest.raises(SemanticConfigurationError, match="does not accept"):
        parse_rays_config(config(tmp_path, "start=0"))


@pytest.mark.parametrize(
    "override",
    [
        "mode=unknown",
        "window_frames=0",
        "ranking_suppression=unknown",
        "plate_max_width=-1",
        "fallback_ray_length_m=0",
        "plate_jpeg_quality=101",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, override: str) -> None:
    with pytest.raises(SemanticConfigurationError):
        parse_rays_config(config(tmp_path, override))


def small_export_config(**kwargs: Any) -> RaysConfig:
    # The synthetic video is 60fps; 0.2s at 1x needs 12 source frames.
    return RaysConfig(
        window_frames=12, time_mode="replay", playback_seconds=0.2, **kwargs
    )


def make_store(tmp_path: Path, *, adapter: bool, imported: bool) -> ClipStore:
    video = tmp_path / "source.mp4"
    save_video_rgb(np.zeros((20, 48, 64, 3), np.uint8), video, fps=60.0)
    videos = [
        {
            "camera_id": cam.camera_id,
            "path": str(video),
            "sha256": dual_sha256(video),
            "num_frames": 20,
            "width": 64,
            "height": 48,
            "fps": 60.0,
        }
        for cam in cameras()
    ]
    store = ClipStore(tmp_path / "store", {"clip_id": "test", "videos": videos})
    codec: ArtifactCodec[dict[str, Any]] = ArtifactCodec(dict)
    refs = {}
    for cam in cameras():
        node = f"ball_detection/{cam.camera_id}"
        provenance = (
            {
                "origin": "import",
                "importer": "video_ball_annotation_v2",
                "review": {"reviewer": "AI-assisted"},
            }
            if imported
            else {"origin": "component", "implementation": "detector-v1"}
        )
        ref = store.publish(
            node,
            {},
            codec,
            schema="ball_detections",
            version=1,
            identity={"annotation_sha256": "example"},
            dependencies={},
            provenance=provenance,
        )
        if adapter:
            ref = store.publish(
                f"ball_points/{cam.camera_id}",
                {},
                codec,
                schema="ball_points",
                version=3,
                identity={},
                dependencies={"detections": ref},
                provenance={"origin": "component"},
            )
        refs[f"ball_{cam.camera_id}"] = ref
    alignment = store.publish(
        "camera_alignment",
        {
            "geometry": {
                "camera_ids": [cam.camera_id for cam in cameras()],
                "cameras": [asdict(cam) for cam in cameras()],
            }
        },
        codec,
        schema="aligned_cameras",
        version=1,
        identity={},
        dependencies=refs,
        provenance={"origin": "component"},
    )
    store.publish(
        "ball_triangulation",
        {"ball": asdict(rank_fixture())},
        codec,
        schema="ball_trajectory",
        version=1,
        identity={},
        dependencies={**refs, "alignment": alignment},
        provenance={"origin": "component"},
    )
    return store


@pytest.mark.parametrize(
    "adapter,imported", [(False, True), (True, True), (True, False)]
)
def test_export_reads_lineage_without_mutating_store_and_writes_partial_plates(
    tmp_path: Path, adapter: bool, imported: bool
) -> None:
    store = make_store(tmp_path, adapter=adapter, imported=imported)
    before = {
        str(path.relative_to(store.root)): dual_sha256(path)
        for path in store.root.rglob("*")
        if path.is_file()
    }
    inputs = load_rays_input(store.root)
    assert {item["kind"] for item in inputs.ball_source["artifacts"].values()} == {
        "imported_annotations" if imported else "detector"
    }
    out = tmp_path / "bundle"
    manifest = export_rays_bundle(
        inputs,
        out,
        start=1,
        stop=13,
        t0=6,
        config=small_export_config(plate_max_width=32),
        resolved_config={"mode": "export"},
    )
    document = json.loads(manifest.read_text())
    assert document["frames"] == [1, 13] and document["t0"] == 6
    assert document["source_window"]["window_frames"] == 12
    assert document["source_window"]["source_speed"] == 1.0
    assert document["origins"]["rays_and_frusta"] == "schematic"
    assert np.load(out / "ray_direction.npy").shape == (3, 12, 3)
    assert np.load(out / "ray_origin.npy").shape == (3, 3)
    assert np.load(out / "ray_state.npy").dtype == np.uint8
    assert np.load(out / "ball_reason.npy").dtype == np.uint8
    assert np.isnan(np.load(out / "ball_position.npy")[7]).all()
    from PIL import Image

    for cam in cameras():
        plates = sorted((out / "plates" / cam.camera_id).glob("*.jpg"))
        assert [p.stem for p in plates] == [f"{i:05d}" for i in range(1, 13)]
        with Image.open(plates[0]) as image:
            assert image.size == (32, 24)
    with pytest.raises(FileExistsError):
        export_rays_bundle(
            inputs,
            out,
            start=1,
            stop=13,
            t0=6,
            config=small_export_config(),
            resolved_config={},
        )
    assert before == {
        str(path.relative_to(store.root)): dual_sha256(path)
        for path in store.root.rglob("*")
        if path.is_file()
    }


def test_export_allows_invalid_midpoint_because_playback_does_not_use_t0(
    tmp_path: Path,
) -> None:
    inputs = load_rays_input(make_store(tmp_path, adapter=False, imported=True).root)
    assert not inputs.ball.trajectory.valid[8]
    manifest = export_rays_bundle(
        inputs,
        tmp_path / "bundle",
        start=0,
        stop=12,
        t0=8,
        config=small_export_config(),
        resolved_config={},
    )
    assert json.loads(manifest.read_text())["t0"] == 8


def test_export_rejects_wrong_window_length_and_budget_before_writing(
    tmp_path: Path,
) -> None:
    inputs = load_rays_input(make_store(tmp_path, adapter=False, imported=True).root)
    with pytest.raises(ValueError, match="interval must match"):
        export_rays_bundle(
            inputs,
            tmp_path / "bundle",
            start=0,
            stop=11,
            t0=5,
            config=small_export_config(),
            resolved_config={},
        )
    with pytest.raises(ValueError, match="1x source budget"):
        export_ranking(
            inputs,
            tmp_path / "ranking.json",
            config=RaysConfig(window_frames=12),
            resolved_config={},
        )
    assert (
        not (tmp_path / "bundle").exists() and not (tmp_path / "ranking.json").exists()
    )


def test_export_failure_removes_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = load_rays_input(make_store(tmp_path, adapter=False, imported=True).root)
    out = tmp_path / "bundle"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("decode failed")

    monkeypatch.setattr(
        "src.tennis_scene.visualization.blender.export.rays._write_plate", fail
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        export_rays_bundle(
            inputs,
            out,
            start=0,
            stop=12,
            t0=6,
            config=small_export_config(),
            resolved_config={},
        )
    assert not out.exists()
    with pytest.raises(ValueError, match="t0"):
        export_rays_bundle(
            inputs,
            out,
            start=0,
            stop=12,
            t0=12,
            config=small_export_config(),
            resolved_config={},
        )
    assert not out.exists()


def test_ranking_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    inputs = load_rays_input(make_store(tmp_path, adapter=False, imported=True).root)
    output = tmp_path / "rank/ranking.json"
    export_ranking(inputs, output, config=small_export_config(), resolved_config={})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_ranking(inputs, output, config=small_export_config(), resolved_config={})
    assert output.read_bytes() == original


def test_completed_scene_export_is_checked(tmp_path: Path) -> None:
    store = make_store(tmp_path, adapter=False, imported=True)
    ball = rank_fixture()
    scene = store.root / "scene.npz"
    sidecar = scene.with_suffix(".metadata.json")
    sidecar.write_text("{}")
    np.savez(
        scene,
        ball_3d=ball.trajectory.positions + 1,
        ball_3d_valid=ball.trajectory.valid,
    )
    store.record_export(
        "scene",
        {"scene": scene, "metadata": sidecar},
        {"ball_triangulation": store.references()["ball_triangulation"]},
    )
    with pytest.raises(ValueError, match="ball_3d disagrees"):
        load_rays_input(store.root)
