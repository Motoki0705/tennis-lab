"""Schematic completion preserves evidence and cannot masquerade as inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.visualization.blender.export.diffusion import (
    concept_arrays,
    export_diffusion_bundle,
    smooth_trajectory,
    tube_geometry,
)
from src.tennis_scene.visualization.blender.export.diffusion_configuration import (
    parse_diffusion_config,
)
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2
from src.tennis_scene.visualization.blender.export.rays import (
    RaysConfig,
    RaysInput,
    export_rays_bundle,
)
from src.tennis_scene.visualization.blender.shots.ray_timeline import (
    plan_timeline,
    source_window,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.configuration import SemanticConfigurationError
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints
from src.utils.paths import PROJECT_ROOT
from src.utils.video.writer import save_video_rgb


def _runtime(root: Path, *overrides: str):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="export_blender_diffusion",
            overrides=[f"paths.project_root={root}", *overrides],
        )
    return parse_diffusion_config(cfg)


def _points() -> np.ndarray:
    time = np.linspace(0, 2, 61)
    points: np.ndarray = np.column_stack(
        (2 * time, time**2, 1 + 3 * time - 2 * time**2)
    )
    points[1:-1] += np.random.default_rng(3).normal(0, 0.04, (59, 3))
    points[20:26] = np.nan
    return points


def test_smoothing_bounds_all_observations_and_fills_gap() -> None:
    points = _points()
    trajectory, fit = smooth_trajectory(
        points, fps=30.0, smoothing_lambda=1.0, tolerance_m=0.08
    )
    valid = np.isfinite(points).all(axis=1)
    assert np.isfinite(trajectory).all()
    assert (
        np.linalg.norm(trajectory[valid] - points[valid], axis=1).max() <= 0.08 + 1e-9
    )
    assert 0 < fit["effective_lambda"] < 1.0
    assert fit["filled_frames"] == list(range(20, 26))
    assert not fit["model_output"] and not fit["physics_model"]
    # A gap is a continuous part of the same flight, not repeated endpoints.
    assert np.all(np.diff(trajectory[19:27, 0]) > 0.02)
    assert (
        np.linalg.norm(np.diff(trajectory[1:19], n=2, axis=0), axis=1).mean()
        < np.linalg.norm(np.diff(points[1:19], n=2, axis=0), axis=1).mean()
    )


def test_noise_endpoints_seed_and_monotone_convergence(tmp_path: Path) -> None:
    config = _runtime(tmp_path).config
    arrays, details = concept_arrays(_points(), 30.0, config)
    begin, end = details["stages"][1]["output_frames"]
    np.testing.assert_array_equal(arrays["process"][begin], arrays["noise"])
    np.testing.assert_array_equal(arrays["process"][end - 1], arrays["trajectory"])
    np.testing.assert_array_equal(arrays["process"][-1], arrays["trajectory"])
    error = np.linalg.norm(arrays["process"] - arrays["trajectory"], axis=(1, 2))
    assert np.all(np.diff(error) <= 1e-10)
    repeat, _ = concept_arrays(_points(), 30.0, config)
    np.testing.assert_array_equal(repeat["noise"], arrays["noise"])
    different, _ = concept_arrays(
        _points(), 30.0, replace(config, seed=config.seed + 1)
    )
    assert not np.array_equal(different["noise"], arrays["noise"])
    np.testing.assert_array_equal(arrays["ball_position"][-1], arrays["trajectory"][-1])
    assert arrays["ball_visible"].sum() == 270


def test_exported_tube_centers_and_stationary_points() -> None:
    process = np.zeros((3, 8, 3))
    process[1:, :, 0] = np.arange(8)
    vertices, faces, centers = tube_geometry(process, radius=0.04, sides=8)
    assert np.isfinite(vertices).all()
    np.testing.assert_allclose(vertices[:, centers], process)
    assert len(faces) == 7 * 8
    # First eight vertices at each sample lie on its specified-radius ring.
    rings = vertices.reshape(3, 8, 9, 3)[:, :, :8]
    np.testing.assert_allclose(
        np.linalg.norm(rings - process[:, :, None], axis=-1), 0.04, atol=1e-6
    )


@pytest.mark.parametrize("case", ["start", "end", "partial_nan", "infinity", "too_few"])
def test_invalid_evidence_does_not_trigger_an_implicit_fallback(case: str) -> None:
    points = _points()
    if case == "start":
        points[0] = np.nan
    elif case == "end":
        points[-1] = np.nan
    elif case == "partial_nan":
        points[0, 0] = np.nan
    elif case == "infinity":
        points[0, 0] = np.inf
    else:
        points[2:] = np.nan
    with pytest.raises(ValueError):
        smooth_trajectory(points, fps=30.0, smoothing_lambda=0.01, tolerance_m=0.2)


def _delivery(
    root: Path, *, fps: float = 30.0, frame_range: tuple[int, int] | None = None
) -> Path:
    """Use the upstream exporter and source clock, including actual JPEG decoding.

    Only the opaque .blend identity is a stub: Blender execution is outside
    these CPU tests. No delivery-level manifest is generated by the upstream.
    """
    source = root / "delivery"
    bundle, shot = source / "bundle", source / "shot"
    if frame_range is None:
        points = _points()
        start, stop = 0, len(points)
        config = RaysConfig(
            window_frames=stop,
            camera_id="cam0",
            two_d_seconds=0.5,
            pullback_seconds=0.5,
            playback_seconds=1.02,
        )
    else:
        start, stop = frame_range
        points = np.tile([1.0, 2.0, 1.5], (stop, 1))
        points[start + 20 : start + 26] = np.nan
        config = RaysConfig(window_frames=stop - start, camera_id="cam0")
    cameras = []
    for index, center in enumerate(
        ([8.0, -18.0, 3.0], [-8.0, -18.0, 3.0], [0.0, 18.0, 3.0])
    ):
        center = np.asarray(center)
        forward = np.array([0.0, 0.0, 1.0]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(forward, right), forward))
        cameras.append(
            PinholeCamera(
                f"cam{index}",
                np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 18.0], [0.0, 0.0, 1.0]]),
                rotation,
                -rotation @ center,
            )
        )
    frames: np.ndarray = np.zeros((stop, 36, 64, 3), np.uint8)
    # Adjacent source frames have distinct plate content, including 940/941.
    frames[:, :, :, 0] = np.where(np.arange(stop) % 2, 220, 20)[:, None, None]
    video = root / "source.mp4"
    save_video_rgb(frames, video, fps=fps)
    store = ClipStore(
        root / "store",
        {
            "clip_id": "test",
            "videos": [
                {
                    "camera_id": camera.camera_id,
                    "path": str(video),
                    "sha256": dual_sha256(video),
                    "num_frames": stop,
                    "width": 64,
                    "height": 36,
                    "fps": fps,
                }
                for camera in cameras
            ],
        },
    )
    reader = StoreReader(store.index_path)
    valid = np.isfinite(points).all(axis=1)
    positions = np.nan_to_num(points)
    visible: np.ndarray = np.ones((len(cameras), stop), bool)
    visible[1:, ~valid] = False
    inliers = visible & valid
    inliers[-1, start + 10] = False
    ball = BallReconstructionResult(
        np.stack([camera.project(positions)[0] for camera in cameras]),
        visible,
        TriangulatedPoints(
            positions,
            valid,
            np.where(valid, 0, 1).astype(np.uint8),
            inliers,
            np.zeros(visible.shape),
        ),
        "ok",
    )
    inputs = RaysInput(
        reader,
        tuple(cameras),
        ball,
        fps,
        {"source_sha256": reader.source_sha256},
        {"kind": "imported_annotations"},
    )
    upstream = export_rays_bundle(
        inputs,
        bundle,
        start=start,
        stop=stop,
        t0=(start + stop - 1) // 2,
        config=config,
        resolved_config={"mode": "export"},
    )
    manifest = json.loads(upstream.read_text())
    settings = config.shot_timing()
    timeline = plan_timeline(manifest, settings)
    shot.mkdir()
    (shot / "shot_manifest.json").write_text(
        json.dumps(
            {
                "shot": asdict(settings),
                "source_frames": timeline.source_frames.tolist(),
                "camera_selection": manifest["camera_selection"],
                "source_window": source_window(fps, settings),
                "stages": [
                    stage.to_dict(settings.fps, fps) for stage in timeline.stages
                ],
                "ball_source": manifest["ball_source"],
            }
        )
    )
    (shot / "ball_rays.blend").write_bytes(b"unit test blend identity")
    return source


@pytest.mark.parametrize("fps,final_frame", [(60.0, 940), (59.94006, 941)])
def test_upstream_export_and_source_clock_feed_diffusion_without_delivery_manifest(
    tmp_path: Path, fps: float, final_frame: int
) -> None:
    source = _delivery(tmp_path, fps=fps, frame_range=(282, 942))
    assert not (source / "manifest.json").exists()
    shot = json.loads((source / "shot/shot_manifest.json").read_text())
    assert shot["source_frames"][-1] == final_frame
    output = tmp_path / "diffusion/bundle"
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    export_diffusion_bundle(
        source, output, font=font, config=_runtime(tmp_path).config, resolved_config={}
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["camera_continuity"]["source_frame"] == final_frame
    assert (
        manifest["camera_continuity"]["source_output_frame"]
        == len(shot["source_frames"]) - 1
    )
    assert (
        manifest["camera_continuity"]["source_blend_sha256"]
        == hashlib.sha256((source / "shot/ball_rays.blend").read_bytes()).hexdigest()
    )
    assert str((source / "manifest.json").resolve()) not in [
        item["path"] for item in manifest["inputs"]
    ]
    for index in range(3):
        expected = source / f"bundle/plates/cam{index}/{final_frame:05d}.jpg"
        assert (output / f"plate_{index}.jpg").read_bytes() == expected.read_bytes()
        assert str(expected.resolve()) in [item["path"] for item in manifest["inputs"]]
        if final_frame == 940:
            assert (output / f"plate_{index}.jpg").read_bytes() != (
                expected.parent / "00941.jpg"
            ).read_bytes()


@pytest.mark.parametrize("frames", [[], [None], [False], [0.0, 60.0], [-1], [61]])
def test_invalid_upstream_source_frames_fail_explicitly(
    tmp_path: Path, frames: list
) -> None:
    source = _delivery(tmp_path)
    path = source / "shot/shot_manifest.json"
    shot = json.loads(path.read_text())
    shot["source_frames"] = frames
    path.write_text(json.dumps(shot))
    with pytest.raises(ValueError, match="source_frames"):
        export_diffusion_bundle(
            source,
            tmp_path / "failed",
            font=Path("unused-font"),
            config=_runtime(tmp_path).config,
            resolved_config={},
        )
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize(
    "missing",
    [
        "bundle/manifest.json",
        "shot/shot_manifest.json",
        "shot/ball_rays.blend",
        "bundle/plates/cam0/00060.jpg",
    ],
)
def test_missing_upstream_files_do_not_fall_back(tmp_path: Path, missing: str) -> None:
    source = _delivery(tmp_path)
    (source / missing).unlink()
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    with pytest.raises(FileNotFoundError):
        export_diffusion_bundle(
            source,
            tmp_path / "failed",
            font=font,
            config=_runtime(tmp_path).config,
            resolved_config={},
        )
    assert not (tmp_path / "failed").exists()


def test_missing_source_frame_field_does_not_infer_bundle_endpoint(
    tmp_path: Path,
) -> None:
    source = _delivery(tmp_path)
    path = source / "shot/shot_manifest.json"
    shot = json.loads(path.read_text())
    del shot["source_frames"]
    path.write_text(json.dumps(shot))
    with pytest.raises(KeyError, match="source_frames"):
        export_diffusion_bundle(
            source,
            tmp_path / "failed",
            font=Path("unused-font"),
            config=_runtime(tmp_path).config,
            resolved_config={},
        )


def test_bundle_preserves_measured_mask_provenance_and_input_hashes(
    tmp_path: Path,
) -> None:
    source = _delivery(tmp_path)
    runtime = _runtime(tmp_path)
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    output = tmp_path / "new/bundle"
    export_diffusion_bundle(
        source,
        output,
        font=font,
        config=runtime.config,
        resolved_config=runtime.resolved_config,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["origins"] == {
        "conditions": "measured",
        "process": "schematic",
        "trajectory": "schematic",
        "overall": "concept",
        "gap_markers": "schematic",
    }
    assert manifest["ball_source"] == {"kind": "imported_annotations"}
    assert manifest["source_frames"] == [0, 61]
    for item in manifest["inputs"]:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    for name, digest in manifest["assets"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    states = np.load(output / "condition_state.npy")
    assert states[10] == 2 and np.all(states[20:26] == 0) and states[0] == 1
    np.testing.assert_array_equal(np.load(output / "condition_position.npy"), _points())
    np.testing.assert_array_equal(
        np.load(output / "condition_valid.npy"), np.isfinite(_points()).all(axis=1)
    )
    cache, start, rate = read_pc2(output / "process.pc2")
    assert start == 0 and rate == 1
    centers = np.load(output / "tube_center_indices.npy")
    np.testing.assert_allclose(
        cache[:, centers], np.load(output / "process.npy"), atol=1e-6
    )
    with pytest.raises(FileExistsError):
        export_diffusion_bundle(
            source, output, font=font, config=runtime.config, resolved_config={}
        )


def test_failed_bundle_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _delivery(tmp_path)
    output = tmp_path / "failed"
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"

    def fail(*args, **kwargs):
        raise RuntimeError("intentional write failure")

    monkeypatch.setattr(
        "src.tennis_scene.visualization.blender.export.diffusion.write_pc2", fail
    )
    with pytest.raises(RuntimeError, match="intentional"):
        export_diffusion_bundle(
            source,
            output,
            font=font,
            config=_runtime(tmp_path).config,
            resolved_config={},
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "override",
    [
        "seed=-1",
        "noise_knots=2",
        "point_tolerance_m=0",
        "fps=0",
        "noise_scale_m=[1,2]",
        "denoise_seconds=0.01",
        "width=1919",
        "gif_fps=60",
    ],
)
def test_invalid_config_fails_before_reading_inputs(
    tmp_path: Path, override: str
) -> None:
    with pytest.raises(SemanticConfigurationError):
        _runtime(tmp_path, override)


@pytest.mark.parametrize(
    "field,value",
    [("smoothing_lambda", float("nan")), ("noise_center_m", [0, float("inf"), 0])],
)
def test_nonfinite_config_fails_before_reading_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    cfg = OmegaConf.create(_runtime(tmp_path).resolved_config)
    cfg[field] = value
    with pytest.raises(SemanticConfigurationError):
        parse_diffusion_config(cfg)


def test_config_path_roles(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        f"paths.artifact_root={tmp_path}/input",
        f"paths.output_root={tmp_path}/output",
        f"paths.external_asset_root={tmp_path}/fonts",
    )
    assert runtime.input_path.is_relative_to(tmp_path / "input")
    assert runtime.output_path.is_relative_to(
        tmp_path / "output/tennis_scene/visualize/export_blender_diffusion"
    )
    assert runtime.font_path == tmp_path / "fonts/opentype/noto/NotoSansCJK-Regular.ttc"
