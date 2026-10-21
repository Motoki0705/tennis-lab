"""Schematic completion preserves evidence and cannot masquerade as inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
from src.utils.configuration import SemanticConfigurationError
from src.utils.paths import PROJECT_ROOT


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


def _delivery(root: Path) -> Path:
    source = root / "delivery"
    bundle, shot = source / "bundle", source / "shot"
    (bundle / "plates/cam0").mkdir(parents=True)
    shot.mkdir()
    points = _points()
    valid = np.isfinite(points).all(axis=1)
    states = np.where(valid, 1, 3).astype(np.uint8)[None, :]
    states[:, 10] = 2
    for name, array in {
        "ball_position": points,
        "ball_reason": np.where(valid, 0, 1).astype(np.uint8),
        "ray_state": states,
        "ray_origin": np.array([[8.0, -18.0, 3.0]]),
    }.items():
        np.save(bundle / f"{name}.npy", array)
    manifest = {
        "schema": "tennis_scene_blender_rays_v1",
        "frames": [0, len(points)],
        "fps": 30.0,
        "ball_source": {"kind": "imported_annotations"},
        "provenance": {"source_sha256": "test"},
        "cameras": [
            {"id": "cam0", "center": [8.0, -18.0, 3.0], "plate_dir": "plates/cam0"}
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "plates/cam0/00060.jpg").write_bytes(b"unit test plate")
    (source / "manifest.json").write_text("{}")
    (shot / "shot_manifest.json").write_text(
        json.dumps(
            {
                "shot": {"width": 1920, "height": 1080},
                "source_frames": list(range(len(points))),
                "camera_selection": {"camera_id": "cam0"},
                "ball_source": manifest["ball_source"],
            }
        )
    )
    (shot / "ball_rays.blend").write_bytes(b"unit test blend identity")
    return source


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
