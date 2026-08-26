"""Opt-in real-CUDA smoke test for BLCS joint Gaussian rasterization."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.composition import GaussianAsset
from src.synthetic_data_generation.dataset.blcs import build_ball_gaussian_asset
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallGaussianSettings,
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTComposedRenderClient,
    NHTComposedRenderCommandRequest,
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform

_ENABLE_ENVIRONMENT_VARIABLE = "TENNIS_RUN_B00_COMPOSED_NHT_GPU_TEST"
_SOURCE_FRAMES = (585, 586, 587)
_CAMERA_INDEX = 4
_OBJECT_INDEX = 4


@pytest.mark.skipif(
    os.environ.get(_ENABLE_ENVIRONMENT_VARIABLE) != "1",
    reason=f"set {_ENABLE_ENVIRONMENT_VARIABLE}=1 inside the shared GPU queue",
)
def test_real_b00_joint_gaussian_rasterization() -> None:
    """Move one real ball asset through B00 and validate joint RGB/depth/ID output."""
    repository = Path(__file__).resolve().parents[3]
    repro_value = os.environ.get("TENNIS_REPRO_DIR")
    if repro_value is None:
        pytest.fail("The real B00 CUDA smoke test must run through training-queue.")
    repro_root = Path(cast(str, repro_value)).resolve() / "b00-composed-nht-smoke"
    repro_root.mkdir(parents=True, exist_ok=False)

    scene_path = (
        repository
        / "data/synthetic_data_generation/scenes/B00/reconstruction/export/scene.json"
    )
    plan_root = (
        repository
        / "data/synthetic_data_generation/scenes/B00/datasets/blcs/samples"
        / "B00-blcs-000000"
    )
    alignment_path = (
        repository
        / "data/synthetic_data_generation/scenes/B00/alignment/alignment.json"
    )
    for required in (scene_path, plan_root / "plan.json", plan_root / "plan.npz", alignment_path):
        if not required.is_file():
            pytest.fail(f"Required B00 CUDA fixture is unavailable: {required}")

    plan = _json_mapping(plan_root / "plan.json")
    alignment = _json_mapping(alignment_path)
    with np.load(plan_root / "plan.npz", allow_pickle=False) as archive:
        positions_scene = np.array(archive["positions_scene"], dtype=np.float64)
        present = np.array(archive["present"], dtype=np.bool_)
        geometric_visible = np.array(archive["geometric_visible"], dtype=np.bool_)
        camera_uv = np.array(archive["camera_uv"], dtype=np.float64)
        camera_depth = np.array(archive["camera_depth"], dtype=np.float64)

    selected = np.asarray(_SOURCE_FRAMES, dtype=np.int64)
    if not bool(np.all(present[selected, _OBJECT_INDEX])) or not bool(
        np.all(geometric_visible[selected, _CAMERA_INDEX, _OBJECT_INDEX])
    ):
        pytest.fail("The fixed B00 smoke-test trajectory is no longer visible.")

    metric_adapter = cast(dict[str, Any], alignment["metric_scene_adapter"])
    nht_units_per_metre = float(metric_adapter["nht_scene_units_per_metre"])
    if not math.isfinite(nht_units_per_metre) or nht_units_per_metre <= 0.0:
        pytest.fail("B00 metric/NHT scale is invalid.")

    composition_root = repro_root / "composition"
    composition_root.mkdir()
    settings = _write_ball_asset(repository, composition_root)
    _write_timeline(
        composition_root,
        plan=plan,
        positions_scene=positions_scene,
        nht_units_per_metre=nht_units_per_metre,
    )
    composition_path = _write_composition_request(
        composition_root,
        settings=settings,
    )
    cameras = _probe_cameras(
        plan=plan,
        nht_units_per_metre=nht_units_per_metre,
    )

    output = repro_root / "render"
    camera_request_path = repro_root / "cameras.json"
    executable_value = os.environ.get("NHT_RENDER_EXECUTABLE")
    executable: str | Path = (
        Path(executable_value) if executable_value is not None else "nht-render"
    )
    request = NHTComposedRenderCommandRequest(
        base=NHTRenderCommandRequest(
            scene_path=scene_path.resolve(strict=True),
            output_directory=output,
            arbitrary_cameras=NHTRenderRequest(cameras=cameras),
            arbitrary_request_path=camera_request_path,
            executable=executable,
        ),
        composition_request_path=composition_path.resolve(strict=True),
    )
    result = NHTComposedRenderClient().render_composed(
        request,
        environment={
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "TENNIS_B00_COMPOSED_CUDA_VISIBLE_DEVICES", "0"
            ),
            "PYTHONPATH": str((repository / "third_party/nht").resolve()),
        },
        timeout_seconds=900.0,
    )

    assert result.scene_id == "B00"
    assert result.appearance_model == "direct_linear_rgb"
    assert result.rasterization == "joint_3dgs_eval3d_transmittance_v1"
    assert result.cuda_peak_bytes > 0
    assert len(result.background.records) == len(cameras)
    assert len(result.chunks) == 1
    arrays = result.chunks[0].load_arrays()
    background = result.background.records[_CAMERA_INDEX].load_arrays()

    expected_uv = camera_uv[selected, _CAMERA_INDEX, _OBJECT_INDEX]
    expected_depth_m = camera_depth[selected, _CAMERA_INDEX, _OBJECT_INDEX]
    centroid_errors: list[float] = []
    depth_errors_m: list[float] = []
    maximum_rgb_deltas: list[float] = []
    foreground_depth_fractions: list[float] = []
    pixel_counts: list[int] = []
    median_joint_rgbs: list[list[float]] = []
    green_pixel_fractions: list[float] = []
    for local_frame in range(len(_SOURCE_FRAMES)):
        sample_index = local_frame * len(cameras) + _CAMERA_INDEX
        start = int(arrays.offsets[sample_index])
        stop = int(arrays.offsets[sample_index + 1])
        pixels = arrays.pixel_indices[start:stop]
        assert len(pixels) >= 4
        pixel_counts.append(len(pixels))
        np.testing.assert_array_equal(arrays.instance_ids[start:stop], 1)

        xs = pixels % cameras[_CAMERA_INDEX].width
        ys = pixels // cameras[_CAMERA_INDEX].width
        centroid = np.asarray((float(np.mean(xs)), float(np.mean(ys))))
        centroid_error = float(np.linalg.norm(centroid - expected_uv[local_frame]))
        assert centroid_error <= 2.0
        centroid_errors.append(centroid_error)

        metric_depth = arrays.depth[start:stop] / nht_units_per_metre
        depth_error = abs(float(np.median(metric_depth)) - expected_depth_m[local_frame])
        assert depth_error <= 0.05
        depth_errors_m.append(depth_error)

        background_rgb = background.rgb.reshape(-1, 3)[pixels]
        rgb_delta = np.abs(arrays.rgb[start:stop] - background_rgb).max(axis=1)
        maximum_rgb_delta = float(np.max(rgb_delta))
        assert maximum_rgb_delta >= 0.03
        maximum_rgb_deltas.append(maximum_rgb_delta)

        background_depth = background.depth.reshape(-1)[pixels]
        positive_background = background_depth > 0.0
        assert bool(np.any(positive_background))
        foreground_fraction = float(
            np.mean(
                arrays.depth[start:stop][positive_background]
                < background_depth[positive_background]
            )
        )
        assert foreground_fraction >= 0.5
        foreground_depth_fractions.append(foreground_fraction)
        assert bool(np.all(arrays.alpha[start:stop] > 0.0))

        joint_rgb = arrays.rgb[start:stop]
        median_rgb = np.median(joint_rgb, axis=0)
        green_fraction = float(
            np.mean(
                (joint_rgb[:, 1] > joint_rgb[:, 0])
                & (joint_rgb[:, 1] > joint_rgb[:, 2])
            )
        )
        assert green_fraction >= 0.5
        assert median_rgb[1] >= median_rgb[0] + 0.05
        assert median_rgb[1] >= median_rgb[2] + 0.25
        median_joint_rgbs.append(median_rgb.tolist())
        green_pixel_fractions.append(green_fraction)

        joint_frame = background.rgb.copy().reshape(-1, 3)
        joint_frame[pixels] = joint_rgb
        joint_image = Image.fromarray(
            (joint_frame.reshape(background.rgb.shape) * 255.0 + 0.5).astype(
                np.uint8
            )
        )
        centre_x, centre_y = expected_uv[local_frame]
        joint_image.crop(
            (
                int(round(centre_x)) - 32,
                int(round(centre_y)) - 32,
                int(round(centre_x)) + 32,
                int(round(centre_y)) + 32,
            )
        ).save(repro_root / f"joint-frame-{local_frame:03d}-camera-04-crop.png")

    absent_sample_start = len(_SOURCE_FRAMES) * len(cameras)
    for camera_index in range(len(cameras)):
        sample_index = absent_sample_start + camera_index
        absent_start = int(arrays.offsets[sample_index])
        absent_stop = int(arrays.offsets[sample_index + 1])
        assert absent_start == absent_stop

    metrics = {
        "schema": "b00_blcs_composed_nht_smoke_v1",
        "source_frames": list(_SOURCE_FRAMES),
        "camera_count": len(cameras),
        "pixel_counts": pixel_counts,
        "centroid_errors_px": centroid_errors,
        "depth_errors_m": depth_errors_m,
        "maximum_rgb_deltas": maximum_rgb_deltas,
        "foreground_depth_fractions": foreground_depth_fractions,
        "median_joint_rgbs": median_joint_rgbs,
        "green_pixel_fractions": green_pixel_fractions,
        "appearance_model": result.appearance_model,
        "rasterization": result.rasterization,
        "cuda_peak_bytes": result.cuda_peak_bytes,
    }
    (repro_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


def _write_ball_asset(
    repository: Path,
    composition_root: Path,
) -> BLCSBallGaussianSettings:
    config = OmegaConf.load(
        repository / "src/synthetic_data_generation/configs/dataset/blcs/production.yaml"
    )
    ball_raw = OmegaConf.to_container(config.assets.ball, resolve=True)
    settings_raw = OmegaConf.to_container(config.assets.settings, resolve=True)
    if not isinstance(ball_raw, dict) or not isinstance(settings_raw, dict):
        raise TypeError("BLCS production ball asset config must be a mapping.")
    settings = BLCSBallGaussianSettings(**settings_raw)
    assets = BLCSCompositionAssets(
        ball=GaussianAsset.from_dict(ball_raw),
        settings=settings,
    )
    ball = build_ball_gaussian_asset(assets)
    np.savez(
        composition_root / "ball-gaussians.npz",
        means_m=ball.means.cpu().numpy(),
        quats_wxyz=ball.quaternions_wxyz.cpu().numpy(),
        log_scales_m=ball.log_scales.cpu().numpy(),
        opacity_logits=ball.opacity_logits.cpu().numpy(),
        colors_linear_rgb=ball.features.cpu().numpy(),
    )
    return settings


def _write_timeline(
    composition_root: Path,
    *,
    plan: dict[str, Any],
    positions_scene: np.ndarray,
    nht_units_per_metre: float,
) -> None:
    target_court = cast(dict[str, Any], plan["target_court"])
    scene_from_court = np.asarray(
        target_court["scene_from_court"], dtype=np.float64
    ).reshape(4, 4)
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, None], 4, axis=0)
    for local_frame, (source_frame, angle) in enumerate(
        zip(_SOURCE_FRAMES, (0.0, 0.7, 1.4), strict=True)
    ):
        rotation = scene_from_court[:3, :3] @ _rotation_y(angle)
        transforms[local_frame, 0, :3, :3] = nht_units_per_metre * rotation
        transforms[local_frame, 0, :3, 3] = (
            nht_units_per_metre * positions_scene[source_frame, _OBJECT_INDEX]
        )
    np.savez(
        composition_root / "timeline.npz",
        transforms_nht_from_asset=transforms,
        present=np.asarray([[True], [True], [True], [False]], dtype=np.bool_),
        instance_ids=np.asarray([1], dtype=np.int32),
    )


def _write_composition_request(
    composition_root: Path,
    *,
    settings: BLCSBallGaussianSettings,
) -> Path:
    payload = {
        "schema": "nht_composed_render_request_v1",
        "asset": {
            "asset_id": "regulation-tennis-ball",
            "coordinate_space": "right_handed_asset_local_metres",
            "appearance_model": "direct_linear_rgb",
            "gaussian_count": 256,
            "tensors": "ball-gaussians.npz",
        },
        "timeline": {
            "coordinate_space": "canonical NHT scene space",
            "frame_count": 4,
            "object_count": 1,
            "object_ids": ["ball-001"],
            "instance_ids": [1],
            "tensors": "timeline.npz",
            "chunks": [{"chunk_index": 0, "frame_indices": [0, 1, 2, 3]}],
        },
        "visibility_threshold": settings.visibility_threshold,
    }
    path = composition_root / "composition.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _probe_cameras(
    *,
    plan: dict[str, Any],
    nht_units_per_metre: float,
) -> tuple[NHTRenderCamera, ...]:
    raw_cameras = cast(list[dict[str, Any]], plan["cameras"])
    result: list[NHTRenderCamera] = []
    for camera_index, record in enumerate(raw_cameras):
        camera_raw = cast(dict[str, Any], record["camera"])
        camera_to_scene = np.asarray(
            camera_raw["camera_to_scene"], dtype=np.float64
        ).reshape(4, 4)
        camera_to_scene[:3, 3] *= nht_units_per_metre
        result.append(
            NHTRenderCamera(
                camera_id=f"b00-blcs-composed-probe-{camera_index:02d}",
                width=int(camera_raw["width"]),
                height=int(camera_raw["height"]),
                intrinsics=tuple(float(value) for value in camera_raw["intrinsics"]),
                camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
            )
        )
    return tuple(result)


def _rotation_y(angle: float) -> NDArray[np.float64]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation: NDArray[np.float64] = np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )
    return rotation


def _json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)
