"""Bundle and verify the actual B00 point cloud and historical LINE projections."""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import yaml
from common import REPO, ROOT, sha256, write_json

sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.synthetic_data_generation.alignment.evidence_source import (
    _estimate_ground_plane,
    _GroundPlane,
    _project_axis_to_plane,
    _project_probability_to_ground,
)
from src.synthetic_data_generation.alignment.settings import (
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

BUNDLE = ROOT / "evidence/alignment_method"
SELECTED = ("frame_000000", "frame_000183", "frame_000306")
CACHE_ID = "eea97d5cf0e9453ee15512d091e08e7673c3e752fb70d7a6eaf38a436c03446e"


def read_bundle() -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(BUNDLE / name) != digest:
            raise ValueError(f"Changed alignment evidence: {name}")
    with np.load(BUNDLE / "arrays.npz") as archive:
        return manifest, {k: archive[k] for k in archive.files}


def ground_fit(
    manifest: dict, arrays: dict
) -> tuple[_GroundPlane, np.ndarray, np.ndarray]:
    """Rerun production RANSAC; derive candidate/support masks for display only."""
    points = arrays["points_xyzrgb"][:, :3].astype(np.float64)
    cameras = {c["camera_id"]: SceneCamera.from_dict(c) for c in manifest["cameras"]}
    fit = tuple(cameras[i] for i in manifest["fit_camera_ids"])
    config = manifest["settings"]["ground_plane"]
    plane = _estimate_ground_plane(
        points,
        fit,
        seed=manifest["settings"]["seed"],
        settings=GroundPlaneSettings(**config),
    )
    # These masks expose the production candidate-selection stage without
    # substituting a new fit or filtering points before RANSAC.
    poses = np.stack([c.camera_to_scene.matrix() for c in fit])
    up = np.mean(poses[:, :3, :3] @ [0.0, -1.0, 0.0], axis=0)
    up /= np.linalg.norm(up)
    u = _project_axis_to_plane(np.array([1.0, 0.0, 0.0]), normal=up)
    basis = np.stack([u, np.cross(up, u)], axis=1)
    uv, camera_uv = points @ basis, poses[:, :3, 3] @ basis
    lo = (
        np.quantile(camera_uv, config["footprint_quantile"], axis=0)
        - config["footprint_margin"]
    )
    hi = (
        np.quantile(camera_uv, 1 - config["footprint_quantile"], axis=0)
        + config["footprint_margin"]
    )
    footprint = np.all((uv >= lo) & (uv <= hi), axis=1)
    height = points @ up
    median = float(np.median(poses[:, :3, 3] @ up))
    eligible = (
        footprint
        & (height >= median - config["maximum_camera_height"])
        & (height <= median - config["minimum_camera_height"])
    )
    edges = np.arange(
        median - config["maximum_camera_height"],
        median - config["minimum_camera_height"] + config["histogram_bin_width"],
        config["histogram_bin_width"],
    )
    counts, edges = np.histogram(height[eligible], bins=edges)
    peak = int(counts.argmax())
    candidates = footprint & (
        np.abs(height - (edges[peak] + edges[peak + 1]) / 2)
        <= config["candidate_half_width"]
    )
    support = footprint & (
        np.abs(points @ plane.normal + plane.offset) <= config["refine_threshold"]
    )
    return plane, candidates, support


def raster_view(manifest: dict, arrays: dict, index: int) -> np.ndarray:
    """Historical reducer: nearest cell, max(probability * proximity) per view."""
    start, stop = arrays["projected_offsets"][index : index + 2]
    uv = arrays["projected_points_uv"][start:stop]
    bounds = np.asarray(manifest["bounds_uv"])
    indices = np.rint((uv - bounds[[0, 2]]) / manifest["grid_spacing"]).astype(int)
    h, w = arrays["evidence_sum"].shape
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < w)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < h)
    )
    result = np.zeros(h * w, dtype=np.float32)
    weighted = (
        arrays["projected_probabilities"][start:stop].astype(np.float64)
        * arrays["proximity_weights"][start:stop]
    )
    np.maximum.at(
        result,
        indices[valid, 1] * w + indices[valid, 0],
        weighted[valid].astype(np.float32),
    )
    return result.reshape(h, w)


def aggregate(manifest: dict, arrays: dict, indices: list[int]) -> np.ndarray:
    result = np.zeros_like(arrays["evidence_sum"])
    for index in indices:
        if not arrays["included_in_aggregate"][index]:
            raise ValueError("Holdout view cannot contribute to fit aggregate")
        result += raster_view(manifest, arrays, index)
    return result


def validate_geometry(manifest: dict, arrays: dict) -> dict:
    plane, candidates, support = ground_fit(manifest, arrays)
    frame = manifest["ground_plane_frame"]
    scale = manifest["nht_scene_units_per_metre"]
    if not np.allclose(
        plane.normal, frame["normal_metric_scene"], atol=1e-10, rtol=0
    ) or not np.allclose(
        plane.origin / scale, frame["origin_metric_scene"], atol=1e-9, rtol=0
    ):
        raise ValueError("RANSAC plane does not match saved alignment")
    config = manifest["settings"]["line_model"]
    # Reproject the historical probabilities with their original extraction
    # settings. Architecture metadata stays in the immutable source manifest;
    # this check does not construct or load either the old or current model.
    model_settings = CourtLineModelSettings(
        checkpoint_path=(REPO / "ckpt" / config["checkpoint_path"]).absolute(),
        device="cpu",
        probability_threshold=config["probability_threshold"],
        maximum_selected_pixels_per_camera=config["maximum_selected_pixels_per_camera"],
    )
    camera_ids = arrays["camera_ids"].tolist()
    cameras = {c["camera_id"]: SceneCamera.from_dict(c) for c in manifest["cameras"]}
    max_error = 0.0
    for camera_id in SELECTED:
        i = camera_ids.index(camera_id)
        projected = _project_probability_to_ground(
            arrays[f"probability_{camera_id}"],
            camera=cameras[camera_id],
            plane=plane,
            model_settings=model_settings,
            projection_settings=LineProjectionSettings(
                **manifest["settings"]["projection"]
            ),
        )
        start, stop = arrays["projected_offsets"][i : i + 2]
        uv = arrays["projected_points_uv"][start:stop]
        actual = projected.points_uv / scale
        if actual.shape != uv.shape or not np.allclose(actual, uv, atol=1e-9, rtol=0):
            raise ValueError(f"Ray-plane projection differs for {camera_id}")
        if not np.array_equal(
            projected.probabilities, arrays["projected_probabilities"][start:stop]
        ) or not np.allclose(
            projected.proximity_weights,
            arrays["proximity_weights"][start:stop],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"Projection weights differ for {camera_id}")
        max_error = max(max_error, float(np.abs(actual - uv).max()))
    fit_indices = np.flatnonzero(arrays["included_in_aggregate"]).tolist()
    if [camera_ids[i] for i in fit_indices] != manifest["fit_camera_ids"]:
        raise ValueError("Fit/holdout camera ownership differs")
    combined = aggregate(manifest, arrays, fit_indices)
    if not np.array_equal(combined, arrays["evidence_sum"]):
        raise ValueError("Fit-only aggregate differs from saved heatmap")
    return {
        "point_count": len(arrays["points_xyzrgb"]),
        "candidate_count": int(candidates.sum()),
        "support_count": int(support.sum()),
        "fit_views": len(fit_indices),
        "holdout_views": int((~arrays["included_in_aggregate"]).sum()),
        "shown_views": list(SELECTED),
        "max_projection_difference_metres": max_error,
        "aggregate_matches_exactly": True,
        "ransac_matches_saved_plane": True,
    }


def collect() -> None:
    """Copy a specific saved v2 run; reject any substituted source or probability."""
    scene = REPO / "data/synthetic_data_generation/scenes/B00"
    cache = scene / "court-line-inference" / CACHE_ID
    inputs = json.loads((cache / "manifest.json").read_text())
    if inputs["schema"] != "court_line_inference_cache_v1":
        raise ValueError("Expected the historical captured-image inference cache")
    cfg = yaml.safe_load((scene / "resolved-config.yaml").read_text())["alignment"][
        "evidence"
    ]
    with np.load(scene / "alignment/line-heatmaps/heatmaps.npz") as original:
        archive = {k: original[k] for k in original.files}
    if str(archive["schema"]) != "alignment_line_heatmaps_v2":
        raise ValueError("Expected the historical v2 heatmap archive")
    with np.load(scene / "alignment/ground-line-map.npz") as ground:
        frame = json.loads(str(ground["ground_plane_frame_json"]))
        scale = float(ground["nht_scene_units_per_metre"])
        fit_ids = ground["fit_camera_ids"].tolist()
    source_cameras = json.loads(
        (scene / "reconstruction/export/cameras.json").read_text()
    )["cameras"]
    cameras = [
        {
            "camera_id": c["camera_id"],
            "source_frame_index": c["source_frame_index"],
            "width": c["width"],
            "height": c["height"],
            "intrinsics": np.asarray(c["intrinsics"]["matrix"]).ravel().tolist(),
            "camera_to_scene": np.asarray(c["camera_to_scene"]).ravel().tolist(),
            "image_path": c["image"],
        }
        for c in source_cameras
        if c["camera_id"] in archive["camera_ids"]
    ]
    keep = (
        "camera_ids",
        "included_in_aggregate",
        "projected_offsets",
        "projected_points_uv",
        "projected_probabilities",
        "proximity_weights",
        "evidence_sum",
    )
    arrays = {k: archive[k] for k in keep}
    arrays["points_xyzrgb"] = np.load(scene / "reconstruction/export/points_scene.npy")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    source_files = [
        scene / p
        for p in (
            "alignment/line-heatmaps/heatmaps.npz",
            "alignment/ground-line-map.npz",
            "alignment/alignment.json",
            "resolved-config.yaml",
            "reconstruction/export/points_scene.npy",
            "reconstruction/export/cameras.json",
        )
    ] + [cache / "manifest.json"]
    for index, view in enumerate(inputs["views"]):
        camera_id = view["camera_id"]
        if camera_id != archive["camera_ids"][index]:
            raise ValueError("Cache camera order differs")
        begin, end = archive["probability_offsets"][index : index + 2]
        probability = archive["probability_values"][begin:end].reshape(
            archive["probability_shapes"][index]
        )
        probability_path = cache / view["probability_file"]
        rgb_path = scene / "reconstruction/export/images" / f"{camera_id}.png"
        if (
            not np.array_equal(probability, np.load(probability_path))
            or sha256(rgb_path) != view["image_sha256"]
        ):
            raise ValueError(f"Historical RGB/probability mismatch: {camera_id}")
        source_files.extend([probability_path, rgb_path])
        if camera_id in SELECTED:
            arrays[f"probability_{camera_id}"] = probability
            shutil.copyfile(rgb_path, BUNDLE / f"{camera_id}.png")
    np.savez_compressed(BUNDLE / "arrays.npz", **arrays)
    manifest = {
        "schema": "court_paper_alignment_method_v1",
        "scene_id": "B00",
        "source_kind": "calibrated_captured_rgb",
        "source_archive_schema": str(archive["schema"]),
        "selection": list(SELECTED),
        "selection_policy": "Three fit views illustrating different observed directions; aggregate uses all 32 fit views.",
        "cameras": cameras,
        "fit_camera_ids": fit_ids,
        "settings": cfg,
        "detector": inputs["inference_identity"],
        "ground_plane_frame": frame,
        "nht_scene_units_per_metre": scale,
        "bounds_uv": archive["bounds_uv"].tolist(),
        "grid_spacing": float(archive["grid_spacing"]),
        "files": {
            p.name: sha256(p)
            for p in [BUNDLE / "arrays.npz", *[BUNDLE / f"{i}.png" for i in SELECTED]]
        },
        "source_files": {str(p.relative_to(REPO)): sha256(p) for p in source_files},
    }
    manifest["verification"] = validate_geometry(manifest, arrays)
    write_json(BUNDLE / "manifest.json", manifest)
    print(json.dumps(manifest["verification"], indent=2))


if __name__ == "__main__":
    collect()
