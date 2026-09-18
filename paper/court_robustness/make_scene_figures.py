"""Project persisted metric court alignment onto genuine saved 3DGS renders.

--collect snapshots the named source owners. Without it, bundled RGB and
camera/alignment evidence suffice to reproduce every figure without a GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import cv2
import numpy as np
from common import REPO, ROOT, sha256, write_json
from PIL import Image

SELECTION = {
    "B00": [720, 1493, 423],
    "B01": [1403, 1918, 1601],
    "B02": [808, 1982, 445],
    "B03": [1112, 1091, 1288],
}
COLORS = [(0, 235, 205), (255, 182, 68), (252, 114, 194)]


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def projected_courts(bundle: dict, sample: dict) -> list[np.ndarray]:
    camera_from_scene = np.linalg.inv(
        np.asarray(sample["camera"]["camera_to_scene"]).reshape(4, 4)
    )
    local = np.asarray(bundle["local_keypoints_3d"])
    return [
        transform(
            transform(local, np.asarray(c["scene_from_court"]).reshape(4, 4)),
            camera_from_scene,
        )
        for c in bundle["alignment"]["layout"]["courts"]
    ]


def validate_projection(bundle: dict, sample: dict) -> float:
    """Independently bind the saved labels to the exact current 3-D alignment."""
    k = np.asarray(sample["camera"]["intrinsics"]).reshape(3, 3)
    cameras = projected_courts(bundle, sample)
    courts = bundle["alignment"]["layout"]["courts"]
    local = np.asarray(bundle["local_keypoints_3d"])
    maximum = 0.0
    for court, camera_points in zip(courts, cameras, strict=True):
        saved = next(
            c
            for c in sample["projection"]["courts"]
            if c["court_instance_id"] == court["court_instance_id"]
        )
        scene_points = transform(
            local, np.asarray(court["scene_from_court"]).reshape(4, 4)
        )
        projected = camera_points @ k.T
        uv = projected[:, :2] / projected[:, 2:3]
        for cls in saved["classes"]:
            for point in cls["points"]:
                index = point["physical_index"]
                if not np.allclose(
                    scene_points[index], point["scene_xyz_m"], atol=2e-5, rtol=0
                ):
                    raise ValueError(
                        "Saved dataset and current alignment disagree in metric coordinates"
                    )
                error = float(np.max(np.abs(uv[index] - point["uv"])))
                maximum = max(maximum, error)
                if not np.isfinite(error) or error > 0.02:
                    raise ValueError(
                        f"Saved camera/label projection mismatch: {error} px"
                    )
    return maximum


def project_segment(
    points: np.ndarray, k: np.ndarray, near: float = 0.05
) -> np.ndarray | None:
    points = points.copy()
    if not np.isfinite(points).all() or np.all(points[:, 2] < near):
        return None
    if np.any(points[:, 2] < near):
        behind = int(np.argmin(points[:, 2]))
        front = 1 - behind
        fraction = (near - points[behind, 2]) / (points[front, 2] - points[behind, 2])
        points[behind] += fraction * (points[front] - points[behind])
    uvh = points @ k.T
    return uvh[:, :2] / uvh[:, 2:3]


def render_overlay(bundle: dict, sample: dict, image: Image.Image) -> Image.Image:
    validate_projection(bundle, sample)
    if image.size != (sample["width"], sample["height"]):
        raise ValueError("Renderer RGB resolution differs from saved camera")
    rgb = np.asarray(image.convert("RGB")).copy()
    h, w = rgb.shape[:2]
    k = np.asarray(sample["camera"]["intrinsics"]).reshape(3, 3)
    for ci, points in enumerate(projected_courts(bundle, sample)):
        for first, second in bundle["segments"]:
            uv = project_segment(points[[first, second]], k)
            if uv is None:
                continue
            start, stop = [
                tuple(np.clip(np.rint(p), -1000000, 1000000).astype(int)) for p in uv
            ]
            ok, start, stop = cv2.clipLine((0, 0, w, h), start, stop)
            if ok:
                cv2.line(
                    rgb,
                    start,
                    stop,
                    COLORS[ci % len(COLORS)],
                    max(2, round(w / 480)),
                    cv2.LINE_AA,
                )
    return Image.fromarray(rgb)


def collect() -> None:
    sys.path.insert(0, str(REPO))
    from src.utils.schema.court import (
        COURT_SKELETON,
        STANDARD_COURT_CONFIG,
        court_keypoints_3d,
    )

    local = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].tolist()
    segments = [(a, b) for a, b in COURT_SKELETON if a < 14 and b < 14]
    for sid, indices in SELECTION.items():
        scene = REPO / "data/synthetic_data_generation/scenes" / sid
        dataset = scene / "datasets/court"
        manifest = dataset / "dataset.json"
        d = json.loads(manifest.read_text())
        alignment_path = scene / "alignment/alignment.json"
        bundle = {
            "scene": sid,
            "dataset_schema": d["schema"],
            "source_dataset": str(manifest.relative_to(REPO)),
            "dataset_sha256": sha256(manifest),
            "source_alignment": str(alignment_path.relative_to(REPO)),
            "alignment_sha256": sha256(alignment_path),
            "alignment": json.loads(alignment_path.read_text()),
            "local_keypoints_3d": local,
            "segments": segments,
            "metrics": d["metrics"],
            "trajectory_group_count": len(d["trajectory_groups"]),
            "selection": "Three distinct trajectory groups chosen for visible line structure and different views; illustrative, not random or an accuracy benchmark.",
            "rgb_kind": "Saved 3DGS novel-view RGB preview from the canonical court dataset; not captured video or a point-cloud plot.",
            "overlay_kind": "All persisted courts projected by the sample's metric camera, with near/image clipping and no depth-occlusion mask.",
            "views": [],
        }
        for index in indices:
            s = next(s for s in d["samples"] if s["sample_index"] == index)
            source = dataset / s["rgb_preview"]
            filename = f"{sid}_{index:06d}.png"
            dest = ROOT / "evidence/scene_sources" / filename
            shutil.copyfile(source, dest)
            error = validate_projection(bundle, s)
            bundle["views"].append(
                {
                    "sample": s,
                    "source_rgb": str(source.relative_to(REPO)),
                    "bundled_rgb": str(dest.relative_to(ROOT)),
                    "rgb_sha256": sha256(dest),
                    "max_label_reprojection_difference_px": error,
                    "figure": f"figures/{sid}_{index:06d}_alignment.png",
                }
            )
        manual = scene / "alignment/manual-confirmation.json"
        bundle["manual_confirmation"] = (
            json.loads(manual.read_text()) if manual.exists() else None
        )
        write_json(ROOT / f"evidence/scene_sources/{sid}.json", bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    summary = {}
    for sid in SELECTION:
        bundle_path = ROOT / f"evidence/scene_sources/{sid}.json"
        bundle = json.loads(bundle_path.read_text())
        views = []
        for view in bundle["views"]:
            source = ROOT / view["bundled_rgb"]
            if sha256(source) != view["rgb_sha256"]:
                raise ValueError(f"Changed renderer RGB: {source}")
            image = Image.open(source).convert("RGB")
            path = ROOT / view["figure"]
            render_overlay(bundle, view["sample"], image).save(path)
            views.append(
                {
                    "sample_index": view["sample"]["sample_index"],
                    "figure": view["figure"],
                    "sha256": sha256(path),
                }
            )
        summary[sid] = {"bundle_sha256": sha256(bundle_path), "views": views}
    write_json(ROOT / "evidence/scene_visualization.json", summary)
    print("Generated 12 genuine 3DGS render/alignment overlays (3 per B00-B03).")


if __name__ == "__main__":
    main()
