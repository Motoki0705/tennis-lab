"""Bounded CPU RGB experiment. Manual annotation opened only after selection freeze."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.synthetic_data_generation.court_calibration.database import (
    POINTS,
    SEGMENTS,
    DatabaseConfig,
    generate,
)
from src.synthetic_data_generation.court_calibration.matching import query_database
from src.tennis_scene.pipeline.court_reference import fit_camera

RECIPE = dict(
    count=96,
    width=512,
    height=288,
    line_width=1,
    top_k=5,
    iterations=100,
    truncation=12.0,
    roi_dilation_px=80,
    tophat_kernel=15,
    tophat_min=20,
    min_value=100,
    max_saturation=100,
    component_min_area=25,
    component_min_extent=25,
    component_min_aspect=3.0,
    downsample_min=0.15,
    center_radius_m=0.3,
    target_radius_m=0.5,
    hfov_radius_deg=1.5,
)
CLIPS = [
    ("video_000", "clip_000"),
    ("video_001", "clip_003"),
    ("video_002", "clip_017"),
]


def project(h):
    p = np.c_[POINTS[:, :2], np.ones(14)] @ h.T
    return p[:, :2] / p[:, 2:]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save(path, im):
    if not cv2.imwrite(str(path), im):
        raise RuntimeError(str(path))


def extract(frame, roi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    raw = (
        (hat >= 20) & (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 100) & (roi > 0)
    ).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    out = np.zeros(raw.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25 or max(w, h) < 25:
            continue
        yy, xx = np.nonzero(labels[y : y + h, x : x + w] == i)
        eig = np.linalg.eigvalsh(np.cov(np.stack([xx, yy])))
        if np.sqrt(eig[1] / max(eig[0], 0.1)) < 3:
            continue
        out[labels == i] = 255
    return (
        cv2.resize(
            out.astype(np.float32) / 255, (512, 288), interpolation=cv2.INTER_AREA
        )
        >= 0.15
    ).astype(np.uint8) * 255


def mask_h(h):
    mask = np.zeros((288, 512), np.uint8)
    pts = project(h)
    if not np.isfinite(pts).all() or np.max(np.abs(pts)) > 1e6:
        raise ValueError("invalid projection")
    pts = np.rint(pts).astype(int)
    for a, b in SEGMENTS:
        cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, 1)
    return mask


def diagnostic(h, observed):
    predicted = mask_h(h)
    if not np.any(predicted) or not np.any(observed):
        return None
    do = cv2.distanceTransform(255 - observed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dp = cv2.distanceTransform(255 - predicted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    a = float(np.minimum(do[predicted > 0], 12).mean())
    b = float(np.minimum(dp[observed > 0], 12).mean())
    return dict(
        predicted_to_rgb_truncated_px=a,
        rgb_to_predicted_truncated_px=b,
        symmetric_mean_px=(a + b) / 2,
    )


def frame_at(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise RuntimeError("decode seek")
        ok, im = cap.read()
        if not ok:
            raise RuntimeError("decode read")
        return im
    finally:
        cap.release()


def self_check():
    im = np.full((1080, 1920, 3), 40, np.uint8)
    cv2.line(im, (200, 300), (1700, 700), (240, 240, 240), 5)
    m = extract(im, np.full(im.shape[:2], 255, np.uint8))
    assert np.count_nonzero(m) > 100
    assert not np.any(
        extract(np.full_like(im, 40), np.full(im.shape[:2], 255, np.uint8))
    )
    # A held-out displacement must worsen the independent line diagnostic.
    h = np.array([[20.0, 0, 256], [0, 8, 144], [0, 0, 1]])
    exact = mask_h(h)
    shifted = h.copy()
    shifted[0, 2] += 10
    assert diagnostic(h, exact)["symmetric_mean_px"] == 0
    assert diagnostic(shifted, exact)["symmetric_mean_px"] > 0
    half = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(project(h @ half @ half), project(h))
    print("fixture checks passed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    cv2.setNumThreads(1)
    if args.check:
        self_check()
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("OMP_NUM_THREADS") == "2"
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "recipe.json", RECIPE)
    records = []
    s = np.array(
        [
            [512 / 1920, 0, (512 / 1920 - 1) / 2],
            [0, 288 / 1080, (288 / 1080 - 1) / 2],
            [0, 0, 1.0],
        ]
    )
    for vi, (video, clip) in enumerate(CLIPS):
        source = (
            root
            / "data/tennis_multivew/processed/meiji_3cam/dataset/videos"
            / video
            / "clips"
            / clip
        )
        cache = (
            root
            / "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005"
            / video
            / clip
        )
        manifest = json.loads((source / "clip.json").read_text())
        with np.load(cache / "court.npz") as a:
            hs = a["homographies"].copy()
        for ci in range(3):
            ident = f"{video}_{clip}_cam{ci}"
            dest = out / ident
            dest.mkdir()
            sample_file = cache / (
                f"cam{ci}_court_refined_samples.npz"
                if ci < 2
                else f"cam{ci}_court_samples.npz"
            )
            with np.load(sample_file) as a:
                samples = set(a["frame_indices"].tolist())
            frames = [
                int(manifest["num_frames"] * fraction) + 1
                for fraction in (1 / 3, 2 / 3)
            ]
            frames = [
                next(j for j in range(i, manifest["num_frames"]) if j not in samples)
                for i in frames
            ]
            local_h = hs[ci]
            half = np.diag([-1.0, -1.0, 1.0]) if ci == 2 else np.eye(3)
            global_h = local_h @ half
            pts = project(local_h)
            roi = np.zeros((1080, 1920), np.uint8)
            cv2.fillConvexPoly(roi, cv2.convexHull(np.rint(pts).astype(np.int32)), 255)
            roi = cv2.dilate(roi, np.ones((161, 161), np.uint8))
            rgb = [frame_at(source / manifest["video_paths"][ci], i) for i in frames]
            masks = [extract(im, roi) for im in rgb]
            for tag, im, mask in zip(("query", "temporal"), rgb, masks, strict=True):
                save(dest / f"{tag}_rgb.jpg", im)
                save(dest / f"{tag}_mask.png", mask)
            fit = fit_camera(pts / [1920, 1080], (1920, 1080), ci == 2)
            center = np.array(fit["camera_center_court_m"])
            rotation = np.array(fit["R"])
            axis = rotation[2]
            if axis[2] >= 0:
                raise ValueError("camera optical axis does not intersect ground")
            target = center - axis * center[2] / axis[2]
            fov = float(
                np.degrees(2 * np.arctan(1920 / (2 * np.array(fit["K"])[0, 0])))
            )
            slot = dict(
                slot_id=ident,
                position_x_m=[center[0] - 0.3, center[0] + 0.3],
                position_y_m=[center[1] - 0.3, center[1] + 0.3],
                height_m=[center[2] - 0.3, center[2] + 0.3],
                look_at_x_m=[target[0] - 0.5, target[0] + 0.5],
                look_at_y_m=[target[1] - 0.5, target[1] + 0.5],
                look_at_height_m=[0.0, 0.0],
                hfov_degrees=[fov - 1.5, fov + 1.5],
            )
            config = DatabaseConfig.from_mapping(
                dict(
                    count=96,
                    seed=42 + vi * 3 + ci,
                    width=512,
                    height=288,
                    line_width=1,
                    camera=slot,
                )
            )
            rec = dict(
                view=ident,
                frames=frames,
                excluded_detector_frames=sorted(samples),
                half_turn=ci == 2,
                approximate_v9_camera=fit,
                database_config=dataclasses.asdict(config),
                baseline_global_H=global_h.tolist(),
                mask_pixels=[int(np.count_nonzero(m)) for m in masks],
                candidates=[],
            )
            try:
                db = generate(config)
                write_json(dest / "database_metadata.json", config.metadata())
                # Database held only in memory; exact config/seed reproduces it.
                matches = query_database(
                    db, masks[0], top_k=5, truncation=12.0, iterations=100
                )
                for match in matches:
                    item = dataclasses.asdict(match)
                    for key in (
                        "initial_world_to_query",
                        "world_to_query",
                        "query_to_template",
                    ):
                        if item[key] is not None:
                            item[key] = item[key].tolist()
                    item["query_diagnostic"] = (
                        None
                        if not match.success
                        else diagnostic(match.world_to_query, masks[0])
                    )
                    rec["candidates"].append(item)
                eligible = [
                    m
                    for m in rec["candidates"]
                    if m["success"] and m["query_diagnostic"] is not None
                ]
                selected = (
                    min(
                        eligible,
                        key=lambda m: m["query_diagnostic"]["symmetric_mean_px"],
                    )
                    if eligible
                    else None
                )
                rec["selected_index"] = None if selected is None else selected["index"]
                init = np.asarray(
                    (selected or rec["candidates"][0])["initial_world_to_query"]
                )
                refine = (
                    None if selected is None else np.asarray(selected["world_to_query"])
                )
                stages = {"baseline": s @ global_h, "initial": init, "refined": refine}
                rec["status"] = (
                    "refinement_returned_not_verified"
                    if selected
                    else "all_refinements_failed"
                )
                rec["stages"] = {}
                for name, h in stages.items():
                    if h is None:
                        rec["stages"][name] = None
                        continue
                    original_h = np.linalg.inv(s) @ h
                    rec["stages"][name] = dict(
                        global_H=original_h.tolist(),
                        local_H=(original_h @ half).tolist(),
                        query=diagnostic(h, masks[0]),
                        temporal=diagnostic(h, masks[1]),
                    )
                panels = []
                for name, h in stages.items():
                    overlay = cv2.resize(rgb[0], (960, 540))
                    if h is not None:
                        pp = np.rint(project(np.linalg.inv(s) @ h) / 2).astype(int)
                        for a, b in SEGMENTS:
                            cv2.line(
                                overlay,
                                tuple(pp[a]),
                                tuple(pp[b]),
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    cv2.putText(
                        overlay,
                        name if h is not None else name + " FAILED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    panels.append(overlay)
                save(dest / "comparison.jpg", np.concatenate(panels, axis=0))
            except (ValueError, RuntimeError) as error:
                rec["status"] = "explicit_failure"
                rec["error"] = str(error)
            records.append(rec)
            write_json(dest / "result.json", rec)
            print(ident, rec["status"], rec["mask_pixels"], flush=True)
    # Freeze all evidence-based choices before opening the held-out annotation.
    write_json(out / "selection_frozen.json", records)
    annotation = (
        root
        / "data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/manual_court_kp_result.json"
    )
    gt = json.loads(annotation.read_text())
    points = np.asarray(gt["keypoints"])
    for ci, rec in enumerate(records[:3]):
        ref = points[ci, 0] * [1920, 1080]
        rec["heldout_manual"] = {
            "independent_static_layouts": 1,
            "points": 14,
            "repeated_frames_not_independent": int(points.shape[1]),
            "stages": {},
        }
        for name, stage in rec.get("stages", {}).items():
            if stage is not None:
                errors = np.linalg.norm(
                    project(np.asarray(stage["local_H"])) - ref, axis=1
                )
                rec["heldout_manual"]["stages"][name] = dict(
                    mean_px=float(errors.mean()),
                    max_px=float(errors.max()),
                    per_point_px=errors.tolist(),
                )
    result = dict(
        recipe=RECIPE,
        execution_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        opencv=cv2.__version__,
        manual_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        views=records,
    )
    write_json(out / "results.json", result)
    bundle = Path(__file__).parent
    write_json(bundle / "results.json", result)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
