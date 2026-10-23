"""Generation-equivalent visibility, ROI diagnostics and paired local review media."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import plan_probe as plan
from PIL import Image, ImageDraw

from src.synthetic_data_generation.dataset.court.components.labels import (
    AmbiguousCameraRelativeNearFarError,
    attach_renderer_visibility,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.utils.schema.court import COURT_SKELETON

GROUND_LINES = [(a, b) for a, b in COURT_SKELETON if a < 14 and b < 14]


def near_clip(polygon: np.ndarray, near: float = 0.01) -> np.ndarray:
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if a[2] >= near:
            result.append(a)
        if (a[2] >= near) != (b[2] >= near):
            result.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return cast(np.ndarray, np.asarray(result).reshape(-1, 3))


def court_roi(record: dict[str, Any]) -> np.ndarray:
    width, height = record["width"], record["height"]
    mask = np.zeros((height, width), np.uint8)
    if not record["points"]:
        return cast(np.ndarray, mask.astype(bool))
    points = {p["physical_index"]: p for p in record["points"]}
    world = np.array([points[i]["scene_xyz_m"] for i in [0, 1, 3, 2]])
    pose = np.array(record["pose"])
    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    clipped = near_clip(camera)
    if len(clipped) < 3:
        return cast(np.ndarray, mask.astype(bool))
    projected = clipped @ np.array(record["K"]).reshape(3, 3).T
    uv = np.ascontiguousarray(projected[:, :2] / projected[:, 2:], dtype=np.float32)
    frame = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    area, polygon = cv2.intersectConvexConvex(uv, frame)
    if area > 0 and polygon is not None:
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    return cast(np.ndarray, mask.astype(bool))


def line_raster(record: dict[str, Any]) -> np.ndarray:
    result = np.zeros((record["height"], record["width"]), np.uint8)
    points = {p["physical_index"]: p for p in record["points"]}
    for a, b in GROUND_LINES:
        if (
            a not in points
            or b not in points
            or not points[a]["in_front"]
            or not points[b]["in_front"]
        ):
            continue
        start = tuple(np.rint(points[a]["uv"]).astype(int).tolist())
        end = tuple(np.rint(points[b]["uv"]).astype(int).tolist())
        valid, p0, p1 = cv2.clipLine(
            (0, 0, record["width"], record["height"]), start, end
        )
        if valid:
            cv2.line(result, p0, p1, 1, 1)
    return cast(np.ndarray, result.astype(bool))


def diagnostics(
    record: dict[str, Any], arrays: NHTRenderArrays, scene: Any
) -> dict[str, Any]:
    alpha, depth, rgb = arrays.alpha[..., 0], arrays.depth[..., 0], arrays.rgb
    roi = court_roi(record)
    valid = (alpha >= 0.01) & (depth > 0)
    result: dict[str, Any] = {
        k: record[k]
        for k in [
            "id",
            "pair_id",
            "scene",
            "condition",
            "kind",
            "index",
            "ordinal",
            "fraction",
            "hfov",
            "shape",
            "center_kind",
            "height_m",
            "hull_clearance_m",
            "target",
            "geometry_target_kp",
            "geometry_all_kp",
            "coverage",
            "existing_pre_gate",
        ]
    }
    result["roi_pixels"] = int(roi.sum())
    for prefix, mask in [("all", np.ones_like(roi)), ("roi", roi)]:
        size = int(mask.sum())
        for key, test in [
            ("alpha_hole", alpha < 0.01),
            ("depth_hole", depth <= 0),
            ("joint_hole", ~valid),
            ("alpha_lt05", alpha < 0.5),
            ("alpha_lt09", alpha < 0.9),
            ("positive_depth", depth > 0),
        ]:
            result[f"{prefix}_{key}"] = float(test[mask].mean()) if size else None
        result[f"{prefix}_alpha_p05"] = (
            float(np.quantile(alpha[mask], 0.05)) if size else None
        )
    projection = None
    try:
        projection = attach_renderer_visibility(
            project_court_semantics_v3(
                plan.camera_from_record(record), scene.alignment.layout
            ),
            alpha=arrays.alpha,
            depth=arrays.depth,
        )
    except AmbiguousCameraRelativeNearFarError:
        if not record["ambiguous"]:
            raise
    result["renderer_target_kp"] = 0
    result["renderer_all_kp"] = 0
    if projection is not None:
        result["renderer_all_kp"] = projection.visible_point_count
        court = next(
            c for c in projection.courts if c.court_instance_id == record["target"]
        )
        result["renderer_target_kp"] = sum(
            p.renderer_visible is True for cls in court.classes for p in cls.points
        )
    result["existing_post_gate"] = result["renderer_all_kp"] > 0
    result["existing_gate"] = (
        result["existing_pre_gate"] and result["existing_post_gate"]
    )
    result["candidate_target4"] = (
        result["existing_gate"] and result["renderer_target_kp"] >= 4
    )
    for threshold in [0.01, 0.05, 0.10]:
        result[f"candidate_roi{int(threshold * 100):02d}"] = (
            result["candidate_target4"]
            and result["roi_joint_hole"] is not None
            and result["roi_joint_hole"] <= threshold
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    ridges = (
        (gray - opening > 0.06)
        & (rgb.max(2) > 0.45)
        & (rgb.max(2) - rgb.min(2) < 0.30)
        & valid
    )
    distance = cv2.distanceTransform((~ridges).astype(np.uint8), cv2.DIST_L2, 5)
    line_mask = line_raster(record) & roi & valid
    distances = np.minimum(distance[line_mask], 24.0)
    result["line_pixels"] = int(line_mask.sum())
    result["line_ridge_distance_median_px"] = (
        float(np.median(distances)) if len(distances) else None
    )
    result["line_ridge_within3px"] = (
        float((distances <= 3).mean()) if len(distances) else None
    )
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    result["roi_laplacian_variance"] = (
        float(laplacian[roi].var()) if roi.any() else None
    )
    return result


def overlay(
    record: dict[str, Any], arrays: NHTRenderArrays, metric: dict[str, Any]
) -> Image.Image:
    rgb = np.round(arrays.rgb * 255).astype(np.uint8)
    output = rgb.copy()
    holes = (arrays.alpha[..., 0] < 0.01) | (arrays.depth[..., 0] <= 0)
    output[holes] = (
        output[holes].astype(float) * 0.3 + np.array([255, 0, 180]) * 0.7
    ).astype(np.uint8)
    # Same projection as the ROI; no inferred or detected line substitutes.
    points = {p["physical_index"]: p for p in record["points"]}
    for a, b in GROUND_LINES:
        if (
            a in points
            and b in points
            and points[a]["in_front"]
            and points[b]["in_front"]
        ):
            p0, p1 = (
                tuple(np.rint(points[a]["uv"]).astype(int)),
                tuple(np.rint(points[b]["uv"]).astype(int)),
            )
            valid, c0, c1 = cv2.clipLine(
                (0, 0, record["width"], record["height"]), p0, p1
            )
            if valid:
                cv2.line(output, c0, c1, (40, 240, 60), 1, cv2.LINE_AA)
    for p in points.values():
        if p["in_frame"]:
            x, y = np.rint(p["uv"]).astype(int)
            x0, x1, y0, y1 = (
                max(0, x - 1),
                min(record["width"], x + 2),
                max(0, y - 1),
                min(record["height"], y + 2),
            )
            supported = np.any(
                (arrays.alpha[y0:y1, x0:x1, 0] >= 0.01)
                & (arrays.depth[y0:y1, x0:x1, 0] > 0)
            )
            cv2.circle(
                output,
                (int(x), int(y)),
                4,
                (0, 225, 255) if supported else (255, 40, 40),
                -1,
            )
    pose = np.array(record["pose"])
    xyz = (np.array(record["look_at"]) - pose[:3, 3]) @ pose[:3, :3]
    uv = np.array(record["K"]).reshape(3, 3) @ xyz
    if uv[2] > 0:
        x, y = np.rint(uv[:2] / uv[2]).astype(int)
        cv2.drawMarker(output, (int(x), int(y)), (255, 240, 0), cv2.MARKER_CROSS, 18, 2)
    return Image.fromarray(output)


def panel(
    record: dict[str, Any],
    arrays: NHTRenderArrays,
    metric: dict[str, Any],
    width: int = 480,
) -> Image.Image:
    h = round(record["height"] * width / record["width"])
    canvas = Image.new("RGB", (width * 2, h + 68), "#151a22")
    canvas.paste(
        Image.fromarray(np.round(arrays.rgb * 255).astype(np.uint8)).resize((width, h)),
        (0, 68),
    )
    canvas.paste(overlay(record, arrays, metric).resize((width, h)), (width, 68))
    draw = ImageDraw.Draw(canvas)
    hole = metric["roi_joint_hole"]
    text = f"{record['scene']} {record['condition']} {record['kind']}:{record['index']:02d} FOV={record['hfov']} target={record['target']}"
    draw.text((8, 5), text, fill="white")
    draw.text(
        (8, 23),
        f"KP geom/render={metric['geometry_target_kp']}/{metric['renderer_target_kp']} coverage={record['coverage']} ROI hole={hole:.2%}"
        if hole is not None
        else text,
        fill="white",
    )
    draw.text(
        (8, 42),
        "RGB | lines:green  KP:cyan/red  look-at:yellow  alpha/depth holes:magenta",
        fill="#c2cbd8",
    )
    return canvas


def load_arrays(directory: Path, record: dict[str, Any]) -> NHTRenderArrays:
    paths = directory / "renders" / record["id"]
    return NHTRenderArrays(
        rgb=np.load(paths / "rgb.npy", allow_pickle=False),
        alpha=np.load(paths / "alpha.npy", allow_pickle=False),
        depth=np.load(paths / "depth.npy", allow_pickle=False),
    )


def write_video(path: Path, frames: Any) -> dict[str, Any]:
    process = None
    count = 0
    try:
        for frame in frames:
            if process is None:
                width, height = frame.size
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-threads",
                        "2",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        f"{width}x{height}",
                        "-framerate",
                        "12",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "2",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(path),
                    ],
                    stdin=subprocess.PIPE,
                )
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
            count += 1
    finally:
        if process is not None:
            assert process.stdin is not None
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {path}")
    if count == 0:
        raise ValueError("Empty video")
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,nb_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = info["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert int(stream["nb_frames"]) == count
    return {"path": str(path), "frames": count, **info}


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for values, group in sorted(groups.items()):
        entry = dict(zip(keys, values, strict=True))
        entry["n"] = len(group)
        for key in [
            "existing_pre_gate",
            "existing_post_gate",
            "existing_gate",
            "candidate_target4",
            "candidate_roi01",
            "candidate_roi05",
            "candidate_roi10",
        ]:
            entry[key + "_pass"] = sum(bool(r[key]) for r in group)
        entry["coverage_counts"] = dict(Counter(r["coverage"] for r in group))
        for key in [
            "all_joint_hole",
            "roi_alpha_hole",
            "roi_depth_hole",
            "roi_joint_hole",
            "roi_alpha_lt05",
            "roi_alpha_lt09",
            "all_positive_depth",
            "roi_positive_depth",
            "roi_alpha_p05",
            "geometry_target_kp",
            "renderer_target_kp",
            "line_ridge_distance_median_px",
            "line_ridge_within3px",
            "hull_clearance_m",
            "height_m",
        ]:
            numbers = [r[key] for r in group if r[key] is not None]
            entry[key] = (
                {
                    "n": len(numbers),
                    "p05": float(np.quantile(numbers, 0.05)),
                    "median": float(np.median(numbers)),
                    "p95": float(np.quantile(numbers, 0.95)),
                }
                if numbers
                else None
            )
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.plan_root.resolve(strict=True)
    media = output / "review"
    media.mkdir(exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    all_plans = {}
    all_metrics = {}
    for sid in ["B00", "B01", "B02", "B03"]:
        scene = plan.prior.load_scene(args.scenes_root.resolve(strict=True), sid)
        for condition in plan.CONDITIONS:
            directory = output / sid / condition
            records = json.loads((directory / "plan.json").read_text())
            continuous = [r for r in records if r["kind"] == "continuous"]
            switches = {
                r["id"]
                for r, previous in zip(
                    continuous, continuous[-1:] + continuous[:-1], strict=True
                )
                if r["target"] != previous["target"]
            }
            metrics = {}
            for record in {r["id"]: r for r in records}.values():
                arrays = load_arrays(directory, record)
                metrics[record["id"]] = diagnostics(record, arrays, scene)
                metrics[record["id"]]["target_switch"] = record["id"] in switches
            all_rows.extend(metrics.values())
            all_plans[sid, condition], all_metrics[sid, condition] = records, metrics
            plan.dump(directory / "qa.json", list(metrics.values()))
            # Contact sheets include visibility extremes, largest hole, weakest ridge, and four phases.
            ids = [
                min(metrics, key=lambda i: metrics[i]["renderer_target_kp"]),
                max(metrics, key=lambda i: metrics[i]["renderer_target_kp"]),
                max(metrics, key=lambda i: metrics[i]["roi_joint_hole"] or 0),
                max(
                    metrics,
                    key=lambda i: metrics[i]["line_ridge_distance_median_px"] or 0,
                ),
            ]
            ids += [
                r["id"]
                for r in records
                if r["kind"] == "continuous" and r["index"] in [0, 24, 48, 72]
            ]
            by_id = {r["id"]: r for r in records}
            panels = [
                panel(by_id[i], load_arrays(directory, by_id[i]), metrics[i], 320)
                for i in ids
            ]
            sheet = Image.new("RGB", (1280, panels[0].height * 4), "#10151d")
            for index, item in enumerate(panels):
                sheet.paste(item, ((index % 2) * 640, (index // 2) * item.height))
            sheet.save(media / f"{sid}-{condition}-contact.jpg", quality=92)
            print("QA", sid, condition, len(metrics), flush=True)
    plan.dump(output / "summary.json", summarize(all_rows, ["scene", "condition"]))
    strata: list[dict[str, Any]] = []
    for row in all_rows:
        row["height_layer"] = (
            "below2"
            if row["height_m"] < 2
            else "2to2.5"
            if row["height_m"] < 2.5
            else "above2.5"
        )
        row["hull_layer"] = (
            "within1m"
            if row["hull_clearance_m"] < 1
            else "1to3m"
            if row["hull_clearance_m"] < 3
            else "above3m"
        )
    for key in [
        "hfov",
        "shape",
        "center_kind",
        "height_layer",
        "hull_layer",
        "coverage",
        "kind",
        "target",
        "target_switch",
    ]:
        strata.extend(
            {"stratum": key, **r}
            for r in summarize(all_rows, ["scene", "condition", key])
        )
    plan.dump(output / "strata.json", strata)
    with (output / "frames.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    videos = []
    motion = []
    for sid in ["B00", "B01", "B02", "B03"]:

        def frames(condition: str, sid: str = sid) -> Any:
            for r in all_plans[sid, condition]:
                if r["kind"] == "continuous":
                    yield panel(
                        r,
                        load_arrays(output / sid / condition, r),
                        all_metrics[sid, condition][r["id"]],
                    )

        videos.append(
            write_video(
                media / f"{sid}-recommended-rgb-overlay.mp4", frames("recommended")
            )
        )
        for condition in ["hull5", "hfov45_90", "jitter0"]:

            def paired(condition: str = condition, frames: Any = frames) -> Any:
                for a, b in zip(frames("recommended"), frames(condition), strict=True):
                    canvas = Image.new("RGB", (a.width, a.height * 2))
                    canvas.paste(a, (0, 0))
                    canvas.paste(b, (0, a.height))
                    yield canvas

            videos.append(
                write_video(media / f"{sid}-compare-{condition}.mp4", paired())
            )
        for condition in plan.CONDITIONS:
            continuous = [
                r for r in all_plans[sid, condition] if r["kind"] == "continuous"
            ]
            axes = np.array([np.array(r["pose"])[:3, 2] for r in continuous])
            angles = np.degrees(
                np.arccos(np.clip((axes * np.roll(axes, 1, axis=0)).sum(1), -1, 1))
            )
            switches = sum(
                a["target"] != b["target"]
                for a, b in zip(
                    continuous, continuous[-1:] + continuous[:-1], strict=True
                )
            )
            motion.append(
                {
                    "scene": sid,
                    "condition": condition,
                    "target_switches": switches,
                    "step_angle_median_degrees": float(np.median(angles)),
                    "step_angle_p95_degrees": float(np.quantile(angles, 0.95)),
                    "step_angle_max_degrees": float(angles.max()),
                }
            )
        print("videos", sid, flush=True)
    plan.dump(output / "videos.json", videos)
    plan.dump(output / "motion.json", motion)
    # Verify every logical comparison shares the point identity and intended variables.
    pair_checks = []
    for sid in ["B00", "B01", "B02", "B03"]:
        reference = all_plans[sid, "recommended"]
        for condition in ["hull5", "hfov45_90", "jitter0"]:
            other = all_plans[sid, condition]
            assert [(r["kind"], r["index"], r["pair_id"]) for r in reference] == [
                (r["kind"], r["index"], r["pair_id"]) for r in other
            ]
            if condition != "hull5":
                assert all(
                    a["position"] == b["position"]
                    for a, b in zip(reference, other, strict=True)
                )
            if condition == "hfov45_90":
                assert all(
                    a["pose"] == b["pose"]
                    for a, b in zip(reference, other, strict=True)
                )
            if condition == "jitter0":
                assert all(
                    a["K"] == b["K"] for a, b in zip(reference, other, strict=True)
                )
            pair_checks.append(
                {
                    "scene": sid,
                    "condition": condition,
                    "logical_pairs": len(reference),
                    "passed": True,
                }
            )
    plan.dump(output / "pair-checks.json", pair_checks)


if __name__ == "__main__":
    main()
