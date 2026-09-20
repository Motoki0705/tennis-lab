"""One-shot CPU pixel-line candidates in a fixed ROI; no automatic white-line adoption."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.checksum import dual_sha256

ROI = (550, 495, 1040, 555)
FRAMES = (0, 453, 907)
SCALE = 3


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def save(path: Path, image: np.ndarray) -> None:
    require(bool(cv2.imwrite(str(path), image)), f"Image write failed: {path}")


def label(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def line_y(endpoints: np.ndarray, x: np.ndarray) -> np.ndarray:
    slope = (endpoints[1, 1] - endpoints[0, 1]) / (endpoints[1, 0] - endpoints[0, 0])
    return np.asarray(endpoints[0, 1] + slope * (x - endpoints[0, 0]))


def compare(endpoints: np.ndarray, baseline: np.ndarray) -> list[dict[str, float]]:
    xs = np.linspace(endpoints[0, 0], endpoints[1, 0], 3)
    ys, hy = line_y(endpoints, xs), line_y(baseline, xs)
    slope = (baseline[1, 1] - baseline[0, 1]) / (baseline[1, 0] - baseline[0, 0])
    return [
        {
            "x": float(x),
            "candidate_y": float(y),
            "H_y": float(h),
            "H_minus_candidate_vertical_px": float(h - y),
            "distance_to_H_line_px": float(abs(h - y) / np.sqrt(1 + slope * slope)),
        }
        for x, y, h in zip(xs, ys, hy, strict=True)
    ]


def draw_segment(
    image: np.ndarray, endpoints: np.ndarray, text: str, color: tuple[int, int, int]
) -> None:
    points = np.rint((endpoints - np.array(ROI[:2])) * SCALE).astype(int)
    cv2.line(image, tuple(points[0]), tuple(points[1]), color, 2, cv2.LINE_AA)
    center = points.mean(0).astype(int)
    label(image, text, (int(center[0]) + 4, int(center[1]) - 5), color)


def panel(
    crop: np.ndarray,
    title: str,
    segments: list[tuple[np.ndarray, str, tuple[int, int, int]]],
) -> np.ndarray:
    image = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    for endpoints, text, color in segments:
        draw_segment(image, endpoints, text, color)
    header = np.zeros((40, image.shape[1], 3), np.uint8)
    label(header, title, (8, 25))
    return np.concatenate([header, image])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be disabled")
    require(
        os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2",
        "Thread limits must be 2",
    )
    cv2.setNumThreads(1)
    root, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    clip = (
        root
        / "data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_002/clips/clip_017"
    )
    court = (
        root
        / "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/video_002/clip_017"
    )
    video = clip / "media/cam0.mp4"
    inputs = [
        video,
        clip / "clip.json",
        court / "court.json",
        court / "court.npz",
        Path(__file__).resolve(),
        bundle / "repro.sh",
        Path.cwd() / "src/utils/checksum.py",
    ]
    before = {str(path): dual_sha256(path) for path in inputs}
    manifest = json.loads((clip / "clip.json").read_text())
    receipt = json.loads((court / "court.json").read_text())
    require(
        before[str(video)] == receipt["identity"]["video_sha256"]["cam0"],
        "Video identity mismatch",
    )
    require(
        before[str(clip / "clip.json")] == receipt["identity"]["clip_sha256"],
        "Clip identity mismatch",
    )
    with np.load(court / "court.npz", allow_pickle=False) as archive:
        baseline = archive["keypoints"][0, 0, [2, 3]] * np.asarray(
            [manifest["width"], manifest["height"]], np.float32
        )
    baseline = baseline[np.argsort(baseline[:, 0])]
    results: dict[str, Any] = {
        "execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "coordinate_system": "full image pixel coordinates",
        "clip": "video_002/clip_017",
        "camera": "cam0",
        "fixed_roi_xyxy": ROI,
        "frame_indices": FRAMES,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "H_baseline_endpoints_px": baseline.tolist(),
        "selection": {
            "detector": "cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD), default parameters, grayscale fixed RGB ROI",
            "candidate_min_length_px": 40,
            "candidate_max_abs_angle_deg": 12,
            "pair_min_overlap_px": 40,
            "pair_max_angle_difference_deg": 3,
            "pair_vertical_separation_range_px": [2, 12],
            "pair_intensity_sampling": "31 x positions in overlap; at proposed middle and 3px beyond each edge",
            "automatic_adoption": False,
            "H_used_for_selection": False,
            "threshold_optimization": False,
        },
        "interpretation": "Pixel white-line position diagnostic only; not manual GT or independent 3D accuracy. All LSD segments are retained. Candidate edge-pair centerlines are unadopted hypotheses, not confirmed white line references.",
        "comparison_sign": "positive H_minus_candidate_vertical_px means H lies below candidate at same x; normal distance is shortest unsigned distance to infinite H baseline line",
        "human_visual_review": "pending",
        "selected_pixel_reference": None,
        "confirmed_quantitative_result": None,
        "frames": [],
    }
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    capture = cv2.VideoCapture(str(video))
    try:
        require(capture.isOpened(), "Video open failed")
        require(
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == manifest["num_frames"],
            "Frame count differs",
        )
        for index in FRAMES:
            require(bool(capture.set(cv2.CAP_PROP_POS_FRAMES, index)), "Seek failed")
            ok, frame = capture.read()
            require(
                ok and frame.shape[:2] == (manifest["height"], manifest["width"]),
                "Decode failed",
            )
            x0, y0, x1, y1 = ROI
            crop = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lines, widths, precisions, _ = detector.detect(gray)
            segments: list[dict[str, Any]] = []
            candidates: list[tuple[int, np.ndarray, float]] = []
            all_draw = []
            for number, local in enumerate(
                [] if lines is None else lines.reshape(-1, 4)
            ):
                points = np.asarray(local, dtype=float).reshape(2, 2) + [x0, y0]
                points = points[np.argsort(points[:, 0])]
                delta = points[1] - points[0]
                length = float(np.linalg.norm(delta))
                angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
                eligible = length >= 40 and abs(angle) <= 12
                segments.append(
                    {
                        "id": number,
                        "endpoints_px": points.tolist(),
                        "length_px": length,
                        "angle_deg": angle,
                        "LSD_width_px": float(widths.reshape(-1)[number]),
                        "LSD_precision_rad": float(precisions.reshape(-1)[number]),
                        "long_near_horizontal": eligible,
                        "provisional_edge_comparison": compare(points, baseline)
                        if eligible
                        else None,
                    }
                )
                color = (0, 255, 0) if eligible else (0, 160, 255)
                all_draw.append((points, str(number), color))
                if eligible:
                    candidates.append((number, points, angle))
            pairs: list[dict[str, Any]] = []
            pair_panels = []
            for first, second in combinations(candidates, 2):
                aid, a, aa = first
                bid, b, ba = second
                start, end = max(a[0, 0], b[0, 0]), min(a[1, 0], b[1, 0])
                if end - start < 40 or abs(aa - ba) > 3:
                    continue
                xs = np.linspace(start, end, 31)
                ay, by = line_y(a, xs), line_y(b, xs)
                low, high = np.minimum(ay, by), np.maximum(ay, by)
                separation = high - low
                if not bool(((separation >= 2) & (separation <= 12)).all()):
                    continue
                center_y = (ay + by) / 2
                # Intensity is diagnostic, not a selection or optimization score.
                intensity = []
                for ys in (center_y, low - 3, high + 3):
                    local_x, local_y = (
                        np.rint(xs - x0).astype(int),
                        np.rint(ys - y0).astype(int),
                    )
                    valid = (
                        (local_x >= 0)
                        & (local_x < gray.shape[1])
                        & (local_y >= 0)
                        & (local_y < gray.shape[0])
                    )
                    intensity.append(
                        float(np.median(gray[local_y[valid], local_x[valid]]))
                        if valid.any()
                        else None
                    )
                center = np.asarray([[start, center_y[0]], [end, center_y[-1]]])
                pair_id = len(pairs)
                pairs.append(
                    {
                        "id": pair_id,
                        "edge_segment_ids": [aid, bid],
                        "center_endpoints_px": center.tolist(),
                        "median_vertical_separation_px": float(np.median(separation)),
                        "median_gray_center_above_below": intensity,
                        "provisional_center_comparison": compare(center, baseline),
                        "adopted": False,
                    }
                )
                pair_panels.append(
                    panel(
                        crop,
                        f"frame {index} UNADOPTED pair {pair_id}: edges {aid},{bid}; magenta middle; cyan H comparison only",
                        [
                            (a, str(aid), (0, 255, 0)),
                            (b, str(bid), (0, 255, 0)),
                            (center, f"P{pair_id}", (255, 0, 255)),
                            (baseline, "H", (255, 255, 0)),
                        ],
                    )
                )
            original_path = output / f"frame{index:05d}_original.png"
            all_path = output / f"frame{index:05d}_all_candidates.png"
            detail_path = output / f"frame{index:05d}_long_segments.png"
            save(original_path, frame)
            save(
                all_path,
                np.concatenate(
                    [
                        panel(
                            crop,
                            f"frame {index} RAW RGB ROI {ROI}; nearest-neighbor 3x; all {len(segments)} LSD segments below",
                            [],
                        ),
                        panel(
                            crop,
                            "All segments: green = length >=40px and |angle|<=12deg; orange = other; IDs are frame-local",
                            all_draw,
                        ),
                    ]
                ),
            )
            details = [
                panel(
                    crop,
                    f"frame {index} raw RGB; candidate selection is independent of H; no reference adopted",
                    [],
                )
            ]
            details.extend(
                panel(
                    crop,
                    f"frame {index} segment {number}; green edge candidate; cyan H comparison only",
                    [
                        (points, str(number), (0, 255, 0)),
                        (baseline, "H", (255, 255, 0)),
                    ],
                )
                for number, points, _ in candidates
            )
            save(detail_path, np.concatenate(details))
            pair_path = None
            if pair_panels:
                pair_path = output / f"frame{index:05d}_edge_pairs.png"
                save(pair_path, np.concatenate(pair_panels))
            results["frames"].append(
                {
                    "frame_index": index,
                    "segments": segments,
                    "eligible_segment_ids": [c[0] for c in candidates],
                    "unadopted_edge_pairs": pairs,
                    "original": str(original_path),
                    "all_candidates": str(all_path),
                    "long_segments": str(detail_path),
                    "pair_hypotheses": str(pair_path) if pair_path else None,
                }
            )
            print(
                f"frame={index}: all_segments={len(segments)} long_horizontal={len(candidates)} unadopted_pairs={len(pairs)}",
                flush=True,
            )
    finally:
        capture.release()
    after = {str(path): dual_sha256(path) for path in inputs}
    require(before == after, "Input changed")
    results.update(
        input_sha256_before=before,
        input_sha256_after=after,
        inputs_unchanged=True,
        image_sha256={
            str(path): dual_sha256(path) for path in sorted(output.glob("*.png"))
        },
        exit_status=0,
    )
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        "Completed once; confirmed quantitative result remains unset pending human candidate review",
        flush=True,
    )


if __name__ == "__main__":
    main()
