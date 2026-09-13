"""Render paired real-clip trajectories and source-video root reprojections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.schema.court import (
    COURT_COORD_SCALE_XYZ,
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)
from src.utils.video.writer import VideoWriter

BASE = (95, 100, 245)
NEW = (230, 210, 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--new-label", default="residual")
    args = parser.parse_args()
    base = np.load(args.comparison / "baseline_clip.npz")
    new = np.load(args.comparison / f"{args.new_label}_clip.npz")
    clip = json.loads((args.clip / "clip.json").read_text())
    frames = clip["num_frames"]
    fps = clip["fps"]
    cameras = [cv2.VideoCapture(str(args.clip / p)) for p in clip["video_paths"]]
    court = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()

    def top(points: np.ndarray) -> np.ndarray:
        return cast(
            np.ndarray,
            np.round(
                np.stack((440 + points[..., 0] * 24, 460 - points[..., 1] * 24), -1)
            ).astype(int),
        )

    canvas: np.ndarray = np.full((900, 1600, 3), (28, 24, 20), dtype=np.uint8)

    def label(
        im: np.ndarray,
        text: str,
        point: tuple[int, int],
        color: tuple[int, int, int] = (235, 235, 235),
        scale: float = 0.65,
    ) -> None:
        cv2.putText(
            im, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
        )

    for a, b in COURT_SKELETON:
        if a < 14 and b < 14:
            cv2.line(
                canvas,
                tuple(top(court[a])),
                tuple(top(court[b])),
                (150, 170, 165),
                2,
                cv2.LINE_AA,
            )
    cv2.line(
        canvas,
        tuple(top(np.array([-5.485, 0]))),
        tuple(top(np.array([5.485, 0]))),
        (210, 210, 210),
        2,
    )
    label(canvas, "PLCS | ground prior + learned offset", (40, 42), scale=0.9)
    label(canvas, "Baseline: direct regression", (40, 77), BASE)
    label(canvas, "New: geometry tokens + residual + hard sampling", (40, 107), NEW)
    label(canvas, "Court top view | root XY in meters", (40, 150))
    label(canvas, "No measured 3D ground truth", (40, 815), scale=0.7)
    label(
        canvas,
        "Cameras: circles = projected root; white cross = observed hips",
        (40, 845),
        scale=0.62,
    )
    selected = set(range(0, frames, 2))
    writer = VideoWriter(args.comparison / "comparison.mp4", fps=fps / 2)
    try:
        for frame in range(frames):
            images = []
            for cap in cameras:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Video decode failed at {frame}")
                images.append(image)
            if frame not in selected:
                continue
            out = canvas.copy()
            for data, color in [(base, BASE), (new, NEW)]:
                for person in range(2):
                    trail = top(
                        data["position"][person, max(0, frame - 90) : frame + 1, :2]
                    )
                    cv2.polylines(out, [trail], False, color, 2, cv2.LINE_AA)
                    point = tuple(trail[-1])
                    cv2.circle(out, point, 7, color, -1, cv2.LINE_AA)
                    label(
                        out,
                        f"P{person + 1}",
                        (point[0] + 10, point[1] - 10),
                        color,
                        0.6,
                    )
            label(
                out,
                f"{frame / fps:5.2f} s / {frames / fps:.2f} s",
                (40, 780),
                scale=0.8,
            )
            for v, im in enumerate(images):
                im = cv2.resize(im, (534, 300))
                for p in range(2):
                    if base["hip_valid"][p, v, frame]:
                        observed = base["hip_observed_uv"][p, v, frame] * [534, 300]
                        cv2.drawMarker(
                            im,
                            tuple(observed.astype(int)),
                            (255, 255, 255),
                            cv2.MARKER_CROSS,
                            10,
                            2,
                        )
                    for data, color in [(base, BASE), (new, NEW)]:
                        uv = data["projected_root_px"][p, v, frame] * [
                            534 / clip["width"],
                            300 / clip["height"],
                        ]
                        if np.isfinite(uv).all() and np.abs(uv).max() < 100000:
                            pt = tuple(np.round(uv).astype(int))
                            cv2.circle(im, pt, 6, color, 2, cv2.LINE_AA)
                label(im, f"cam{v}", (12, 24), scale=0.6)
                out[v * 300 : (v + 1) * 300, 1066:] = im
            writer.write_frame(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if frame in [0, 250, 500, 750]:
                cv2.imwrite(str(args.comparison / f"preview_{frame:04d}.jpg"), out)
    finally:
        writer.close()
        for cap in cameras:
            cap.release()
    base_test = np.load(args.comparison / "baseline_test.npz")
    new_test = np.load(args.comparison / f"{args.new_label}_test.npz")
    np.testing.assert_array_equal(base_test["target"], new_test["target"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for data, name, plot_color in [
        (base_test, "Baseline", "#ef6461"),
        (new_test, "Residual", "#168c9e"),
    ]:
        error = np.linalg.norm(
            (data["position"] - data["target"]) * np.array(COURT_COORD_SCALE_XYZ),
            axis=-1,
        ).ravel()
        axes[0].plot(
            np.sort(error),
            np.linspace(0, 1, len(error)),
            label=name,
            color=plot_color,
        )
    axes[0].set(
        xlim=(0, 2), xlabel="Synthetic 3D error (m)", ylabel="Cumulative fraction"
    )
    axes[0].legend()
    for data, name, plot_color in [
        (base, "Baseline", "#ef6461"),
        (new, "Residual", "#168c9e"),
    ]:
        e = data["reprojection_error_px"]
        valid = data["hip_valid"]
        mean = np.where(valid, e, np.nan).reshape(-1, frames)
        axes[1].plot(
            np.arange(frames) / fps,
            np.nanmean(mean, axis=0),
            label=name,
            color=plot_color,
            alpha=0.85,
        )
    axes[1].set(xlabel="Clip time (s)", ylabel="Mean root reprojection error (px)")
    anchors = base_test["anchor"]
    target = base_test["target"]
    hard = (
        np.linalg.norm(
            (anchors - target)[..., :2] * np.array(COURT_COORD_SCALE_XYZ[:2]), axis=-1
        )
        > 1
    )
    hard &= base_test["anchor_valid"].astype(bool)
    for data, name, plot_color in [
        (base_test, "Baseline", "#ef6461"),
        (new_test, "Residual", "#168c9e"),
    ]:
        xy = np.linalg.norm(
            (data["position"] - target)[..., :2] * np.array(COURT_COORD_SCALE_XYZ[:2]),
            axis=-1,
        )
        axes[2].bar(name, xy[hard].mean(), color=plot_color)
    axes[2].set(ylabel="XY error for prior error >1m (m)")
    fig.tight_layout()
    fig.savefig(args.comparison / "accuracy_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
