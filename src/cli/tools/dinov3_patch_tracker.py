"""CLI tool for DINOv3-based patch-token tracking and segmentation.

This tool loads a local DINOv3 backbone via ``load_dinov3``, lets the user
interactively select an ROI on the first frame of a video using OpenCV's
``selectROI`` UI, and then runs patch-token-based tracking + segmentation on
all subsequent frames.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from src.tools.dinov3_patch_tracker import Dinov3PatchTracker, TrackerConfig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DINOv3-based single-object tracking and segmentation using "
            "patch-token cosine similarity."
        )
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help=(
            "Optional path for output video. If empty, a file named "
            "'<stem>_dinov3_track.mp4' is created next to the input video."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="dinov3_vits16",
        help="DINOv3 backbone architecture (torch.hub name).",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=(
            "third_party/dinov3/checkpoints/"
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        ),
        help="Path to pretrained DINOv3 weights (local .pth file).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine-similarity threshold for foreground segmentation.",
    )
    parser.add_argument(
        "--template-update-alpha",
        type=float,
        default=0.0,
        help=(
            "EMA factor for template updates per frame. "
            "Set to 0.0 to disable template updates."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional limit on the number of frames to process (negative = all).",
    )
    return parser.parse_args(argv)


def _make_output_path(video_path: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    stem = video_path.stem
    return video_path.with_name(f"{stem}_dinov3_track.mp4")


def _draw_overlay(
    frame: np.ndarray,
    mask_uint8: np.ndarray | None,
    bbox_xywh: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Overlay segmentation mask and bounding box on the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    if mask_uint8 is not None and mask_uint8.shape[:2] == (h, w):
        colored = np.zeros_like(vis)
        colored[:, :] = (0, 255, 0)
        fg = mask_uint8 > 0
        alpha = 0.4
        vis[fg] = (
            (1.0 - alpha) * vis[fg].astype(np.float32)
            + alpha * colored[fg].astype(np.float32)
        ).astype(np.uint8)

    if bbox_xywh is not None:
        x, y, bw, bh = bbox_xywh
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    return vis


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive DINOv3 tracker CLI.

    Args:
        argv (Sequence[str] | None): Optional command-line arguments. Uses
            ``sys.argv`` when ``None``.

    Returns:
        int: Zero when the video is processed successfully, or a non-zero
        exit code if initialization/tracking fails.

    """
    args = _parse_args(argv)

    video_path = Path(args.video_path)
    if not video_path.exists():
        sys.stderr.write(f"[error] Video not found: {video_path}\n")
        return 2

    out_path = _make_output_path(video_path, args.output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.stderr.write(f"[error] Failed to open video: {video_path}\n")
        return 1

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        sys.stderr.write("[error] Failed to read first frame from video.\n")
        cap.release()
        return 1

    win_name = "DINOv3 Tracker - Select ROI and press ENTER"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, first_frame)
    bbox = cv2.selectROI(win_name, first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_name)

    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        sys.stderr.write("[error] No valid ROI selected.\n")
        cap.release()
        return 2

    cfg = TrackerConfig(
        arch=str(args.arch),
        weights_path=str(args.weights_path) if args.weights_path else None,
        threshold=float(args.threshold),
        template_update_alpha=float(args.template_update_alpha),
        device=str(args.device),
    )
    tracker = Dinov3PatchTracker(cfg)

    try:
        tracker.set_template(first_frame, (float(x), float(y), float(w), float(h)))
    except Exception as exc:  # pragma: no cover - interactive failure path
        sys.stderr.write(f"[error] Failed to initialize template: {exc}\n")
        cap.release()
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        sys.stderr.write(f"[error] Failed to create VideoWriter for: {out_path}\n")
        cap.release()
        return 1

    frame_idx = 0
    current_frame = first_frame
    max_frames = int(args.max_frames)

    try:
        while current_frame is not None:
            try:
                mask, bbox_tracked, _ = tracker.track(current_frame)
            except Exception as exc:  # pragma: no cover - runtime failure path
                sys.stderr.write(
                    f"[warn] Tracking failed at frame {frame_idx}: {exc}\n"
                )
                mask = None
                bbox_tracked = None

            overlay = _draw_overlay(current_frame, mask, bbox_tracked)
            writer.write(overlay)

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break
            current_frame = frame

    finally:
        cap.release()
        writer.release()

    print(f"[dinov3-tracker] Wrote video to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
