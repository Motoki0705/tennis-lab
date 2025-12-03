"""Demo script for ball detection using WASBPredictor.

This script detects tennis ball positions in a video and overlays
the detection results on the output video.

Usage:
    python demo/wasb_ball_detection.py \
        --video path/to/input.mp4 \
        --checkpoint third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar \
        --output path/to/output.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.wasb.inference import HRCNetWASBPredictor, WASBPredictor

PREDICTOR_REGISTRY = {
    "wasb": WASBPredictor,
    "hrcnet": HRCNetWASBPredictor,
}


def load_video_frames(video_path: str | Path) -> tuple[np.ndarray, dict]:
    """Load all frames from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (frames array, video metadata).
        frames: RGB frames with shape (T, H, W, 3).
        metadata: Dictionary with fps, width, height, frame_count.

    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }

    return np.array(frames, dtype=np.uint8), metadata


def draw_ball_marker(
    frame: np.ndarray,
    x: float,
    y: float,
    visible: bool,
    score: float,
    radius: int = 10,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a ball marker on the frame.

    Args:
        frame: BGR frame to draw on.
        x: Ball x coordinate in pixels.
        y: Ball y coordinate in pixels.
        visible: Whether the ball is visible.
        score: Detection confidence score.
        radius: Circle radius.
        thickness: Circle line thickness.

    Returns:
        Frame with ball marker drawn.

    """
    if not visible:
        return frame

    # Color based on confidence (green=high, yellow=medium, red=low)
    if score > 0.7:
        color = (0, 255, 0)  # Green
    elif score > 0.4:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 165, 255)  # Orange

    center = (int(x), int(y))

    # Draw filled circle with transparency effect
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw circle outline
    cv2.circle(frame, center, radius, color, thickness)

    # Draw crosshair
    cv2.line(
        frame,
        (center[0] - radius - 5, center[1]),
        (center[0] - radius, center[1]),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (center[0] + radius, center[1]),
        (center[0] + radius + 5, center[1]),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (center[0], center[1] - radius - 5),
        (center[0], center[1] - radius),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (center[0], center[1] + radius),
        (center[0], center[1] + radius + 5),
        color,
        thickness,
    )

    # Draw score text
    score_text = f"{score:.2f}"
    text_pos = (center[0] + radius + 5, center[1] - 5)
    cv2.putText(
        frame,
        score_text,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[tuple[float, float]],
    max_points: int = 15,
) -> np.ndarray:
    """Draw ball trajectory trail on frame.

    Args:
        frame: BGR frame to draw on.
        trajectory: List of (x, y) coordinates.
        max_points: Maximum number of points to draw.

    Returns:
        Frame with trajectory drawn.

    """
    if len(trajectory) < 2:
        return frame

    # Use last N points
    points = trajectory[-max_points:]

    # Draw fading trail
    for i in range(1, len(points)):
        alpha = i / len(points)
        color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
        thickness = max(1, int(3 * alpha))

        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    return frame


def draw_info_panel(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    visible: bool,
    score: float,
) -> np.ndarray:
    """Draw information panel on frame.

    Args:
        frame: BGR frame to draw on.
        frame_idx: Current frame index.
        total_frames: Total number of frames.
        visible: Whether ball is visible.
        score: Detection confidence score.

    Returns:
        Frame with info panel drawn.

    """
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text
    cv2.putText(
        frame,
        f"Frame: {frame_idx + 1}/{total_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    status = "DETECTED" if visible else "NOT DETECTED"
    status_color = (0, 255, 0) if visible else (0, 0, 255)
    cv2.putText(
        frame,
        f"Ball: {status}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
    )

    if visible:
        cv2.putText(
            frame,
            f"Confidence: {score:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


def create_output_video(
    frames: np.ndarray,
    results: dict,
    output_path: str | Path,
    fps: float,
    show_trajectory: bool = True,
) -> None:
    """Create output video with ball detections overlaid.

    Args:
        frames: Original RGB frames with shape (T, H, W, 3).
        results: Detection results from WASBPredictor.
        output_path: Path to save output video.
        fps: Output video frame rate.
        show_trajectory: Whether to draw ball trajectory.

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Build frame-indexed results
    frame_results = {}
    for i, frame_idx in enumerate(results["frame_indices"]):
        frame_results[int(frame_idx)] = {
            "xy": results["ball_xy_px"][i],
            "visible": results["visibility"][i],
            "score": results["score"][i],
        }

    trajectory: list[tuple[float, float]] = []

    for frame_idx in tqdm(range(len(frames)), desc="Creating output video"):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR)

        # Get detection for this frame
        if frame_idx in frame_results:
            result = frame_results[frame_idx]
            x, y = result["xy"]
            visible = result["visible"]
            score = result["score"]

            if visible:
                trajectory.append((x, y))

            # Draw trajectory
            if show_trajectory:
                frame_bgr = draw_trajectory(frame_bgr, trajectory)

            # Draw ball marker
            frame_bgr = draw_ball_marker(frame_bgr, x, y, visible, score)

            # Draw info panel
            frame_bgr = draw_info_panel(
                frame_bgr, frame_idx, len(frames), visible, score
            )
        else:
            # No detection for this frame
            frame_bgr = draw_info_panel(frame_bgr, frame_idx, len(frames), False, 0.0)

        writer.write(frame_bgr)

    writer.release()
    print(f"Output saved to: {output_path}")


def main() -> None:
    """Run the ball detection demo."""
    parser = argparse.ArgumentParser(
        description="Detect tennis ball in video using WASB model"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar",
        help="Path to WASB checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wasb",
        choices=list(PREDICTOR_REGISTRY.keys()),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video (default: input_detected.mp4)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detection score threshold",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory visualization",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Set default output path
    if args.output is None:
        output_path = video_path.parent / f"{video_path.stem}_detected.mp4"
    else:
        output_path = Path(args.output)

    print(f"Loading video from {video_path}...")
    frames, metadata = load_video_frames(video_path)
    print(
        f"  Loaded {len(frames)} frames ({metadata['width']}x{metadata['height']}, {metadata['fps']:.1f} fps)"
    )

    # Limit frames if specified
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
        print(f"  Limited to {len(frames)} frames")

    predictor_cls = PREDICTOR_REGISTRY.get(args.model)
    if predictor_cls is None:
        raise ValueError(f"Unknown model: {args.model}")
    print(f"Loading {args.model} model from {args.checkpoint}...")
    predictor = predictor_cls.load_from_checkpoint(
        args.checkpoint,
        device="cuda",
        score_threshold=args.score_threshold,
    )

    print("Running ball detection...")
    results = predictor.predict(frames)

    # Count detections
    visible_count = int(results["visibility"].sum())
    print(f"  Detected ball in {visible_count}/{len(frames)} frames")

    print("Creating output video...")
    create_output_video(
        frames,
        results,
        output_path,
        fps=metadata["fps"],
        show_trajectory=not args.no_trajectory,
    )

    print("Done!")


if __name__ == "__main__":
    main()
