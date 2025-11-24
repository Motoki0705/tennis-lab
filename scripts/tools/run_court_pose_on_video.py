import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from third_party.court_pose.dino_fpn_v2.dino_fpn_v2_loader import (
    DinoFpnV2LoadConfig,
    load_dino_fpn_v2_with_ckpt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Court Pose (DINO FPN v2) on a video and overlay keypoints.",
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("output", type=Path, help="Output video path")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to DinoFpnV2LoadConfig YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint for court pose model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--draw-radius",
        type=int,
        default=4,
        help="Circle radius for keypoint drawing (pixels)",
    )
    parser.add_argument(
        "--draw-thickness",
        type=int,
        default=2,
        help="Circle thickness for keypoint drawing (pixels)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum heatmap value to draw a keypoint (after argmax)",
    )
    return parser.parse_args()


def load_court_pose_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = DinoFpnV2LoadConfig.from_yaml(cfg_path)
    cfg.checkpoint_path = ckpt_path
    cfg.device = device

    model, transform, device_obj = load_dino_fpn_v2_with_ckpt(cfg)
    return model, transform, device_obj


def heatmaps_to_keypoints(heatmaps: torch.Tensor) -> list[tuple[int, int, float]]:
    """Convert a single-frame heatmap tensor [K, H, W] to list of (x, y, score)."""
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [K, H, W] heatmap, got shape={tuple(heatmaps.shape)}")

    k, h, w = heatmaps.shape
    coords: list[tuple[int, int, float]] = []
    # Flatten each channel and take argmax
    for c in range(k):
        channel = heatmaps[c]
        value, index = torch.max(channel.reshape(-1), dim=0)
        y = int((index // w).item())
        x = int((index % w).item())
        coords.append((x, y, float(value.item())))
    return coords


def draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints: list[tuple[int, int, float]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
    thickness: int = 2,
    score_threshold: float = 0.0,
) -> np.ndarray:
    """Draw keypoints on a BGR frame in-place and return the frame."""
    h, w, _ = frame_bgr.shape
    for x, y, score in keypoints:
        if score < score_threshold:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(frame_bgr, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
    return frame_bgr


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    # Load model + transform
    model, transform, device = load_court_pose_model(
        cfg_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {args.output}")

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB PIL.Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Preprocess
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                heatmaps = model(tensor)  # [1, K, H, W]

            heatmaps_0 = heatmaps[0].detach().cpu()
            keypoints = heatmaps_to_keypoints(heatmaps_0)

            # Draw on original-resolution frame
            draw_keypoints(
                frame_bgr,
                keypoints,
                radius=args.draw_radius,
                thickness=args.draw_thickness,
                score_threshold=args.score_threshold,
            )

            writer.write(frame_bgr)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    main()
