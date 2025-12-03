#!/usr/bin/env python
"""Simplified BLCS demo for quick testing.

This script demonstrates the annotation and visualization workflow
without requiring actual model checkpoints.

Usage:
    python -m src.blcs.demo.simple_demo --video input.mp4

"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from src.blcs.demo.court_annotator import CourtAnnotator, QuickAnnotator
from src.blcs.demo.video_processor import VideoProcessor
from src.blcs.inference.visualization import TrajectoryVisualizer


def run_simple_demo(video_path: str) -> None:
    """Run simplified demo without model inference.

    This demo:
    1. Loads video
    2. Allows court keypoint annotation
    3. Shows dummy 3D trajectory visualization

    Args:
        video_path: Path to video file.

    """
    print("=" * 60)
    print("BLCS Simple Demo (No Model Required)")
    print("=" * 60)

    # Load video
    print(f"\nLoading video: {video_path}")
    processor = VideoProcessor(max_frames=300)  # Limit for demo
    frames, fps, total = processor.load_video(video_path)
    print(f"  Loaded {len(frames)} frames at {fps:.1f} fps")

    # Show first frame and let user pick reference
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frames[0])
    ax.set_title("Reference frame (close to continue)")
    plt.show()

    # Court annotation
    print("\nStarting court annotation...")
    print("Choose annotation mode:")
    print("  1. Full annotation (20 points)")
    print("  2. Quick annotation (4 corners)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        annotator = CourtAnnotator()
        court_kp = annotator.annotate(frames[0])
    else:
        annotator = QuickAnnotator()
        court_kp = annotator.annotate(frames[0])

    valid = (court_kp[:, 0] >= 0).sum()
    print(f"\nAnnotated {valid}/20 keypoints")

    # Generate dummy 3D trajectory for visualization demo
    print("\nGenerating dummy 3D trajectory for visualization...")
    T = len(frames)
    t = np.linspace(0, 2 * np.pi, T)

    # Simulate a ball trajectory (parabolic arc)
    ball_3d = np.zeros((T, 3), dtype=np.float32)
    ball_3d[:, 0] = 3 * np.sin(t)  # X: side-to-side
    ball_3d[:, 1] = -10 + 20 * (t / (2 * np.pi))  # Y: baseline to baseline
    ball_3d[:, 2] = 2 + 3 * np.sin(t) ** 2  # Z: height with bounce

    # Visualize
    print("\nGenerating visualizations...")
    visualizer = TrajectoryVisualizer()

    fig_3d = visualizer.plot_trajectory_3d(
        ball_3d,
        title="Demo: Simulated 3D Ball Trajectory",
        show_court=True,
    )

    fig_2d = visualizer.plot_trajectory_2d(
        ball_3d,
        title="Demo: Trajectory Top View",
        show_court=True,
    )

    # Show frame with overlay
    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 8))
    ax_overlay.imshow(frames[0])
    ax_overlay.set_title("Reference Frame with Court Keypoints")

    # Draw keypoints
    h, w = frames[0].shape[:2]
    valid_mask = court_kp[:, 0] >= 0
    xs = court_kp[valid_mask, 0] * w
    ys = court_kp[valid_mask, 1] * h
    ax_overlay.scatter(
        xs, ys, c="lime", s=100, marker="o", edgecolors="black", linewidths=2
    )

    # Add labels
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_overlay.text(
            x + 5, y - 5, str(i), color="yellow", fontsize=8, fontweight="bold"
        )

    plt.show()

    print("\nDemo complete!")
    print("In a full demo, the 3D trajectory would be predicted by the BLCS model.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BLCS Simple Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    run_simple_demo(args.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
