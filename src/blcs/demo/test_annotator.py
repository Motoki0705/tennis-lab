#!/usr/bin/env python
"""Test script for court annotator UI.

Generates a dummy court image to test annotation without needing a real video.

Usage:
    python -m src.blcs.demo.test_annotator

"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from src.blcs.demo.court_annotator import (
    COURT_KEYPOINT_NAMES,
    CourtAnnotator,
    QuickAnnotator,
)


def create_dummy_court_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a dummy court image for testing.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        RGB image array, shape (H, W, 3).

    """
    # Create green background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]  # Forest green

    # Draw simplified court lines (white)
    line_color = [255, 255, 255]

    # Court boundaries (centered, scaled)
    cx, cy = width // 2, height // 2
    court_w, court_h = int(width * 0.7), int(height * 0.85)
    x1, y1 = cx - court_w // 2, cy - court_h // 2
    x2, y2 = cx + court_w // 2, cy + court_h // 2

    # Draw baselines
    img[y1 : y1 + 3, x1:x2] = line_color
    img[y2 - 3 : y2, x1:x2] = line_color

    # Draw sidelines
    img[y1:y2, x1 : x1 + 3] = line_color
    img[y1:y2, x2 - 3 : x2] = line_color

    # Draw net (center horizontal)
    img[cy - 1 : cy + 2, x1:x2] = [128, 128, 128]

    # Draw service lines
    service_y_offset = court_h // 4
    img[
        cy - service_y_offset : cy - service_y_offset + 2,
        x1 + court_w // 6 : x2 - court_w // 6,
    ] = line_color
    img[
        cy + service_y_offset : cy + service_y_offset + 2,
        x1 + court_w // 6 : x2 - court_w // 6,
    ] = line_color

    # Draw center service line
    img[cy - service_y_offset : cy + service_y_offset, cx - 1 : cx + 2] = line_color

    # Singles sidelines
    singles_offset = court_w // 10
    img[y1:y2, x1 + singles_offset : x1 + singles_offset + 2] = line_color
    img[y1:y2, x2 - singles_offset - 2 : x2 - singles_offset] = line_color

    return img


def test_full_annotator() -> None:
    """Test the full 20-point annotator."""
    print("=" * 60)
    print("Testing Full Court Annotator (20 points)")
    print("=" * 60)
    print("\nKeypoint order:")
    for i, name in enumerate(COURT_KEYPOINT_NAMES):
        print(f"  {i:2d}: {name}")

    print("\nCreating dummy court image...")
    img = create_dummy_court_image()

    print("Opening annotator...")
    print("Click on the image to add points in order.")
    print("Use 'Undo' to remove last point, 'Reset' to start over.")
    print("Click 'Done' when finished.\n")

    annotator = CourtAnnotator()
    keypoints = annotator.annotate(img)

    print("\nAnnotation results:")
    valid = (keypoints[:, 0] >= 0).sum()
    print(f"  Valid points: {valid}/20")

    if valid > 0:
        print("\n  Keypoint coordinates (normalized):")
        for i, (x, y) in enumerate(keypoints):
            if x >= 0:
                print(
                    f"    {i:2d} ({COURT_KEYPOINT_NAMES[i][:20]:20s}): ({x:.4f}, {y:.4f})"
                )


def test_quick_annotator() -> None:
    """Test the quick 4-corner annotator."""
    print("=" * 60)
    print("Testing Quick Court Annotator (4 corners)")
    print("=" * 60)

    print("\nCreating dummy court image...")
    img = create_dummy_court_image()

    print("Opening quick annotator...")
    print("Click on the 4 doubles corners.\n")

    annotator = QuickAnnotator()
    keypoints = annotator.annotate(img)

    print("\nAnnotation results (interpolated):")
    valid = (keypoints[:, 0] >= 0).sum()
    print(f"  Valid points: {valid}/20")

    # Visualize interpolated points
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    h, w = img.shape[:2]
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))

    for i, (x, y) in enumerate(keypoints):
        if x >= 0:
            px, py = x * w, y * h
            ax.scatter([px], [py], c=[colors[i]], s=80, marker="o", edgecolors="black")
            ax.text(px + 5, py - 5, str(i), color="white", fontsize=8)

    ax.set_title("Interpolated Court Keypoints")
    plt.show()


def main() -> int:
    """Main entry point."""
    print("BLCS Court Annotator Test")
    print("=" * 60)
    print("\nChoose test mode:")
    print("  1. Full annotator (20 points)")
    print("  2. Quick annotator (4 corners)")
    print("  3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        test_full_annotator()
    elif choice == "2":
        test_quick_annotator()
    else:
        test_full_annotator()
        test_quick_annotator()

    print("\nTest complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
