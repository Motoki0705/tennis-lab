"""Whole-frame shape baseline using local contrast and frame differencing."""

import cv2
import numpy as np

OFFSETS = (-6, -4, -2, 0, 2, 4, 6)
SCALES = (0.8, 1.4, 2.5, 4.0)


def shape_peak(frames: list[np.ndarray]) -> tuple[tuple[float, float], float]:
    """Return the strongest candidate before the presence threshold."""
    center = len(OFFSETS) // 2
    if len(frames) != len(OFFSETS):
        raise ValueError("Shape baseline requires the seven declared frames")
    height, width = frames[center].shape[:2]
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    background = np.median([g for i, g in enumerate(gray) if i != center], axis=0).astype(np.float32)
    image = gray[center].astype(np.float32)
    difference = np.minimum(1, np.abs(image - background) / 36)

    hue, saturation, _ = cv2.split(cv2.cvtColor(frames[center], cv2.COLOR_BGR2HSV))
    yellow = (hue >= 18) & (hue <= 44) & (saturation >= 35)
    color = np.where(yellow | (saturation <= 76), 1., 0.55).astype(np.float32)

    scores = np.zeros((height, width), dtype=np.float32)
    for sigma in SCALES:
        smooth = cv2.GaussianBlur(image, (0, 0), sigma)
        surround = cv2.GaussianBlur(image, (0, 0), 1.8 * sigma)
        contrast = np.minimum(1, np.abs(smooth - surround) / 26)
        xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        # Equal-sign and equal-magnitude curvatures give a roundness of 1.
        roundness = np.clip(4 * (xx * yy - xy * xy) / ((xx + yy) ** 2 + 1e-6), 0, 1)
        response = contrast * roundness * color * (0.15 + 0.85 * difference)
        np.maximum(scores, response, out=scores)
    y, x = np.unravel_index(scores.argmax(), scores.shape)
    return (x / (width - 1), y / (height - 1)), float(scores[y, x])
