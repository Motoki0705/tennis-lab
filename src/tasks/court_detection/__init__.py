"""Court detection module.

Provides three tasks:

* **seg** — Court cell segmentation (7 classes).
* **kp**  — Court keypoint heatmap regression (14 keypoints).
* **line** — Court white-line segmentation (binary).
"""

from src.tasks.court_detection.model_io.factory import build_court_detection_pair

__all__ = ["build_court_detection_pair"]
