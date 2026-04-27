"""Court detection module.

Provides three tasks:

* **seg** — Court cell segmentation (7 classes).
* **kp**  — Court keypoint heatmap regression (14 keypoints).
* **line** — Court white-line segmentation (binary).
"""

from src.tasks.court_detection.models.court_fpn import CourtFPN

__all__ = ["CourtFPN"]  # noqa: F401
