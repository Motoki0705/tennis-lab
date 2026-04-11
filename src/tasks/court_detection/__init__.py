"""Court detection module.

Provides three tasks:

* **seg** — Court cell segmentation (7 classes).
* **kp**  — Court keypoint heatmap regression (14 keypoints).
* **line** — Court white-line segmentation (binary).
"""

from src.tasks.court_detection.models.court_unet import CourtUNet

__all__ = ["CourtUNet"]
