"""Read-only ACCAD (AMASS/SMPL-H) motion review UI.

The package serves a local browser viewer for the raw ACCAD mocap archives:
a searchable directory list on the left and the selected motion reconstructed
in the world coordinate system on the right.
"""

from .dataset_service import PLCSDatasetReviewService
from .dataset_web import create_dataset_app
from .service import ReviewService
from .web import create_app

__all__ = [
    "PLCSDatasetReviewService",
    "ReviewService",
    "create_app",
    "create_dataset_app",
]
