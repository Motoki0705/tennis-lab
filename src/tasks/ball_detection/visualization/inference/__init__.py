"""Backend for the ball-detection dataset review and inference web UIs.

``DetectionService`` is the single entry point the shared detection app
(``src.tasks.base.visualization.detection``) imports: it serves the dataset and
checkpoint catalogs, original-size ground truth previews, frame images, and one
bounded inference window per request.
"""

from .loader import (
    BallInferenceCheckpointError,
    LoadedBallModel,
    load_ball_model,
)
from .peaks import FramePeaks, decode_frame_peaks, peaks_to_points
from .rasters import Raster, probability_raster
from .service import (
    PREVIEW_FRAME_LIMIT,
    TASK,
    TITLE,
    DetectionRequestError,
    DetectionService,
    WindowMode,
    WindowPlan,
    dataset_ids,
)

__all__ = [
    "PREVIEW_FRAME_LIMIT",
    "TASK",
    "TITLE",
    "BallInferenceCheckpointError",
    "DetectionRequestError",
    "DetectionService",
    "FramePeaks",
    "LoadedBallModel",
    "Raster",
    "WindowMode",
    "WindowPlan",
    "dataset_ids",
    "decode_frame_peaks",
    "load_ball_model",
    "peaks_to_points",
    "probability_raster",
]
