"""Explicit method selection; encoders and feature extraction are independent."""

from src.tasks.person_tracking.botsort_pose import BotSortPose, BotSortPoseConfig
from src.tasks.person_tracking.contracts import TrackingMethod


def build_tracker(method: str, *, fps: float, config: BotSortPoseConfig) -> TrackingMethod:
    if method == "botsort_pose":
        return BotSortPose(fps, config)
    raise ValueError(f"Tracking method {method!r} is not implemented; available: botsort_pose")
