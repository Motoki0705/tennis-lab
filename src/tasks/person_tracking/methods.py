"""Explicit method selection; encoders and feature extraction are independent."""

from src.tasks.person_tracking.botsort_pose import BotSortPose, BotSortPoseConfig
from src.tasks.person_tracking.contracts import TrackingMethod
from src.tasks.person_tracking.deep_ocsort_pose import (
    DeepOCSortPose,
    DeepOCSortPoseConfig,
)


def build_tracker(method: str, *, fps: float,
                  config: BotSortPoseConfig | DeepOCSortPoseConfig | None = None) -> TrackingMethod:
    if method == "botsort_pose":
        if config is not None and not isinstance(config, BotSortPoseConfig):
            raise TypeError('botsort_pose requires BotSortPoseConfig')
        return BotSortPose(fps, config or BotSortPoseConfig())
    if method == 'deep_ocsort_pose':
        if config is not None and not isinstance(config, DeepOCSortPoseConfig):
            raise TypeError('deep_ocsort_pose requires DeepOCSortPoseConfig')
        return DeepOCSortPose(fps, config or DeepOCSortPoseConfig())
    raise ValueError(f"Tracking method {method!r} is not implemented; available: botsort_pose, deep_ocsort_pose")
