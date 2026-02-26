"""Task-specific heads for ball multi-task models."""

from src.developing.ball_multitask.models.heads.trajectory_head import Trajectory3DHeadAdapter
from src.developing.ball_multitask.models.heads.uv_head import UVCompletionHead
from src.developing.ball_multitask.models.heads.event_head import EventLogitsHeadAdapter
from src.developing.ball_multitask.models.heads.in_frame_head import InFrameHead

__all__ = ["UVCompletionHead", "Trajectory3DHeadAdapter", "EventLogitsHeadAdapter", "InFrameHead"]
