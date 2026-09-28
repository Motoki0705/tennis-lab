"""External inputs and supervision, independent of camera/track identity."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from src.tasks.ball_detection.model_io.contracts import BallCandidates


@dataclass(frozen=True)
class Refiner2DInput:
    """One independent camera-window per batch row; no temporal padding.

    Coordinates use source x/(W-1), y/(H-1). Pose joints are COCO17 indices
    (7, 8, 9, 10): left/right elbows then left/right wrists. D is an unordered
    person axis (D=0 is allowed); C is the fixed, ordered court KP axis.
    Context coordinates may extend outside the image. All float inputs must
    be finite float32, including masked slots, whose values are ignored.
    """

    candidates: BallCandidates
    timestamps_seconds: Tensor  # B,T; strictly increasing, real PTS seconds
    pose_uv: Tensor  # B,T,D,4,2
    pose_confidence: Tensor  # B,T,D,4; [0,1]
    pose_valid: Tensor  # B,T,D,4; bool, observed and usable
    court_uv: Tensor  # B,C,2; frame-0 prior, no time axis
    court_confidence: Tensor  # B,C; [0,1]
    court_valid: Tensor  # B,C; bool


@dataclass(frozen=True)
class Refiner2DTarget:
    """Amodal in-frame presence and conditional normalized-uv supervision.

    position_valid implies presence_valid and presence=1. Unknown coordinates
    may contain NaN; the loss removes them before arithmetic. weight is a
    finite nonnegative frame weight (e.g. pseudo-label quality), shared by
    the joint likelihood terms. Zero weight explicitly excludes a frame.
    """

    uv: Tensor  # B,T,2; float32/64
    position_valid: Tensor  # B,T; bool
    presence: Tensor  # B,T; bool, not detector visibility
    presence_valid: Tensor  # B,T; bool
    weight: Tensor  # B,T; float32/64
