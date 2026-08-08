"""Typed rejection reasons for stochastic full-physics BLCS proposals."""


class FullPhysicsProposalError(RuntimeError):
    """A sampled full-physics proposal is invalid and may be resampled."""


class FullPhysicsLandingError(FullPhysicsProposalError):
    """A sampled shot produced no valid full-physics landing."""


class FullPhysicsReturnTimingError(FullPhysicsProposalError):
    """A sampled return time places the hitter on the wrong court side."""


__all__ = [
    "FullPhysicsLandingError",
    "FullPhysicsProposalError",
    "FullPhysicsReturnTimingError",
]
