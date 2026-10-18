"""Pure phase schedule: frame index -> every layer's progress value."""

from __future__ import annotations

from dataclasses import dataclass

from src.synthetic_data_generation.visualization.reveal.contracts import (
    REVEAL_PHASE_ORDER,
    RevealDurations,
    RevealPhase,
)


@dataclass(frozen=True, slots=True)
class NHTRenderState:
    """Everything one ``nht-render`` image depends on; equal states share it.

    ``view_progress`` places the camera on the free-viewpoint path (0 is the
    fixed reveal viewpoint).
    """

    splat_progress: float
    view_dependence: float
    view_progress: float


@dataclass(frozen=True, slots=True)
class RevealFrameState:
    """Every progress value needed to composite one frame (all in ``[0, 1]``).

    ``camera_fraction`` and ``sfm_fraction`` reveal prefixes of the capture
    cameras and SfM points. ``*_opacity`` fade whole layers. ``color_progress``
    moves Gaussian points from one uniform colour to their decoded colour, and
    ``matte_release`` moves from compositing the NHT alpha over the background
    to the raw NHT RGB (the exact ``nht-render`` output). ``alignment_progress``
    draws the accepted court overlay on top of the render.
    """

    phase: RevealPhase
    camera_fraction: float
    camera_opacity: float
    sfm_fraction: float
    sfm_opacity: float
    gaussian_opacity: float
    color_progress: float
    matte_release: float
    alignment_progress: float
    nht: NHTRenderState

    @property
    def needs_nht(self) -> bool:
        """Whether the Gaussian layer is visible in this frame."""
        return self.gaussian_opacity > 0.0


def smoothstep(value: float) -> float:
    """Ease a linear ``[0, 1]`` parameter with zero slope at both ends."""
    t = min(max(value, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def phase_frame_counts(durations: RevealDurations, fps: float) -> tuple[int, ...]:
    """Return at least one frame per phase, rounded from its duration."""
    return tuple(max(1, round(seconds * fps)) for seconds in durations.seconds)


def _state(phase: RevealPhase, t: float) -> RevealFrameState:
    """Return the state ``t`` (eased, in ``(0, 1]``) of the way through a phase."""
    index = REVEAL_PHASE_ORDER.index(phase)

    def after(target: RevealPhase) -> float:
        """1 once ``target`` is complete, ``t`` while it runs, 0 before it."""
        target_index = REVEAL_PHASE_ORDER.index(target)
        if index > target_index:
            return 1.0
        if index == target_index:
            return t
        return 0.0

    crossfade = after(RevealPhase.SFM_TO_GAUSSIANS)
    return RevealFrameState(
        phase=phase,
        camera_fraction=after(RevealPhase.CAMERAS),
        camera_opacity=1.0 - crossfade,
        sfm_fraction=after(RevealPhase.SFM),
        sfm_opacity=1.0 - crossfade,
        gaussian_opacity=crossfade,
        color_progress=after(RevealPhase.COLORIZE),
        matte_release=after(RevealPhase.SPLAT),
        alignment_progress=after(RevealPhase.ALIGNMENT),
        nht=NHTRenderState(
            splat_progress=after(RevealPhase.SPLAT),
            view_dependence=after(RevealPhase.VIEW_DEPENDENCE),
            view_progress=after(RevealPhase.CAMERA_MOVE),
        ),
    )


def build_timeline(durations: RevealDurations, fps: float) -> tuple[RevealFrameState, ...]:
    """Return one eased state per output frame for the complete reveal.

    Frame ``j`` of an ``n``-frame phase sits at ``(j + 1) / n``, so every
    phase ends exactly on its completed state and the next phase starts
    strictly after it.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    frames: list[RevealFrameState] = []
    for phase, count in zip(
        REVEAL_PHASE_ORDER, phase_frame_counts(durations, fps), strict=True
    ):
        frames.extend(
            _state(phase, smoothstep((step + 1) / count)) for step in range(count)
        )
    return tuple(frames)


def frame_index_at(seconds: float, fps: float, frame_count: int) -> int:
    """Return the frame displayed at ``seconds`` from the start."""
    index = int(seconds * fps)
    if not 0 <= index < frame_count:
        raise ValueError(f"{seconds} s lies outside the {frame_count}-frame timeline.")
    return index
