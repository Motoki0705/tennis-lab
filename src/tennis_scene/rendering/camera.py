"""Virtual camera control for 3D tennis scene animations.

Provides named viewpoint presets and per-frame camera motion (static shot,
slow orbit, or keyframed moves with smoothstep easing) for the matplotlib-3D
scene renderer. A :class:`CameraController` maps a frame index to a
:class:`CameraView3D`; the renderer applies it via ``Axes3D.view_init`` and
the ``zoom`` argument of ``Axes3D.set_box_aspect``.

Court coordinates: XY is the ground plane, +Z up, far side at +Y. matplotlib
azimuth -90 therefore looks from the near baseline toward the far side (the
classic broadcast framing).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CameraMode = Literal["static", "orbit", "keyframes"]

_VALID_MODES: tuple[str, ...] = ("static", "orbit", "keyframes")


@dataclass(frozen=True)
class CameraView3D:
    """A single virtual-camera pose for a matplotlib 3D axis.

    Attributes:
        elev: Elevation angle in degrees (``Axes3D.view_init``).
        azim: Azimuth angle in degrees (``Axes3D.view_init``).
        zoom: Zoom factor passed to ``Axes3D.set_box_aspect``; > 1 zooms in.
    """

    elev: float
    azim: float
    zoom: float = 1.0

    def __post_init__(self) -> None:
        if self.zoom <= 0.0:
            raise ValueError(f"zoom must be positive, got {self.zoom}")


CAMERA_PRESETS: dict[str, CameraView3D] = {
    "broadcast": CameraView3D(elev=18.0, azim=-90.0, zoom=1.6),
    "side": CameraView3D(elev=12.0, azim=0.0, zoom=1.3),
    "top": CameraView3D(elev=90.0, azim=-90.0, zoom=1.1),
    "corner": CameraView3D(elev=28.0, azim=-135.0, zoom=1.3),
    "behind_far": CameraView3D(elev=18.0, azim=90.0, zoom=1.6),
}


def resolve_camera_view(view: CameraView3D | str) -> CameraView3D:
    """Return ``view`` itself, or look up a preset by name.

    Raises:
        KeyError: If ``view`` is a string not present in ``CAMERA_PRESETS``.
    """
    if isinstance(view, CameraView3D):
        return view
    if view not in CAMERA_PRESETS:
        raise KeyError(
            f"Unknown camera preset '{view}'. Available: {sorted(CAMERA_PRESETS)}"
        )
    return CAMERA_PRESETS[view]


@dataclass(frozen=True)
class CameraKeyframe:
    """A camera pose pinned to an animation frame index."""

    frame: int
    view: CameraView3D


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _lerp_view(a: CameraView3D, b: CameraView3D, t: float) -> CameraView3D:
    s = _smoothstep(t)
    return CameraView3D(
        elev=a.elev + (b.elev - a.elev) * s,
        azim=a.azim + (b.azim - a.azim) * s,
        zoom=a.zoom + (b.zoom - a.zoom) * s,
    )


class CameraController:
    """Compute the camera pose for each animation frame.

    Modes:
        - ``static``: always the base view.
        - ``orbit``: base view with the azimuth revolving at a constant rate
          (one full turn per ``orbit_period_s`` seconds).
        - ``keyframes``: smoothstep interpolation through an increasing
          sequence of :class:`CameraKeyframe`; clamped outside the range.

    Azimuth interpolation is linear in degrees without wrap-around handling,
    so keyframes spanning more than 180 degrees take the long way round.
    """

    def __init__(
        self,
        base: CameraView3D | str = "broadcast",
        *,
        mode: CameraMode = "static",
        orbit_period_s: float = 24.0,
        keyframes: Sequence[CameraKeyframe] | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown camera mode '{mode}'. Available: {_VALID_MODES}")
        self.base = resolve_camera_view(base)
        self.mode: CameraMode = mode

        if mode == "orbit" and orbit_period_s <= 0.0:
            raise ValueError(f"orbit_period_s must be positive, got {orbit_period_s}")
        self.orbit_period_s = orbit_period_s

        self.keyframes: list[CameraKeyframe] = list(keyframes or [])
        if mode == "keyframes":
            if len(self.keyframes) < 2:
                raise ValueError(
                    "keyframes mode requires at least 2 keyframes, "
                    f"got {len(self.keyframes)}"
                )
            frames = [kf.frame for kf in self.keyframes]
            if any(b <= a for a, b in zip(frames, frames[1:], strict=False)):
                raise ValueError(
                    f"keyframe frames must be strictly increasing, got {frames}"
                )
        elif self.keyframes:
            raise ValueError(
                f"keyframes were provided but mode is '{mode}'; "
                "set mode='keyframes' to use them"
            )

    def view_at(self, frame_idx: int, fps: float) -> CameraView3D:
        """Camera pose for ``frame_idx`` of an animation running at ``fps``."""
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")

        if self.mode == "static":
            return self.base

        if self.mode == "orbit":
            t_seconds = frame_idx / fps
            azim = self.base.azim + 360.0 * t_seconds / self.orbit_period_s
            return CameraView3D(elev=self.base.elev, azim=azim, zoom=self.base.zoom)

        first, last = self.keyframes[0], self.keyframes[-1]
        if frame_idx <= first.frame:
            return first.view
        if frame_idx >= last.frame:
            return last.view
        for kf_a, kf_b in zip(self.keyframes, self.keyframes[1:], strict=False):
            if kf_a.frame <= frame_idx < kf_b.frame:
                t = (frame_idx - kf_a.frame) / (kf_b.frame - kf_a.frame)
                return _lerp_view(kf_a.view, kf_b.view, t)
        raise AssertionError("unreachable: keyframes cover the clamped range")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> CameraController:
        """Build a controller from a plain mapping (e.g. a Hydra config).

        Expected keys:
            - ``preset`` (str) or explicit ``elev``/``azim`` for the base view;
              ``zoom`` optionally overrides the base zoom in both cases.
            - ``mode``: one of ``static`` / ``orbit`` / ``keyframes``.
            - ``orbit_period_s``: seconds per revolution (orbit mode).
            - ``keyframes``: list of mappings, each with ``frame`` plus either
              ``preset`` or ``elev``/``azim`` (and optional ``zoom``).
        """
        base = _view_from_mapping(cfg, context="camera")
        mode = str(cfg.get("mode", "static"))
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown camera mode '{mode}'. Available: {_VALID_MODES}")

        keyframes: list[CameraKeyframe] | None = None
        raw_keyframes = cfg.get("keyframes")
        if raw_keyframes is not None:
            keyframes = []
            for i, raw in enumerate(raw_keyframes):
                if "frame" not in raw:
                    raise ValueError(f"keyframes[{i}] is missing required key 'frame'")
                keyframes.append(
                    CameraKeyframe(
                        frame=int(raw["frame"]),
                        view=_view_from_mapping(raw, context=f"keyframes[{i}]"),
                    )
                )

        orbit_period_s = float(cfg.get("orbit_period_s", 24.0))
        return cls(
            base,
            mode=mode,  # type: ignore[arg-type]  # validated against _VALID_MODES above
            orbit_period_s=orbit_period_s,
            keyframes=keyframes,
        )


def _view_from_mapping(raw: Mapping[str, Any], *, context: str) -> CameraView3D:
    """Resolve a view from ``preset`` or explicit ``elev``/``azim`` keys."""
    preset = raw.get("preset")
    has_angles = raw.get("elev") is not None or raw.get("azim") is not None
    if preset is not None and has_angles:
        raise ValueError(
            f"{context}: specify either 'preset' or explicit 'elev'/'azim', not both"
        )
    if preset is not None:
        view = resolve_camera_view(str(preset))
    elif raw.get("elev") is not None and raw.get("azim") is not None:
        view = CameraView3D(elev=float(raw["elev"]), azim=float(raw["azim"]))
    else:
        raise ValueError(
            f"{context}: requires 'preset' or both 'elev' and 'azim', got {dict(raw)}"
        )
    zoom = raw.get("zoom")
    if zoom is not None:
        view = CameraView3D(elev=view.elev, azim=view.azim, zoom=float(zoom))
    return view
