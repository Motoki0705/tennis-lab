"""Virtual camera control for matplotlib-3D scene rendering.

Single shared 3D-viewpoint API for every scene renderer (tennis_scene, BLCS,
PLCS): named viewpoint presets, per-frame camera motion (static shot, slow
orbit, or keyframed moves with smoothstep easing), and application of a view
to a 3D axis. A :class:`CameraController` maps a frame index to a
:class:`CameraView3D`; renderers apply it via :func:`apply_scene_camera`
after every ``ax.clear()``.

Issue #630 extends this module (not a parallel implementation) with
``look_at`` / ``scene_camera`` view modes derived from
``src.utils.projection.camera_projector`` conventions.

Court coordinates: XY is the ground plane, +Z up, far side at +Y. matplotlib
azimuth -90 therefore looks from the near baseline toward the far side (the
classic broadcast framing).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

CameraMode = Literal["static", "orbit", "keyframes"]

_VALID_MODES: tuple[str, ...] = ("static", "orbit", "keyframes")

# Fixed court framing shared by scene renderers: court plus a small run-off
# margin, with a 4 m ceiling that keeps ball apexes and players in frame.
DEFAULT_VIEW_MARGIN: float = 2.0
DEFAULT_VIEW_Z_LIMIT: float = 4.0


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

        self.keyframes = [] if keyframes is None else list(keyframes)
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
        allowed = {
            "preset",
            "elev",
            "azim",
            "zoom",
            "mode",
            "orbit_period_s",
            "keyframes",
        }
        unknown = set(cfg) - allowed
        missing = allowed - set(cfg)
        if missing or unknown:
            raise ValueError(
                "Invalid camera controller keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )

        base = _view_from_mapping(cfg, context="camera")
        mode = cfg["mode"]
        if type(mode) is not str:
            raise TypeError(
                f"camera.mode must be exactly str, got {type(mode).__name__}"
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown camera mode '{mode}'. Available: {_VALID_MODES}")

        keyframes: list[CameraKeyframe] | None = None
        raw_keyframes = cfg["keyframes"]
        if raw_keyframes is not None:
            if not isinstance(raw_keyframes, Sequence) or isinstance(
                raw_keyframes, str | bytes
            ):
                raise TypeError("camera.keyframes must be a sequence or null.")
            keyframes = []
            for i, raw in enumerate(raw_keyframes):
                if not isinstance(raw, Mapping):
                    raise TypeError(f"keyframes[{i}] must be a mapping.")
                unknown_keyframe = set(raw) - {
                    "frame",
                    "preset",
                    "elev",
                    "azim",
                    "zoom",
                }
                if unknown_keyframe:
                    raise ValueError(
                        f"keyframes[{i}] has unknown keys: {sorted(unknown_keyframe)}"
                    )
                if "frame" not in raw:
                    raise ValueError(f"keyframes[{i}] is missing required key 'frame'")
                frame = raw["frame"]
                if type(frame) is not int:
                    raise TypeError(
                        f"keyframes[{i}].frame must be exactly int, "
                        f"got {type(frame).__name__}"
                    )
                keyframes.append(
                    CameraKeyframe(
                        frame=frame,
                        view=_view_from_mapping(raw, context=f"keyframes[{i}]"),
                    )
                )

        orbit_period_s = cfg["orbit_period_s"]
        if type(orbit_period_s) is not float:
            raise TypeError(
                "camera.orbit_period_s must be exactly float, "
                f"got {type(orbit_period_s).__name__}"
            )
        return cls(
            base,
            mode=cast(CameraMode, mode),
            orbit_period_s=orbit_period_s,
            keyframes=keyframes,
        )


def _optional_mapping_value(raw: Mapping[str, Any], key: str) -> Any:
    """Read one schema-declared optional key without synthesizing a value."""
    try:
        return raw[key]
    except KeyError:
        return None


def _view_from_mapping(raw: Mapping[str, Any], *, context: str) -> CameraView3D:
    """Resolve a view from ``preset`` or explicit ``elev``/``azim`` keys."""
    preset = _optional_mapping_value(raw, "preset")
    elev = _optional_mapping_value(raw, "elev")
    azim = _optional_mapping_value(raw, "azim")
    zoom = _optional_mapping_value(raw, "zoom")
    has_angles = elev is not None or azim is not None
    if preset is not None and has_angles:
        raise ValueError(
            f"{context}: specify either 'preset' or explicit 'elev'/'azim', not both"
        )
    if preset is not None:
        if type(preset) is not str:
            raise TypeError(
                f"{context}.preset must be exactly str, got {type(preset).__name__}"
            )
        view = resolve_camera_view(preset)
    elif elev is not None and azim is not None:
        if type(elev) is not float or type(azim) is not float:
            raise TypeError(f"{context}.elev and {context}.azim must be exactly float.")
        view = CameraView3D(elev=elev, azim=azim, zoom=1.0)
    else:
        raise ValueError(
            f"{context}: requires 'preset' or both 'elev' and 'azim', got {dict(raw)}"
        )
    if zoom is not None:
        if type(zoom) is not float:
            raise TypeError(
                f"{context}.zoom must be exactly float, got {type(zoom).__name__}"
            )
        view = CameraView3D(elev=view.elev, azim=view.azim, zoom=zoom)
    return view


def apply_scene_camera(
    ax: Axes3D,
    view: CameraView3D,
    *,
    margin: float = DEFAULT_VIEW_MARGIN,
    z_limit: float = DEFAULT_VIEW_Z_LIMIT,
) -> None:
    """Apply ``view`` and the fixed court framing to a 3D axis.

    Sets the viewpoint via ``view_init``, pins the axis limits to the court
    extended by ``margin`` on the ground plane and ``z_limit`` vertically, and
    applies real-world box proportions with the view's zoom. Renderers must
    call this after every ``ax.clear()`` so animations keep an explicit,
    frame-stable camera instead of whatever state matplotlib restores.
    """
    x_half_span = float(HALF_DOUBLES_WIDTH + margin)
    y_half_span = float(HALF_LENGTH + margin)

    ax.view_init(elev=view.elev, azim=view.azim)
    ax.set_xlim(-x_half_span, x_half_span)
    ax.set_ylim(-y_half_span, y_half_span)
    ax.set_zlim(0.0, z_limit)
    ax.set_box_aspect(
        (x_half_span * 2.0, y_half_span * 2.0, z_limit),
        zoom=view.zoom,
    )
