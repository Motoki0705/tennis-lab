"""Typed request/result contracts for the staged scene-reveal animation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SCENE_REVEAL_METADATA_SCHEMA = "scene_reveal_animation_v2"


class RevealPhase(StrEnum):
    """Ordered phases from capture cameras to free-viewpoint court overlays."""

    CAMERAS = "cameras"
    SFM = "sfm"
    SFM_TO_GAUSSIANS = "sfm_to_gaussians"
    COLORIZE = "colorize"
    SPLAT = "splat"
    VIEW_DEPENDENCE = "view_dependence"
    HOLD = "hold"
    ALIGNMENT = "alignment"
    CAMERA_MOVE = "camera_move"


REVEAL_PHASE_ORDER: tuple[RevealPhase, ...] = tuple(RevealPhase)


class RevealMode(StrEnum):
    """Publish the full video or only selected still frames."""

    VIDEO = "video"
    STILLS = "stills"


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _vector3(value: tuple[float, ...], *, name: str) -> tuple[float, float, float]:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    x, y, z = (_finite(item, name=name) for item in items)
    return (x, y, z)


def _rgb(value: tuple[float, ...], *, name: str) -> tuple[float, float, float]:
    rgb = _vector3(value, name=name)
    if any(not 0.0 <= channel <= 1.0 for channel in rgb):
        raise ValueError(f"{name} channels must lie in [0, 1].")
    return rgb


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class RevealDurations:
    """Seconds spent in every phase, in :data:`REVEAL_PHASE_ORDER`."""

    seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        seconds = tuple(
            _positive(value, name=f"durations[{phase.value}]")
            for phase, value in zip(REVEAL_PHASE_ORDER, self.seconds, strict=True)
        )
        object.__setattr__(self, "seconds", seconds)

    @classmethod
    def from_mapping(cls, mapping: dict[str, float]) -> RevealDurations:
        """Build from a mapping keyed by every phase name exactly once."""
        expected = {phase.value for phase in REVEAL_PHASE_ORDER}
        if set(mapping) != expected:
            raise ValueError(
                "Reveal durations must name every phase; "
                f"missing={sorted(expected - set(mapping))}, "
                f"unknown={sorted(set(mapping) - expected)}."
            )
        return cls(tuple(mapping[phase.value] for phase in REVEAL_PHASE_ORDER))

    def of(self, phase: RevealPhase) -> float:
        """Return one phase duration."""
        return self.seconds[REVEAL_PHASE_ORDER.index(phase)]


@dataclass(frozen=True, slots=True)
class RevealViewpoint:
    """Fixed look-at camera expressed in one accepted court's metric frame.

    Court frames are metres with the origin at the net centre on the ground,
    +x across the court, +y along it, and +z up.
    """

    anchor_court_id: str
    eye_court_m: tuple[float, float, float]
    target_court_m: tuple[float, float, float]
    up_court: tuple[float, float, float]
    horizontal_fov_deg: float

    def __post_init__(self) -> None:
        if not self.anchor_court_id:
            raise ValueError("anchor_court_id must be non-empty.")
        eye = _vector3(self.eye_court_m, name="eye_court_m")
        target = _vector3(self.target_court_m, name="target_court_m")
        up = _vector3(self.up_court, name="up_court")
        forward = [t - e for t, e in zip(target, eye, strict=True)]
        if math.hypot(*forward) <= 1.0e-6:
            raise ValueError("eye_court_m and target_court_m must differ.")
        if math.hypot(*up) <= 1.0e-6:
            raise ValueError("up_court must be non-zero.")
        cross = (
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        )
        if math.hypot(*cross) <= 1.0e-6 * math.hypot(*forward) * math.hypot(*up):
            raise ValueError("The viewing direction must not be parallel to up_court.")
        fov = _finite(self.horizontal_fov_deg, name="horizontal_fov_deg")
        if not 1.0 <= fov <= 170.0:
            raise ValueError("horizontal_fov_deg must lie in [1, 170].")
        object.__setattr__(self, "eye_court_m", eye)
        object.__setattr__(self, "target_court_m", target)
        object.__setattr__(self, "up_court", up)
        object.__setattr__(self, "horizontal_fov_deg", fov)

    def moved_to(self, keyframe: RevealKeyframe) -> RevealViewpoint:
        """Return this viewpoint looking from a camera-path keyframe."""
        return RevealViewpoint(
            anchor_court_id=self.anchor_court_id,
            eye_court_m=keyframe.eye_court_m,
            target_court_m=keyframe.target_court_m,
            up_court=self.up_court,
            horizontal_fov_deg=self.horizontal_fov_deg,
        )


@dataclass(frozen=True, slots=True)
class RevealKeyframe:
    """One camera-path waypoint in the viewpoint's anchor court frame (metres)."""

    eye_court_m: tuple[float, float, float]
    target_court_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "eye_court_m", _vector3(self.eye_court_m, name="eye_court_m")
        )
        object.__setattr__(
            self, "target_court_m", _vector3(self.target_court_m, name="target_court_m")
        )


@dataclass(frozen=True, slots=True)
class RevealCameraPath:
    """Waypoints of the free-viewpoint move after the alignment overlay.

    The path starts at :class:`RevealViewpoint` and passes through every
    keyframe; anchor court, up direction and field of view stay fixed.
    """

    keyframes: tuple[RevealKeyframe, ...]

    def __post_init__(self) -> None:
        keyframes = tuple(self.keyframes)
        if not keyframes:
            raise ValueError("camera_path needs at least one keyframe.")
        if any(not isinstance(item, RevealKeyframe) for item in keyframes):
            raise TypeError("camera_path keyframes must be RevealKeyframe values.")
        object.__setattr__(self, "keyframes", keyframes)


@dataclass(frozen=True, slots=True)
class RevealOutput:
    """Encoded video geometry and quality."""

    width: int
    height: int
    fps: float
    crf: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = _positive_int(getattr(self, name), name=name)
            if value % 2:
                raise ValueError(f"{name} must be even for yuv420p encoding.")
        _positive(self.fps, name="fps")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int):
            raise TypeError("crf must be an integer.")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must lie in [0, 51].")


@dataclass(frozen=True, slots=True)
class RevealStyle:
    """Colours and sizes of the procedural layers drawn by tennis-lab."""

    background_rgb: tuple[float, float, float]
    gaussian_point_rgb: tuple[float, float, float]
    sfm_point_radius_px: int
    camera_rgb: tuple[float, float, float]
    camera_frustum_depth_m: float
    camera_frustum_stride: int
    camera_line_width_px: int
    sfm_crop_court_m: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "background_rgb", _rgb(self.background_rgb, name="background_rgb")
        )
        object.__setattr__(
            self,
            "gaussian_point_rgb",
            _rgb(self.gaussian_point_rgb, name="gaussian_point_rgb"),
        )
        object.__setattr__(self, "camera_rgb", _rgb(self.camera_rgb, name="camera_rgb"))
        _positive_int(self.sfm_point_radius_px, name="sfm_point_radius_px")
        _positive(self.camera_frustum_depth_m, name="camera_frustum_depth_m")
        _positive_int(self.camera_frustum_stride, name="camera_frustum_stride")
        _positive_int(self.camera_line_width_px, name="camera_line_width_px")
        crop = tuple(
            _finite(value, name="sfm_crop_court_m") for value in self.sfm_crop_court_m
        )
        if len(crop) != 6 or any(crop[2 * i] >= crop[2 * i + 1] for i in range(3)):
            raise ValueError(
                "sfm_crop_court_m must be (xmin, xmax, ymin, ymax, zmin, zmax) "
                "with min < max on every axis."
            )
        object.__setattr__(self, "sfm_crop_court_m", crop)


@dataclass(frozen=True, slots=True)
class RevealAlignmentStyle:
    """Court-line and court-axes overlay drawn from the accepted alignment."""

    line_rgb: tuple[float, float, float]
    line_width_px: int
    outline_px: int
    axis_length_m: float
    label_font_px: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_rgb", _rgb(self.line_rgb, name="line_rgb"))
        _positive_int(self.line_width_px, name="line_width_px")
        if (
            isinstance(self.outline_px, bool)
            or not isinstance(self.outline_px, int)
            or self.outline_px < 0
        ):
            raise ValueError("outline_px must be a non-negative integer.")
        _positive(self.axis_length_m, name="axis_length_m")
        _positive_int(self.label_font_px, name="label_font_px")


@dataclass(frozen=True, slots=True)
class RevealNHTSettings:
    """Public ``nht-render`` invocation and staged-appearance parameters."""

    executable: str
    point_radius_m: float
    splat_stagger: float
    states_per_invocation: int

    def __post_init__(self) -> None:
        if self.executable != "nht-render" and not Path(self.executable).is_absolute():
            raise ValueError(
                "executable must be exactly nht-render or an absolute path to it."
            )
        _positive(self.point_radius_m, name="point_radius_m")
        stagger = _finite(self.splat_stagger, name="splat_stagger")
        if not 0.0 < stagger <= 1.0:
            raise ValueError("splat_stagger must lie in (0, 1].")
        _positive_int(self.states_per_invocation, name="states_per_invocation")


@dataclass(frozen=True, slots=True)
class SceneRevealRequest:
    """One complete staged reveal of a canonical scene workspace."""

    scene_root: Path
    output_video: Path
    work_directory: Path
    mode: RevealMode
    still_times_seconds: tuple[float, ...]
    output: RevealOutput
    durations: RevealDurations
    viewpoint: RevealViewpoint
    camera_path: RevealCameraPath
    style: RevealStyle
    alignment: RevealAlignmentStyle
    nht: RevealNHTSettings

    def __post_init__(self) -> None:
        # Every waypoint must be a valid look-at with the shared up direction.
        for keyframe in self.camera_path.keyframes:
            self.viewpoint.moved_to(keyframe)
        for name in ("scene_root", "output_video", "work_directory"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path.")
        if self.output_video.suffix != ".mp4":
            raise ValueError("output_video must be an .mp4 path.")
        if not isinstance(self.mode, RevealMode):
            raise TypeError("mode must be a RevealMode.")
        stills = tuple(
            _finite(value, name="still_times_seconds")
            for value in self.still_times_seconds
        )
        total = sum(self.durations.seconds)
        if self.mode is RevealMode.STILLS:
            if not stills:
                raise ValueError("stills mode requires still_times_seconds.")
            if any(not 0.0 <= value < total for value in stills):
                raise ValueError(
                    f"still_times_seconds must lie in [0, {total}) seconds."
                )
        elif stills:
            raise ValueError("still_times_seconds is only valid in stills mode.")
        object.__setattr__(self, "still_times_seconds", stills)
        reconstruction = (self.scene_root / "reconstruction").resolve(strict=False)
        if self.work_directory.resolve(strict=False).is_relative_to(reconstruction):
            raise ValueError(
                "work_directory must stay outside the reconstruction workspace."
            )

    @property
    def metadata_path(self) -> Path:
        """Return the JSON metadata path published next to the video."""
        return self.output_video.with_suffix(".json")

    @property
    def stills_directory(self) -> Path:
        """Return the directory that receives still PNG frames."""
        return self.output_video.with_suffix("")


@dataclass(frozen=True, slots=True)
class SceneRevealResult:
    """Published outputs of one reveal run."""

    video_path: Path | None
    still_paths: tuple[Path, ...]
    metadata_path: Path
    frame_count: int
    nht_state_count: int
