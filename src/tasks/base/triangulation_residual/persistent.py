"""Persistent camera-local false detections with intervals measured in seconds.

An event chooses one, two, or all views (0.75/0.20/0.05). Its interval and
confidence regime are shared, while each affected view draws its own false
trajectory. Both tasks use the configured duration range without a hidden cap.
There is only one person in the source data, so PLCS never invents another
person's pose. Missing source observations are recorded explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.triangulation_residual.configuration import V2Config


class PersistentKind(IntEnum):
    NONE = 0
    SELF_OFFSET = 1
    CONTRALATERAL = 2
    FROZEN_JOINT = 3
    BALL_DECOY = 4
    STUCK_BALL = 5


@dataclass(frozen=True)
class PersistentEvent:
    kind: PersistentKind
    start_frame: int
    stop_frame: int
    start_seconds: float
    duration_seconds: float
    views: tuple[int, ...]
    joints: tuple[int, ...]
    high_confidence: bool
    mean_confidence: float


@dataclass(frozen=True)
class PersistentObservations:
    observations_px: NDArray[np.float64]
    scores: NDArray[np.float64]
    mask: NDArray[np.bool_]
    kind: NDArray[np.uint8]
    source_missing: NDArray[np.bool_]
    events: tuple[PersistentEvent, ...]


def sample_persistent_events(
    frames: int,
    views: int,
    rng: np.random.Generator,
    config: V2Config,
    *,
    fps: float,
    task: str,
    severity: float = 1.0,
    force_event: bool = False,
) -> tuple[PersistentEvent, ...]:
    """Poisson arrivals, bounded holding times, and a shared multi-view interval.

    The arrival intensity is divided by the expected affected-view count so
    ``persistent_rate_per_view_second`` describes each view, rather than
    multiplying the rate again when an event affects several cameras. Events
    may overlap; the later event owns a point where their supports overlap.
    For a clip shorter than a holding time, its duration is explicitly clipped.
    """
    if frames < 1 or views < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Persistent events need frames, views, and positive FPS")
    if task not in {"plcs", "blcs"}:
        raise ValueError("Persistent events support only plcs and blcs")
    if not np.isfinite(severity) or severity < 0:
        raise ValueError("Persistent severity must be finite and nonnegative")
    seconds = frames / fps
    expected_views = 0.75 + 0.20 * min(2, views) + 0.05 * views
    event_count = int(
        rng.poisson(
            config.persistent_rate_per_view_second
            * severity
            * seconds
            * views
            / expected_views
        )
    )
    if force_event:
        event_count = max(1, event_count)
    events = []
    for _ in range(event_count):
        duration = min(
            float(
                rng.uniform(
                    config.persistent_min_seconds, config.persistent_max_seconds
                )
            ),
            seconds,
        )
        start = float(rng.uniform(0, seconds - duration))
        begin = int(np.floor(start * fps))
        end = min(frames, max(begin + 1, int(np.ceil((start + duration) * fps))))
        view_count = int(rng.choice([1, min(2, views), views], p=[0.75, 0.20, 0.05]))
        affected = tuple(
            sorted(int(v) for v in rng.choice(views, view_count, replace=False))
        )
        if task == "plcs":
            kind = PersistentKind(int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])))
            pair = ((9, 10), (7, 8), (15, 16))[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
            joints = (
                pair
                if kind == PersistentKind.CONTRALATERAL
                else (int(rng.choice(pair)),)
            )
        else:
            kind = PersistentKind(int(rng.choice([4, 5], p=[0.8, 0.2])))
            joints = (0,)
        high = bool(rng.random() < config.persistent_high_confidence_probability)
        mean = float(
            np.clip(
                rng.normal(0.78 if high else 0.42, 0.08 if high else 0.10), 0.05, 0.98
            )
        )
        events.append(
            PersistentEvent(
                kind, begin, end, start, duration, affected, joints, high, mean
            )
        )
    return tuple(sorted(events, key=lambda event: event.start_seconds))


def _ar_noise(
    rng: np.random.Generator, frames: int, channels: int, *, fps: float, tau: float
) -> NDArray[np.float64]:
    values = rng.normal(size=(frames, channels))
    rho = np.exp(-1.0 / (fps * tau))
    for frame in range(1, frames):
        values[frame] = rho * values[frame - 1] + np.sqrt(1 - rho**2) * values[frame]
    return np.asarray(values, dtype=np.float64)


def _torso_scale(reference: np.ndarray, visible: np.ndarray) -> float | None:
    needed = (5, 6, 11, 12)
    valid = visible[:, needed].all(axis=1) & np.isfinite(reference[:, needed]).all(
        axis=(1, 2)
    )
    if not valid.any():
        return None
    shoulder = reference[valid][:, (5, 6)].mean(axis=1)
    hip = reference[valid][:, (11, 12)].mean(axis=1)
    lengths = np.linalg.norm(shoulder - hip, axis=-1)
    lengths = lengths[lengths > 1e-6]
    return float(np.median(lengths)) if len(lengths) else None


def _observed_anchor(
    reference: np.ndarray,
    visible: np.ndarray,
    start: int,
    stop: int,
    joint: int,
) -> tuple[NDArray[np.float64] | None, int]:
    """Hold the latest observed source; if absent, wait for its first detection."""
    available = visible[:, joint] & np.isfinite(reference[:, joint]).all(axis=-1)
    past = np.flatnonzero(available[: start + 1])
    if len(past):
        return np.asarray(reference[past[-1], joint], dtype=np.float64), start
    later = np.flatnonzero(available[start:stop])
    if len(later):
        frame = start + int(later[0])
        return np.asarray(reference[frame, joint], dtype=np.float64), frame
    return None, stop


def _false_path(
    event: PersistentEvent,
    view: int,
    joint: int,
    reference: np.ndarray,
    visible: np.ndarray,
    image_size: np.ndarray,
    rng: np.random.Generator,
    *,
    fps: float,
    offset_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    start, stop = event.start_frame, event.stop_frame
    count = stop - start
    path: NDArray[np.float64] = np.full((count, 2), np.nan, dtype=np.float64)
    available: NDArray[np.bool_] = np.zeros(count, dtype=bool)
    missing: NDArray[np.bool_] = np.zeros(count, dtype=bool)
    source = reference[view]
    source_visible = visible[view]
    seconds = np.arange(count) / fps
    if event.kind == PersistentKind.CONTRALATERAL:
        opposite = joint + 1 if joint % 2 else joint - 1
        available = source_visible[start:stop, opposite] & np.isfinite(
            source[start:stop, opposite]
        ).all(axis=-1)
        path[available] = source[start:stop, opposite][available]
        return path, available, ~available
    if event.kind == PersistentKind.SELF_OFFSET:
        torso = _torso_scale(source[start:stop], source_visible[start:stop])
        if torso is None:
            return path, available, np.ones(count, dtype=bool)
        direction, drift_direction = rng.uniform(-np.pi, np.pi, 2)
        offset = (
            np.array([np.cos(direction), np.sin(direction)])
            * rng.uniform(0.25, 1.25)
            * torso
            * offset_scale
        )
        velocity = (
            np.array([np.cos(drift_direction), np.sin(drift_direction)])
            * rng.uniform(0, 0.35)
            * torso
            * offset_scale
        )
        available = source_visible[start:stop, joint] & np.isfinite(
            source[start:stop, joint]
        ).all(axis=-1)
        path[available] = (
            source[start:stop, joint][available]
            + offset
            + seconds[available, None] * velocity
        )
        return path, available, ~available
    anchor, first = _observed_anchor(source, source_visible, start, stop, joint)
    missing[: first - start] = True
    if anchor is None:
        missing[:] = True
        if event.kind == PersistentKind.FROZEN_JOINT:
            return path, available, missing
        # A ball decoy can exist even when no ball has ever been observed. Its
        # explicit image-supported prior is independent of hidden ball GT.
        anchor = rng.uniform(0.15, 0.85, 2) * image_size[view]
        first = start
    if event.kind == PersistentKind.FROZEN_JOINT:
        path[first - start :] = anchor
    else:
        scale = image_size[view, 1] / 1080 * offset_scale
        angle, velocity_angle = rng.uniform(-np.pi, np.pi, 2)
        offset = np.array([np.cos(angle), np.sin(angle)]) * rng.uniform(10, 80) * scale
        velocity = (
            np.array([np.cos(velocity_angle), np.sin(velocity_angle)])
            * rng.uniform(0, 60)
            * scale
        )
        if event.kind == PersistentKind.STUCK_BALL:
            path[first - start :] = anchor + offset
        else:
            velocity_noise = _ar_noise(rng, count, 2, fps=fps, tau=0.25) * 8 * scale
            displacement = np.cumsum(velocity_noise / fps, axis=0)
            path[first - start :] = (
                anchor + offset + seconds[:, None] * velocity + displacement
            )[first - start :]
    available[first - start :] = np.isfinite(path[first - start :]).all(axis=-1)
    return path, available, missing


def apply_persistent_events(
    observations_px: np.ndarray,
    scores: np.ndarray,
    reference_px: np.ndarray,
    reference_visible: np.ndarray,
    image_size: np.ndarray,
    rng: np.random.Generator,
    events: tuple[PersistentEvent, ...],
    *,
    fps: float,
    offset_scale: float = 1.0,
) -> PersistentObservations:
    """Replace selected observations, retaining false points when GT is hidden.

    Moving offsets and swaps require their named source at the current frame.
    Frozen detections and ball decoys retain an observed anchor after the true
    target disappears. A ball with no observed anchor uses an explicit interior
    image prior and marks ``source_missing``. Final false-detection validity
    depends on its own finite/in-image pixels, never target clean visibility.
    """
    obs = np.asarray(observations_px, dtype=np.float64).copy()
    conf = np.asarray(scores, dtype=np.float64).copy()
    reference = np.asarray(reference_px, dtype=np.float64)
    visible = np.asarray(reference_visible, dtype=bool)
    size = np.asarray(image_size)
    if (
        obs.ndim != 4
        or obs.shape[-1] != 2
        or conf.shape != obs.shape[:-1]
        or reference.shape != obs.shape
        or visible.shape != conf.shape
        or size.shape != (len(obs), 2)
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or not np.isfinite(fps)
        or fps <= 0
        or not np.isfinite(offset_scale)
        or offset_scale < 0
    ):
        raise ValueError(
            "Invalid persistent observation shapes, FPS, image size, or scale"
        )
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=2)
    motion_rng, confidence_rng = (np.random.default_rng(int(seed)) for seed in seeds)
    mask = np.zeros(conf.shape, dtype=bool)
    kind = np.zeros(conf.shape, dtype=np.uint8)
    source_missing = np.zeros(conf.shape, dtype=bool)
    for event in events:
        start, stop = event.start_frame, event.stop_frame
        if not 0 <= start < stop <= obs.shape[1]:
            raise ValueError(
                "Persistent event interval is outside the observation window"
            )
        if (
            event.kind == PersistentKind.NONE
            or not 0.05 <= event.mean_confidence <= 0.98
        ):
            raise ValueError(
                "Persistent event must have a false-detection kind and confidence"
            )
        for view in event.views:
            if not 0 <= view < len(obs):
                raise ValueError("Persistent event view is outside the observation rig")
            for joint in event.joints:
                if not 0 <= joint < obs.shape[2]:
                    raise ValueError(
                        "Persistent event joint is outside the observation skeleton"
                    )
                path, available, missing = _false_path(
                    event,
                    view,
                    joint,
                    reference,
                    visible,
                    size,
                    motion_rng,
                    fps=fps,
                    offset_scale=offset_scale,
                )
                inside = (
                    np.isfinite(path).all(axis=-1)
                    & (path >= 0).all(axis=-1)
                    & (path <= size[view]).all(axis=-1)
                )
                valid = available & inside
                confidence = np.clip(
                    event.mean_confidence
                    + 0.035
                    * _ar_noise(confidence_rng, stop - start, 1, fps=fps, tau=0.25)[
                        :, 0
                    ],
                    0.05,
                    0.98,
                )
                # An out-of-image generated path is a missing detection, not a
                # reason to silently restore the original target observation.
                obs[view, start:stop, joint][available] = np.where(
                    valid[available, None], path[available], np.nan
                )
                conf[view, start:stop, joint][available] = np.where(
                    valid[available], confidence[available], 0.0
                )
                mask[view, start:stop, joint][available] = valid[available]
                kind[view, start:stop, joint][available] = np.where(
                    valid[available], int(event.kind), 0
                )
                source_missing[view, start:stop, joint] |= missing
    return PersistentObservations(obs, conf, mask, kind, source_missing, events)


def corrupt_persistent(
    observations_px: np.ndarray,
    scores: np.ndarray,
    reference_px: np.ndarray,
    reference_visible: np.ndarray,
    image_size: np.ndarray,
    rng: np.random.Generator,
    config: V2Config,
    *,
    fps: float,
    task: str,
    severity: float = 1.0,
    force_event: bool = False,
) -> PersistentObservations:
    schedule_seed, path_seed = rng.integers(0, np.iinfo(np.int64).max, size=2)
    events = sample_persistent_events(
        observations_px.shape[1],
        len(observations_px),
        np.random.default_rng(int(schedule_seed)),
        config,
        fps=fps,
        task=task,
        severity=severity,
        force_event=force_event,
    )
    return apply_persistent_events(
        observations_px,
        scores,
        reference_px,
        reference_visible,
        image_size,
        np.random.default_rng(int(path_seed)),
        events,
        fps=fps,
        offset_scale=config.persistent_offset_scale * severity,
    )
