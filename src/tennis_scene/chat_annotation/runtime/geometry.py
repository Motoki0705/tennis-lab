"""Short, explicitly reviewed ball interpolation in presentation time."""

from __future__ import annotations

from fractions import Fraction

from .contracts import Annotation, Ball, ClipManifest


def interpolation_values(
    annotation: Annotation,
    manifest: ClipManifest,
    track_id: str,
    start: int,
    stop: int,
) -> dict[int, list[float]]:
    frames = {frame.frame_index: frame for frame in annotation.frames}
    if (
        not 0 <= start < stop < len(manifest.frames)
        or stop <= start + 1
        or any(index not in frames for index in range(start, stop + 1))
    ):
        raise ValueError(
            "interpolation requires two endpoints enclosing consecutive clip frames"
        )
    selected = [frames[index] for index in range(start, stop + 1)]
    if any(frame.interpolation_break for frame in selected):
        raise ValueError(
            "interpolation cannot cross hits, bounces, cuts or play boundaries"
        )
    if any(not frame.reviewed for frame in selected):
        raise ValueError("inspect every interpolation frame before using the helper")
    balls: list[Ball] = []
    for frame in selected:
        matches = [ball for ball in frame.balls if ball.track_id == track_id]
        if len(matches) != 1:
            raise ValueError(
                "same alive ball must be explicitly present throughout the interval"
            )
        balls.append(matches[0])
    first, last = balls[0], balls[-1]
    if (
        first.status != "visible"
        or last.status != "visible"
        or first.center_px is None
        or last.center_px is None
    ):
        raise ValueError("interpolation endpoints must be directly visible")
    start_pts, stop_pts = (
        manifest.frames[start].clip_pts,
        manifest.frames[stop].clip_pts,
    )
    elapsed = (stop_pts - start_pts) * Fraction(manifest.time_base)
    if elapsed > Fraction(str(manifest.policies.ball_max_gap_seconds)):
        raise ValueError("interpolation gap exceeds configured time limit")
    return {
        index: [
            float(
                a
                + (b - a)
                * Fraction(
                    manifest.frames[index].clip_pts - start_pts, stop_pts - start_pts
                )
            )
            for a, b in zip(first.center_px, last.center_px, strict=True)
        ]
        for index in range(start + 1, stop)
    }


def interpolate_ball(
    annotation: Annotation, manifest: ClipManifest, track_id: str, start: int, stop: int
) -> Annotation:
    values = interpolation_values(annotation, manifest, track_id, start, stop)
    result: Annotation = annotation.model_copy(deep=True)
    for frame in result.frames:
        if frame.frame_index not in values:
            continue
        ball = next(ball for ball in frame.balls if ball.track_id == track_id)
        if ball.center_px is not None or ball.status == "out_of_frame":
            raise ValueError(
                "interpolation must not overwrite a position or fill an out-of-frame ball"
            )
        ball.center_px = values[frame.frame_index]
        ball.status = "interpolated"
        ball.interpolation_frames = [start, stop]
    return result
