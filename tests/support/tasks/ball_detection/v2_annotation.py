"""Small asymmetric legacy judgments shared by converter/publication contracts."""

from fractions import Fraction
from typing import Any

from src.tasks.ball_detection.annotation.video import VideoAnnotationInput


def legacy_data(expected: VideoAnnotationInput) -> dict[str, Any]:
    source = expected.source
    fps = Fraction(source.nominal_fps)
    kinds = ("observed", "interpolated", "occlusion_estimated", "unresolved", "observed")
    visibility = ("visible", "not_independently_visible", "occluded", "unknown", "visible")
    points = ((9.0, 13.0), (22.5, 7.25), (31.75, 8.0), None, (49.25, 16.5))
    frames = []
    for i, frame in enumerate(expected.frames):
        point = points[i]
        frames.append(dict(
            frame_index=i, pts=frame.pts, track_id=expected.target.track_id, label="tennis_ball",
            status=kinds[i], visibility=visibility[i],
            center_px=None if point is None else dict(x=point[0], y=point[1]),
            center_normalized=None if point is None else dict(x=point[0] / source.width, y=point[1] / source.height),
            support_frames=[0, 2] if i == 1 else [0, 4] if i == 2 else [],
            break_before=i in {0, 4}, notes="legacy note" if i == 2 else "", image_score=123.5 if i == 0 else None,
        ))
    return {
        "schema_version": "video_ball_annotation.v2",
        "source": dict(file_name=source.file_name, sha256=source.media_sha256,
                       width=source.width, height=source.height, frame_count=5,
                       rotation=0, time_base=source.time_base,
                       fps_numerator=fps.numerator, fps_denominator=fps.denominator),
        "target": dict(track_id=expected.target.track_id, label="tennis_ball", scope="main-court rally ball"),
        "coordinate_system": dict(origin="top_left", x_axis="right", y_axis="down", frame_index="zero_based",
                                  normalization={"x": "x_px / width", "y": "y_px / height"}),
        "summary": dict(frame_count=5, all_frames_reviewed=True,
                        counts=dict(observed=2, interpolated=1, occlusion_estimated=1, unresolved=1)),
        "review": {"reviewer": "historical fixture; no new review"},
        "frames": frames,
    }
