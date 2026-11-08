"""Tiny camera-specific VFR originals for direct registration contracts (CPU only)."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from src.tasks.annotation.configuration import LocalVideoConfiguration, NewBudget
from src.tasks.annotation.contracts import (
    LocalVideoInput,
    StageCost,
    WorkerRuntimeSettings,
)
from src.tasks.ball_detection.annotation.video import (
    BallTarget,
    FrameIdentity,
    VideoAnnotationInput,
    VideoIdentity,
)
from src.tasks.base.annotation.contracts import FrameRange
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    ExecutionPlan,
    ResourceBudget,
    StageCapacity,
    file_digest,
)


def local_configuration(
    layout: AnnotationLayout, *, review_timeout: float = 10, three_cameras: bool = False
) -> LocalVideoConfiguration:
    videos = []
    cameras = (
        (
            ("cam0", 64, 48, 1, 4),
            ("cam2", 80, 32, 2, 4),
        )
        if not three_cameras
        else (
            ("cam0", 64, 48, 0, 12),
            ("cam1", 80, 32, 0, 12),
            ("cam2", 48, 64, 0, 12),
        )
    )
    for camera, width, height, start, stop in cameras:
        path = layout.repository / (camera + ".mp4")
        pts: tuple[int, ...] = (0, 640, 1920, 2560, 4480)
        durations: tuple[int, ...] = (640, 1280, 640, 1920, 640)
        if three_cameras:
            pts = tuple(index * 640 for index in range(12))
            durations = (640,) * 12
        random = np.random.default_rng(17 if camera == "cam0" else 213)
        frames = [
            av.VideoFrame.from_ndarray(
                random.integers(0, 256, (height, width, 3), dtype=np.uint8),
                format="rgb24",
            )
            for _ in pts
        ]
        encode_video(
            path,
            width=width,
            height=height,
            time_base=Fraction(1, 16000),
            rate=Fraction(25),
            frames=zip(frames, pts, durations, strict=True),
            crf=0,
            preset="ultrafast",
        )
        timeline = probe_video(path)
        videos.append(
            LocalVideoInput(
                path=str(path),
                annotation=VideoAnnotationInput(
                    source=VideoIdentity(
                        recording_id="recording-a",
                        camera_id=camera,
                        file_name=path.name,
                        media_sha256=file_digest(path),
                        size_bytes=path.stat().st_size,
                        width=width,
                        height=height,
                        frame_count=len(pts),
                        time_base=str(timeline.time_base),
                        nominal_fps=str(timeline.rate),
                    ),
                    target=BallTarget(
                        target_semantics="active_ball", track_id="play-ball"
                    ),
                    frame_range=FrameRange(start=start, stop=stop),
                    frames=tuple(
                        FrameIdentity(frame_index=index, pts=timeline.pts[index])
                        for index in range(start, stop)
                    ),
                ),
            )
        )
    stages = (
        StageCapacity(stage="review", kind="codex", parallel=1, threads_per_process=1),
        StageCapacity(stage="render", kind="cpu", parallel=1, threads_per_process=1),
    )
    return LocalVideoConfiguration(
        campaign=ExecutionPlan(
            campaign_id="local-fixture",
            phase="production",
            annotation_root=str(layout.root),
            code_commit="a" * 40,
            code_diff_digest="b" * 64,
            agent_model="gpt-6.1-sol",
            agent_effort="max",
            resources=ResourceBudget(
                codex_workers=1,
                additional_storage_bytes=64 * 1024**2,
                minimum_free_bytes=1,
                total_pss_bytes=256 * 1024**2,
                gpu_seconds=None,
                stages=stages,
            ),
        ),
        videos=tuple(videos),
        publish_dataset=False,
        budget=NewBudget(
            scope_id="local-scope",
            roots=(str(layout.repository),),
            previously_allocated_bytes=123,
            previously_used_gpu_seconds=0,
            max_pending_per_stage=4,
        ),
        runtime=WorkerRuntimeSettings(
            codex_binary="codex",
            codex_home=str(layout.path("cache", "codex")),
            costs=tuple(
                StageCost(
                    stage=s.stage,
                    pss_bytes=16 * 1024**2,
                    storage_bytes=1024**2,
                    timeout_seconds=review_timeout if s.stage == "review" else 10,
                    gpu_seconds=0,
                )
                for s in stages
            ),
        ),
    )
