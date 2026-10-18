"""Registered, unreviewed predictions on a real tiny VFR clip; no GPU required."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np

from src.tasks.ball_detection.annotation.contracts import (
    InferenceConfig,
    MDDInferenceConfig,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    FrameMap,
    FrameRange,
    SourceIdentity,
    SourceProvenance,
    content_digest,
)
from src.tasks.base.annotation.jpeg_media import write_jpeg_archive
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.predictions import Predictions
from src.tasks.player_detection.annotation.contracts import DinoInferenceConfig


@dataclass
class Fixture:
    layout: AnnotationLayout
    plan: CampaignPlan
    manifest: ClipManifest
    reference: ArtifactRef
    service: Predictions


def example(
    tmp_path: Path,
    count: int = 65,
    *,
    task: Literal["ball", "player"] = "ball",
    jpeg: bool = False,
    inference_config: InferenceConfig | None = None,
) -> Fixture:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    inputs = layout.path("campaigns", "inputs")
    placeholder = write_asset(layout, inputs / "placeholder", b"fixture")
    inference = write_document(
        layout,
        inputs / "inference.json",
        (
            inference_config or MDDInferenceConfig(
                stride=16,
                device="cpu",
                precision="float32",
                decode="opencv_jpeg_rgb24_native" if jpeg else "pyav_rgb24_native",
            )
            if task == "ball"
            else DinoInferenceConfig(
                confidence_threshold=0.3, short_side=32, max_long_side=64
            )
        ),
    )
    ref = placeholder.model_dump()
    plan = CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": "fixture",
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 1,
                    "videos_per_channel": 1,
                    "max_clips_per_video": 1,
                },
                "selection": ref,
                "preparation": ref,
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "max",
                "resources": {
                    "codex_workers": 1,
                    "additional_storage_bytes": 30 * 1024**3,
                    "minimum_free_bytes": 1024**2,
                    "total_pss_bytes": 22 * 1024**3,
                    "gpu_seconds": 43200,
                    "stages": [
                        {
                            "stage": "render",
                            "kind": "cpu",
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
                    ],
                },
                "tasks": [
                    {
                        "task": task,
                        "model": {
                            "task": task,
                            "version": "v1",
                            "training_weights": ref,
                            "inference_weights": ref,
                            "configuration": ref,
                        },
                        "inference": inference.model_dump(),
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": task,
                                "dataset_task": task + "_detection",
                                "dataset_version": "v1",
                                "component": "frames",
                                "base_snapshot": None,
                                "generation_policy": ref,
                            }
                        ],
                    }
                ],
            }
        )
    )
    pts = [0]
    durations = [100 if index % 2 == 0 else 200 for index in range(count)]
    for duration in durations[:-1]:
        pts.append(pts[-1] + duration)
    video = inputs / "clip.mp4"
    encode_video(
        video,
        width=64,
        height=32,
        time_base=Fraction(1, 1000),
        rate=Fraction(10),
        frames=(
            (
                av.VideoFrame.from_ndarray(
                    np.full((32, 64, 3), index, np.uint8), format="rgb24"
                ),
                pts[index],
                durations[index],
            )
            for index in range(count)
        ),
        crf=25,
        preset="ultrafast",
    )
    media = annotation_reference(layout, video)
    timeline = probe_video(video)
    source = SourceIdentity(
        provider="local",
        channel_id="fixture",
        video_id="fixture",
        media=media,
        origin_url="local:fixture",
    )
    jpeg_storage = None
    if jpeg:
        archive = inputs / "frames.bin"
        jpeg_storage = write_jpeg_archive(archive, video, timeline, 0, count)
        media = annotation_reference(layout, archive)
        video.unlink()
    manifest = ClipManifest(
        schema_version="annotation.clip.v2" if jpeg else "annotation.clip.v1",
        source=SourceProvenance.from_source(source) if jpeg else source,
        media=media,
        jpeg=jpeg_storage,
        preparation_digest="e" * 64,
        width=64,
        height=32,
        time_base=str(timeline.time_base),
        nominal_fps=str(timeline.rate),
        source_range=FrameRange(start=0, stop=count),
        target_range=FrameRange(start=1, stop=count - 1),
        frames=tuple(
            FrameMap(
                frame_index=index,
                source_frame_index=index,
                source_pts=pts,
                clip_pts=pts,
                duration_pts=duration,
            )
            for index, (pts, duration) in enumerate(
                zip(timeline.pts, timeline.durations, strict=True)
            )
        ),
    )
    reference = write_document(layout, inputs / "manifest.json", manifest)
    service = Predictions(layout, plan)
    # The collection/preparation path is tested independently; register its exact ledger output here.
    with service.registry.transaction() as database:
        database.execute(
            "INSERT INTO sources (id,provider,channel_id,video_id,campaign_id,candidate,state) VALUES (?,?,?,?,?,?,?)",
            (
                source.canonical_id,
                "local",
                "fixture",
                "fixture",
                plan.campaign_id,
                "{}",
                "prepared",
            ),
        )
        database.execute(
            "INSERT INTO clips VALUES (?,?,?,?,?)",
            (
                manifest.clip_id,
                source.canonical_id,
                plan.campaign_id,
                reference.canonical_bytes().decode(),
                media.digest,
            ),
        )
        database.execute(
            "INSERT INTO annotation_tasks VALUES (?,?,?,?,?,?,?)",
            (
                content_digest([task, manifest.clip_id]),
                plan.campaign_id,
                task,
                manifest.clip_id,
                plan.tasks[0].model.sha256,
                inference.digest,
                "pending",
            ),
        )
    return Fixture(layout, plan, manifest, reference, service)
