"""One tiny real-video/fixed-geometry scene, with synthetic model judgments only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.annotation.contracts import LocalVideoInput
from src.tasks.annotation.scene_workflow import SceneAnnotationRequest
from src.tasks.ball_detection.annotation.video import (
    BallTarget,
    FrameIdentity,
    VideoAnnotationInput,
    VideoIdentity,
)
from src.tasks.ball_detection.annotation.video.agent.contracts import AnnotationPolicy
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    SourceSplit,
)
from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.contracts import FrameRange, content_digest
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import ProductPlan
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.source import build_clip_source
from src.utils.checksum import dual_sha256
from tests.support.tasks.local_annotation import local_configuration
from tests.support.tennis_scene.ball_imports import ImportFixture, make_import_fixture


@dataclass
class DirectWorkflowFixture:
    layout: AnnotationLayout
    request: SceneAnnotationRequest
    config: PipelineRuntimeConfig
    fixed: ImportFixture
    hydra_config: DictConfig

    @property
    def store(self) -> Path:
        root: Path = self.config.cache_directory
        return root

    @property
    def decisions(self) -> dict[str, list[dict]]:
        return {
            c: json.loads(p.read_text())["frames"]
            for c, p in self.fixed.annotations.annotations.items()
        }


def direct_fixture(root: Path, *, publish: bool = True) -> DirectWorkflowFixture:
    layout = AnnotationLayout.testing(root / "annotation")
    # Reuse resource/worker setup; its independent VFR media are not scene inputs.
    configuration = local_configuration(layout, review_timeout=5400)
    budget = configuration.campaign.resources.model_copy(
        update={
            "total_pss_bytes": 1536 * 1024**2,
            "additional_storage_bytes": 128 * 1024**2,
        }
    )
    configuration = configuration.model_copy(
        update={
            "campaign": configuration.campaign.model_copy(update={"resources": budget}),
            "runtime": configuration.runtime.model_copy(
                update={
                    "costs": tuple(
                        c.model_copy(
                            update={
                                "pss_bytes": 256 * 1024**2,
                                "timeout_seconds": 5400 if c.stage == "review" else 30,
                            }
                        )
                        for c in configuration.runtime.costs
                    )
                }
            ),
        }
    )
    videos = []
    for view, camera in enumerate(("right", "near", "far")):
        path = layout.repository / f"{camera}.mp4"
        frames = [
            av.VideoFrame.from_ndarray(
                np.full((240, 320, 3), 30 + view * 50 + t * 5, np.uint8), format="rgb24"
            )
            for t in range(4)
        ]
        encode_video(
            path,
            width=320,
            height=240,
            time_base=Fraction(1, 15360),
            rate=Fraction(30),
            frames=zip(frames, (0, 512, 1024, 1536), (512,) * 4, strict=True),
            crf=0,
            preset="ultrafast",
        )
        timeline = probe_video(path)
        videos.append(
            LocalVideoInput(
                path=str(path),
                annotation=VideoAnnotationInput(
                    source=VideoIdentity(
                        recording_id="synthetic-recording",
                        camera_id=camera,
                        file_name=path.name,
                        media_sha256=dual_sha256(path),
                        size_bytes=path.stat().st_size,
                        width=320,
                        height=240,
                        frame_count=4,
                        time_base=str(timeline.time_base),
                        nominal_fps=str(timeline.rate),
                    ),
                    target=BallTarget(
                        target_semantics="active_ball", track_id="active"
                    ),
                    frame_range=FrameRange(start=0, stop=4),
                    frames=tuple(
                        FrameIdentity(frame_index=t, pts=p)
                        for t, p in enumerate(timeline.pts)
                    ),
                ),
            )
        )
    configuration = configuration.model_copy(
        update={"videos": tuple(videos), "publish_dataset": publish}
    )
    source = build_clip_source(
        [Path(v.path) for v in videos],
        [v.annotation.source.camera_id for v in videos],
        clip_id="tiny-scene",
    )
    fixed = make_import_fixture(root, source=source, omit_court_arrays=True)
    policy = write_document(
        layout,
        layout.path("campaigns", "policies", "active-ball.json"),
        BallPublicationPolicy(
            jpeg_quality=90,
            target_semantics="active_ball",
            source_splits=(
                SourceSplit(
                    source_id=content_digest("synthetic-recording"), split="val"
                ),
            ),
        ),
    )
    request = SceneAnnotationRequest(
        configuration=configuration,
        policy=AnnotationPolicy(
            policy_id="adaptive",
            version="scene-test-v1",
            instructions="Explore this camera with temporal context.",
        ),
        publication=ProductPlan(
            product="active-ball",
            dataset_task="ball_detection",
            dataset_version="tiny-active",
            component="frames",
            base_snapshot=None,
            generation_policy=policy,
        )
        if publish
        else None,
    )
    ref = write_document(
        layout, layout.path("campaigns", "requests", "scene.json"), request
    )
    from src.utils.paths import PROJECT_ROOT

    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")
    ):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "device=cpu",
                "gvhmr.enabled=false",
                "person_observations.enabled=false",
                "player_reconstruction.enabled=false",
                f"paths.data_root={root}",
                f"paths.artifact_root={root}",
                f"paths.output_root={root}",
                f"paths.checkpoint_root={root / 'ckpt'}",
                f"paths.external_asset_root={root / 'third_party'}",
                "output_directory=run",
                "cache.directory=stages",
                "ball_detection.method=gpt_annotation",
                "ball_detection.checkpoint=null",
                f"video_paths=[{','.join(Path(v.path).relative_to(root).as_posix() for v in videos)}]",
                f"camera_ids=[{','.join(source.camera_ids)}]",
                f"clip_id={source.clip_id}",
                f"direct_annotation.request={layout.artifact_path(ref).relative_to(root)}",
                f"direct_annotation.request_sha256={ref.digest}",
                f"direct_annotation.bundle={fixed.fixed.bundle_path.relative_to(root)}",
                f"direct_annotation.bundle_sha256={fixed.fixed.bundle_sha256}",
                "direct_annotation.court_detection=verified_calibration_frame0",
                f"direct_annotation.publish_dataset={str(publish).lower()}",
            ],
        )
    return DirectWorkflowFixture(
        layout, request, PipelineRuntimeConfig.from_config(cfg), fixed, cfg
    )


def complete_annotations(fixture: DirectWorkflowFixture):
    """Seed completed model judgments via real revision APIs; no worker execution."""
    from src.tasks.annotation.benchmark.blank import measure_blank_plan
    from src.tasks.annotation.configuration import configure_local_videos
    from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
    from src.tasks.annotation.scene_workflow import (
        CompletedSceneAnnotations,
        SceneAnnotationWorkflow,
    )
    from src.tasks.ball_detection.annotation.video import VideoBallFrame
    from src.tasks.ball_detection.annotation.video.agent.contracts import (
        AnnotationRun,
        EditQuery,
    )
    from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
    from src.tasks.base.annotation.artifacts import read_document

    layout = fixture.layout
    workflow = SceneAnnotationWorkflow(layout, fixture.request)
    registration = configure_local_videos(
        layout, fixture.request.configuration, annotation_policy=fixture.request.policy
    )
    registered = read_document(layout, registration, LocalVideoConfigurationReceipt)
    completions = []
    by_camera = {
        v.annotation.source.camera_id: v.annotation
        for v in fixture.request.configuration.videos
    }
    for entry in registered.registrations:
        from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext

        context = read_document(layout, entry.context, BlankAnnotationContext)
        expected = read_document(layout, context.task_input, VideoAnnotationInput)
        camera = expected.source.camera_id
        run = AnnotationRun(
            context=entry.context,
            plan=registered.plan,
            runtime=registered.runtime,
            policy=fixture.request.policy,
            timeout_seconds=5400,
        )
        ref = write_document(
            layout, layout.artifact_path(entry.context).parent / "video-run.json", run
        )
        store = VideoRevisionStore(layout, ref)
        revision = store.initialize()
        rows = tuple(
            VideoBallFrame.model_validate_json(
                json.dumps(dict(row, pts=by_camera[camera].frames[i].pts))
            )
            for i, row in enumerate(fixture.decisions[camera])
        )
        revision = store.apply(
            EditQuery(
                camera_id=camera,
                expected_revision=revision,
                rows=rows,
                reason="Synthetic model judgments",
            ),
            workflow.reference,
        )
        completions.append(
            store.complete(
                revision, workflow.reference, "All synthetic frames processed"
            )
        )
    write_document(
        layout,
        workflow.directory / "annotations.json",
        CompletedSceneAnnotations(
            request=workflow.reference,
            registration=registration,
            completions=tuple(completions),
            measurements=measure_blank_plan(layout, registered.plan),
        ),
    )
    return workflow
