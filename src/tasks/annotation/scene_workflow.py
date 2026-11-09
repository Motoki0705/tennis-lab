"""Compose saved local-camera annotation and optional publication, without learning."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    read_annotation,
)
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AnnotationPolicy,
    AnnotationRun,
    VideoCompletion,
)
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    SavedVideoAnnotation,
    VideoPublishRequest,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, Document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import ProductPlan

from .annotation_policy import read_registered_annotation_policy
from .configuration import LocalVideoConfiguration, configure_local_videos
from .contracts import LocalAnnotationRuntime
from .local_controller import LocalAnnotationController
from .local_videos import LocalVideoConfigurationReceipt

LOGGER = logging.getLogger(__name__)


class SceneAnnotationRequest(Document):
    configuration: LocalVideoConfiguration
    policy: AnnotationPolicy
    publication: ProductPlan | None


class CompletedSceneAnnotations(Document):
    request: ArtifactRef
    registration: ArtifactRef
    completions: tuple[ArtifactRef, ...]
    measurements: ArtifactRef


class SceneAnnotationWorkflow:
    def __init__(
        self, layout: AnnotationLayout, request: SceneAnnotationRequest
    ) -> None:
        self.layout, self.request = layout, request
        cfg = request.configuration
        if layout.review_namespace is not None or cfg.campaign.phase != "production":
            raise ValueError(
                "Scene publication requires a production annotation run; comparison runs stay isolated"
            )
        if cfg.publish_dataset != (request.publication is not None):
            raise ValueError("Publication selection and explicit output must agree")
        self.directory = layout.path(
            "campaigns", cfg.campaign.campaign_id, "scene-workflow"
        )
        # Pin policy and output selection before registration or worker admission.
        self.reference = write_document(
            layout, self.directory / "request.json", request
        )

    def saved(self) -> CompletedSceneAnnotations:
        completed: CompletedSceneAnnotations = read_document(
            self.layout,
            annotation_reference(self.layout, self.directory / "annotations.json"),
            CompletedSceneAnnotations,
        )
        if completed.request != self.reference:
            raise ValueError("Completed annotations belong to another workflow request")
        receipt = read_document(
            self.layout, completed.registration, LocalVideoConfigurationReceipt
        )
        runtime = read_document(self.layout, receipt.runtime, LocalAnnotationRuntime)
        if (
            read_registered_annotation_policy(self.layout, runtime)
            != self.request.policy
        ):
            raise ValueError(
                "Saved annotations differ from the registered annotation policy"
            )
        if len(completed.completions) != len(self.request.configuration.videos):
            raise ValueError("Completed annotation camera set is incomplete")
        by_input = {v.annotation: v for v in self.request.configuration.videos}
        seen = set()
        for ref in completed.completions:
            completion = read_document(self.layout, ref, VideoCompletion)
            run = read_document(self.layout, completion.run, AnnotationRun)
            if (
                completion.context not in {e.context for e in receipt.registrations}
                or run.plan != receipt.plan
                or run.runtime != receipt.runtime
                or run.policy != self.request.policy
            ):
                raise ValueError(
                    "Saved completion differs from the pinned annotation run"
                )
            store = VideoRevisionStore(self.layout, completion.run)
            store.validate_completion(ref, completion.revision)
            if store.current() != (completion.revision, ref):
                raise ValueError(
                    "Saved completion differs from the completed revision head"
                )
            annotation = read_annotation(
                verify_reference(self.layout, completion.annotation),
                require_complete=True,
            )
            if (
                annotation.input_identity not in by_input
                or annotation.input_identity in seen
            ):
                raise ValueError(
                    "Completed annotations differ from registered camera inputs"
                )
            seen.add(annotation.input_identity)
        if (
            seen != set(by_input)
            or read_document(self.layout, receipt.request, LocalVideoConfiguration)
            != self.request.configuration
        ):
            raise ValueError("Completed annotations differ from registered inputs")
        verify_reference(self.layout, completed.measurements)
        return completed

    def annotate(
        self,
        *,
        controller_factory: Callable[
            ..., LocalAnnotationController
        ] = LocalAnnotationController,
        wait: Callable[[LocalAnnotationController], None] | None = None,
    ) -> CompletedSceneAnnotations:
        if (self.directory / "annotations.json").exists():
            return self.saved()
        registration = configure_local_videos(
            self.layout,
            self.request.configuration,
            annotation_policy=self.request.policy,
        )
        receipt = read_document(
            self.layout, registration, LocalVideoConfigurationReceipt
        )
        runtime = read_document(self.layout, receipt.runtime, LocalAnnotationRuntime)
        controller = controller_factory(
            self.layout, runtime, annotation_policy=self.request.policy
        )
        last, printed = "", 0.0
        while True:
            progress = controller.cycle()
            summary = json.dumps(
                {
                    "state": progress["state"],
                    "cameras": [
                        {
                            key: row.get(key)
                            for key in ("camera_id", "state", "summary", "reason")
                        }
                        for row in progress["reviews"]
                    ],
                },
                sort_keys=True,
            )
            if summary != last or time.monotonic() - printed >= 30:
                LOGGER.info("Direct annotation: %s", summary)
                last, printed = summary, time.monotonic()
            if progress["state"] == "complete":
                break
            if progress["state"] == "blocked" and not progress["live"]:
                raise RuntimeError(f"Local annotation needs recovery: {progress}")
            if wait is None:
                time.sleep(0.2)
            else:
                wait(controller)
        from .benchmark.blank import measure_blank_plan

        completions = tuple(
            ArtifactRef.model_validate(row["completion"]) for row in progress["reviews"]
        )
        completed = CompletedSceneAnnotations(
            request=self.reference,
            registration=registration,
            completions=completions,
            measurements=measure_blank_plan(self.layout, receipt.plan),
        )
        write_document(self.layout, self.directory / "annotations.json", completed)
        return self.saved()

    def inputs(
        self, completed: CompletedSceneAnnotations
    ) -> tuple[dict[str, Path], dict[str, VideoAnnotationInput]]:
        # Reopen the formal completion/annotation receipts; never consume worker memory.
        if completed != self.saved():
            raise ValueError("Annotations differ from the saved workflow completion")
        paths, expected = {}, {}
        for video in self.request.configuration.videos:
            expected[video.annotation.source.camera_id] = video.annotation
        for ref in completed.completions:
            completion = read_document(self.layout, ref, VideoCompletion)
            annotation = read_annotation(
                verify_reference(self.layout, completion.annotation),
                require_complete=True,
            )
            paths[annotation.source.camera_id] = verify_reference(
                self.layout, completion.annotation
            )
        return paths, expected

    def publish(self) -> ArtifactRef | None:
        completed = self.saved()
        output = self.request.publication
        if output is None:
            return None
        registration = read_document(
            self.layout, completed.registration, LocalVideoConfigurationReceipt
        )
        contexts = {e.context for e in registration.registrations}
        saved = []
        for ref in completed.completions:
            completion = read_document(self.layout, ref, VideoCompletion)
            if completion.context not in contexts:
                raise ValueError("Publication completion differs from its registration")
            saved.append(
                SavedVideoAnnotation(
                    context=completion.context, annotation=completion.annotation
                )
            )
        request = VideoPublishRequest(
            plan=registration.plan,
            output=output,
            annotations=tuple(sorted(saved, key=lambda s: s.context.digest)),
        )
        write_document(
            self.layout, self.directory / "publication-request.json", request
        )
        # The publisher owns transactional recovery and duplicate prevention,
        # including interruption after publishing but before this receipt exists.
        result = BallPublisher(self.layout).publish(request)
        write_document(self.layout, self.directory / "publication.json", result)
        return result
