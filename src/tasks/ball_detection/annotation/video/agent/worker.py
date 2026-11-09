"""Task adapter for a private single-camera GPT worker, without prediction review."""

from __future__ import annotations

from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    canonical_bytes,
)
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, references
from src.tasks.base.annotation.storage.references import distinct
from src.tasks.base.annotation.telemetry.capture import verify_report
from src.tasks.base.annotation.telemetry.contracts import RenderedImage, ToolEvidence
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)

from ..contracts import VideoBallFrame
from .contracts import (
    AgentDecision,
    AgentTask,
    AgentView,
    AnnotationPolicy,
    AnnotationRun,
    ServiceEvent,
)
from .services import image_catalog, resolve_attempt
from .store import VideoRevisionStore, progress_summary


class BallVideoWorker:
    name = "ball_video_annotation"
    version = "3"

    def __init__(self, policy: AnnotationPolicy | None = None) -> None:
        self.policy = (
            None
            if policy is None
            else AnnotationPolicy.model_validate_json(policy.canonical_bytes())
        )

    def input_references(
        self, layout: AnnotationLayout, plan: LocalAnnotationPlan, request: WorkRequest
    ) -> tuple[ArtifactRef, ...]:
        from src.tasks.annotation.contracts import LocalVideoManifest
        from src.tasks.annotation.reviews import read_local_context
        from src.tasks.base.annotation.workers.isolation import require_private_paths

        fixed_inputs = read_document(layout, plan.input_manifest, LocalVideoManifest)
        require_private_paths(
            (layout.repository, *(Path(video.path) for video in fixed_inputs.videos))
        )

        context = read_document(layout, request.payload, BlankAnnotationContext)
        manifest = read_document(layout, context.manifest, ClipManifest)
        if (
            (request.adapter, request.adapter_version, request.stage)
            != (self.name, self.version, "review")
            or read_document(layout, context.plan, LocalAnnotationPlan) != plan
            or manifest.clip_id != request.subject_id
            or request.inputs
            != (context.manifest, context.task_input, context.source_media)
        ):
            raise ValueError("blank worker differs from its fixed camera/plan/input")
        read_local_context(layout, request.payload, context.plan)
        result: tuple[ArtifactRef, ...] = distinct(
            layout, (*references(context), manifest.media)
        )
        return result

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: LocalAnnotationPlan,
        request: WorkRequest,
        directory: Path,
        *,
        timeout_seconds: float,
    ) -> PreparedWorker:
        from src.tasks.annotation.annotation_policy import (
            read_registered_annotation_policy,
        )
        from src.tasks.annotation.contracts import LocalAnnotationRuntime

        self.input_references(layout, plan, request)
        context = read_document(layout, request.payload, BlankAnnotationContext)
        runtime_ref = annotation_reference(
            layout,
            layout.path("campaigns", plan.campaign_id, "configuration", "runtime.json"),
        )
        runtime = read_document(layout, runtime_ref, LocalAnnotationRuntime)
        policy = read_registered_annotation_policy(layout, runtime)
        if self.policy is not None and self.policy != policy:
            raise ValueError(
                "worker policy differs from its registered annotation policy"
            )
        if (
            runtime.plan != context.plan
            or runtime.cost("review").timeout_seconds != timeout_seconds
        ):
            raise ValueError("worker timeout/conditions differ from the frozen runtime")
        run = AnnotationRun(
            context=request.payload,
            plan=context.plan,
            runtime=runtime_ref,
            policy=policy,
            timeout_seconds=timeout_seconds,
        )
        run_ref = write_document(
            layout, layout.artifact_path(request.payload).parent / "video-run.json", run
        )
        store = VideoRevisionStore(layout, run_ref)
        revision = store.initialize()
        body = AgentTask(work=request, run=run_ref, initial_revision=revision)
        task_ref = write_document(layout, directory / "request.json", body)
        (directory / "input").mkdir(exist_ok=True)
        (directory / "exchange").mkdir(exist_ok=True)
        (directory / "work").mkdir(exist_ok=True)
        marker = write_asset(
            layout, directory / "work/.camera-input-root", b"private camera workspace\n"
        )
        safe = write_asset(
            layout,
            directory / "input/tool-input.json",
            canonical_bytes(
                {
                    "task_request": task_ref.model_dump(mode="json"),
                    "timeout_seconds": timeout_seconds,
                    "input": store.expected.model_dump(mode="json"),
                    "initial_revision": revision.model_dump(mode="json"),
                    "summary": progress_summary(store.annotation(revision)),
                    "policy": policy.model_dump(mode="json"),
                }
            )
            + b"\n",
        )
        bridge = write_asset(
            layout,
            directory / "input/tools.py",
            (Path(__file__).parent / "scripts/client.py").read_bytes(),
        )
        schema = write_asset(
            layout,
            directory / "input/frame-schema.json",
            canonical_bytes(VideoBallFrame.model_json_schema()) + b"\n",
        )
        instructions = Path(__file__).with_name("instructions.md").read_text()
        prompt = (
            instructions
            + "\n\nFixed input and saved progress:\n"
            + verify_reference(layout, safe).read_text()
            + f"\nSupervisor attempt limit: {timeout_seconds:g} seconds. Save edits and the result before this deadline.\n"
        )
        return PreparedWorker(
            prompt=write_asset(layout, directory / "prompt.md", prompt.encode()),
            inputs=(task_ref, run_ref, safe, bridge, schema, revision, marker),
            isolate_inputs=True,
        )

    def telemetry_images(
        self,
        layout: AnnotationLayout,
        plan: LocalAnnotationPlan,
        request: WorkRequest,
        directory: Path,
    ) -> tuple[RenderedImage, ...]:
        attempt = resolve_attempt(
            layout, annotation_reference(layout, directory / "request.json")
        )
        if (
            attempt.task.work != request
            or read_document(layout, attempt.launch.plan, LocalAnnotationPlan) != plan
        ):
            raise ValueError("image telemetry belongs to another camera/plan")
        result: tuple[RenderedImage, ...] = image_catalog(layout, attempt)
        return result

    def classify_tool(self, name: str | None, raw_input: str) -> str:
        if name and "view_image" in name:
            return "video_image_presentation"
        for operation in ("video_handoff", "video_rows", "video_image", "video_edit"):
            if "tools.py" in raw_input and operation in raw_input:
                return operation
        return "video_inspection"

    def tool_evidence(
        self, layout: AnnotationLayout, launch: ArtifactRef, category: str, text: str
    ) -> ToolEvidence | None:
        from src.tasks.base.annotation.telemetry.tools import output_envelopes

        directory = layout.artifact_path(launch).parent
        results: list[ArtifactRef] = []
        dependencies: list[ArtifactRef] = []
        for value in output_envelopes(text, ("video_service", "result")):
            digest = value.get("video_service")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                continue
            event_ref = annotation_reference(
                layout, directory / "services" / digest / "event.json"
            )
            event = read_document(layout, event_ref, ServiceEvent)
            attempt = resolve_attempt(layout, event.task)
            if (
                attempt.actor != launch
                or ArtifactRef.model_validate(value.get("result")) != event.result
            ):
                raise ValueError(
                    "tool output differs from its committed camera service"
                )
            view = read_document(layout, event.result, AgentView)
            results.extend((event_ref, event.result))
            dependencies.extend(
                (event.task, event.request, view.revision, *references(view))
            )
        if results:
            return ToolEvidence(
                artifacts=distinct(layout, results),
                dependencies=distinct(layout, dependencies),
            )
        return None

    def validate(
        self,
        layout: AnnotationLayout,
        plan: LocalAnnotationPlan,
        request: WorkRequest,
        directory: Path,
        result: ArtifactRef,
        execution: ExitReceipt,
        telemetry: ArtifactRef | None,
    ) -> WorkOutcome:
        attempt = resolve_attempt(
            layout, annotation_reference(layout, directory / "request.json")
        )
        self.input_references(layout, plan, request)
        if (
            attempt.task.work != request
            or execution.status != "succeeded"
            or execution.exit_code != 0
            or execution.launch_digest != attempt.actor.digest
        ):
            raise ValueError(
                "blank completion requires its successful matching attempt"
            )
        if telemetry is None:
            if attempt.launch.test_command is None:
                raise ValueError(
                    "real worker completion requires actual rollout telemetry"
                )
        else:
            report = verify_report(layout, telemetry)
            if (
                report.launch != attempt.actor
                or read_document(layout, report.execution, ExitReceipt) != execution
            ):
                raise ValueError(
                    "blank completion telemetry belongs to another attempt"
                )
        decision = read_document(layout, result, AgentDecision)
        store = attempt.store
        head, sealed = store.current()
        if decision.revision != head or not decision.reason.strip():
            raise ValueError(
                "worker result must name the current saved revision with a reason"
            )
        if decision.status == "deferred":
            if sealed is not None:
                raise ValueError("a complete annotation cannot be deferred")
            return WorkOutcome(
                status="deferred", receipt=result, reason=decision.reason
            )
        adoption = store.complete(head, attempt.actor, decision.reason)
        return WorkOutcome(status="complete", receipt=result, adoption=adoption)
