# Single-camera blank annotation

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`BallVideoWorker` implements `ball_video_annotation / 3 / review` for registered
`BlankAnnotationContext` inputs. It does not use prediction receipts, overlays or
`confirm_predictions`. States and provenance belong to the [v3 contract](../README.md).

## Tools and exploration policy

[`instructions.md`](instructions.md) defines the live-ball target, honest provenance,
unknown minimization and tool protocol. `AnnotationPolicy(policy_id, version,
instructions)` supplies the exploration strategy. Temporal pages, arbitrary anchors,
crop/zoom, inference and corrections use the same tools without a fixed viewing order.

`AnnotationRun` pins context, plan, runtime, policy and tool/worker versions. The plan
fixes model, effort, code and resources; runtime fixes worker/service conditions.
New attempts reuse the run; changed policies or conditions are rejected.

The parent validates camera, original frame/PTS, crop geometry and ownership before
returning plain images. All frame arguments use **original source indices**, including
nonzero starts. Only supplied same-camera frames may be requested or cited as supports.
Image receipts retain native crop, scale and source PTS. The worker uses the
[private input boundary](../../../../base/annotation/workers/README.md#private-input-workers).

## Parent integration and durable progress

Construct `WorkerPool` with `BallVideoWorker(policy)`, the saved `LocalAnnotationPlan`
and runtime's Codex binary/home, review timeout, resource scope/reservations and code
revision. Production still requires the shared ResourceManager. `ServiceBroker`
accepts this adapter's inbox; `Operations` executes its `ReviewOperation` in the CPU
render stage. Blank plans permit camera services only; conditions use `benchmark-stage`.
Bypassing the standalone stdlib client in `input/tools.py` cannot bypass validation.

Every `video_edit` validates the complete v3 result against registered input, writes
immutable payload/command/revision files and commits the head with SQLite CAS before
replying. Identical retries reuse revisions. Saved progress, old payloads and image/tool
provenance remain retained even when the final worker response is interrupted.

```python
from src.tasks.ball_detection.annotation.video.agent import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.usage import is_unprocessed

store = VideoRevisionStore(layout, task.run)  # task: saved AgentTask
revision, completion = store.current()
annotation = store.annotation(revision)
remaining = tuple(row.frame_index for row in annotation.frames if is_unprocessed(row))
```

`video_handoff` pages only unprocessed indices, excluding final unknowns. The next
attempt's `initial_revision` is the saved head. Automatic time-limit continuation is
#1709's responsibility; this module neither schedules attempts nor retries errors.

The final `AgentDecision` names the current revision. Machine validation checks worker
and rollout identity, syntax, input/frame/PTS, state and provenance. Estimated frames
need no independent visual review or human QA. Zero unprocessed rows permits completion
with final unknowns. `WorkOutcome.adoption` points to `VideoCompletion`; its `annotation`
is the exact v3 artifact for scene/dataset consumers. Completion-file, seal and final
receipt interruptions reuse the original completion. Dataset publication is separate.

## Strategy comparison

The [blank benchmark API](../../../../annotation/benchmark/blank.py) reuses benchmark
namespaces and common telemetry usage aggregation.

```python
from src.tasks.annotation.benchmark.blank import (
    compare_blank_conditions,
    register_blank_condition,
)

condition_a = register_blank_condition(layout, source_plan, "overview", policy_a)
condition_b = register_blank_condition(layout, source_plan, "adaptive", policy_b)
# Run each with its saved plan/runtime/policy and benchmark layout.
report = compare_blank_conditions(
    layout, (condition_a, condition_b), reference_annotations=references
)
```

Conditions share images and exact frame identities with independent worker/revision
ledgers. `phase=benchmark` and disabled publication prevent normal dataset/progress
changes. References are parent-only scoring inputs; workers never receive them.

Reports link attempt/thread/response IDs, input/output/cache usage and saved revisions
to processed/unknown/usable counts, image requests, presentations, revisits and native
pixel reference errors. Requests count committed image services; request revisits count
repeated source-frame tiles; response revisits count repeated presented source frames.
Errors use coordinate pairs, with pair counts and camera/source-frame/PTS rows. Missing
usage/reference coordinates remain null; known token subtotals are separate. Telemetry
deduplicates usage. Worker kind distinguishes synthetic/real/mixed/unstarted attempts;
acceptance remains `unverified`.

Tests use short synthetic videos, real file/process boundaries, fake workers and
synthetic usage. GPT accuracy, token savings and real-video acceptance remain in #1715.
