# Direct ball annotation in a scene workflow

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`run_pipeline` defaults to `ball_detection.method=model`. Select `gpt_annotation`
to register original camera videos, annotate without predictions, automatically
continue saved timeout progress, reopen completed v3 annotations, reconstruct with
fixed geometry, and optionally publish an active-ball dataset. No additional
training is submitted by this path. People/body features retain their own settings.

## Pin the request

Generate the input schema with:

```bash
.venv/bin/python -m src.tasks.annotation schema --type scene_annotation
```

`SceneAnnotationRequest` contains:

- `configuration`: the [local video configuration](../../../tasks/annotation/README.md#ローカル動画の白紙入力登録),
  including an explicit execution plan, budget selection, worker/service costs,
  original paths and `VideoAnnotationInput` for every camera, and `publish_dataset`.
- `policy`: an `AnnotationPolicy` with an ID, version and exploration instructions.
- `publication`: a `ProductPlan` when publishing, otherwise `null`. Its policy must
  use `target_semantics=active_ball`, unchanged Q90 JPEGs and explicit recording
  splits or a partition recipe. It may extend a compatible active-ball snapshot.

Use the same recording ID for all cameras and related clips. The scene request
requires exactly one original per camera, starting at source frame 0, covering
the scene frame count. Pin original bytes, native dimensions, exact decoded PTS,
time base and FPS using the local registration contract. An explicit `max_frames`
must match the pinned annotation scope. Budgets and worker costs come from the
operator's measured inputs; no fixture budget or test clock is a production setting.

```python
from src.tasks.annotation.scene_workflow import SceneAnnotationRequest
from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.paths import AnnotationLayout

layout = AnnotationLayout.production()
# local_configuration, exploration_policy and output_product are explicitly
# prepared objects of the types above; output_product is None without publication.
request = SceneAnnotationRequest(configuration=local_configuration,
    policy=exploration_policy, publication=output_product)
reference = write_document(layout,
    layout.path("campaigns", "scene-inputs", "request.json"), request)
```

The request is immutable for the campaign. Registration rejects changed video
identity, time axes, resource scope or publication selection. The policy is
pinned before worker admission and stays the same across attempts.
Registration receipts pin the policy before controller initialization. Recovery
reads that reference and checks the scene request or benchmark condition; missing,
corrupt or conflicting records stop explicitly. A missing controller mirror never
selects a default policy. Only an initial registration with no specified policy
chooses and saves the registration default.

## Run one workflow

All derived paths below are relative to their declared roots: videos to DATA;
request/bundle to ARTIFACT. Use the saved references for both digest arguments.
The `clip_id` must equal the fixed bundle's manifest clip ID.

```bash
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[samples/cam0.mp4,samples/cam1.mp4,samples/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' 'clip_id=meiji_3cam/video_000/clip_000' \
  ball_detection.method=gpt_annotation ball_detection.checkpoint=null \
  direct_annotation.request=chat_annotation/campaigns/scene-inputs/request.json \
  direct_annotation.request_sha256="$REQUEST_SHA256" \
  direct_annotation.bundle=tennis_scene/evaluate/ball3d_comparison/clip000_20261103/fixed_input_bundle_v1/bundle.json \
  direct_annotation.bundle_sha256="$BUNDLE_SHA256" \
  direct_annotation.court_detection=verified_calibration_frame0 \
  direct_annotation.publish_dataset=true
```

The given clip000 bundle omits court arrays and requires the explicit
`verified_calibration_frame0` storage policy. Other bundles default to
`stored_artifact`. Fixed courts/calibration/side/alignment become load-only leaves;
the [import contract](../imports/README.md) verifies input identities and provenance.
No court or ball checkpoint is needed for these imported inputs. GPU features
still follow the repository's training queue rules; for a CPU ball-only run select
`device=cpu person_observations.enabled=false player_reconstruction.enabled=false gvhmr.enabled=false`.

## Receipts and recovery

The component store's `run.json` reports independent `annotation`, `scene`,
`dataset` and `learning` states. Scene metadata retains its normal `ok`/`partial`
meaning. Annotation completion is saved under the campaign's
`scene-workflow/annotations.json`, including per-camera completion and measurement
references. `scene-completion.json` pins the scene artifact and exported archive.
`scene-workflow/publication-request.json` and `publication.json` identify the
optional publisher request and receipt. `dataset=not_requested` is successful;
`learning=not_requested` never starts training.

Ordinary timeouts continue automatically from saved revisions. A no-progress,
quota, corrupt-save or external-service stop requires the [local controller's
recovery](../../../tasks/annotation/README.md#白紙注釈の自動継続).
The integrated annotation CLI's `run`, `tick`, `status`, `recover`, `resume-budget`
and `report` accept the saved `LocalAnnotationRuntime` and its digest.

If only publication failed, rerun the same scene command with
`direct_annotation.resume=publication` and the same request, geometry and store.
This validates completed annotations and the saved scene export, then resumes
the publisher without building scene stages or admitting a worker. Interruption
after publisher commit but before the workflow receipt resolves to the original
snapshot. A new output request never overwrites saved annotations.

## Exploration trials and measurements

Use `blank-register --operation PATH --sha256 VALUE` with a
`BlankConditionRegistration` (source plan, distinct condition ID, versioned policy).
The CLI schemas are `blank_condition` and `blank_comparison`. Each returned
`BlankCondition` pins an isolated runtime. Run that runtime with the usual
`run --runtime PATH --sha256 VALUE --max-seconds N`; it serves only the condition's
camera inputs in its benchmark namespace. Shared images and budget are reused;
progress and revisions remain isolated. Its plan fixes publication off.

`blank-compare --operation PATH --sha256 VALUE` accepts `BlankComparisonRequest`
with condition references and optional parent-only reference annotations. It
returns an immutable measurement report. Production completion likewise saves
`BlankRunMeasurements`: policy/run, attempts, image requests/revisits, coordinate
coverage, response usage and reference error where supplied. Missing token usage
stays null. Synthetic, real, mixed and unstarted workers remain distinguishable;
measurement reports keep `acceptance=unverified`. Comparison runs cannot enter
the scene publication composition or publish a normal dataset automatically.

## Coverage and follow-up acceptance

`annotation-summary.json` contains total and per-camera counts, state breakdowns,
and half-open unknown/unprocessed intervals. `summarize_video_annotations` uses
final states; coordinate-unavailable means processed without usable coordinates,
including occlusion, out-of-frame and no-active-ball. Unprocessed rows are counted
separately. `summarize_reference` counts final `reference_kind`, including adopted
estimates whose original `source_status` was unknown.

`clip000_acceptance(report)` applies the agreed limits only to the full
cam0/cam1/cam2 × frames 0–1009 scope: unavailable ≤81, final unknown ≤25 and
unprocessed =0 across all cameras. Small fixtures return `not_applicable`.
The workflow saves counts without imposing a recognition gate.

The representative CPU E2E uses 3×4 synthetic frames and fixed geometry. Only
model responses and the 5400-second clock boundary are synthetic; registration,
services, revisions, rereading, geometry, scene export, publisher and dataset
reader execute. It verifies estimated contributions and a frame with fewer than
two usable views. Integration covers publication off/on and interruption before
the workflow publication receipt. These results do not measure GPT recognition.

Follow-up #1714 needs old annotations, original video/frame/PTS identities,
recording groups/splits and migration provenance; it must not invent v3 states.
Follow-up #1715 needs all three original videos, a frozen plan/policy/budget,
the clip000 fixed bundle and its reference, plus parent-only ground truth from
`human_annotations/ground_truth_v1/ground_truth.json`. Compare strategy measurements
before selecting the production policy, then retain all 3030 judgments, summary,
new 3D arrays/masks/provenance, snapshot/readback and overlays. Real annotation,
migration, recognition acceptance and model training remain unexecuted here.
