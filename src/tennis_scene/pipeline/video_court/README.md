# Video-shared court artifacts

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`court_kp.mode=clip` remains the default. `video_shared` prepares one court for
each explicitly identified dataset/video/camera. Model H candidates initialize
RGB white-stripe fitting; finite line endings and the centre service segment
check semantic correspondence. Equivalent multi-frame solutions are selected
by line residual, without averaging H coefficients. Numeric primitives and
their units are documented in [shared geometry](../../../tasks/court_detection/geometry/video_shared/README.md).

The current build input is a pinned manifest of **frozen model candidates**.
Build does not capture new neural predictions or switch to a model when inputs
are missing. This allows CPU preparation while a caller owns model execution.

Choose acquisition explicitly under `court_kp.video_shared`:

|Field|Build|Load|
|---|---|---|
|`action`|`build`|`load`|
|`directory`|ARTIFACT-root-relative video store|Same store|
|`dataset_id`, `video_id`|Required scope|Required scope|
|`request`, `request_sha256`|Pinned ARTIFACT-root-relative build request|`null`|
|`artifact_digest`|`null`|Digest returned by preparation|
|`rebuild`|Explicitly recompute; permits repair of corrupt objects|`false`|

For example, add these overrides to the ordinary single-clip command:

```bash
court_kp.mode=video_shared \
court_kp.video_shared.action=load \
court_kp.video_shared.directory=tennis_scene/court_shared/match \
court_kp.video_shared.dataset_id=match \
court_kp.video_shared.video_id=video_000 \
court_kp.video_shared.artifact_digest="$ARTIFACT_DIGEST" \
clip_id=video_000/clip_000
```

`run.json` records the chosen bundle and camera digests. A load does not require a
Court checkpoint, run a detector, build another artifact, or repair corruption.
Both acquisition modes use camera-content digests in the clip node identity and
bundle camera references in dataset publication identity. Changing one camera
invalidates its court and dependent nodes; independent person/ball roots remain
reusable with the same implementation. The existing global Python fingerprint
still invalidates all component caches when source code changes.

The build request has exactly these fields:

```json
{
  "schema": "video_court_build_v1",
  "profile": "temporal_rgb_candidate_consensus_v1",
  "dataset_directory": "tennis_multiview/processed/match/dataset",
  "video_id": "video_000",
  "candidate_manifests": {},
  "seed_clip_ids": null,
  "fit_clip_ids": null,
  "bootstrap_clip_ids": null,
  "fit_numerators": [1, 2, 3],
  "diagnostic_numerators": [0, 2, 5],
  "denominator": 5,
  "workers": 1
}
```

Fill `candidate_manifests` with **every** camera's `{path, sha256}` reference,
relative to the ARTIFACT root. Null clip selections mean all clips in this video;
explicit selections reject empty, repeated, foreign, or unknown IDs. Sampling
uses `floor(numerator * (N-1) / denominator)` and rejects duplicate source frames.
Null bootstrap selection uses the first, middle, and last available fit clips.

Candidate files use `court_seed_manifest_v1`, with dataset/video/camera scope,
`size_wh`, `provenance`, and `candidates`. Each candidate records `candidate_id`,
`clip_id`, `frame_index`, crop-pixel `matrix` (or null), `model_accepted`,
`region_xyxy`, and `source_sha256`. Provenance must include
`source_frames["<clip_id>/<frame_index>"] = {media_sha256, rgb_sha256}`. Preparation
checks this against the actual video and decoded RGB. Finite rejected model H
values may be initializers, but failed solves are never fabricated.

`prepare_video_court(config, resolver, code_identity=...)` is the shared API.
Pass its artifact to `standard_definition(..., video_court=artifact)` for each
`ClipExecution` in one `ExecutionPlan`. The orchestrator prepares and binds it for
single-clip and existing dataset entry points. The future batch CLI owns its
connection to this API; this module does not implement that CLI or model lifetime.

An unavailable common H remains a failure. A coherent shift above 3px is a
warning; reliable measured displacement above mean 10px or maximum 25px fails
the camera/clip. Limits scale from 1920×1080. Set
`court_kp.video_shared.unobservable_policy` to `warn` (default) or `fail`.
`warn` uses the shared H when drift is unobservable or observed lines conflict,
preserving the reason and frame as a warning; `fail` retains strict rejection.
Neither policy changes a reliable tolerance failure. Camera motion partway
through a clip may go undetected by the automatic check.
There is no time segmentation or local-H substitution. Shared court observations
use schema v3, with derived frame-zero coordinates and explicit video provenance;
calibration does not relabel them as a direct frame-zero observation.

Fixed-geometry import, direct annotation, and the old ball3d court-observation
reader reject shared provenance. They are not extended by this mode.

## Separate drift evidence for an immutable court

`drift_build.build_video_drift(base, sources, frame_plan=..., directory=...,
provenance=..., config=DriftConfig(), workers=...)` measures explicitly selected
frames on CPU. `frame_plan[camera][clip]` lists distinct source-frame indices for
**every** camera and clip in the base bundle. Callers choose and record exclusions
before evaluation. The API checks media hashes, source membership and timelines
before measurement, checks inputs again afterward, and publishes into a separate
store. It never calls the common-H estimator or rewrites the original objects.

The [signed-line diagnostic](../../../tasks/court_detection/geometry/video_drift/README.md)
uses fixed line correspondences, four motion parameters and worst-frame clip
aggregation. Load its result together with the original court using both:

```bash
court_kp.video_shared.drift_directory=tennis_scene/court_drift/match \
court_kp.video_shared.drift_digest="$DRIFT_DIGEST"
```

These options require `action=load`. Omitting both explicitly retains the
historical diagnostics embedded in the original v1 artifact. Specifying only
one, a missing/corrupt object, or a diagnostic for another base H is an error;
there is no fallback to embedded diagnostics. `prepare_video_court()` on the
orchestrator returns the bound, diagnosed artifact. Multi-clip plans must pass
that same artifact and matching runtime configuration.

Clip cache identity contains the original camera digest **and** the separate
diagnostic camera digest. Dataset markers include diagnostic camera references.
The receipt and court observation record both digests and every frame verdict.
Changing a camera's diagnostic invalidates that court and its descendants while
the immutable geometry digest stays unchanged. The original build format and
its embedded v1 diagnostic remain readable for historical reproduction.

The execution policy is also part of court-node and publication identities.
Changing it recalculates court consumers while both immutable bundles remain
unchanged. `policy.apply_drift_policy` preserves each frame's
`measurement_status`, values and reasons, then aggregates the effective verdict.
Receipts and gallery views expose the policy and affected frames. This does not
certify that an unobservable camera stayed still. The default follows the human
approval of the shared H documented in [run 78](../../../../knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md).
