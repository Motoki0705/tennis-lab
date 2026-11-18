# Video-shared court side

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

With fixed cameras, `court_side.mode=video_shared` estimates one set of half-turns for a video.
It requires `court_kp.mode=video_shared` and inherits that court's dataset and video IDs. The default is `court_side.mode=clip`.
Saved configurations containing just the six court-side thresholds retain their clip semantics. In the current format,
`mode` and `video_shared` are a required pair.

Each clip uses the ordinary sampled frame grid, pixel scaling, static-detection filter and ball-only hypothesis scores.
Source frames are identified by clip ID and local frame index. For each assignment the builder weights cost and support by the
number of distinct multi-view frames in each clip, adds the camera-pair counts, then applies the existing thresholds to the video.
All clips contribute by default. The video artifact references a separate immutable diagnostic object with each clip's
scored frame indices, ranked hypotheses, pairs, margin and standalone stop reason. A clip with zero scored frames has an empty ranking and a null margin.

## Use from a single clip

Supply the usual [shared court overrides](../video_court/README.md) and the target `clip_id`.
To load an existing decision, add the following overrides to `src.tennis_scene.scripts.run_pipeline`.

```bash
court_side.mode=video_shared \
court_side.video_shared.action=load \
court_side.video_shared.directory=tennis_scene/court_side_shared/match \
court_side.video_shared.artifact_digest="$SIDE_DIGEST"
```

For a build, set `action=build`, `directory`, `request` and `request_sha256`. Both paths are relative to the ARTIFACT root.
The request lists the stores with completed `court_calibration` and `ball_points/<camera>` for **every clip** in the shared court.
Use `PinnedSideInputs.from_store(directory).digest` for the revision pin. It covers the source and active input references, so
publishing a downstream scene in the same store leaves the pin valid. A one-clip-video request has this shape.

```json
{
  "schema": "video_court_side_inputs_v1",
  "clips": {
    "video_000/clip_000": {
      "role": "data",
      "directory": "match/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene",
      "inputs_digest": "<revision digest>"
    }
  },
  "evidence_clip_ids": null
}
```

`role` accepts `data` and `artifact`. An explicit nonempty list of contributing clip IDs limits the pooled evidence.
The builder still measures every clip. Identical builds reuse the pinned result. `rebuild=true` explicitly recomputes and
allows repair of corrupt objects. Load reads the two content-addressed objects and verifies their hashes.

## Clip outcomes and multi-clip callers

Build and load also check the pinned court drift revision used for calibration.
The clip node checks its sampled ball observations, calibrated camera matrices and reference against the frozen inputs.
A decided standalone result that disagrees with the video fails with `court_side_video_shared_conflict`.
An undecided clip uses the video result and records a warning. `court_side.video_shared.unobservable_policy=fail`
also rejects undecided clips. An undecided video fails for every clip under either policy. `run.json` records both
rankings, margins, the effective verdict and the two object digests. The side node and dataset marker include the
video artifact digest and execution policy in their identities.

For the preliminary plan, use the same shared court and set `court_side.mode=clip`. Run only `court_calibration` and
`ball_points/<camera>` for every clip. When the inputs are ready, the batch caller can pass the
mapping of clip IDs to `PinnedSideInputs` to `build_video_court_side`, or use the configured build/load API
`prepare_video_court_side`. Pass the returned artifact as `video_court_side` alongside `video_court` to
`standard_definition` for each clip. A changed media source, geometry, ball stream or numerical setting requires
matching evidence. The receipt for a prepared clip is available through `artifact.clip_report(clip_id, policy)`.

Reusing a build or binding a prepared build to a plan verifies the request hash, evidence selection, current revisions and array checksums of every calibration and ball_points input, including diagnostic-only clips. An explicit video load verifies the decision and diagnostics at the requested digest.

Both actions apply the current `court_kp.video_shared.unobservable_policy` to every camera in every contributing clip. A rejected observation stops adoption and identifies the clip, camera and drift reason. Pass this setting as `court_policy` to the direct APIs (default `warn`). Each excluded clip must also pass its own court policy when it runs or publishes.

Automatic clip cache hits and explicit component loads require the shared component's output with matching dependencies, shared digest and both execution policies. The shared side rejects imports and component overrides. Creating or reusing a dataset completion marker verifies the clip verdict and the scene export's dependency on that component.
