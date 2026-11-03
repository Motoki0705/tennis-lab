# Frozen ball 3D evaluation

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`ball3d/` compares reconstruction methods on the same historical component
artifacts. It performs CPU replay and user-reference 2D scoring. It does not change
the standard pipeline or execute detector inference.

| Module | Contract |
|---|---|
| `bundle.py` | Extract and validate fixed inputs, original identities, source hashes and complete dependency descriptors |
| `contracts.py` | Selected-view method inputs and masked 3D results |
| `baseline.py` | Replay pinned detector observations through production consensus, LOESS and triangulation; independently check native-map decoding |
| `runner.py` | Run all three views and each of the three two-view LOVO folds; save/load typed, checksummed results |
| `annotations.py` | Adapt strict user-reference selections into scoring-only arrays |
| `scoring.py` | Split-aware reprojection, coverage, motion/event diagnostics and the annotation-only geometric reference |
| [Comparison visualization](../visualization/ball3d/) | N-method RGB overlays, top/side trajectories, figures, crops and portable HTML |

## Fixed inputs and provenance

`create_bundle` copies the source `scene.json`, the selected component descriptors
and their arrays, plus all transitive dependency descriptors for audit. It preserves
original paths, identities, code provenance and checksums. It never relabels a
historical store as a current pipeline run. The original store remains untouched.
Unselected dependency arrays are unnecessary for replay and are not copied.

The loader validates the source document, descriptor lineage, file inventory,
array checksums/types, camera/frame axes, evidence window provenance, court side,
calibration and observation masks. Annotation/import origins and undeclared files
are rejected. Pin the externally recorded manifest digest through
`load_bundle(path, expected_sha256=...)` when exchanging a bundle. Bind relocated
videos explicitly by camera ID with `verify_videos`; historical video paths are
provenance, not automatic lookup locations.

`FixedInputBundle` belongs to the evaluation side. Its `baseline_reference` is a
saved method prediction used only for replay verification. Pass only its `inputs`
to methods. Neither the reference, annotations, annotation missingness nor store
paths are present in `MethodInput`.

## Method interface

The optional [candidate GMM method](../ball_gmm/README.md) uses production Method A.
`ball3d/uncertainty.py` propagates its covariance and measures ellipse coverage on
the same scored camera/frame cohort; nonzero time offsets require a temporal posterior and are rejected.

Implement `BallMethod.name`, `config`, explicit bool `requires_court_lines`, and `__call__(inputs) -> Ball3DResult`, then
call `run_method` or `run_comparison` with bundle/code digests. No annotation
argument exists on this runner. LOVO passes only its two retained `ViewInput`s;
the excluded camera's heatmaps and observations are absent. The fixed calibration
was estimated by the source run and is shared by every method.

Each view contains its aligned camera, local calibration/side, model evidence,
pinned post-threshold/gate detector observations, accepted ball points and their
recorded replay settings. Coordinates are native video pixels. Raw heatmaps retain
their own H/W grid; candidate pixels use the detector's W−1/H−1 convention.
All inputs preserve the complete, zero-based source timeline and read-only arrays.

`ViewInput.court` retains camera-local raw KP14 pixels, detector scores/validity,
source homography selection and template orientation on actual measured frames
(frame 0 here). Fitted/broadcast court geometry is not a raw observation. Camera,
frame, native-size and field mismatches are errors; LOVO retains only the selected
views' court objects. The original bundle already contains these raw KP records
in validated calibration diagnostics, so its bytes and digest remain unchanged.

Raw LINE observations have a typed representation, but the historical store saved
only LINE support summaries. `court.lines is None` carries an explicit
`line_missing_reason`; summaries are never converted into measurements.
The runner checks every selected view before calling a method that declares
`requires_court_lines=True`. Direct callers can use `court.require_lines()` or
`load_bundle(..., require_court_lines=True)`, which fails when evidence is absent.
The baseline uses fixed cameras and does not require LINE. Recovering original
LINE measurements requires additional source assets; this evaluation does not
rerun inference or synthesize LINE measurements.

Results contain court-metre `positions (T,3)`, `valid (T)` and
`used_views (V,T)` in explicit `camera_ids` order. Invalid coordinates are zero;
valid points require at least two used views. Optional fields are PSD covariance
`(T,3,3)`, changed cameras, per-camera offsets in seconds and diagnostics.
Changed calibration must keep the original court coordinate gauge. Camera frame
`t` observes world time `t/fps + offset`. Projection may interpolate adjacent valid
3D samples for a fractional offset; it never bridges gaps or extrapolates.
Held-out cameras keep the original calibration and zero offset.

The baseline reuses pinned decoded/gated observations and recomputes production
consensus, LOESS and triangulation. `verify_baseline` separately re-decodes saved
native maps and compares source points, positions and masks. Its receipt states
numerical tolerances and bitwise equality separately. `without_loess` retains the
same consensus mask and raw accepted detector coordinates, isolating LOESS rather
than changing which observations are accepted.

## Scoring reference

`annotations.py` is a thin adapter over the strict [user-reference reader](../ground_truth/README.md).
`load_annotations(path, bundle, expected_sha256=...)` builds `ReferenceIdentity` from
verified bundle inputs and video hashes, calls `load_user_reference`, and selects
explicit camera/frame axes through `UserReference.select`. There is no second JSON
schema or provenance validator. Source CSVs must remain beside the reference JSON.

Only `valid_for_scoring` controls scoring. Approved `estimated` points are included
and retain their kind; they are never renamed `observed`. Missing/excluded coordinates
remain NaN and are never interpolated. Every fold and segment reports `all`, `observed`
and `estimated` distributions, eligible/scored counts and coverage. Original camera
specific event rows, notes and estimation segments remain in the scoring report.

`score_comparison(runs, bundle, annotations)` and `error_samples(run, bundle,
annotations)` require the verified `FixedInputBundle`, not a free `MethodInput`.
They check every result's bundle digest and camera/frame axes, and bind reference
identity (clip ID, cameras, frame count, FPS, native size and video hashes) to the
same bundle. `annotation_oracle`, `plot_comparison` and `render_extremes` also take
the bundle. Mixing equal-sized results from another bundle is an error.

The default split is tuning `[0,505)`, primary evaluation `[505,1010)`, and a
supplementary whole-clip result. LOVO scores only the excluded camera, pooling
individual residuals across folds rather than averaging medians. Metrics include
median/p90, fraction strictly above 20 px, all-view errors, valid-3D rate, gross
reprojection frame rate, adjacent motion/acceleration and their outlier rates,
event-window errors and full-run CPU time. Gross frame rate means any scored
reprojection above 20 px divided by frames with a scored projection; independent
3D truth is unavailable. Motion stencils never cross missing frames or split boundaries.
Diagnostic defaults are 65 m/s, 500 m/s² and event radius 3 frames, recorded without
tuning on the evaluation half.

Event metrics use the union of the original camera-row windows for each kind.
A residual contributes once, including when two rows describe the same event
with a one-frame difference. `source_row_count` is not a physical event count;
`physical_event_count` is deliberately unavailable. Source notes remain intact.

`annotation_oracle` is scoring only: each fold triangulates the scoreable labels
of its retained cameras, including approved estimates, with the same fixed cameras.
It never calls a prediction method. Approximate calibration and noisy references
make it a geometric reference, not a guaranteed mathematical lower bound.

The main HTML and standalone metric table show LOVO scored/eligible counts and
coverage beside LOVO errors. All-view 3D validity has its own heading. Their shared
table generator prevents different values in the two media. A second table reports
LOVO errors on the intersection of scored (held-out camera, source frame) pairs
across all listed methods, including the oracle when supplied. Both tables explain
that missing predictions are excluded and that common-sample results remain
conditional on availability. The cohort identities and observed/estimated
breakdowns are saved; empty intersections have unavailable errors, never zero.

Visualization supports N methods. Source overlays use green crosses for observed
references and white diamonds for approved estimates; predictions use coloured rings.
Crops rank held-out errors and also show LOVO crosses. Trajectories never connect
across missing frames. The HTML embeds media, figures, tables and metrics for portable
viewing; sibling downloads are optional. Shared video readers/frame selection are
reused. `MultiviewSceneRenderer` requires SMPL/SceneResult, so the ball-only comparison
composes its own top/side panels without inventing player meshes.

## Reproduction and tests

The clip-specific entrypoint, test design and actual evidence live in
[`knowledge/runs/run-ball3d-foundation-review-20261103`](../../../knowledge/runs/run-ball3d-foundation-review-20261103/).
Run `run.py --help` with the repository on `PYTHONPATH`; `bundle` and `compare`
require explicit local inputs and new output directories. The compare command requires `--bundle-sha256`, `--annotations` and
`--annotations-sha256` from the trusted input record, plus `--checkpoints` for the historical checkpoint audit. Review reproduction additionally requires
`--prior-comparison` to verify that predicted arrays remain unchanged from v1. Real assets, full video
decode and replay checks are separate from lightweight CI fixtures.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
