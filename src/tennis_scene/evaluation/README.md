# Frozen ball 3D evaluation

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`ball3d/` compares reconstruction methods on the same historical component
artifacts. It performs CPU replay and independent 2D scoring. It does not change
the standard pipeline or execute detector inference.

| Module | Contract |
|---|---|
| `bundle.py` | Extract and validate fixed inputs, original identities, source hashes and complete dependency descriptors |
| `contracts.py` | Selected-view method inputs and masked 3D results |
| `baseline.py` | Replay pinned detector observations through production consensus, LOESS and triangulation; independently check native-map decoding |
| `runner.py` | Run all three views and each of the three two-view LOVO folds; save/load typed, checksummed results |
| `annotations.py` | Load scoring-only native-pixel labels, explicit review states and provenance |
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

Implement `BallMethod.name`, `config` and `__call__(inputs) -> Ball3DResult`, then
call `run_method` or `run_comparison` with bundle/code digests. No annotation
argument exists on this runner. LOVO passes only its two retained `ViewInput`s;
the excluded camera's heatmaps and observations are absent. The fixed calibration
was estimated by the source run and is shared by every method.

Each view contains its aligned camera, local calibration/side, model evidence,
pinned post-threshold/gate detector observations, accepted ball points and their
recorded replay settings. Coordinates are native video pixels. Raw heatmaps retain
their own H/W grid; candidate pixels use the detector's W−1/H−1 convention.
All inputs preserve the complete, zero-based source timeline and read-only arrays.

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

## Scoring labels

`load_annotations` accepts `tennis_lab_ball2d_annotations.v1` JSON with:

- `clip_id`, ordered `camera_ids`, `num_frames`, `fps`, `source_size_wh`,
  `video_sha256` keyed by camera, and `provenance` containing
  `kind: direct_video_review`, `model` and `effort`.
- `observations`: records with `camera_id`, `frame_index`, `status`, optional
  `uv_px`, `reviewed` and `evidence`. States are `observed`, `occluded`,
  `out_of_frame`, `unknown`, `unreviewed`. Reviewed states require evidence;
  only observed points have coordinates. Missing records remain unreviewed.
- Optional `events`: `frame_index`, `kind` (`bounce`/`hit`), `evidence`.

Duplicate records, wrong sources/axes, unreviewed observed points, outside-image
coordinates and detector/test-fixture provenance are rejected. This schema records
review evidence; it cannot establish the truth of a claimed image review by itself.
Unit-test fixtures are not real evaluation labels.

The default split is tuning `[0,505)`, primary evaluation `[505,1010)`, and a
supplementary whole-clip result. Only observed 2D labels are scored. LOVO scores the
excluded view and pools individual residuals across folds; it never averages fold
medians. Outputs include median/p90, fraction strictly above 20 px, observed/scored
counts and coverage, all-view residuals, valid-3D rate, frame gross reprojection
rate, adjacent motion/acceleration, event windows and full-run runtime. Motion
stencils cannot cross missing frames or split boundaries. Default diagnostic
thresholds are 65 m/s, 500 m/s² and event radius 3 frames; these are recorded and
have not been tuned on the evaluation half.

`annotation_oracle` is an explicit scoring-only path: each fold triangulates only
retained-view labels. Approximate calibration and noisy labels mean it is a
geometric reference, not a guaranteed mathematical error lower bound. Missing
labels/events produce unavailable values and coverage, never manufactured zeros.
Independent 3D accuracy is unavailable without independent 3D truth.

## Reproduction and tests

The clip-specific entrypoint, test design and actual evidence live in
[`knowledge/runs/run-ball3d-evaluation-foundation-20261103`](../../../knowledge/runs/run-ball3d-evaluation-foundation-20261103/).
Run `run.py --help` with the repository on `PYTHONPATH`; `bundle` and `compare`
require explicit local inputs and new output directories. Add `--annotations`
only when independently reviewed labels are available. Real assets, full video
decode and replay checks are separate from lightweight CI fixtures.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
