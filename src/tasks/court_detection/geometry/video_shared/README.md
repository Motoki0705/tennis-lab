# Shared court geometry

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

CPU geometry primitives for fixed-camera observations. They accept explicit
model initializers and source RGB measurements; they do not load a detector,
reference homography, manual point, or camera-specific mask. Dataset selection,
scene configuration, artifact publication, and GPU scheduling belong to callers.

- `candidates.read_frozen_candidates` validates a pinned manifest and scope,
  preserves the model acceptance flag, and applies the crop translation once.
  Rejected finite model H values may initialize a fit; null H values are not repaired.
- `observations.measure_frame` proposes projective grids from paired white-stripe
  edges and checks other lines. `endpoints.measure_endings` measures both inside
  and outside finite segments. Unobserved endpoints are not evidence of an ending.
- `fit.fit_common_court` uses equal line/clip/frame weights and explicit finite
  baseline/service evidence. Two transverse lines also require a centre service
  segment to establish service order. `SharedFit.matrix` is absent on failure; the separate
  `fitted_matrix` is diagnostic only. Optimizer convergence alone is insufficient.
  Multi-frame residuals remain diagnostics for clip drift inspection; the 6px
  line p95 gate applies to the reliability of each single-frame local fit.
- `fit.choose_common_solution` requires admissible projections to agree within
  3px on all 14 points, then selects by residual. This is the production policy.
- `consensus.merge_measurements` compares actual stripes bidirectionally and
  merges observations within their original frame/line group. It neither averages
  homography coefficients nor counts repeated initializers as new frames.
- `drift.inspect_drift` fits diagnostic local geometry with correlated line
  uncertainty, using only retained lines. It records each measured line's signed
  median and p95 residual against the common H, flagging excluded lines. It reports
  warning/failure/unobservable without replacing the common
  H. Its 3/10/25px limits scale from the diagonal of 1920×1080; paint measurement
  tolerances in `SharedPaintConfig` and `PaintConfig` are source-image pixels.

Line order and court metres follow `geometry.homography`; pixels are unnormalized
source coordinates. Line identity, spatial support, temporal support, and drift
are separate checks. Successful geometry is not an independent accuracy result.
An unseen line is extrapolation, and a static white stripe may still be a net.

CPU contract tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit/tasks/court_detection/geometry/test_video_shared.py
```
