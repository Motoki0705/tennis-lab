# Court/ball bundle adjustment

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Set `ball_reconstruction.calibration=court_ball_ba` in the standard pipeline.
The default `fixed` path retains its original component, schema and numerical
output. This optional stage refines only the ball path; player reconstruction
continues to use the original alignment. The ball artifact uses schema v2 and
the final scene metadata records its separate cameras and clock offsets.

`bundle_adjustment.py::refine_court_ball` jointly optimizes seven parameters per
camera (one common focal multiplier, rotation, camera centre), relative clocks,
and a latent 3D position per correspondence. Principal point, aspect ratio and
zero distortion stay fixed, preserving the undistorted `PinholeCamera` contract.
Metric CourtKP14 fixes the spatial gauge. The lexically first **selected** camera
has zero offset; camera frame t observes world time t/fps + offset_seconds.
Every LOVO fold fits independently and has its own selected-camera clock anchor.
No calibration or clock is inferred for an excluded camera.

The shared raw court contract and historical diagnostic reader live in `court.py`;
evaluation imports the same definitions. Fit uses valid source-homography-selected
keypoints. All other valid raw keypoints remain unused by this optimization.
The final source homography already maps the original template; PROSAC
`x_reflected` is provenance, not an instruction to reflect labels again. Apply
only the decided court half-turn. Four noncollinear selected points provide at
least eight residuals for seven camera parameters; a numerical rank-seven check
also rejects degenerate camera geometry. Raw LINE coordinates are not required.
The evaluation holdout is conditional on the historical detector/selection and
initial calibration, rather than an independent human court reference.

Ball correspondences come from the standard consensus/LOESS/triangulation
inliers. Each fit correspondence requires inliers in both neighbours in each
used view. Support is fixed before optimization, so changing a clock cannot
remove residuals. At most `max_ball_frames` are selected uniformly over the full
clip; connected moving support is mandatory. Fractional times use piecewise
linear interpolation of the adjacent accepted 2D measurements. Final output
uses the original triangulation and speed gate after resampling at t-offset.
Neither fitting nor output resampling extrapolates or bridges missing points.
Interpolation can reduce coverage; fitted 3D latent points never fill output gaps.
The ball artifact and `SceneResult.ball_uv/ball_vis` keep the original source-frame
2D pixels and masks. Resampled pixels/visibility belong only to the 3D solver and
explicit `resampled_uv_px/resampled_visibility` diagnostics.

Residuals use soft-L1 loss after court/ball pixel scaling. Scores select the
historical inliers but are not treated as calibrated probabilities in the BA.
The dimensionless prior weight scales camera priors with standard deviations
0.05 in log focal, 0.02 radians in rotation, 0.5 metres in centre and 0.5 frames
in clock offset. Camera bounds are four prior units; clocks are bounded by
`max_offset_frames < 1`. Pixel scales, prior weight and CPU budgets are explicit
under `ball_reconstruction.bundle_adjustment`. Solver failure, geometric
degeneracy and insufficient support raise; there is no substitute result.

The Hydra pipeline composition owns all six defaults. `BundleAdjustmentConfig`
and the evaluation adapter require explicit settings; replay composes the same
YAML and uses `PipelineRuntimeConfig.from_config` before constructing the method.

`CalibrationRefinement` saves camera order, offsets in seconds, source fit frames
and solver/support/boundary diagnostics. The evaluation adapter returns these
through the existing `Ball3DResult` fields and shared typed runner. No ground
truth, reference masks, video paths or excluded-view evidence enter the solver.
Clip-specific metrics, raw court holdout overlays and reproduction belong to
`knowledge/runs/`, not this module.
