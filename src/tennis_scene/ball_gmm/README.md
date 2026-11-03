# Candidate GMM ball reconstruction

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Select `ball_reconstruction.method=candidate_gmm` in the standard pipeline.
The default remains `triangulation`. Only the ball reconstruction component
switches to saved `ball_detections` v2 evidence; side/alignment keep ball points.
This is CPU inference over existing evidence, with no learned covariance model.

`CandidateGMMConfig` and `configs/pipeline.yaml` define all settings. Choose
`estimator=map|mean` and `temporal=none|kalman|rts` inside
`ball_reconstruction.candidate_gmm`. Supply calibrated `covariance_scale`,
`presence_slope` and `presence_bias` explicitly; production never reads labels.
The clip experiment and tuning-only fitting live in
[`knowledge/runs/run-ball3d-candidate-gmm-20261103`](../../../knowledge/runs/run-ball3d-candidate-gmm-20261103/).

## Distribution and masks

For each camera/frame, inspect saved candidates in score order and retain the
first K=2 or 3 whose 3x3 log-probability quadratic has full rank and positive
precision. Joint curvature retains the cross-axis term. The subpixel optimum is
bounded to its native cell. Covariance eigenvalues are explicitly bounded by
`min_variance_cells2` / `max_variance_cells2`; bounded/rejected counts are saved.
Insufficient boundary support, flat maps and saddles are rejected without an
isotropic fallback. Native pixels use `(W-1)/(w-1)` and `(H-1)/(h-1)` for both
means and full covariance. Dense heatmaps are never copied by this conversion.

Component weights are normalized peak scores. The score function is
`sigmoid(presence_slope*logit(max retained score)+presence_bias)`, with score
endpoints clipped to [1e-7,1-1e-7]. No usable peak means presence=0. This is a
calibrated proxy for usable candidate support; the available labels cannot
identify physical ball existence or independently separate occlusion.

The existing [Method A](../../utils/geometry/probabilistic_triangulation/README.md)
enumerates Bernoulli subsets and component products. The mandatory court-metre
prior is configured by `prior_mean_m` (default [0,0,2]) and `prior_std_m`
(default [8,16,6], diagonal covariance). Its influence is part of the method.
The `(K+1)^V` enumeration budget is checked by Method A before solving.
An optimizer failure rejects the entire frame with its reason; it never drops
one failed component or switches solvers. Budget/contract errors raise.

Only the posterior conditional on at least two active cameras supplies a
triangulated point. Its original probability mass must exceed
`min_multiview_mass` and is recorded. Prior-only and single-view terms remain
outside this triangulation result. `map` means the MAP point of the
highest-weight component, **not** the mode of the complete mixture. `mean`
uses mixture moments including between-component variance. MAP covariance is
local to its selected component. The mean's used views are the union of its
positive-weight supporting subsets; MAP uses its selected subset.

The shared constant-velocity Kalman/RTS model uses seconds, metres, explicit
acceleration and initial velocity scales. It resets at every missing frame and
preserves validity and used-view masks. It does not export predicted gap points
or infer event labels; a bounce can still be oversmoothed.

## Outputs and evaluation

`CandidateGMMOutput` keeps the usual `ball` field plus the MAP/mean posterior,
selected covariance, camera support probability, candidate indices/means/full
covariances/weights, curvature diagnostics and whole-frame solver failures.
The component's 2D diagnostic points are nearest retained candidate means to
the exported projection; their distances are diagnostics, not an extra gate.
Existing scene export consumes `ball`; detailed covariance stays in the
component artifact and the common evaluation result.

The evaluation adapter calls this same production algorithm. Ground truth,
its missingness and the held-out view are absent from its inputs. The run fits
only scalar parameters on [0,505), separately using each fold's retained views,
then freezes parameters before primary scoring. The <=20 px localized cohort
fits covariance scale; all scoreable tuning points fit usable-support scores.

The common evaluator reports all-view/LOVO errors and availability coverage.
`score_uncertainty` reuses its scored camera/frame identities and provenance
checks to measure 50/90/95% Gaussian ellipse coverage by split/fold/reference
kind. Pinhole Jacobians propagate posterior covariance only: reference noise,
calibration uncertainty, nonlinear tails and mode-selection errors are omitted.
Nonzero camera time offsets are explicitly unsupported by covariance projection.
Coverage is conditional on scored predictions and differs from availability.
Common rendering adds 95% ellipses to camera/top/side panels only when covariance
exists; portable HTML can embed supplementary uncertainty figures.

The synthetic unit/integration tests under `tests/{unit,integration}/tennis_scene`
and `tests/unit/utils/geometry/test_kalman.py` cover Gaussian curvature, camera
order, missing support, calibration isolation, covariance propagation, smoothing,
public configuration/codec and frozen main output arrays.
