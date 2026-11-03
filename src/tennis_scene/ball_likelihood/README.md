# Dense heatmap likelihood

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This opt-in CPU method estimates court-metre ball positions directly from saved
model heatmaps. The standard pipeline still defaults to `triangulation`.
`ball_reconstruction.method=dense_likelihood` selects the new component. Its
parameters and their defaults live under `ball_reconstruction.dense_likelihood`
in [`pipeline.yaml`](../configs/pipeline.yaml). [`DenseLikelihoodConfig`](config.py)
requires every value explicitly and validates the composed configuration.

The component consumes pre-gate `ball_detections` v2. Court side and camera
alignment keep their original `ball_points` inputs. Dense reconstruction never
uses 2D LOESS, detector observation masks, annotations or annotation events.
The evaluation adapter calls the same `reconstruct_dense` function and uses the
[fixed-input runner and scoring contracts](../evaluation/README.md).

## Likelihood and interpolation

For each frame and view, sigmoid probabilities are clipped symmetrically at
half FP32 epsilon (approximately 5.96e-8) before conversion to finite logits.
Exact ones are counted in diagnostics. Clipping cannot recover lost saturated
curvature. `quadratic_log` instead interpolates log probabilities; temperatures
and support drops then refer to that log domain.

`bicubic_logit` uses a tensor cubic spline over logits. `quadratic_logit` and
`quadratic_log` fit a full quadratic, including the xy cross term, to the actual
3x3 neighbourhood of the nearest grid cell. Image edges use one-sided real
samples. Quadratic fits are local and can have discontinuities at neighbourhood
changes; a failed line search invalidates the result rather than silently
returning its seed. There is no bilinear interpolation in this method.

Let f be the interpolated log-domain map, T the view temperature, and N the
native pixel count. We maximize the sum across views of
`log(epsilon + (1-epsilon) * exp(f/T - logsumexp(grid/T) + log(N)))`.
This is the log likelihood of a signal/uniform mixture relative to uniform.
Out-of-image and behind-camera samples contribute `log(epsilon)` and zero
gradient; flat/low-peak maps provide no positional support. Thus a corrupt view
has a bounded penalty. Sigmoid scores remain uncalibrated model evidence.

`view_temperatures` and `view_backgrounds` are sequences of `[camera_id, value]`
pairs. They are keyed by identity, preserving their meaning in LOVO folds.
Global temperature/background fill unspecified cameras explicitly. At least two
informative in-image views within `support_logit_drop * temperature` of their
native peak and a nondegenerate ray angle are required for a valid point.

## Search and time

`initialization=candidates` triangulates every valid candidate pair, ranks the
resulting seeds by the dense objective, and continuously optimizes a bounded
number of separated starts. Candidate coordinates do not constrain the final
position. `initialization=coarse` instead starts from a bounded court volume,
keeps its best cells and refines their neighbourhoods before continuous
optimization. It does not read candidate coordinates. Neither search guarantees
a global optimum. Court bounds and search budgets are explicit; a coarse volume
larger than 200000 cells is rejected.

`temporal_weight>0` jointly optimizes windows of already supported frames with a
Cauchy 3D acceleration prior in m/s². Input-derived acceleration change points
split windows at impacts; gaps split them too and are never filled. The robust
prior limits undetected impacts, but event detection can oversegment noisy
paths. Windows are disjoint and capped by `window_size`; there is no physical
ballistic model or fitted synchronization. Nonconvergence is a missing result
with a rejection code, never an implicit return to the independent solution.

Dense maps are converted one frame or short window at a time; the entire video
heatmap tensor is never duplicated. Volume points, gradients and interpolator
workspaces are bounded by the configured budgets.

## Results and validation

The shared `DenseTrajectory` contains float32 positions, validity, selected-view
support, log likelihood, event boundaries, convergence, exact-one counts and
rejection codes on the complete source timeline. Invalid coordinates and used
views are zero/false. The standard component persists this diagnostic object
beside the existing ball trajectory. Its 2D coordinates are projections of the
dense mode supported by those maps; they are not original detector peaks.

Synthetic subpixel, corrupt-view, gradient, gap/impact and main-regression tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/ball_likelihood \
  tests/integration/tennis_scene/test_ball3d_dense_likelihood.py
```

The clip-specific tuning, commands, figures and findings belong to
[`knowledge/runs/run-ball3d-dense-likelihood-20261103`](../../../knowledge/runs/run-ball3d-dense-likelihood-20261103/),
not to production defaults. Adoption awaits the human integrated comparison.
