# Ball inference and native heatmap decoding

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`predictor.py` owns checkpoint inference; `trajectory_gate.py` accepts one temporal
observation stream. `heatmap_decoding.py` optionally re-decodes the selected native
sigmoid maps on CPU. It never executes a model or uses annotations.

The scene pipeline selects `ball_detection.decoding`. Its defaults belong to
[pipeline.yaml](../../../tennis_scene/configs/pipeline.yaml); the Python config
requires every field explicitly. The comparison script composes this same YAML
and applies only the explicit trial differences. The default scene path keeps the predictor's
arrays exactly; its `subpixel_refine` option retains the existing behaviour.
Other methods require `ball_detection.subpixel_refine=true`.

| method | Local position estimator |
|---|---|
| `legacy` | Existing axis-wise three-point log parabola |
| `axis_logit` | Axis-wise three-point quadratic in logit probability |
| `quadratic_log` / `quadratic_logit` | Least-squares quadratic surface, including XY cross term, on a 3×3 or 5×5 patch |
| `centroid` | Centroid weighted by probability minus the patch minimum |
| `softargmax` | Spatial softmax of log background-subtracted probability divided by temperature; T=1 equals centroid |
| `gaussian` | Nonlinear 5×5 fit of background plus anisotropic Gaussian; positive precision via a Cholesky factor |

```bash
# Run GPU inference through the shared training queue, as for the default pipeline.
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[cam0.mp4,cam1.mp4,cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' \
  ball_detection.decoding.method=quadratic_log \
  ball_detection.decoding.patch_size=5
```

Coordinates use x/(W−1), y/(H−1); peak scores, dense maps and top-K evidence retain
their original meaning. Candidates retain the predictor's original decoder, while
the selected single observation can use another decoder. Threshold, trajectory gate,
temporal consensus, LOESS and triangulation continue to operate on that observation.
The immutable component identity records the full decoding configuration.

Only observed cells enter a truncated border patch. Quadratic fits require rank six,
a negative definite Hessian and a stationary point within one cell of the lattice
peak. Gaussian centres have the same one-cell bound and at most 100 function
evaluations. A rejected fit keeps the lattice peak and returns a `FitStatus` reason:
degenerate, non-concave, outside-local-cell, optimizer-failed or insufficient-patch.
Axis fits keep the border/degenerate axis and mark partial refinement. Scene logs
status counts; replay results retain each camera/frame status. These decisions are
explicit; no different estimator is silently substituted.

`coordinate_correction=stride` applies the half-pixel stride-centre hypothesis
`((u*(N−1)+0.5)*s−0.5)/(N*s−1)` per axis. It is an optional hypothesis about the
convolution grid, not a verified claim about the learned model's bias.
`affine` applies `u*scale_xy+offset_xy`; callers must explicitly supply constants.
Corrections are clipped to [0,1]. Label-based fitting belongs to evaluation tuning,
outside the decoder. Raw labels are never a decoder argument.

Float32 sigmoid saturation has already lost information. Logit fitting clips
probabilities to [1e−7,1−1e−7] and cannot reconstruct lost logits. Border centroids
are biased by truncation; the nonlinear fit assumes one Gaussian plus constant
background. Neither fit guarantees improvement on a learned, multi-modal map.
CPU replay copies at most 64 maps at a time; no full-video RGB buffer is used.

The shared evaluation adapter is
`src.tennis_scene.evaluation.ball3d.heatmap_decoding.HeatmapDecodingMethod`.
It reuses the [fixed-input runner and scorer](../../../tennis_scene/evaluation/README.md)
and production downstream components. Clip-specific comparison scripts and evidence
live in `knowledge/runs/run-i1559-heatmap-decoding-20261103/`.
