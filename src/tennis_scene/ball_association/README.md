# Top-K ball association

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`topk.py` associates saved, pre-gate candidates in court metres without RGB,
detector inference, dense-map sampling, annotations or learned parameters.
Select `ball_reconstruction.method=topk_multihypothesis` in the standard pipeline.
The default remains `triangulation`. Settings live in `ball_reconstruction.topk`;
`TopKConfig` validates them. Pixel thresholds use the source diagonal relative to
1920×1080, motion uses metres/seconds, and missing limits use source frames.

Every valid candidate up to K≤8 is eligible, including zero-score candidates.
Zero is floored at 1e-8 only for numerical log evaluation, never used as a gate.
All camera-pair candidate combinations are triangulated by unit-weight DLT.
Each pair also proposes an extension with the best reprojection/score candidate
in each other view. Pair-only hypotheses retain explicit omitted views. The
joint refit rejects selected-camera depth failures, <1° ray separation at the
seed pair, reprojection above the configured limit, height outside [-0.2,20]m,
or absolute court X/Y above 40m. Unique candidate-index tuples are ranked by
mean squared reprojection / sigma² plus mean negative log score and omitted-view
cost. Retain at most `max_hypotheses` per frame. These are pair-seeded proposals,
not an exhaustive Cartesian search across every camera.

The temporal beam minimizes cumulative emission, speed and acceleration costs.
Speed above `max_speed_mps` forbids a link. Acceleration is the difference of
two velocities divided by their mean time interval; its squared cost saturates
at `turn_cost_cap`, so bounces/hits have finite cost without using event labels.
The beam keeps the cheapest history for each (last, penultimate) hypothesis pair,
then at most `beam_width` pairs. It approximates second-order Viterbi; pruning
can remove the globally optimal path. Ties use stable candidate and cost order.

Null states pay `missing_frame_cost`. Up to `max_missing_frames` retain motion
history; actual source-frame spacing determines velocity/acceleration across
these gaps. Exceeding the limit discards history and charges `restart_cost`.
Reacquisition is explicit. Gaps and endpoints are never interpolated into 3D.
Fewer than `ball_reconstruction.min_frames` supported outputs invalidate the
whole trajectory, matching the standard minimum-support policy.

`AssociationResult` records selected source candidate indices (-1 unused), raw
pixels, geometric residuals, hypothesis count, emission cost, exp(-cost) score,
and temporal restart flags. Scores are uncalibrated relative weights, not ball
existence probabilities. Invalid 3D/pixels/residuals are zero. Every valid 3D
point uses at least two selected candidates. No post-selection 2D LOESS is applied.
The optional pipeline component saves this typed result beside the unchanged
ball payload; the evaluation adapter uses exactly the same function.

CPU cost is bounded by pair proposals, hypothesis cap and beam width. Only small
candidate arrays are copied; native heatmaps/patches are shared read-only. Source
videos are streamed by the [foundation renderer](../evaluation/README.md), whose
optional candidate display shows grey ranks and coloured selected squares;
method rings remain 3D reprojections. Existing rendering defaults stay unchanged.

Synthetic contracts are in `tests/unit/tennis_scene/ball_association/test_topk.py`;
configuration, component/codec, LOVO isolation and main-array regression are in
`tests/integration/tennis_scene/test_ball3d_topk.py`. Clip-specific commands and
evidence belong to `knowledge/runs/run-ball3d-topk-multihypothesis-20261103/`.
