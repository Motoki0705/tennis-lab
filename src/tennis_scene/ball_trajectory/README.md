# Segment physics and relative camera time

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This optional CPU method fits image evidence with piecewise flights in court
metres: `dX/dt = V`, `dV/dt = (0,0,-g) - k |V| V`. Each flight has six initial
state parameters. One nonnegative drag coefficient and relative camera offsets
are shared across the clip. Camera calibration is fixed.

Select `ball_reconstruction.method=segment_physics` and
`ball_reconstruction.segment_physics.observation=candidates` or `heatmap`.
The default remains `method=triangulation`. The optional component depends on
both accepted `ball_points` and pre-gate `ball_detections` v2. Model evidence is
required; annotation imports are rejected. Initialization, side and alignment
retain the standard pipeline. All settings live in `configs/pipeline.yaml`.

`segments.py` detects velocity changes from independent quadratic fits on each
side of each baseline 3D frame. Nonmaximum suppression selects changes above
the configured velocity threshold. A low event with downward then upward
vertical velocity is a bounce; other changes are labelled hit candidates.
These are heuristic predictions, not verified tennis events. Long missing runs
split the fit. Maximum-duration cuts bound integration cost and are not events.
Short missing runs may be crossed internally, while output validity and used
views never exceed the baseline's real-observation support.

`fit.py` jointly optimizes the six states per flight, common drag and clocks,
using bounded sparse least squares and soft-L1 residuals. Candidate mode chooses
the valid peak minimizing squared pixel distance / sigma² minus twice log score.
It fits two coordinate residuals and the score penalty. Heatmap mode fits the
square root of negative twice log bilinear native-map likelihood, normalized by
the peak, with an outside-image barrier. It reads four map cells per sample.
No detector, GPU, video or reference label is used. RK4 integrates bounded
flight durations without a dense timeline/heatmap copy.

Offsets are seconds, with camera frame `f` observing `X(f/fps + offset)`.
The lexicographically first **retained** camera fixes zero; each other offset
has absolute value below one frame. Fitting ignores one boundary frame at either
end of a segment so offset queries do not cross a discontinuous event. Outputs
remain on frame times. Independent flights can disagree at an event; the gap
and inferred vertical restitution are recorded, not constrained away.

The optional component saves the fit and diagnostics beside the usual `ball`
field. The [evaluation adapter](../evaluation/ball3d/segment_physics.py) uses the
same solver, exposing offsets through the foundation result contract. Each
LOVO run estimates only its retained cameras. A held-out camera retains zero
offset under the foundation scorer. Relative time gauges can therefore differ
between folds; absolute synchronization is unidentifiable from this fit.

Diagnostics include segments/events, image residuals, physical acceleration,
states, drag, offsets, optimizer status and local information rank. Conditional
curvature supplies offset covariance/standard errors when full rank. Rank
deficiency is explicitly unavailable; active parameter bounds are flagged.
These are local approximations, not calibrated confidence intervals. Failed
optimization returns an explicit status and invalid output; no baseline fallback
is substituted. Flat heatmaps, short flights and calibration error can defeat
the fit. Magnus force, rolling shutter and calibration refinement are absent.

Clip-specific experiments, synthetic test design and measurements live in
[`knowledge/runs/run-i1565-segment-physics-offset-20261103`](../../../knowledge/runs/run-i1565-segment-physics-offset-20261103/).
