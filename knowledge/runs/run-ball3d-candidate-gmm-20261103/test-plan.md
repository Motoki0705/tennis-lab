# #1563 test design (before implementation)

- Contract: a sampled Gaussian's joint log curvature gives its subpixel mean and
  full covariance, in native pixels using `(W-1)/(w-1)` and `(H-1)/(h-1)`.
  Unit tests use isotropic and rotated anisotropic Gaussians, unequal pixel scales,
  noncentral peaks, a boundary, a flat patch and unequal candidate counts.
  Detect x/y transposition, missing cross curvature, invalid padding and silent
  covariance replacement. Budget: 6 cases, each <1 s, no real assets.
- Contract: selected cameras' candidate evidence reaches existing Method A;
  only the posterior conditional on >=2 camera support supplies a triangulated
  point. Preserve missing frames, report conditional mass and rejected candidates;
  MAP of the highest-weight component and mixture moments stay distinct.
  Unit: asymmetric camera/frame fixture with distractors and a known trajectory;
  detect swapped cameras, rank collapse and prior-only points. Budget: 4 cases,
  each <5 s. Reuse utils geometry probabilistic triangulation tests for conjugate
  posterior, evidence normalization, Bernoulli subsets and scaling invariance.
- Contract: constant-velocity Kalman filtering and RTS smooth only contiguous
  valid segments, use real FPS, retain covariance and never create observations.
  Unit in utils/geometry: hand-computed scalar update, asymmetric 3D motion and
  gap reset. Budget: 3 cases, each <1 s. No existing Kalman/RTS implementation found.
- Contract: public config -> definition -> assembler -> production algorithm ->
  typed codec and evaluation adapter agree; LOVO has no held-out evidence/GT.
  Integration: 2 cases using tiny synthetic evidence, plus a frozen main default
  output generated before modifying production. Compare positions, masks,
  inliers, raw pixels, rejection codes and reprojection arrays exactly.
  Reuse `test_scoring.py::test_method_runner_excludes_labels_reference_and_heldout_camera`,
  `test_ball3d_bundle.py::test_real_component_codec_bundle_and_cpu_runner_roundtrip`,
  and `components/test_ball_points.py` masking contracts. Budget: <5 s/case.
- Contract: projected covariance uses the pinhole Jacobian and native px²;
  coverage uses the foundation's scored camera/frame identities, includes
  observed/approved-estimated separately, and never alters predictions.
  Unit: independent analytic projection, exact chi-square boundaries and
  missing samples; rotated ellipse endpoints. Budget: 3 cases, <1 s/case.
  Reuse existing synchronized video/full-decode/HTML integration.
- Calibration stays in the run script, restricted to [0,505). Unit: corrupt all
  later labels and verify unchanged parameters; selected-view calibration uses
  only retained camera IDs. Budget: 1 case <1 s. Calibrated score denotes usable
  top-K support, not independently identified physical existence.
- New settings rejection: representative bad K, scale/prior/variance/budget/
  selector values, at the cheapest unit layer. Budget: <=10 parametrized cases.

Measure collection with `--collect-only` and all setup/call/teardown durations
with `--durations=0 --durations-min=0`; pytest `-n 0` (at most 2). Record exact
nodes/counts/times. No GPU, detector inference, or concurrent heavy jobs.

Real-data qualification: pinned bundle/GT digests, baseline bitwise equality to
foundation_v2, 3 cameras x 1010 frames, all-view/LOVO and three splits; common
LOVO cohort and scored/eligible coverage accompany errors. Save MAP/mean and
both RTS variants, 50/90/95% projected ellipse coverage, 95% ellipses in video
camera/top/side panels, all-frame decode and portable HTML audit. Outputs use
a new candidate_gmm version directory. No parameter change after primary scoring.

Prior evidence: #1670 has top-8 recovery headroom but no guaranteed joint
recovery. #1672's beam null states removed 296/505 full-view frames and worsened
common-cohort errors. This design avoids temporal path/null search; smoothing
preserves the geometric validity mask and gaps. All results, including negative
ones and coverage changes, will be reported. No dependence on #1559.

Presentation follow-up: extend the existing media integration with covariance
and embedded supplementary figures. One cheap unit case draws an intentionally
broad ellipse into a court panel and requires all RGB/adjacent-panel pixels to
remain untouched (<1 s, no video fixture). This guards the observed panel-boundary
risk without adding another video integration. Representative ellipsoid frames
may be selected by prediction availability only; no error-based selection.
