# Drift of a fixed shared court

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`inspect_frame(rgb, shared_h, supported_lines, DriftConfig())` diagnoses one
source-resolution frame. `shared_h` and named physical lines come from an
accepted shared fit. No homography, line identity, endpoint or net mask is
re-estimated. Existing paired-edge extraction supplies stripe-centre proposals
and competing stripes along those known lines.

Signed normal offsets constrain an inverse image similarity (translation,
rotation, uniform scale: four parameters). Three physical lines are needed to
initialize scale; two intersecting lines leave scale about their intersection
unobserved. The complete fit needs at least four lines, one transverse and two
longitudinal lines, two directions separated by 12 degrees, and rank four.
Alternatives remain explicit. Contradictory observed lines, distinct equally
supported motions, deficient rank or excessive uncertainty fail as
`drift_unobservable`; there is no translation-only or local-H fallback.

Each physical line contributes two spatial anchors at its 10th/90th percentiles.
Its weight does not grow with the pixel sample count. A 1px anchor-noise floor
and residual variance define parameter covariance, propagated through the
inverse transform to visible court keypoints. Maximum projection standard
deviation must be at most 5px. Line residuals must be at most 3px. Pixel settings
and the 48px search radius scale by image diagonal from 1920×1080.

Mean displacement minus two standard deviations above 3px gives
`camera_moved_possible` (warning). Reliable mean displacement above 10px or
maximum above 25px fails. Unobservable frames also fail. The estimated transform
is diagnostic only: downstream coordinates always use the original shared H.

`summarize_clip(frames)` retains every measured frame and chooses the worst
verdict (failure before warning before ok). It never pools temporal observations
or assumes equal camera poses across frames. Sparse sampling cannot certify
unmeasured frames. The four-parameter model can reject genuine projective
motion as model conflict; such a rejection does not establish that the camera
moved or that it stayed still.

The support and uncertainty profile was frozen on separate artificial
**development** views before run 78's heldout and real-video evaluation. Evidence,
selection and limitations are in [run 78](../../../../../knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md).
Artifact preparation and explicit loading are documented in the
[video court boundary](../../../../tennis_scene/pipeline/video_court/README.md).
