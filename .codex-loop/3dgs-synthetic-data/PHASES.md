# 3DGS-native synthetic-data phases

Phase changes are allowed only after every measurable gate in the active phase
passes. A failed experiment stays in the same phase with a recorded hypothesis.

## P0 — Isolated NHT runtime and reproducible training

Status: complete (cycle 01).

Acceptance gates:

- `third_party/nht` tracks exact upstream/toolchain pins and keeps `.venv`,
  checkout, and artifacts local.
- CUDA NHT rasterization plus deferred-shader forward/backward is finite.
- A real COLMAP scene completes at least one immutable training step and emits a
  loadable finite checkpoint, PLY/deferred asset, validation renders, metrics,
  and trajectory video.
- Setup and training refuse modified upstream state and non-empty outputs.

## P1 — Remove the overlay-era implementation, preserve alignment

Status: complete (cycle 01).

Acceptance gates:

- A reviewed inventory proves the deletion boundary before removal.
- Old `dataset/`, `provider/`, and `rendering/` namespaces and their dedicated
  scripts/configs/tests are absent.
- Scene export lives under alignment ownership; no compatibility shim preserves
  the old namespace.
- A new real B00 export loads strictly with 491 cameras/images.
- Alignment unit/e2e tests, Ruff, mypy, and script convention review pass.

## P2 — Native Gaussian scene composition boundary

Status: complete (cycle 02).

Acceptance gates:

- Versioned file contracts describe background and movable Gaussian assets,
  coordinate frames, units, transforms, appearance payloads, and provenance.
- Rigid Sim(3) transforms update Gaussian means, rotations, and scales with
  dtype/device behavior covered by unit tests.
- Background and at least one asset are composed before one renderer call; no
  RGB overlay path exists.
- A deterministic CUDA integration smoke emits RGB, alpha/depth or visibility
  evidence, per-instance labels, and an immutable manifest.

## P3 — BLCS single/multi-ball generation

Status: complete (cycle 09).

Acceptance gates:

- Gate revision authorized by the user on 2026-07-28: P3 may use an explicitly
  generated prototype ball instead of a real user-ball asset, and the existing
  user-approved court alignment does not block completion.
- A deterministic centred 6.7 cm prototype Gaussian asset has explicit
  non-user provenance, a verified metric three-sigma envelope, and passes the
  production capture, feature-fit, ingestion, and registry boundaries.
- Physically valid court-space trajectories support single and multi-object
  scenes without inter-frame identity drift; same-seed regeneration is
  byte-identical.
- RGB plus complete 2D/3D trajectory, instance, visibility, camera, and transform
  labels pass schema, exact-contribution-mask, and projection-consistency tests.
- The accepted alignment references the exact export scene/camera fingerprint,
  and the ball is composed with the background Gaussian tensors before native
  rendering with `rgb_overlay_used=false`.

## P4 — PLCS avatar method research and asset construction

Status: complete (cycle 11). The official-method comparison, two-candidate
geometry screen, 4,096-Gaussian/55-joint SMPL-X asset, explicit NHT feature
fit, three controlled native poses, repeated-run tolerance, and strict
acceptance report pass.

Acceptance gates:

- Primary papers and official implementations are compared with title, URL,
  official code, pinned commit, applicability, limitations, and failure results.
- At least two technically viable avatar/control candidates are exercised on
  representative motion; selection is based on recorded visual and geometric
  metrics rather than availability alone.
- A selected human Gaussian asset can be controlled from SMPL and/or COCO17
  motion without silently dropping unsupported joints or frames.

## P5 — PLCS single/multi-person generation

Status: complete (cycle 12). Deterministic single/multi plans, native
background-plus-person NHT renders, exact contribution labels, byte-identical
repeats, and the thresholded P5 acceptance report pass.

Acceptance gates:

- Single/multi-person placement, rotation, pose, identity, and visibility labels
  are complete and deterministic from the seed.
- Gaussian deformation and rendering occur in the 3D scene, with projection and
  temporal-consistency metrics meeting documented thresholds.

## P6 — Safe SfM-neighborhood camera research and sampling

Status: complete (cycle 14 extension). Cycle 13 established the conservative
support-bounded baseline. Following the user's multi-court revision, cycle 14
added two-court inward-looking circle/ellipse families at 0.75/1.00/1.30 of
the measured SfM envelope, multiple heights and targets, intentional partial
views, native NHT representative renders, and a mechanics-scoped visual
review. The one-step background remains explicitly non-production.

Acceptance gates:

- Primary papers and official implementations are recorded with pinned commits,
  applicability, limitations, and failed sampling hypotheses.
- Candidate poses are bounded by scene support, court framing, collision/near
  plane constraints, and explicit extrapolation scores.
- Distribution and coverage metrics show material novel-view expansion without
  exceeding the selected safety threshold.
- Bold circle and ellipse families are derived from the measured SfM support,
  target either the complex or a verified court instance, and retain full,
  near-full, partial, and sparse second-court coverage without requiring all
  fourteen points.
- Representative native NHT renders cover every family. Visual acceptance
  separates geometric/render-mechanics evidence from production appearance
  quality and records the one-step checkpoint limitation.

## P7 — Court-detection dataset generation

Status: complete (cycle 15). The 428-frame native NHT release and independent
repeat are byte-identical, family-disjoint train/validation/test splits cover
all shape/scale/target semantics and all coverage buckets, two-court physical
annotations and seven-channel targets pass strict projection/visibility/
integrity gates, and corrected visual diagnostics pass at mechanics scope.

Acceptance gates:

- Large deterministic camera trajectories render RGB from one Gaussian scene
  and save projected 2D court keypoints with renderer-derived visibility.
- Verified physical courts retain stable `court_instance_id` in annotations.
  Within each court, the 14 physical line points map to seven near/far-symmetric
  classes, allowing two peaks per class per court and four peaks per class for
  the current two-court scene.
- The model target is exactly seven multi-peak heatmap channels. Court-instance
  grouping, homography fitting, geometry assignment, and Hungarian matching are
  post-processing only and are excluded from training targets.
- Partial views are first-class samples; acceptance is based on useful visible
  physical points/classes rather than all fourteen points.
- Projection round trips, image bounds, occlusion/visibility, pose provenance,
  family-disjoint split leakage, and artifact-integrity checks pass.

## P8 — Integrated dataset release gate

Status: complete (cycle 16). All three task families rehash to the same
491-camera export, scene fingerprint, NHT composition and renderer commit.
Same-seed plans/renders/labels are byte-identical, distinct seeds pass
task-appropriate measured diversity gates, all 15 integrated checks pass, and
the visualization-first final report is published under
`docs/3dgs-native-synthetic-data`.

Acceptance gates:

- BLCS, PLCS, and court generation share the versioned scene/camera boundary.
- Repeated seeds reproduce manifests and labels; distinct seeds achieve measured
  diversity.
- Smoke, visual, metric, schema, and artifact-integrity reports pass without
  overwriting prior runs.

## P9 — Production pipeline refactor

Status: complete (cycle 18). The user selected Architecture A and NHT boundary
N1. BLCS, PLCS, court, and future generators now share one dataset registry,
vertical domain slices, config-selectable algorithms, immutable command plans,
and a pinned subprocess renderer boundary.

Acceptance gates:

- All generators are owned under
  `src/synthetic_data_generation/dataset/<dataset-name>` and are registered in
  exactly one central registry; scripts are grouped by alignment or dataset
  workflow.
- Prototype candidates are stable config-selectable algorithms. Unknown names
  fail closed, and phase numbers or prototype file names are absent from the
  production routing API.
- The NHT submodule owns no tennis dataset logic. The parent project invokes
  project-owned workers with the exact clean submodule commit and logical NHT
  venv interpreter through a shell-free subprocess.
- Export-first reproduces all 491 cameras/images and the accepted scene
  fingerprint into a new immutable bundle.
- A real NHT runtime probe and native RGB render execute through the new
  dataset pipeline. Failed AOV attempts remain artifacts and do not weaken the
  production default.
- Unit, integration, end-to-end, Ruff, mypy, script-convention, and diff checks
  pass, and the visualization-first report documents both successes and
  remaining AOV limitations.
