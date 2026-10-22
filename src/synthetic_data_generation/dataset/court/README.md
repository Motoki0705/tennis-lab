# Court dataset v1/v2/v3 contract

Status: implemented. This document is the canonical production contract for
all explicitly selected Court dataset versions. It is also the sole prose
authority for the camera-view KP14 semantics and their migration policy.

## Visual review

See [Court Review](review/README.md) for the local 3D trajectory and image-label browser.

## Portable scene assets and Colab rendering

`assets.py` packages immutable rendering inputs with schema
`court_scene_assets_v1`. Each scene carries exactly six files: the public NHT
`scene.json`, `cameras.json`, `points_scene.npy`, the checkpoint and runtime config
referenced by that scene, and the accepted `alignment/alignment.json`. Camera
calibration, normalization, metric similarities, court poses and accepted
candidate inventories remain in their original authorities. All six files are
copied byte for byte, including automatic and human-confirmed alignment schemas;
there is no geometry migration or alignment rerun. Unknown alignment formats fail
in the existing reader.

The manifest records size and SHA-256 for every carried file and pins the producing
tennis-lab/NHT commits. Existing run/config/dataset descriptors, the sparse index,
and alignment evidence remain source-relative references with digests, rather than
copied inventories. These references describe provenance; they do not claim that
the full alignment evidence has been transferred or refitted. Captured images,
videos, extracted frames, SfM intermediates, duplicate PLYs, training logs and
alignment heatmap rasters are excluded. The public sparse points are retained
because both scene validators require them.

This bundle is a rendering/label dependency closure, not a completed scene-pipeline
workspace. Load the public export and stored alignment directly; do not fabricate
missing ingest, reconstruction or alignment owners to pass pipeline validation.
Use `load_alignment_result` for every supported stored alignment schema. Convert
exported camera poses with `alignment.metric_adapter.metric_from_nht_camera` and
divide rendered depth by `nht_scene_units_per_metre` before comparing metric depth.

Package into a **new** directory outside the source workspace, then pin the
returned manifest digest for every subsequent verification. The packaging CLI
requires a clean, committed worktree so its producer commit identifies the code:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets pack \
  --source-root /read-only/data/synthetic_data_generation/scenes \
  --scene-ids B00 B01 B02 B03 --destination tmp/court-scene-assets-v1
.venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify \
  --root tmp/court-scene-assets-v1 --manifest-sha256 MANIFEST_SHA256
```

The verifier checks exact inventory, every size/hash, exported geometry and the
stored accepted alignment. Missing/extra files, duplicate paths, symlinks, unknown
schemas and changed manifests fail. It never writes to the input or re-creates a
missing file. The source workspaces remain read only.

Use [tennis-drive](../../../../.agents/skills/tennis-drive/SKILL.md) for transfer,
authentication and verification. Choose the role/path according to its
[storage policy](../../../../.agents/skills/tennis-drive/references/storage-policy.md).
A new bundle awaiting Colab verification is a staging candidate; promote by a
verified copy into a new input-data version after rendering equivalence passes.
Never overwrite an existing asset version. After upload, verify the remote bytes
with the Drive CLI; after download, run the manifest verifier again with the
original digest. Retain the Drive ID and transfer receipts outside the bundle.

The accepted B00–B03 input is
`data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified`
(relative to the Drive project root and the repository checkout). Its manifest
SHA-256 is `a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b`.
The [adoption record](../../../../knowledge/runs/run-court-assets-colab-l4-20261022/adoption.json)
owns the Drive ID, user-decision source and verification receipts; `assets.json`
remains the file/geometry authority. The staging copy is retained.

Colab session creation, exact Git checkout, persistence and GPU release follow
[tennis-colab](../../../../.agents/skills/tennis-colab/SKILL.md). Use an explicitly
requested L4 and a pushed tennis-lab commit whose pinned NHT commit is also
fetchable. In the disposable VM, prepare the repository and NHT with:

```bash
export PATH="$(uv tool dir --bin):$PATH"
uv sync --locked
.venv/bin/python -m spin setup-nht --render-only
nht-render --help
```

Repeat the explicit tool-bin `PATH` export in every Colab shell job that invokes
`nht-render`; a fresh login shell does not inherit another job's environment.

`spin setup-nht` owns the pinned Python/Torch/tiny-cuda-nn versions and recursive
NHT/gsplat checkout; do not duplicate those versions in an alternate installer.
The explicit `--render-only` mode skips the trainer environment; see the
[development CLI](../../../../.spin/README.md).
Run it only inside the Colab checkout for this workflow. A local worktree must
not replace the shared editable NHT tool. For setup timing, bash `time` is
available even when `/usr/bin/time` is absent. Preserve installation stderr,
CUDA/compiler versions and the observed GPU in the session's declared outputs.
Measured setup/transfer/render times and the failed first installation are in the
[L4 verification record](../../../../knowledge/nodes/synthetic_data_generation/000026-run-court-assets-colab-l4-20261022.md).
The successful setup measurement used an already warmed VM; it is not a cold
installation estimate. The pinned NHT image-free change must be fetchable even
while its upstream PR is pending; until it is merged, NHT `main` does not supply it.
Keep actively growing installation logs outside the `--persist` tree and publish
them by rename after each command exits; copying an actively changing file can
fail the worker's periodic Drive save and stop the command.

After setup, download the accepted input to the disposable VM and verify it
before passing any scene to a generator. Use the configured Drive connection
from the skill above; never embed its credentials in a notebook or manifest:

```bash
COURT_ASSET_PATH=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
COURT_ASSET_SHA256=a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download \
  "$COURT_ASSET_PATH" "$COURT_ASSET_PATH" --verify
.venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify \
  --root "$COURT_ASSET_PATH" --manifest-sha256 "$COURT_ASSET_SHA256"
```

The equivalent Python preflight is
`verify_scene_assets(Path(asset_root), manifest_sha256=expected_digest)` from
`src.synthetic_data_generation.dataset.court.assets`. Both the root and expected
digest are explicit generator inputs. Download into a new directory; a prior
directory's presence is not proof of a completed, verified transfer.
Then render existing exported cameras:

```bash
nht-render --scene "$COURT_ASSET_PATH/scenes/B00/reconstruction/export/scene.json" \
  --without-source-images --camera-id EXPORTED_CAMERA_ID --batch-size 1 \
  --output outputs/synthetic_data_generation/profile/assets-check/B00
```

The explicit image-free mode retains camera/reference/model validation and records
`source_images_explicitly_omitted` in the NHT result. Default export validation
still requires the captured images. In Python, use
`NHTRenderCommandRequest(..., without_source_images=True)` and
`NHTRenderClient.validate_scene(..., without_source_images=True)`;
`CourtNHTRenderer(..., without_source_images=True)` propagates that choice.
Image-free and complete validations have separate process-local cache entries.

### Rendering equivalence contract

For equivalence, render the **same camera requests** locally and on L4, with the
same checkpoint/runtime and camera batch size. Local GPU rendering must use the
shared training queue. Compare original float32 RGB, alpha and depth before JPEG
encoding, plus projected/renderer-visible KP. Packed publications retain JPEG
quality 95 and sparse labels, but no alpha/depth rasters; JPEG differences alone
cannot validate depth or visibility. A queue-rendered baseline supplies that
missing evidence. Report both raw and JPEG RGB errors. Record per-scene results,
setup/transfer/render times, transferred bytes and GPU memory in the experiment
record. Declare probe outputs with Colab `--persist`, verify terminal job/Drive
receipts, and always `stop` the session when finished.

Separate checkpoint deserialization, resident renderer initialization, the first
camera (including lazy CUDA compilation), subsequent render calls, and the public
call including validation/PNG/array writes. A few existing-camera probes measure
runtime compatibility; they do not establish production generation throughput or
unseen-view quality. Record PyTorch peak allocated/reserved memory separately from
sampled device memory, with the sampling interval; sampled peaks are lower bounds.

Keep full-frame distributions and worst-pixel coordinates when checking numeric
agreement. Inspect RGB, alpha, metric depth and the production visibility threshold
at those coordinates. Court polygons and common alpha masks are useful diagnostic
subsets, but do not replace the full-frame result. In particular, finite positive
depth at zero alpha is not evidence of a valid surface. Repeat each GPU's render to
separate repeat variation from reproducible differences between devices. If the
acceptance checks below fail, retain the staging inputs and report the unresolved
comparison before promoting them.

The user decision on 2026-10-22 in
[#1351](https://github.com/Motoki0705/tennis-lab/issues/1351) accepts cross-GPU
equivalence by teacher agreement and RGB error. All checks must pass separately
for each repeat, without excluding background pixels or outliers:

- The complete scene/camera/court/KP inventory, class and physical IDs, projected
  UV, metric KP camera depth, `in_front`, `in_frame` and `renderer_visible` must
  match exactly. Renderer visibility retains the production alpha >= 0.01,
  positive-depth, radius-1px rule. There is no epsilon for teacher agreement.
- Raw float32 RGB in [0, 1] must have **PSNR >= 57 dB per scene**, pooling all
  full-frame RGB channel errors of the declared probe cameras before computing
  RMSE. Every individual full-frame view must also have **PSNR >= 55 dB**, so a
  good average cannot hide a degraded camera. PSNR is `-20 log10(RMSE)` with
  peak=1; exact RGB equality passes with infinite PSNR (JSON `null` only when
  RMSE=0). Missing views or non-finite evidence fail.
- Alpha/depth distributions, threshold crossings, full-frame and court-region
  outliers remain diagnostic evidence. Raster equality between GPUs is not an
  acceptance condition. Generation still applies its normal alpha/depth quality
  gates to each L4 output; this decision does not change those gates.

The thresholds round down the measured minimum scene PSNR 57.986 dB and minimum
individual-view PSNR 55.859 dB to whole-dB floors. The scene floor limits RGB
RMSE to 0.001413, and the view floor to 0.001778. The measured JPEG quality-95
round trip (including uint8 quantization) has RMSE 0.003537–0.004983, or
PSNR 46.05–49.03 dB. Even the view floor permits roughly half the smallest
observed JPEG RMSE; this is the input-image error budget justification, not a
claim that GPU depth differences are noise or that training accuracy is unchanged.
The approximately 58 dB value is a pooled scene result, not a per-view minimum.
Measurements, maxima and limitations remain in the L4 verification record above.

The saved comparison can be checked without a GPU using
[`court_assets_acceptance.py`](../../../../knowledge/runs/run-court-assets-colab-l4-20261022/court_assets_acceptance.py)
(`--help`). It checks the exact teacher inventories and the two RGB floors for
each supplied comparison; it does not rerender or modify the original evidence.
Use the recorded B00–B03 probe inventory and both repeats when validating these
assets. Production generation uses **Colab L4 only** with the pinned runtime;
new-view quality, throughput and chunked generation/restart belong to the
generator and Colab generation workflow.

## JPEG shard publication

Generation evaluates the original renderer RGB, alpha and depth before accepting
samples. The assembler then publishes RGB as JPEG quality 95 (using exactly
`np.round(rgb * 255).astype(np.uint8)`), with the original dimensions. Alpha,
depth and duplicate preview images are not retained after the generation gates.
No renderer runs during training.

The fixed owner still contains `dataset.json`, `samples/` and `diagnostics/`.
`dataset.json` explicitly declares `court_image_store_v1` and the unchanged
v1/v2/v3 geometry schema. `samples/shards/*.bin` concatenate independent JPEGs;
`samples/index.npz` stores byte offsets, lengths,
checksums and losslessly compressed sparse JSON records. KP coordinates, physical
IDs, geometry/renderer visibility, camera intrinsics and pose, target binding,
trajectory groups, splits and generation QA results remain authoritative there.
There is one copy of each sparse sample record, rather than separate copies in
both a manifest and per-image labels. The index is pickle-free and DEFLATE
compressed; readers decode only the requested JPEG from a worker-local mmap.

`sample_store.py` is the shared storage boundary for training, visualization,
3D review and the canonical validator. `read_court_manifest` exposes the logical
geometry manifest, `read_court_labels` exposes sparse labels, and
`read_court_rgb` returns stored uint8 RGB. Storage versions are dispatched by
explicit schema, never by guessing files or substituting preview images.

Pose supervision and 3D review retain their geometry contracts. Full validation
of a packed publication checks shard/image hashes, all JPEG dimensions and the
sparse geometry/semantic/visibility inventories. Recomputing renderer visibility
from alpha/depth is a **generation-time** check: those rasters are deliberately
absent after publication. Historical generation performance remains recorded;
published byte counts describe the current packed owner.

Existing float32 owners can be imported once with
`migration.migrate_court_store(source, destination)`. It validates the old
alpha/depth evidence before using the same writer as generation, decodes every
new JPEG and compares all sparse records and splits. It neither re-renders nor
deletes its source. After validation, replace the fixed owner and remove the old
copy; the migration report belongs outside that owner. Legacy float32 readers
remain available only for explicitly versioned old publications and migration.

## Purpose

Court dataset v2 introduced three related changes without changing the shared
scene or court geometry contracts.

1. Provide an explicit configuration that uses only `court_center` as the
   camera target.
2. Resolve the target court from each generated camera position. A
   court-centred trajectory keeps its centre court; a complex-centred
   trajectory selects the nearest court for each camera sample.
3. Replace the seven two-point semantic classes with fourteen camera-relative
   Court keypoint classes.

The generator must continue to produce the current v1 dataset when v1 is
selected. It must not auto-convert, dual-write, or silently fall back between
versions.

Court dataset v3 retains v2's sample-level target-court planning and fourteen
singleton channels, but corrects the legacy left/right definition. V2 is
frozen as the court-local-X/Y-only legacy family. V3 defines left/right in the
canonical court frame and near/far by camera distance to the baseline. Its
camera-side switch is one proper half-turn in court XY, shared by the KP
permutation and canonical camera transform. V2 artifacts and checkpoints are
never treated as V3.

## SfM-bounded camera generation

Select `dataset/court=sfm_bounded` for the v3 label family with bounded camera
placement and independently sampled look-at offsets. Legacy `train`, `v1`,
`v2`, and `v3` presets keep their existing generation settings. The numeric
policy is owned by `configs/dataset/court/sfm_bounded.yaml`.

`trajectory.sfm_boundary_margin_m` explicitly enables the boundary heuristic.
In this mode the complex orbit centre is the mean of accepted court centres,
projected onto the reference court plane. Scene bounds can include distant
background geometry and therefore do not define the bounded orbit centre.
Legacy presets keep their previous scene-bounds centre.
`trajectory.sfm_complex_center_on_hull=true` explicitly places the complex
orbit centre at the captured-camera XY hull's vertex mean instead. This helps
asymmetric capture footprints use the observed space on both sides of a court.
Per-court orbit centres and look-at targets retain the actual court geometry;
the setting does not shift court labels or relax the boundary. It requires
explicit SfM bounds. The default is false.
`trajectory.sfm_boundary_expansion_percent` allows controlled extrapolation:
`5.0` expands the horizontal hull by a linear factor of 1.05 about the mean of
its vertices, before applying the inward metric margin. This origin belongs
to the hull, not each orbit, and is independent of camera density. Thus all
court centres share the same permitted boundary in a common court plane.
The captured q90 radius cap expands by the same factor. Zero (also the default
when omitted) preserves the previous boundary exactly. Expansion is an upper
allowance, not a requirement to place cameras outside: the margin and nested
radius scales can keep a path inside. Negative or non-finite values and an
expansion without explicit SfM bounds are rejected.
For example, use `dataset.court.trajectory.sfm_boundary_expansion_percent=5.0`
with the `sfm_bounded` preset. The percentage is a distance scale, not an area
percentage; a factor of 1.05 increases hull area by 1.05 squared.

Captured public-export camera centres are first converted to metres. For each
complex/court centre, their XY convex hull in that centre's court plane defines
inward-offset half-planes. Each candidate's shape, orientation and axis ratio
are used to solve the largest admissible major half-extent analytically; the
resolved base radius is the smaller of that limit and the captured q90 radius.
Configured radius scales then generate nested shapes inside that boundary.
The supported footprints are circles, ellipses, rectangles (including squares),
and superellipses with unit boundary `|x|^4 + |y|^4 = 1`. The latter have flattened
sides and rounded corners; they are not rectangles with circular fillets.
`radius_x_m` and `radius_y_m` denote half-extents for every family, so rectangle
width and height are twice those values. The analytic support is L2 for
circles/ellipses, L1 for rectangles, and L(4/3) for superellipses. Rectangle
corners are constrained even if an arc-length sample does not land on them. The entire curve satisfies the margin, including points between
samples. Heights and vertical profiles remain config-owned. Fewer than three
cameras, collinear camera support, an orbit centre outside the inset hull, or
radius scales above one fail explicitly; no unconstrained orbit is substituted.

`trajectory.spatial_coverage_cell_m` enables position-aware selection after
boundary fitting. Every path is mapped into one shared XY grid in the first
resolved centre's metric plane. Within the existing coverage-mode objective,
new typed parameter values have priority, then shape-use balance, then the
number of previously unoccupied cells, then the existing balance/tie criteria.
This prevents new shapes from displacing all circle/ellipse variants while
using more of the available camera area. The grid is shared across courts;
per-court grids must not count identical scene locations as different cells.
The selector retains the proposal budget, minimum frames/groups and complete
typed-parameter coverage. Cheapest remaining-group costs are computed once per
iteration, preserving the same reservation semantics without repeated sorting
for every candidate.

This is a horizontal camera-position heuristic. A convex hull can bridge
unobserved gaps, and remaining height/view-direction extrapolation can still
produce artifacts. It does not certify 3DGS image quality. Moving cameras inward
also changes framing, so the bounded preset uses a wider HFOV range; existing
pre-render semantic and dataset acceptance gates remain active. Before the first
NHT render, the geometric evaluation rejects plans whose remaining candidates
cannot meet the minimum accepted frame count/fraction, leave a trajectory group
empty, or lack full/near-full/partial geometric coverage. Errors report candidate
and required counts plus missing coverage. This is only an upper-bound feasibility
check: renderer visibility and the existing post-render release gates still
control acceptance.

`view.look_at_jitter_radius_m` is an explicit v3-only extension. Each sample
looks at a point drawn uniformly by area from a disk in its resolved court's
local XY plane, at the configured look-at height. The RNG uses
`SeedSequence([target_court.binding.selection_seed, sample_index])`; this
reproduces the same target independently of render batching and rejection.
A positive radius is serialized in each view record and validated against the
requested configuration. Its absence denotes the existing exact-centre view;
v1/v2 reject jitter. The renderer and persisted-dataset reader recompute the
same point and validate the camera forward axis. KP14 channel identities and
the canonical camera transform remain unchanged. Position samples still belong
to closed shape groups, but rectangle paths have corners and independent
look-at offsets do not promise smooth temporal camera motion.

Implementation: `components/camera_sampling/shapes.py` owns the shared unit
boundaries and support functions; `sfm_bounds.py` fits them to the captured
boundary; `sampling.py` samples their 3-D arc length; `selection.py` applies
bounds and budgeted spatial selection; `targeting.py` owns target offsets. Unit tests cover full-curve containment,
scene-frame invariance, degenerate/outside support, bounded uniform-area jitter,
altered-target rejection, analytic shape supports, non-planar rectangle sampling,
and shared-grid coverage improvement with all shape families retained.
The renderer integration test includes the
`sfm_bounded` preset through publication and strict rereading.

## Confirmed v1 behaviour

The current label payload is often described as seven keypoints, but its exact
contract is seven semantic classes with two physical points per class:

| Class | Physical CourtKP indices |
|---|---:|
| `doubles_left` | 0, 2 |
| `doubles_right` | 1, 3 |
| `singles_left` | 4, 5 |
| `singles_right` | 6, 7 |
| `service_left` | 8, 10 |
| `service_right` | 9, 11 |
| `service_t` | 12, 13 |

Thus all fourteen ground points are already nested under
`courts[].classes[].points[]`; what v2 changes is their semantic identity and
channel structure.

Other v1 properties that must remain unchanged are:

- `dataset.court.view.target_modes` contains all four target modes.
- `TrajectoryGroupPlan.target_court` is selected once per trajectory group.
- A court-centred trajectory is bound to its centre court.
- Complex-centred trajectories are assigned to courts with seeded global and
  per-split balancing.
- `canonical_court_dataset_v1`, `canonical_court_orbit_plan_v1`,
  `canonical_court_sample_v1`, and `court_renderer_semantic_manifest_v1` are
  strict schemas.
- Each v1 class contains exactly two points, and existing v1 validators,
  diagnostics, and overlays retain that interpretation.

## Terminology

Two existing concepts have similar names and must remain distinct.

| Concept | Type | Meaning |
|---|---|---|
| trajectory centre kind | `OrbitCenterKind` | The centre about which the camera path is generated: `court` or `complex`. |
| view target mode | `OrbitTargetMode` | The point at which the camera looks: `court_center`, `complex_center`, `near_baseline`, or `far_baseline`. |
| target court | resolved sample data | The accepted `CourtInstance` that owns a court-relative look-at target and target metadata. |

In the v2 preset, the only view target mode is `court_center`, while both
trajectory centre kinds remain enabled. “Court-centred trajectory” below always
means `OrbitCenterKind.COURT`, not merely a view whose target mode is
`court_center`.

## Design decisions

| Area | v1 | v2 legacy | v3 camera-view |
|---|---|---|---|
| Generation selector | explicit | explicit | explicit |
| Court Detection unsuffixed source | legacy explicit source only | `synthetic_court_v2` | `synthetic_court` default |
| View targets | all four modes | only `court_center` | only `court_center` |
| Target-court authority | trajectory group | camera sample | camera sample |
| Court-centred path | centre court | centre court | centre court |
| Complex-centred path | seeded balanced assignment | nearest court per camera sample | nearest court per camera sample |
| Label channels | 7 classes × 2 points | 14 classes × 1 point | 14 classes × 1 point |
| `near` / `far` | fixed physical points grouped together | camera-side Y swap | camera-side Y swap |
| `left` / `right` | symmetric class | fixed court-local X | canonical-frame X after `I | Rz(pi)` |
| Compatibility | exact v1 | exact v2; no fallback | exact v3; no v2 alias/remap |

The target and label flow for v2 is:

```text
versioned Hydra config
        |
        v
trajectory path -> camera centre -> target-court resolver
                                      | court path: centre court
                                      | complex path: nearest court
                                      v
                              court-centre look-at pose
                                      |
                                      v
                           per-court 14-point projection
                                      |
                                      v
                         camera-relative near/far mapping
                                      |
                                      v
                        v2 serializer + strict validator
```

## Configuration and version selection

### Typed configuration

`CourtDatasetConfiguration` gains a required
`schema_version: CourtDatasetSchemaVersion`, with the only accepted values
`v1`, `v2`, and `v3`. The version is selected once at the configuration boundary and
is passed explicitly to planning, assembly, diagnostics, validation, and
visualization. Inferring the version from the shape of a label payload is not
allowed.

The Hydra group keeps the shared numeric policy and compatibility v1 selector
values in one canonical file:

```text
configs/dataset/court/
├── base.yaml   # shared policy plus schema_version=v1 and all four targets
├── v1.yaml     # alias to base.yaml
├── v2.yaml     # composes base and overrides version plus target modes
├── v3.yaml     # corrected camera-view KP14 family
└── train.yaml  # compatibility alias to base.yaml (therefore v1)
```

`run_scene_pipeline.yaml` continues to resolve v1 by default. All versions are
selectable without editing a file:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  dataset/court=v1

.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  dataset/court=v2

.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  dataset/court=v3
```

Tests and runtime code must use Hydra composition for these files. They must not
load `base.yaml`, `v1.yaml`, `v2.yaml`, or `v3.yaml` as standalone mappings.

### Validation rules

- v1 requires the exact existing set of four target modes. This preserves the
  legacy generation contract rather than merely its JSON shape.
- v2 and v3 require exactly `[court_center]`. Overrides to `complex_center`,
  `near_baseline`, or `far_baseline` fail at typed configuration validation.
- Coverage modes remain exactly `full`, `near_full`, and `partial` in all
  versions.
- Both `complex` and `court` trajectory centre kinds remain required.
- `_views_for_group()` only adds its first-group cross-target variant when a
  configured target of the other target kind exists. A singleton target list
  creates one view rather than raising `StopIteration`.
- The generated target-mode inventory must still equal the configured
  target-mode inventory exactly.

The v2 and v3 presets therefore guarantee that every accepted generated camera looks
at the centre of its resolved target court. The existing configurable
`look_at_height_m` remains the local court Z coordinate; “centre” means local
`(x, y) = (0, 0)` rather than forcing the look-at height to zero.

## Per-camera target-court resolution (v2 and v3)

### Resolution point and distance

Target resolution happens inside the camera-sample loop, after a trajectory
point provides `camera_center_scene_m` and before the camera rotation is built.
For every accepted court:

```text
camera centre = camera_center_scene_m
court centre  = court.scene_from_court applied to (0, 0, 0)
distance      = 3-D Euclidean distance in metric scene coordinates
```

Three-dimensional scene distance is used because it is invariant to each
court's orientation and remains correct if accepted court planes have small Z
offsets. It also uses the complete camera position specified by the request.

The resolver is deterministic:

1. For `OrbitCenterKind.COURT`, select
   `trajectory.center_court_instance_id` for every sample, even if another
   court becomes closer along the path. This is the explicit exception in the
   requirement.
2. For `OrbitCenterKind.COMPLEX`, compute the distance to every accepted court
   for each camera sample and select the minimum.
3. Distances within `1e-9 m` of the minimum are treated as a tie. Select the
   lexicographically smallest `court_instance_id` and increment a nearest-court
   tie diagnostic. Every tied court is equally nearest, so this rule preserves
   the nearest-court invariant while making the result reproducible.
4. Do not use `primary_court_instance_id`, split balance, insertion order, or a
   random seed as a fallback.

The look-at point is then computed in the resolved court frame:

```text
target_scene = resolved_court.scene_from_court(0, 0, look_at_height_m)
```

The OpenCV camera forward axis must equal the normalized vector from the camera
centre to this point, within the existing numerical tolerance.

### v2/v3 typed ownership

The group-level v1 binding cannot represent a complex trajectory whose nearest
court changes partway around the orbit. v2 and v3 therefore use an explicit policy at
group level and a resolved binding at sample level:

```text
TrajectoryGroupPlanV2
  target_court_policy:
    mode: trajectory_center_court | nearest_camera
    centre_court_instance_id: string | null

PlannedCourtSampleV2
  target_court:
    binding: TargetCourtBinding
    resolution_policy: trajectory_center_court | nearest_camera
    camera_to_court_center_distance_m: float
```

`centre_court_instance_id` is required only for
`trajectory_center_court`. This is a discriminated contract, not an optional
field whose missing value triggers a fallback.

`OrbitViewSpecV2` retains target mode, coverage, HFOV, and look-at height, but a
complex-centred view does not carry one static `target_court_instance_id`.
`PlannedCourtSampleV2.target_court` is the authority used by:

- the camera look-at pose;
- accepted and rejected sample records;
- label metadata;
- NHT alignment-inventory checks;
- report target-court aggregation;
- target-court diagnostics and validation.

The existing v1 types and serialized fields remain exact. Implementation may
share primitives, but the public boundary must use explicit v1/v2/v3 plan types or
an equally strict discriminated union; it must not make the current group
binding nullable across versions.

### Validation and diagnostics

v2 and v3 remove the v1 court-balance release gate. They replace it with a geometric
recomputation over every planned, accepted, and rejected sample:

- court-centred samples reference their trajectory's centre court;
- complex-centred samples reference the deterministic nearest court;
- the stored binding matches the accepted alignment inventory;
- the stored distance matches the camera and court transforms;
- the camera forward axis points at the selected court centre target.

Diagnostics record sample counts per target court, target switches per
trajectory, the two resolution-policy counts, and nearest-distance tie counts.
They are evidence, not balancing objectives.

## Fourteen camera-relative Court keypoints

### Shared Court schema boundary

`src/utils/schema/court.py` remains the authority for court dimensions,
CourtKP20 geometry, physical indices, and names. The existing
`COURT_KP_NAMES`, `COURT_KP_IDX`, `court_keypoints_3d()`, and
`COURT_SKELETON` must not change their current indices or coordinate meanings,
because Court Detection, SLCS, BLCS, PLCS, and shared rendering consume them.

The shared schema may add aliases/constants that expose, rather than redefine,
the existing ground contract:

```text
NUM_GROUND_COURT_KP = 14
GROUND_COURT_KP_NAMES = COURT_KP_NAMES[:14]
OPPOSITE_COURT_END_INDEX = (2, 3, 0, 1, 5, 4, 7, 6, 10, 11, 8, 9, 13, 12)
CAMERA_VIEW_HALF_TURN_INDEX = (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 12)
```

The scene/camera transformation and dataset-specific label construction stay
under `src/synthetic_data_generation/dataset/court/`; they depend on
`CourtInstance`, `SceneCamera`, and the versioned dataset schema and therefore do not
belong in `src/utils`.

### Shared v2/v3 semantic class order

v2 and v3 contain fourteen singleton classes in the exact order of
`COURT_KP_NAMES[:14]`:

```text
 0 far_doubles_left
 1 far_doubles_right
 2 near_doubles_left
 3 near_doubles_right
 4 far_singles_left
 5 near_singles_left
 6 far_singles_right
 7 near_singles_right
 8 far_service_left
 9 far_service_right
10 near_service_left
11 near_service_right
12 far_service_t
13 near_service_t
```

`class_id` is the stable camera-relative semantic channel ID. `physical_index`
remains the stable physical index in `court_keypoints_3d()`. They are equal only
when the camera is on the court's negative-Y side.

### Legacy v2 near/far algorithm

Near/far is resolved separately for every generated camera and every accepted
court, not only for the sample's target court.

1. Read the camera centre from `camera.camera_to_scene[:3, 3]`.
2. Transform it with `court.court_from_scene` to
   `camera_center_court`.
3. If local `y < -1e-6 m`, the physical negative-Y end is `near` and the
   physical positive-Y end is `far`; the semantic-to-physical mapping is the
   identity mapping.
4. If local `y > +1e-6 m`, the physical positive-Y end is `near` and the
   physical negative-Y end is `far`; use
   `OPPOSITE_COURT_END_INDEX`.
5. If `abs(y) <= 1e-6 m`, near/far is not geometrically defined. Reject the
   proposal before rendering with an explicit
   `ambiguous_camera_relative_near_far:<court_instance_id>` reason. Do not use
   court-axis convention or camera orientation as a silent tie-break.

Because no valid v2 class permutation exists in that case, its rejected sample
record keeps the resolved sample-level `target_court` and stores
`projection: null`. Rejected proposals have no stored image.

In v2 only, left/right remains the court-local X convention. Only near/far
changes with the camera position. This Y-only permutation is preserved solely
so existing v2 artifacts retain their exact meaning; it is not the corrected
camera-view contract.

For every serialized court, strict validation requires:

- class IDs and names are exactly the fourteen-item order above;
- every class contains exactly one point;
- physical indices are exactly the set `0..13`, with no duplicate or omission;
- the class-to-physical-index mapping equals the recomputed camera-relative
  permutation;
- coverage still counts the same fourteen physical points;
- renderer visibility is recomputed from validated NHT alpha and depth.

### Corrected v3 camera-view algorithm

V3 resolves one side decision independently for every accepted court in a
sample. Let `C_court` be the camera center transformed by
`court.court_from_scene`. If `C_court.y < -1e-6 m`, the semantic-to-physical
mapping is identity and the canonical transform `S = I`. If
`C_court.y > +1e-6 m`, both axes in the court plane rotate by pi:

```text
semantic_to_physical = CAMERA_VIEW_HALF_TURN_INDEX
S = Rz(pi) = diag(-1, -1, +1)
```

`S` is a proper right-handed rotation (`det(S) = +1`), not a reflection. The
same side decision produces the KP permutation, `canonical_from_court`,
`camera_from_canonical`, and the canonical camera center. Consequently a known
court point projected through the original court/camera transforms has the
same pixel coordinates and depth when projected through the canonical chain.
Symmetric cameras at opposite ends canonicalize their positions, rotations,
look-at targets, and unchanged focal lengths to the same convention.

The class names remain `COURT_KP_NAMES[:14]`, but their V3 meanings are:

- `near_*` belongs to the baseline closer to the camera center and `far_*` to
  the more distant baseline;
- `left` and `right` are the negative-X and positive-X identities in the
  canonical court frame. End views and baseline-exterior off-axis views satisfy
  `u(left) < u(right)`, which is useful evidence for those fixtures;
- finite lateral views remain valid even when perspective reverses a
  baseline-specific projected-u order. They retain the same side-derived
  physical identities and canonical projection round-trip;
- `abs(C_court.y) <= 1e-6 m`, missing or non-finite transforms, non-finite
  projections, and incomplete/duplicate physical inventories are explicit
  rejections.

Horizontal image flip applies the existing single channel flip permutation to
names, points, visibility, and physical indices together. It does not apply a
second court half-turn. Target-court filtering occurs only after each court's
mapping is validated, so adding non-target courts cannot change the target
court's identities.

### Singleton projection JSON

The outer projection keys are retained in v2 and v3. The exact dataset/sample
schema identifies which singleton mapping owns the payload. A shortened legacy
v2 example is:

```json
{
  "camera_id": "court-sample-000123",
  "resolution": [1280, 720],
  "coverage_modes": ["full", "partial"],
  "visible_class_names": [
    "far_doubles_left",
    "near_doubles_left"
  ],
  "visible_point_count": 8,
  "courts": [
    {
      "court_instance_id": "court-001",
      "coverage_mode": "full",
      "classes": [
        {
          "class_id": 0,
          "class_name": "far_doubles_left",
          "renderer_visible": true,
          "points": [
            {
              "physical_index": 2,
              "uv": [421.5, 119.0],
              "camera_depth_m": 28.4,
              "scene_xyz_m": [4.2, 17.1, 0.0],
              "in_front": true,
              "in_frame": true,
              "renderer_visible": true
            }
          ]
        }
      ]
    }
  ]
}
```

The real `classes` list always contains all fourteen entries; the example is
shortened only for readability. `visible_class_names` is the ordered subset of
the fourteen semantic names with renderer-visible supervision across any
court. `visible_point_count` retains its existing meaning.

The projection must contain every accepted court. An empty `courts` list
remains invalid.

## Versioned serialization and readers

The version registry under the Court dataset package owns all related schema
identifiers and semantic cardinality:

| Artifact | v1 | v2 legacy | v3 camera-view |
|---|---|---|---|
| dataset | `canonical_court_dataset_v1` | `canonical_court_dataset_v2` | `canonical_court_dataset_v3` |
| orbit plan | `canonical_court_orbit_plan_v1` | `canonical_court_orbit_plan_v2` | `canonical_court_orbit_plan_v3` |
| sample labels | `canonical_court_sample_v1` | `canonical_court_sample_v2` | `canonical_court_sample_v3` |
| semantic manifest | `court_renderer_semantic_manifest_v1` | `court_renderer_semantic_manifest_v2` | `court_renderer_semantic_manifest_v3` |
| performance | `court_dataset_performance_v2` | `court_dataset_performance_v3` | `court_dataset_performance_v4` |
| shard attempt | `court_render_shard_attempt_v1` | `court_render_shard_attempt_v2` | `court_render_shard_attempt_v3` |
| acceptance diagnostics | `court_acceptance_diagnostics_v1` | `court_acceptance_diagnostics_v2` | `court_acceptance_diagnostics_v3` |
| semantic classes | 7 × 2 | 14 × 1 | 14 × 1 |

The performance evidence schema changes with each family because
`visible_points_by_class` changes cardinality or semantic meaning. Attempt-local
shard metadata must include the selected dataset schema so a shard cannot be
reused by another version's attempt. Split, parameter-table, and semantic-visibility
diagnostics that encode group target or class cardinality receive new schema
versions as well.

Dispatch rules are strict:

- Generation uses only `configuration.schema_version`.
- Validation accepts the exact requested dataset, plan, sample, manifest,
  performance, and diagnostic schemas.
- A reader that supports multiple versions dispatches on the exact top-level
  `schema` value, then invokes a version-specific parser. It does not try one
  parser and fall back to the other.
- no version is upgraded, aliased, or down-converted in place.
- One dataset owner directory contains one version. Rerunning with another
  version goes through normal stage invalidation and transactional publication;
  files from two versions may not coexist.
- The global `canonical_scene_pipeline_v1` and `multi_court_layout_v1` schemas
  do not change because their contracts are unaffected.

The v2/v3 logical manifest and sparse sample labels add the sample-level resolved
target-court record. `trajectory_groups[].target_court` is replaced in v2/v3 by
the explicit target-court policy. The v1 field layout remains unchanged.

The report adapter aggregates unique target bindings from v2/v3 samples instead
of trajectory groups. Semantic manifests, accepted records, rejected records,
and logical sample labels must agree exactly on each sample's target binding.

### V3 regeneration and checkpoint policy

There is no in-place V2-to-V3 metadata rewrite because the physical identity of
left/right channels changes. To adopt V3, regenerate the Court dataset with
`dataset/court=v3`, then regenerate every derived Court Detection target,
preview, diagnostic overlay, and semantic manifest from that V3 artifact.
Finally retrain any model that consumes the fourteen KP channels.

Court Detection distinguishes the bundles exactly as
`synthetic_camera_view_kp14_v3` and
`synthetic_camera_view_kp14_v3_target_court`. The unsuffixed
`data/source=synthetic_court` selects V3; legacy V2 requires
`data/source=synthetic_court_v2`. A V2 checkpoint is not compatible with either
V3 bundle even though both have fourteen channels. Loading it as V3 must fail
at the target-bundle snapshot check. No channel-weight permutation, inferred
shape compatibility, or silent remap is supported.

## Court-only pipeline regeneration

A request may change only `dataset.court` plus the request's target/cursor
fields when the cursor is exactly `court_dataset` and the target set is exactly
`{court}`. The sole exception to equality for all other resolved authority is
the requested-only addition of `nht.training_python_path` and/or
`nht.trainer_path`, each as a non-empty string. This exception accommodates
retained legacy resolved configurations that predate these typed-required
fields.

Existing NHT keys or values may not be removed or changed. Unrelated NHT
additions, invalid added values, any other cursor or target set, and every
unrelated resolved-authority change are rejected. The runner revalidates
retained ingest, reconstruction, and alignment owners and their declared
outputs before invalidating Court/report. It does not run or invalidate
reconstruction, alignment, BLCS, or PLCS; rejected changes fail before
publication mutation.

## Visualization

The visualizer first validates and selects the exact dataset schema.

- v1 keeps the current seven-class two-point overlay and legend.
- v2 and v3 validate fourteen singleton classes, index points by
  `physical_index`, and draws the ground-court connections from the 0–13 subset
  of `COURT_SKELETON`. V3 validates the shared identity/full-half-turn inventory
  before drawing. Point labels and the legend use the selected version's
  fourteen-class meanings.
- No overlay reshapes another schema in memory as a compatibility
  conversion.

## Implementation ownership

| Area | Responsibility |
|---|---|
| `configs/dataset/court/*.yaml` | Shared base plus explicit v1/v2/v3 selection; v2/v3 singleton target presets. |
| `configuration.py` | Parse schema version and enforce version-specific target-mode rules. |
| `dataset/court/schema.py` (new) | Court dataset version registry, exact schema identifiers, semantic cardinality, and dispatch metadata. |
| `dataset/court/contracts.py` | Preserve v1/v2 public meanings and give the V3 plan an exact new identity. |
| `components/camera_sampling/targeting.py` (new) | Pure task-local target-court distance, tie, and invariant logic. |
| `components/camera_sampling/selection.py` | Resolve target inside the sample loop, build look-at pose, and handle singleton target lists. |
| `components/camera_view.py` | Own the per-court V3 side decision, proper canonical transform, and finite projection boundary. |
| `components/labels.py` | Keep v1/v2 projections and add explicit V3 camera-view singleton projection. |
| `src/utils/schema/court.py` | Expose legacy and full-half-turn mappings without changing CourtKP20. |
| `assembler.py` / `semantic_manifest.py` | Version-specific sample layout, strict visibility/class validation, and manifest construction. |
| `rendering/nht.py` / `shards.py` | Validate sample bindings against alignment; prevent cross-version shard reuse. |
| `diagnostics.py` / `performance.py` | Record exact versioned sample geometry and class metrics. |
| `pipeline/handlers.py` | Aggregate v2/v3 target bindings from samples. |
| `visualization/sources.py` / `overlays.py` | Exact v1/v2/v3 reader dispatch and version-specific overlays. |

The nearest-court resolver stays task-local because it depends on the synthetic
scene's `MultiCourtLayout` and singleton planning contracts. Only stable CourtKP
geometry vocabulary belongs in `src/utils/schema/court.py`.

## Test design

Tests follow the repository's unit/integration/e2e boundaries.

### Unit

- `tests/unit/utils/schema/test_court.py`
  - ground-name order and the opposite-end permutation;
  - permutation is an involution and covers `0..13` exactly;
  - existing CourtKP20 indices and coordinates do not change.
  - the V3 full-half-turn mapping is an exact bijective involution and maps
    `(x, y, z)` to `(-x, -y, z)`.
- `tests/unit/synthetic_data_generation/dataset/court/components/camera_sampling/test_targeting.py`
  - court-centred paths remain fixed even when another court is closer;
  - a complex trajectory changes target court between camera samples;
  - 3-D nearest distance and lexicographic equal-distance tie behaviour;
  - stored distance and selected binding agree with layout transforms.
- existing `test_selection.py`
  - `[court_center]` creates one valid view and no cross-kind variant;
  - camera forward axes point at each resolved court centre;
  - v1 balanced group assignment remains unchanged.
- existing `test_labels.py`
  - v1 golden shape remains 7 × 2;
  - v2 shape is 14 × 1;
  - v3 shape is 14 × 1 with the exact full-half-turn mapping;
  - cameras on opposite court ends swap near/far physical indices;
  - end views preserve projected left/right evidence while lateral views keep
    canonical-X identities without an image-order rejection;
  - mid-plane ambiguity is rejected explicitly;
  - renderer-visible summaries use the selected version's ordered names.
- existing `test_contracts.py`, `test_assembler.py`,
  `test_semantic_manifest.py`, `test_shards.py`, and `test_performance.py`
  - exact v1/v2/v3 keys and schemas;
  - cross-version inputs and mixed artifacts are rejected;
  - sample-level targets and fourteen-class metrics are recomputed rather than
    trusted.

### Integration

- `tests/integration/synthetic_data_generation/test_configuration.py`
  composes `dataset/court=v1`, `dataset/court=v2`, and `dataset/court=v3`, checks the four-versus-one
  target modes, and rejects unknown versions or invalid empty target lists.
- `tests/integration/synthetic_data_generation/test_court_dataset.py` runs the
  lightweight renderer boundary for the versioned families. v1 keeps the
  current group balance assertions; v2/v3 check per-sample nearest/fixed
  selection, exact 14KP labels, accepted/rejected propagation, and deterministic
  same-seed output. V3 also checks the full-half-turn and finite lateral
  acceptance.
- `tests/integration/synthetic_data_generation/test_dataset_visualization.py`
  validates the versioned overlays without GPU training.
- `tests/integration/synthetic_data_generation/test_dataset_performance.py`
  verifies version-specific semantic key inventories and the new performance
  evidence schema.

### E2E

`tests/e2e/synthetic_data_generation/test_run_scene_pipeline.py` and
`test_visualize_dataset.py` cover the Hydra selectors and exact reader
dispatch at command level. Existing GPU acceptance remains a release gate, not
the primary test for pure target or label logic.

## Implementation sequence

1. Preserve the version enum, composed v1/v2 configs, schema registry, and v1 golden
   tests before changing behaviour.
2. Introduce v2 group/sample contracts and the pure target resolver; update
   camera planning and geometric validation.
3. Add the V3 shared full-half-turn authority and camera-view canonicalization.
4. Add V3 labels and exact version-specific assembly, manifests, diagnostics, performance evidence,
   shard binding, reporting, and visualization.
5. Run focused unit tests, the synthetic-data integration suite, pre-commit,
   and finally the existing Court GPU acceptance when an implementation is
   ready for release.

## Acceptance criteria

Implementation is complete only when all of the following hold:

1. `dataset/court=v1` produces the existing schemas, four target modes,
   group-level balanced target assignment, and seven two-point classes.
2. `dataset/court=v2` produces only `court_center` views by default and all
   output artifacts carry their exact v2 schemas.
3. Every accepted v2 court-centred sample targets its trajectory centre court.
4. Every accepted v2 complex-centred sample targets the recomputed nearest
   court for that camera position, including when the target changes within one
   trajectory.
5. Every accepted v2 camera forward axis points at local `(0, 0,
   look_at_height_m)` of its resolved target court.
6. Every v2 projected court contains exactly fourteen singleton classes in
   `COURT_KP_NAMES[:14]` order and exactly one copy of every physical index
   `0..13`.
7. Near/far swaps when the camera crosses the court mid-plane; ambiguous
   mid-plane proposals are rejected with an explicit reason.
8. Validators, manifests, diagnostics, reports, shards, performance evidence,
   and visualization all use the selected version and reject mixed or unknown
   schemas.
9. Same input, config version, and seed produce exactly equal semantic
   manifests.
10. `dataset/court=v3` uses the exact full-half-turn mapping on the positive-Y
    side and identity on the negative-Y side. It rejects the court mid-plane,
    non-finite geometry, or an invalid inventory, while finite lateral views do
    not depend on projected left/right order.
11. V2 artifacts and checkpoints are rejected by V3 readers and target-bundle
    checks; migration requires regeneration and retraining.

## Explicit scale generation profile

`generate_court_scale` is a separate **train-only** generation entrypoint. It does
not change the `v1/v2/v3/sfm_bounded` presets or the scene's fixed
`datasets/court` owner. Production and comparison recipes are selected explicitly;
neither has an implicit method or quality-policy fallback.

### Production recipe and downstream entrypoint

The production recipe agreed in #1343 is **`stratified_counter_production_v1`**,
stored in
[`scale_stratified_counter_production_v1.json`](../../configs/dataset/court/scale_stratified_counter_production_v1.json).
It selects `stratified_counter` alone and pins:

- **D2:** captured-camera hull expansion 0%, inward margin 0.5 m, HFOV layers
  75/92.5/110 degrees.
- **D5:** target-court **renderer-visible KP >= 4 AND ROI hole fraction <= 0.01**.
  A hole is `alpha < 0.01 OR depth <= 0` inside the clipped target-court ROI.
  An absent ROI (`roi_hole_fraction=null`) fails the gate, including when other
  courts have visible KP. Alpha-below-0.9 area remains diagnostic only.
  Target-KP-zero negatives are rejected. Rejected candidates are never refilled.
- **D6:** stable-point-ID-derived uniform-area disk jitter, radius 1 m.

The recipe also explicitly specifies target heights, resolution and arc precision.
Its SHA-256 covers the **complete** name/version, method, geometry, jitter rule
and QA policy. Production planning rejects changes to D2/D5/D6, missing fields,
other methods, and comparison geometry/quality/method overrides.

#1355 should freeze its requests with this entrypoint (the n/k values below are
a small verification condition, not the main experiment size):

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale plan \
  --scenes-root /path/to/scenes --scenes B00 B01 B02 B03 \
  --production-recipe src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json \
  --output-root tmp/court-scale-production \
  --generation-seed 42 --trajectory-count 6 --density-level 0
```

`planned-requests.json` lists the frozen production request. The render command
below consumes that exact request. A distributed caller can instead load
`scale.recipes.ProductionRecipe.from_dict`, pass its `method/geometry` to
`persist_candidate_plan`, and call `freeze_request(...,
quality=recipe.quality, production_recipe=recipe, renderer=...)`. Both entrypoints
check that every candidate plan and quality policy match the production recipe.
`execute_request(root, request_path, render_callback)` owns the same acceptance
and publication contract; #1355 owns Colab sharding and Drive durability.

D5 is versioned as `target_kp4_roi_holes_1pct_v1` in
`court_scale_quality_policy_v2`. Production requests and dataset versions use
their explicit `v2` schemas, with the complete `production_recipe`, its
`recipe_sha256`, the separate `geometry_recipe_sha256`, `quality_rule_version`
and `quality_policy_sha256`. The latter two fields also bind each v2 QA ledger
and cache completion; each rendered sample's v2 diagnostics pins `rule_version`
and the policy hash. D5 failures become `qa_rejected` with explicit reasons and
are absent from accepted membership and JPEG stores. Existing comparison QA is
not production evidence: its schema, policy and cache key differ. RGB/alpha/depth
must be evaluated under D5 explicitly; JPEG-only legacy caches cannot be
re-certified from their old diagnostics. QA does not change cameras or point IDs.

The selected method rarely shows a whole court in the preview. #1356 must check
real-image metrics by court framing before fixing main-experiment conditions;
the decision and its rationale remain in #1343.

### Comparison recipes

Select every comparison method explicitly:
`grid_budget_replay`, `stratified_counter`, `sobol`, `capture_local`, or
`open_bezier`. Unknown methods and exhausted finite sequences fail; rejected
candidates keep their original ordinal and are never replaced to reach a quota.
The five methods are defined by `scale/strategies.py` and remain available for
Court Review comparisons. Only the named production recipe above is the
main-experiment authority.

`scale_geometry.json` owns the explicit geometry/arc settings for comparison
previews. Saved grid replay
retains the source config's footprint, HFOV and target-height distribution.
`scale_quality.json` keeps the existing all-court geometry/renderer gates and
records target KP count and clipped target ROI alpha/depth holes as diagnostics.
New target-KP/ROI hard thresholds default to `null`; selecting them creates an
explicit new quality recipe. QA never changes camera IDs or geometry.

The geometry identity uses versioned sorted UTF-8 canonical JSON, round-trippable
binary64 numbers, full SHA-256, reduced rational arc fractions and field-local
PCG64/SeedSequence randomness. NumPy/SciPy versions, Sobol bits/directions
implementation and distributions are pinned in each plan. A trajectory's frozen
arc table and base count `N0` are independent of n, k, worker and shard counts;
level k has exactly `N0*2**k` half-open points. Both open and closed paths exclude
q=1 as a camera; the Review curve can display the open endpoint. Point IDs omit
birth level, storage order, frame index and QA. Curve/view fingerprints diagnose
cyclic/orientation and open-path reverse duplicates (binary64 canonical geometry
rounded to 12 decimals); later duplicates are explicitly rejected. Near points
are retained. D2/D6 variants carry a separate `comparison_pair_id`; different
methods never claim index correspondence.

Freeze the complete request before queueing any GPU work (run from this
worktree; choose a fresh output root outside the read-only scenes root):

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale plan \
  --scenes-root /path/to/scenes --scenes B00 B01 B02 B03 \
  --methods grid_budget_replay stratified_counter sobol capture_local open_bezier \
  --output-root tmp/court-scale-preview \
  --recipe src/synthetic_data_generation/configs/dataset/court/scale_geometry.json \
  --quality src/synthetic_data_generation/configs/dataset/court/scale_quality.json \
  --generation-seed 42 --trajectory-count 6 --density-level 0 \
  --method-prefix capture_local=16 --method-prefix open_bezier=4 --method-density open_bezier=1 \
  --scene-prefix capture_local:B01=80 --scene-prefix capture_local:B02=157 \
  --scene-prefix capture_local:B03=153 \
  --legacy-hull-center B01=false
```

The B01 option explicitly adds the missing legacy
`sfm_complex_center_on_hull=false` to an in-memory copy and records the addition;
it neither rewrites the saved config nor supplies an implicit default. Do not
pass it for an input that already declares this field. Prefixes are fixed before
rendering. `planned-requests.json` lists the frozen request paths. Queue the
following command using the shared [training queue](../../../../.agents/skills/training-queue/SKILL.md):

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.generate_court_scale render \
  --output-root /absolute/path/to/tmp/court-scale-preview \
  --request /absolute/path/to/requests/HASH.json ... --batch-size 1
```

The GPU entrypoint requires the queue's `TENNIS_RUN_ID`. NHT's GPU microbatch is
explicitly 1 by default (local preview allows 1–4); one validated RGB/alpha/depth
payload is evaluated and JPEG encoded at a time. Rendering uses the public NHT
file boundary and checks pinned scene assets, launcher/runtime sources and
Court generator sources before running. No assets or source images are modified.
Failures leave a diagnostic staging directory and prevent version publication;
this local entrypoint does not implement automatic retry or Drive durability.
Those orchestration responsibilities remain with the distributed caller.

### Dataset-version consumer contract

A version is an explicit `*.dataset.json` under the output root. Its
explicit `schema=court_scale_dataset_version_v1` (comparison) or
`court_scale_dataset_version_v2` (production), content-derived `dataset_version`,
`request_path/sha256`, aggregate `plan_sha256`, `method`, `recipe_sha256`,
`generation_seed`, `split=train`, and `scenes[]` pin the complete condition.
Every scene includes its asset and candidate-plan references/hashes,
`stores[]`, `accepted_ids_path/sha256`, render-cache completion references,
counts, rejection reasons and the number of empty trajectories.
In v1, `recipe_sha256` identifies geometry only; in v2 it identifies the complete
production recipe as described above. Readers dispatch by schema without
reinterpreting historical comparison versions or their QA records.

Candidate plans stream one trajectory/table at a time into hash-pinned `curves/`
files (`trajectory_storage=curve_files_v1`); the plan header retains references
only. Planning and strict rereading do not retain all dense arc tables or sparse
projections in RAM.

All references are relative to **the version file's parent (the explicit output
root)**. Absolute paths, `..` and escaping symlinks are rejected. Each store
entry contains `path`, `dataset_sha256`, `index_sha256`; training's minimum input
is the explicit list `[version.parent / store['path'] for scene in
version['scenes'] for store in scene['stores']]`. There is no latest-version or
scene-owner discovery. Every row of these stores belongs to that condition's
accepted membership, exactly once; the consumer must not mix roots from other
versions. `scale.publication.load_dataset_version(path, verify_images=True)`
validates membership, counts, hashes, the pinned RNG/Sobol runtime, the explicit
QA policy, ledger/store record agreement, v3 cameras/targets/KP and all JPEGs
before training. `verify_images=False` omits JPEG decoding for bounded Review reads.

Stores use the existing `court_image_store_v1` JPEG shards plus sparse index and
`canonical_court_dataset_v3` geometry. Metadata explicitly declares
`scale_schema=court_scale_train_store_v1`; its RNG/trajectory and train-only
contract requires the scale validator, rather than the legacy index-jitter /
three-split validator. Sparse `scale` and `qa` records have their own schemas;
v3 KP14, metric camera pose, target binding and on-the-fly dense/pose teachers
retain their meanings. No dense teacher rasters are saved.

Shared stores are immutable. Each cache shard covers a fixed 64-candidate chunk
of one trajectory's density **birth level**, independent of requested n/k and
worker completion order. Common IDs reference the same JPEG and sparse row;
expanding a condition renders only new chunks. A shard's fingerprint includes
assets, renderer, QA recipe and exact planned points. Empty shards retain a QA
ledger without fabricating an image store. Identical completed requests reuse
the same data version. Reordered completion yields the same packed row order.

Counts distinguish `logical_records`, `candidate_unique`,
`geometry_pass/rejected`, `rendered_unique`, `render_cache_reused`,
`qa_accepted/rejected`, `pending/failed`. Completed versions require
`candidate_unique=geometry_pass+geometry_rejected`,
`geometry_pass=qa_accepted+qa_rejected=rendered_unique+render_cache_reused`, and
`pending=failed=0`. All accepted rows are train; zero-accepted scene conditions
fail and zero-accepted trajectories remain diagnostic. Counts are measured,
without quota selection, refill or synthetic validation/test sets.
