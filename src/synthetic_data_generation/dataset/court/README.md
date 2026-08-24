# Court dataset v1/v2/v3 contract

Status: implemented. This document is the canonical production contract for
all explicitly selected Court dataset versions. It is also the sole prose
authority for the camera-view KP14 semantics and their migration policy.

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
`projection: null`. No `labels.json` is published for a rejected proposal.

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

The v2/v3 `dataset.json` and per-sample `labels.json` add the sample-level resolved
target-court record. `trajectory_groups[].target_court` is replaced in v2/v3 by
the explicit target-court policy. The v1 field layout remains unchanged.

The report adapter aggregates unique target bindings from v2/v3 samples instead
of trajectory groups. Semantic manifests, accepted records, rejected records,
and label files must agree exactly on each sample's target binding.

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
