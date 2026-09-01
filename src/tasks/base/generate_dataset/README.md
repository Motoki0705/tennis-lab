# BLCS / PLCS CourtKP20 contract

This directory is the single source of truth for the CourtKP20 semantics shared
by standalone BLCS and PLCS dataset generation, model samples, inference, and
checkpoints. Task READMEs should link here and describe only task-specific array
shapes and targets.

## Version selection

The public Hydra selector is `court_keypoints=physical_v1|camera_view_v2`.
`physical_v1` remains the default. The selector resolves to exact semantic and
model-target IDs; IDs are never inferred from a 20- or 14-point shape.

| Selector | Semantic contract ID | Model target-frame ID |
|---|---|---|
| `physical_v1` | `physical_courtkp20_v1` | `physical_court_v1` |
| `camera_view_v2` | `camera_view_courtkp20_rzpi_v1` | `reference_camera_court_rzpi_v1` |

The fixed isotropic `court_coordinate_normalization` contract controls numeric
scale only. It is a separate metadata field from the CourtKP contract, so every
supported CourtKP selector uses the same current normalization identity without
sharing its version axis. Camera side and reference transforms are determined
in physical metres before normalization. The numeric normalization contract is
documented once in `src/utils/README.md`.

## Camera-local disk semantics

`CameraProjector.project_court_keypoints()` continues to return physical
CourtKP20 order. A writer applies the immutable `semantic_to_physical` mapping
once to both UV and visibility. Ball/player UV, physical 3-D targets, and saved
camera `C/R` remain in the physical court frame.

- `physical_v1` always uses identity ordering and identity rotation.
- For `camera_view_v2`, finite `C` with `C_y < -1e-6 m` uses identity.
- For `camera_view_v2`, finite `C` with `C_y > +1e-6 m` uses exact `Rz(pi)` and
  `(3,2,1,0,7,6,5,4,11,10,9,8,13,12,14,17,18,15,16,19)`.
- Non-finite centres and inclusive `abs(C_y) <= 1e-6 m` are rejected. Invalid,
  non-bijective, or non-involutive mappings are errors, never side fallbacks.

The first 14 entries agree with the corrected camera-view ground contract from
#782/#788. `OPPOSITE_COURT_END_INDEX` belongs to the legacy Synthetic Court
Y-reflection behavior and remains available, but it is not used for this
right-handed contract. No Synthetic Court `Court`/`SceneCamera` types are part
of this package.

New artifacts store exact root and scene `court_keypoints` records plus ordered
scene `court_keypoint_views` records. The root/scene dataset schema ID,
CourtKP20 contract, target-frame ID, point count, disk coordinate frame, stable
camera ID, finite physical camera centre, permutation, and canonical rotation
must all parse and match exactly. Missing, unknown, mixed, malformed, or
cross-level mismatched records fail before arrays are consumed.

## Model reference semantics

Camera-view v2 selects exactly one stable camera ID after the view subset is
known. Its local index is resolved independently of view order. With per-camera
semantic-to-physical mappings `H_v` and reference `H_r`, each disk Court channel
is reordered by `H_v^-1 o H_r` before a standard consumer keeps 20 points or a
tracking consumer keeps the aligned first 14.

The reference rotation `S_r` is then applied consistently:

```text
point_ref   = S_r point_phys
vector_ref  = S_r vector_phys
C_ref       = S_r C_phys
R_cam<-ref  = R_cam<-phys S_r^T
```

PLCS heading `(cos(yaw), sin(yaw))` and court-space world joints use the same
proper rotation. Object UV/visibility and player-local `canonical_pose_3d` do
not change. `CourtReferenceFrameProvenance` records the stable reference ID,
local index, target frame, and validated forward/inverse matrices so prediction,
metric, visualization, and integration consumers can restore physical metres.

## Public API for task implementations

Import from `src.tasks.base.generate_dataset`:

- `resolve_court_keypoint_contract()` at config composition.
- `build_court_view_record()` after a physical camera proposal is accepted, and
  `apply_court_view_record()` exactly once for Court UV and visibility.
- `CourtKeypointArtifactMetadata.from_contract()`,
  `inject_court_keypoint_artifact_metadata()`, and
  `inject_scene_court_keypoint_metadata()` in writers.
- `validate_dataset_court_keypoint_contract[_documents]()` before readers index
  scene payloads.
- `resolve_reference_court_view()`,
  `align_court_keypoints_to_reference()`, and
  `build_reference_frame_provenance()` after selecting views.
- `court_points_*`, `court_vectors_*`, `court_headings_*`,
  `court_world_joints_*`, and `camera_extrinsics_*` for reversible transforms.

Import checkpoint/direct-runtime helpers from `src.tasks.base.model_io`:
`write_model_artifact_court_keypoint_contract()`,
`validate_model_artifact_court_keypoint_contract()`, and
`resolve_model_artifact_court_keypoint_contract()`. Metadata-free input can be
resolved only when a known legacy caller explicitly supplies the canonical
`physical_v1` runtime. Camera-view v2 never treats identity or shape as proof.

Camera-view v2 datasets and checkpoints are separate artifacts. Existing v1
weights are not auto-remapped, dual-written, or upgraded in place. Changing
semantics requires dataset regeneration and model retraining; rollback is an
explicit return to `physical_v1` with matching v1 artifacts.

## Human-readable dataset samples

Generated PLCS and BLCS datasets keep machine payloads under `scenes/` and a
small human-inspection surface beside it:

```text
<dataset>/
├── scenes/
└── samples/
    ├── *.gif
    └── manifest.json
```

`src.tasks.{plcs,blcs}.scripts.generate_dataset_samples` are the canonical
entry points. Their default configs cover the four physical-v1 production
layouts plus `multi_object_camera_view_v2`. Each dataset contributes one scene
from every cell of a 3×3 stratification rather than the first N scene IDs.

- PLCS single: motion category × within-category frame-count tercile.
- BLCS single: deuce/ad/behind-baseline first-hit region × within-region
  frame-count tercile.
- PLCS/BLCS multi: track-count tercile × within-band total-active-frame
  tercile.

Visibility and a task-owned auxiliary statistic are offset quantile tie-breaks,
and the rendered camera itself is selected at a low/mid/high visibility rank.
GIF timelines include both endpoints and at most 120 evenly spaced frames;
playback FPS is bounded while approximately preserving source duration.
`manifest.json` records every threshold, stratum population, scene metric,
camera choice, and exact source-frame index. Missing strata, malformed timing,
and invisible-only camera sets are errors rather than fallback selections.

## Reference-camera track-query model contract

CourtKP and track-query RoPE are independently versioned. Only these exact
combinations are valid:

| Model contract | Court / target contract | Spatial coordinates | Forward |
|---|---|---|---|
| `time_camera_role_v1` | `physical_courtkp20_v1` / `physical_court_v1` | query `(t,0,0)`, object `(t,v+1,1)` | the original five tensors |
| `time_camera_reference_selector_v1` | `camera_view_courtkp20_rzpi_v1` / `reference_camera_court_rzpi_v1` | query `(t,0,0)`; reference objects `(t,v+1,0)`; other objects `(t,v+1,1)` | the same five tensors plus required `reference_view_index: int64[B]` |

The v2 selector is clip-level and is repeated over every time and object token;
it does not depend on visibility. Query-first flattening, time order, local
camera coordinate `v+1`, and the compressed spatial width are unchanged.
`rope_dim` must be even and at least 6 so the generic round-robin allocator
assigns a pair to time, camera, and selector.

Each v2 sample carries one typed selection with canonical string IDs,
`reference_view_index`, `view_camera_ids`, `reference_camera_id`,
`reference_from_physical`, and its transpose `physical_from_reference`.
Integer IDs are collision-free ranks in the complete lexicographically ordered
scene ID table; `-1` is reserved only for padded `view_camera_ids`. Missing,
unknown, mixed, out-of-range, padded, or identity/index-inconsistent records are
errors. Checkpoints persist Court, target-frame, RoPE, and selector markers as
independent fields and require the canonical
`fixed_query_track_compressed_v1` architecture marker. Metadata-free and
pre-promotion checkpoints are rejected. Matching tensor shapes never authorize
a semantic or architecture migration.

Training chooses the reference from the selected valid views using the
caller-owned seeded worker RNG after subset selection; the candidate IDs are
sorted before the draw, so view permutation does not change the random choice.
Validation and test use the stable `data.evaluation_reference_camera_id`.
Direct inference and prediction visualization require an explicit stable
`reference_camera_id`; multi-view code never defaults to local index zero or a
sorted/first camera. The selected ID, local index, complete ID table,
forward/inverse transforms, target-frame marker, RoPE marker, and selector mode
are serialized with predictions so downstream consumers can restore physical
court coordinates exactly.
