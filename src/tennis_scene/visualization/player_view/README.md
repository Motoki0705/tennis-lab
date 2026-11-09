# Saved-scene bundle for player views

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This CPU exporter snapshots a saved SceneResult v2 for the Blender and 3DGS
player-view backends. It uses the existing mesh placement, PC2 writer and court
geometry. The head-camera exporter below consumes only that bundle; rendering
is handled by downstream consumers.

```bash
.venv/bin/python -m src.tennis_scene.visualization.player_view \
  --input /absolute/path/to/store \
  --output outputs/tennis_scene/visualize/player_view/example/bundle \
  --smpl-faces /absolute/path/to/body_models/smplh/neutral/model.npz \
  --smpl-joint-regressor /absolute/path/to/body_models/gvhmr/smpl_neutral_J_regressor.pt \
  --smplx-model /absolute/path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
  --frames 10 110 --player-ids 7 42
```

`--input` accepts an NPZ plus its mandatory metadata sidecar, a store directory,
or `scene.json`. Store resolution uses `indexed_scene_path`: a completed export,
its file checksums and current component lineage are required. Missing or stale
exports are rejected. Source videos, camera reprojection checks and inference are
not needed. The existing [reveal exporter](../blender/export/README.md) retains
its plate and reprojection checks.

All asset paths are explicit. The faces asset supplies triangle topology; the
24-row SMPL regressor must match the saved mesh. The neutral SMPL-X NPZ supplies
the skeleton for the 21 body-pose joints. Only numeric model fields are read;
no body model is initialized. The bundle contains the derived numeric assets,
so consumers need no original model files or source store.

Omitting `--frames` selects the full scene; otherwise it is the half-open source
interval `[start,stop)`. Omitting `--player-ids` keeps all saved IDs in source
order. A selection preserves the requested order and can contain any positive
number of players. IDs must be unique and known. An existing output is rejected.
Validation precedes writing, and a complete temporary directory is published
only after success. Failed writes remove that temporary output. Inputs are read
only.

## Schema

`manifest.json` has schema `tennis_scene_player_view_bundle_v1` and records:

- `players`: bundle `index`, original `source_index`, `player_id`, PC2 filename,
  valid-frame count and recorded per-person `scale` (dimensionless).
- `frames`: source `start`, exclusive `stop`, and `source_num_frames`;
  `num_frames` and `fps` retain native timing.
- `arrays`: each array's relative `path`, `shape` and `dtype`.
- `coordinates`, `smpl`, `pc2`, `validity`, and standard `court` geometry.
- Requested `config`, input archive/metadata digests, and, for a store, its index
  digest and adopted export references. `source_metadata.json` preserves the
  source metadata including placement settings and diagnostics.

In `config`, `null` selects the full input. `frames` and `players` record the
resolved interval and player selection.

Let `P` be the selected player count, `F=stop-start`, `V` the mesh vertex count,
and `N` the number of faces. Array keys match their `.npy` basenames.

| Key/file | Shape | Type / units / meaning |
|---|---|---|
| `source_frames` | F | int64, original zero-based frame numbers |
| `source_time_seconds` | F | float64, original frame / source FPS, seconds |
| `player_<index>.pc2` | F,V,3 | float32, placed mesh, metres |
| `player_observed`, `player_valid`, `player_heading_valid`, `player_smpl_valid` | P,F | bool, unchanged v2 validity |
| `player_kp_3d_vis` | P,F,17 | bool, unchanged v2 joint validity |
| `player_rejection_code`, `player_kp_3d_rejection_code` | P,F / P,F,17 | uint8, unchanged v2 reasons |
| `ball_positions`, `ball_3d_valid`, `ball_rejection_code` | F,3 / F / F | float32 metres / bool / uint8 |
| `smpl_body_pose` | P,F,63 | float32, parent-relative axis-angle, radians |
| `smpl_betas` | P,10 | float32, dimensionless shape coefficients |
| `smpl_global_orient` | P,F,3 | float32, v2 renderer compensation, axis-angle radians |
| `player_position` | P,F,3 | float32, court position of SMPL joint0, metres; `smpl.parameters.transl` |
| `player_yaw` | P,F | float32, court +Z yaw, radians |
| `smpl_root_local` | P,F,3 | float32, row-0 regressed root of saved local mesh, metres |
| `body_to_court_rotation` | P,F,3,3 | float32, decoded rotation, column-vector convention |
| `smpl_faces`, `smpl_joint_regressor` | N,3 / 24,V | int32 indices / float32 weights |
| `smplx_parents` | 22 | int32, skeleton parents, root=-1 |
| `smplx_rest_joints`, `smplx_joint_shapedirs` | 22,3 / 22,3,10 | float32, unit-scale SMPL-X rest joints in metres and shape basis |

PC2 starts at cache frame 0 with sample rate 1: sample `i` corresponds to
`source_frames[i]`. There is no resampling or gap filling. Hide a mesh/ball when
its mask is false; zero coordinates do not imply validity. Invalid PC2/ball
samples are zero. Rotation/root arrays are zero when mesh validity is false.
Other scene arrays retain their own original masks; a valid player root does
not imply a valid mesh.

## Placement and the head-pose boundary

World coordinates are right-handed court metres: XY ground, +Z up, X across
the court width and Y along its length, with the origin below the net centre. Saved local
mesh is canonical posed SMPL geometry with the fitted scale **already applied**.
For column vectors, the existing renderer applies:

```text
A = [[1,0,0], [0,0,-1], [0,1,0]]    # SMPL Y-up -> court Z-up
R = Rz(player_yaw) @ A @ Rodrigues(smpl_global_orient).T
world_vertex = R @ (saved_local_vertex - smpl_root_local) + player_position
```

`body_to_court_rotation` stores `R`. `smpl_global_orient` is a compensation
rotation, not the original GVHMR incam root rotation. `player_position` is the
court root translation, not the original incam `transl`. Raw incam parameters
are not present in SceneResult v2 and are not synthesized.

For downstream head orientation, compose `R` with the parent-relative
`smpl_body_pose` rotations along the saved skeleton. `smpl.body_joint_names`
names root plus 21 joints; pose triple `j-1` controls joint `j`. The skeleton is
checked against the shared named body skeleton. Shaped rest joints are
`smplx_rest_joints + einsum('jck,k->jc', smplx_joint_shapedirs, betas)`.

Head position can be regressed directly from the placed mesh using the included
24-row SMPL regressor. SMPL-X FK joints and the converted SMPL mesh have different
topologies; their positions should not be assumed equal. Use the saved scale
for unit-scale skeletal/offset calculations, and do not scale the PC2 again.
A player with valid mesh frames must have a recorded positive placement scale.
An entirely invalid player may have `scale: null`; it remains invalid.

## Head cameras

Generate the authoritative camera sequence once, then give the same saved poses
to Blender and 3DGS. This CPU command reads only a scene bundle v1; no original
store, video, body model, or inference is required. All bundled players retain
their original IDs and source timeline. Output must be new and outside the input
bundle; publication is atomic and failures remove temporary output.

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_player_view_cameras \
  --bundle outputs/tennis_scene/visualize/player_view/example/bundle \
  --output outputs/tennis_scene/visualize/player_view/example/cameras \
  --width 1280 --height 720 --horizontal-fov-degrees 90 \
  --face-offset-head-m 0 0.08 0.12 --near-m 0.05 --far-m 100
```

### Configuration and orientation

The command shows the prototype defaults. The face offset is in **unit-scale
SMPL head axes** (metres before person scale), relative to the regressed head
joint. Near/far are camera +Z depths in court metres, independent of person
scale. The manifest records all values. Pixels are square, skew is zero, the
principal point is `(width/2,height/2)`, and `fx=fy=width/(2*tan(horizontal_fov/2))`.
There is no lens distortion.

The named skeleton and mesh joint order are checked before resolving `head`.
Pose triple `j-1` controls joint `j`; the ordered chain is
`spine1 → spine2 → spine3 → neck → head`. For column vectors:

```text
H = body_to_court_rotation @ R_spine1 @ R_spine2 @ R_spine3 @ R_neck @ R_head
head_position = smpl_joint_regressor[mesh_head_row] @ placed_PC2_vertices
camera_center = head_position + H @ (player_scale * face_offset_head_m)
camera_to_court[:3,:3] = H @ diag(-1,-1,1)
camera_to_court[:3,3] = camera_center
```

SMPL/SMPL-X neutral head axes are +Y up, +Z face-forward and +X towards the
subject's left. Thus OpenCV camera +X right, +Y down, +Z forward correspond to
head −X, −Y, +Z. The face direction was checked against the neutral models'
nose/eye/ear landmarks using the upstream [vertex IDs](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py).
The saved 63-value pose is the SMPL-X body articulation used before the linear
SMPL-X→SMPL topology conversion; it is not a SMPL 69-value pose. Rotational FK
follows that skeleton; position follows the **converted, placed SMPL mesh**.
Do not substitute SMPL-X FK position, recenter the PC2, apply scale to it again,
or interpret the encoded global orientation as the head rotation.

Each original frame has a static pinhole. Roll is preserved, with no additional
smoothing, interpolation, stabilization, or ball targeting. Looking outside the
court or losing the opponent/ball from view is allowed by the head pose.

### Camera output schema

`manifest.json` has schema `tennis_scene_player_view_head_cameras_v1`. It records
the source bundle reference and consumed-file digests, original frame interval,
FPS, ID/index correspondence, person scale, valid counts and reason counts,
configuration, axes, projection, head method and missing-value policy.
Each `players[index].path` is a portable NPZ with these arrays (`F` frames):

| Key | Shape | Type / meaning |
|---|---|---|
| `K` | 3,3 | float64, constant intrinsic matrix, pixels |
| `camera_to_court` | F,4,4 | float64, homogeneous OpenCV camera→court transform, translation in metres |
| `camera_valid` | F | bool, required before using a pose |
| `camera_rejection_code` | F | uint8, codes below |
| `head_position` | F,3 | float64, head regressed from placed mesh, metres |
| `source_frames` | F | int64, original zero-based frame numbers |
| `source_time_seconds` | F | float64, original frame/FPS |
| `source_player_rejection_code` | F | uint8, unchanged source player reason, independent of camera reason |

For a valid pose `T`, `X_camera=T[:3,:3].T @ (X_court-T[:3,3])`, then project with
`K`; +Z is depth. Screen clipping and near/far are rendering choices using the
saved configuration, not changes to the poses. Invalid transforms and head
positions are **all NaN**, never identity/zero poses; K stays defined.

Codes in precedence order are `0 VALID`, `1 MESH_INVALID`,
`2 HEAD_POSE_NONFINITE`, `3 PLACEMENT_ROTATION_INVALID`,
`4 HEAD_POSITION_NONFINITE`. A missing mesh wins over its zero-filled numeric
values. Only finite proper placement rotations and finite head-chain poses can
define a camera. Inconsistent skeletons, regressors or timeline dimensions
reject the export. A wholly missing player may have null scale and remains
wholly invalid. There is no yaw, neighbouring-frame, or fixed-camera fallback.

### Reading cameras

```python
from pathlib import Path
from src.tennis_scene.visualization.player_view.camera_bundle import load_head_cameras
from src.tennis_scene.visualization.blender.export.camera import to_blender_camera

saved = load_head_cameras(Path("outputs/tennis_scene/visualize/player_view/example/cameras"), player_id=7)
index = 0  # cache index; original frame is saved.source_frames[index]
if saved.series.camera_valid[index]:
    camera = saved.series.pinhole(index)
    blender_pose = to_blender_camera(camera, sensor_width_mm=36.0)
```

The reader validates the saved transforms and masks and needs no source bundle.
`pinhole()` raises for an invalid sample; renderers must retain its time slot
and show a missing-camera frame. Blender uses camera rotation
`T[:3,:3] @ diag(1,-1,-1)` and location `T[:3,3]`, with the saved resolution,
intrinsics and clipping. Neither backend should recompute head kinematics.

## Optional camera stabilization

Keep the raw C2 export and derive a separate camera bundle before rendering:

```bash
.venv/bin/python -m src.tennis_scene.scripts.stabilize_player_view_cameras \
  --cameras outputs/tennis_scene/visualize/player_view/example/cameras \
  --output outputs/tennis_scene/visualize/player_view/example/cameras-smoothed \
  --sigma-seconds 0.06
```

The strength is required; `--sigma-seconds 0` explicitly copies the raw poses.
The rotational filter is a Gaussian-weighted Markley quaternion mean, with
`ceil(truncate_sigma * sigma_seconds * fps)` samples on either side
(`--truncate-sigma 3` by default). It uses future frames offline. Faster turns
can be attenuated or begin earlier, so compare the actual videos before choosing
a strength. This is a display camera, not a correction to the source body pose.

Only rotation changes. Camera centers, raw face offsets, head positions, roll
without horizon locking, intrinsics, IDs, times, masks and rejection codes keep
their input meanings. Missing poses stay NaN. Each contiguous valid run is
filtered separately. At its edges, only existing samples in that run contribute,
with normalized weights; a singleton retains its orientation. There is no gap
filling, padding or alternative-pose fallback. A nonunique quaternion mean fails
the export. An already stabilized input is rejected.

The geometric schema remains `tennis_scene_player_view_head_cameras_v1`. The
manifest records the versioned algorithm, complete settings, source-camera
digests and original scene provenance, numerical implementation, run intervals
and correction angles. `stabilization_support_frames` in each NPZ is int64 F×2
(left/right support counts; both zero when missing or strength zero). The reader
checks processing metadata and support. Consumers use this derived directory
with the same camera reader. Output publication is atomic and existing outputs
are refused.
