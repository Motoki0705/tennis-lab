# Saved-scene bundle for player views

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This CPU exporter snapshots a saved SceneResult v2 for the Blender and 3DGS
player-view backends. It uses the existing mesh placement, PC2 writer and court
geometry. Head cameras and rendering are handled by downstream consumers.

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

The exporter preserves the inputs required for head rotation and mesh-based
head position. Defining the head camera, face-forward axis, offset and validity
policy belongs to the head-camera consumer.
