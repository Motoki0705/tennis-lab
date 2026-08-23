# Exploration

- Issue: #786
- Attempt: 1
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Scope and Issue interpretation

This is a versioned normalised court-position contract, not a physical court/frame change. A single resolver must return `v1=(5.485,11.885,1.07)m` or `v2=(11.885,11.885,11.885)m`, reject unknown versions, and support `(...,3)` normalize/denormalize. Position and normalized velocity use the same per-axis scale; world position stays m and world velocity m/s. Court geometry, UV/camera coordinates, yaw, and local/root-relative canonical poses remain unchanged.

Current `src/utils/schema/court.py` globally exposes only the v1 tuple, so all consumers silently mean v1. The implementation must make runtime selection explicit through BLCS/PLCS/SLCS generation, training, evaluation and inference, bind generated datasets/checkpoints to the resolved version/scale, preserve default v1 numeric behavior, and fail before use on missing/unknown/mixed/mismatched contracts. Metadata-free artifacts are legacy v1 only under explicit v1 runtime; no value/shape inference or automatic conversion.

Smallest safe topology: typed resolver/value object and conversion helpers in `src/utils/schema/court.py`; reusable base dataset/checkpoint compatibility validation; one Hydra normalization config included by each task root; and explicit contract injection into task constructors. Keep `COURT_COORD_SCALE_*` only as documented v1 aliases during migration, never as a mutable active selection.

## Relevant files and symbols

| Area | Facts / symbols | Impact |
|---|---|---|
| Shared schema | `src/utils/schema/court.py`: physical dimensions, `court_keypoints_3d`, fixed `COURT_COORD_SCALE_X/Y/Z/XYZ`. | Sole resolver/API; physical values and CourtKP20 remain metres. |
| Court pose | `src/utils/geometry/court_pose.py::{court_position_to_world_translation,canonical_pose_to_world_pose,world_pose_to_canonical_pose}` import fixed scales. | Inject scale only for position translation; canonical pose stays metres. |
| Base storage | `src/tasks/base/data/{dataset_writer,scene_dataset}.py`, `src/utils/data/scene_io.py`. | Add root/scene metadata validation before dataset index/payload use. |
| BLCS generation/data | `generate_dataset/{scene_generator,simulation/ball_physics,io/dataset_io}.py`, `data/{dataset,tracking_dataset}.py`. `ball_pos_world` [m], `ball_pos_norm`, `ball_vel_world` [m/s]. | Versioned generation, artifact naming, metadata and loader guard. |
| BLCS model paths | `models/components/differentiable_projection.py`, `training/{losses,metrics,tracking_losses,tracking_matching,tracking_metrics}.py`, `inference/{predictor,tracking_predictor}.py`, `model_io/checkpoints.py`. | Standard and tracking scale/gravity/metric/predictor/checkpoint propagation. |
| PLCS generation/data | `generate_dataset/{scene_generator,io/dataset_io}.py`, `data/{targets,dataset,tracking_dataset,types}.py`. | Contract only `position`; `canonical_pose_3d` is local root-relative metres. |
| PLCS model/render | `training/{losses,metrics,tracking_losses,tracking_matching,tracking_metrics}.py`, `inference/*`, `visualization/rendering/scene_renderer.py::_world_positions`, `model_io/*`. | Standard/tracking decoding, metre metrics, matching and render conversion must agree. |
| SLCS | `data/dataset.py::load_clip_arrays`, `data/types.py`, `model_io/adapter.py::to_physical`, `training/metrics.py`, `evaluation/evaluate.py`, `inference/predictor.py`. | Version every meter↔normalized conversion and scalar uncertainty conversion. |
| Integration | `src/tennis_scene/schema.py::SceneResult`, `pipeline/orchestrator.py`. | Public `player_position`/`ball_3d` remain court/world [m]. |
| Config roots | `src/tasks/{blcs,plcs,slcs}/configs/{train,generate_dataset,visualize,evaluate,predict_clip}.yaml`, each `configuration.py`, each `scripts/`. | Explicit v1 default / v2 override in every applicable entrypoint. |

## Entry points and execution paths

- BLCS generation: `python -m src.tasks.blcs.scripts.generate_dataset -> parse_generation_run -> build_generator_config -> SceneGenerator/parallel workers -> BLCSDatasetWriter`. `SceneGenerator.generate_scene()` obtains `ball_pos_norm` from `BallPhysics.normalize_position(rally_result.trajectory)`. It writes `config.yaml`, root metadata/splits, then scene metadata/scalars/arrays.
- BLCS standard training: `scripts/train.py -> TrainingRuntimeConfig.from_config + validate_training_boundary -> BLCSTrainingRunner`; `BLCSLoss` defaults `height_scale=COURT_COORD_SCALE_Z`, `DifferentiableProjection` has a scale buffer, `BLCSPredictor` has an optional scale argument. Tracking has independent loss/matching/metric/predictor code; `BLCSTrackingPredictor` imports the global tuple directly.
- PLCS generation: `scripts/generate_dataset.py -> PLCSGenerationConfig -> SceneGenerator/PLCSDatasetWriter`. `_transform_motion_to_court()` divides court metres by fixed XYZ scales; `data.targets` reverses them. Multi-object generation composes the same normalized arrays.
- PLCS train: `scripts/train.py -> PLCSTrainingConfig -> PLCSTrainingRunner`; standard and tracking predictors independently decode. `SceneRenderer._world_positions()` directly multiplies by physical constants and can disagree with a migrated predictor unless supplied the contract.
- SLCS: `load_clip_arrays()` reads metre-valued `SceneResult` player/ball labels then divides by the fixed tuple. Train/evaluate/predict have separate Hydra roots. Evaluation and `SLCSMetrics`, and `SLCSModelIOAdapter.to_physical`, each use both the tuple and its mean for scalar position uncertainty.
- Integrated pipeline: `pipeline/orchestrator.py` places PLCS/BLCS outputs in `SceneResult`; its schema documents court coordinates. All task predictors therefore must denormalize before this boundary.
- Required operational commands are the three generation scripts and BLCS/PLCS/SLCS train/evaluate/predict/visualize scripts. AC-019 training must use the shared training queue, never direct concurrent GPU execution.

## Data, configuration, and interface contracts

### Existing facts

- BLCS/PLCS root `meta.json`, per-scene `meta.json`, `scalars.json`, and arrays have no normalization version, scale or units. `BaseDatasetWriter.save_meta_json()` owns root metadata; task writers own scene metadata. `SceneDatasetBase` indexes scene headers only and does not load/compare root metadata.
- BLCS persists `ball_pos_world` [m], `ball_pos_norm`, and `ball_vel_world` [m/s]. PLCS persists normalized `position`, `rotation`, and `canonical_pose_3d`; `PLCSSceneMeta` currently has no contract field. `canonical_pose_3d` is generated pelvis-root-relative/yaw-canonical in metres.
- BLCS explicitly requires checkpoint `hyper_parameters.config` in `model_io/checkpoints.py`; PLCS/SLCS use generic Lightning-module loading. No inspected checkpoint path independently saves or validates normalization metadata.

### Required mathematical/data contract

```text
position_norm = position_m / scale_xyz
position_m    = position_norm * scale_xyz
velocity_norm = velocity_m_per_s / scale_xyz
velocity_mps  = velocity_norm * scale_xyz
```

For v2: doubles sideline `±5.485/11.885 = ±0.4615…`, baseline `±1`, post top `1.07/11.885 = 0.0900…`; `court_keypoints_3d()` stays metre-valued. Both versions need `(...,3)` round-trip max error <= `1e-5m`.

Recommended artifact schema is the same object in root and every new scene: `court_coordinate_normalization: {version, scale_xyz, position_unit: "m", velocity_unit: "m/s"}`. Checkpoint metadata needs version/scale plus schema marker. This is integrity evidence; the resolver remains the only mathematical source. Central validation should exact-resolve runtime version/scale, require root and every scene to be complete/equal, allow all-missing only with explicit v1 runtime, and reject partial/unknown/mixed/mismatch/v2-missing before model/data use. Current data has no metadata, so root-plus-scene ownership is an implementation decision, not an established legacy format.

There is no inspected repository-wide common Hydra config root. Safest propagation is an identical shared config group/package selected before `_self_` in all BLCS/PLCS/SLCS roots (including tracking, evaluate, inference and visualization); typed configuration boundaries must require it, rather than synthesizing a Python default. Default composition must name `v1`. Persist resolved config in generator output/checkpoint and use version-qualified output names (e.g. `norm-v2`) without overwriting v1.

### Loss/gravity/matching/uncertainty facts

- `BLCSLoss` uses `ballistic_second_difference(gravity, dt, height_scale)`, which yields `-g*dt^2/scale_z`, but its default is fixed v1 Z. Tracking instead compares to literal `loss.gravity_target: -0.01` from `configs/loss/tracking.yaml`.
- BLCS standard defaults position axis weights to uniform (`null`); tracking config explicitly uses `[1,1,0.5]` and passes it to Hungarian matching. PLCS has separate tracking matching/loss. v2 defaults must remove scale-derived nonuniform weights while v1 retains its existing configuration.
- Existing Smooth L1 calls often use PyTorch default beta. AC-013 requires an explicit documented physical Huber transition for v2 (and a v1 compatibility decision), not merely isotropic v2 scale.
- SLCS scalar `log_b` presently converts by `exp(log_b) * mean(scale_xyz)`. That is exact only as a scalar convention, not an axiswise conversion under v1; v2 naturally uses `HALF_LENGTH`.

## Existing tests and fixtures

- `tests/unit/utils/schema/test_court.py` fixes physical CourtKP20; add resolver, unknown rejection, two-version round trips and v2 aspect-ratio tests. `tests/unit/utils/geometry/test_court_pose.py` already tests translation/canonical round trip; parameterize and assert canonical pose is not rescaled.
- BLCS: `tests/unit/tasks/blcs/training/{test_losses,test_tracking_losses,test_tracking_matching,test_tracking_metrics}.py`, `inference/{test_predictor,test_tracking_predictor}.py`, `model_io/test_checkpoints.py`, generation tests, `visualization/rendering/test_scene_renderer.py`; integration config/physics tests are under `tests/integration/tasks/blcs/`.
- PLCS: `tests/unit/tasks/plcs/data/{test_dataset,test_targets,test_types}.py`, training/predictor/tracking/renderer/model-io tests; generation coordinates are also covered in `tests/unit/synthetic_data_generation/dataset/plcs/test_coordinates.py` and `tests/integration/synthetic_data_generation/test_plcs_coordinate_contract.py`.
- SLCS: `tests/unit/tasks/slcs/data/test_dataset.py`, `model_io/test_adapter.py`, `inference/test_predictor.py`, training/model tests; integration smoke/config tests are `tests/integration/tasks/slcs/{test_training_smoke,test_training_config}.py`.
- Base/pipeline evidence locations include `tests/unit/utils/data/test_scene_io.py`, `tests/integration/tasks/base/test_model_io_lifecycle.py`, and `tests/unit/tennis_scene/pipeline/components/{test_blcs,test_plcs}.py`. Existing tests do not cover v2 or normalization metadata.
- AC-020 needs new bounded v1/v2 CPU fixtures proving dataset load -> forward -> loss -> metric -> physical decode -> BLCS projection / PLCS render. AC-018 needs generated versioned sample round-trip evidence; AC-019 requires queued run records and axis/aggregate metre comparisons, beyond unit tests.

## Invariants and compatibility constraints

- Court dimensions, axes/origin, CourtKP20, camera/world metres and UV conventions do not change.
- v1 is default. Metadata-free `data/blcs_broadcast`/`data/plcs_broadcast` and old checkpoints are v1-only when runtime explicitly selects v1; v2 rejects before interpretation.
- Runtime, root, every scene, and checkpoint must agree on both version and scale; never infer, silently fallback, or auto-convert.
- `ball_pos_world` remains m and `ball_vel_world` m/s. Any normalized velocity needs the same selected scale and accurate unit metadata.
- PLCS canonical/local pose, world joints, rotation/yaw and excluded production metre arrays are never rescaled; only global translation/position is.
- `SceneResult.player_position`/`ball_3d` remain metres. Metadata can state provenance but must not alter public values.
- v2 BLCS/PLCS default position loss and Hungarian cost have no inherited axis-scale compensation; v1 regression behavior/weights remain.

## Risks and likely impact radius

1. Many fixed imports/default arguments mean mutating a global tuple or active-version singleton can silently corrupt consumers; explicit injection is required.
2. Header-only dataset indexing can skip validating scenes not sampled in a short test. Validate every split scene when constructing/indexing the dataset.
3. Checkpoint plumbing differs by task, so a BLCS-only helper would leave PLCS/SLCS resume/evaluate/inference unguarded.
4. Tracking has independent BLCS predictor/gravity and PLCS matching/loss code; standard-path evidence is insufficient.
5. PLCS renderer direct scaling and `court_pose.py` are separate conversions; migrate both or visual/integrated metres diverge.
6. Scalar SLCS uncertainty is under-specified for anisotropic v1. Preserve/document current mean-scale legacy convention unless authority permits a breaking per-axis head change.
7. AC-019 is operational, requiring separate versioned artifacts and queued training/evaluation evidence, not only code/tests.

## Unresolved questions

1. **Root vs scene metadata:** recommend required identical contract at root and every new scene; validator permits all-missing legacy v1 only. Confirm redundancy is desired.
2. **Legacy declaration:** recommend a required composed v1 config field, not a Python fallback. Confirm a missing config group must error even if caller wants legacy.
3. **Velocity:** writer persists world velocity only, while model docs/predictor support normalized velocity. Decide whether metadata must cover a new persisted normalized velocity or model-output-only units.
4. **Tracking gravity:** recommend derive/validate target as `-gravity*dt^2/scale_z`, preserving literal `-0.01` only if a v1 config proves the same formula. Freeze source of tracking `dt`.
5. **PLCS compact production data:** inspected PLCS writer stores normalized `position`, while `tennis_scene` contracts carry metre arrays. Identify exact compact producer/loader before adding normalization metadata; excluded metre-only artifacts must not be rescaled.
6. **SLCS uncertainty:** recommended v1 legacy scalar `exp(log_b)*mean(scale_xyz)`, v2 `exp(log_b)*HALF_LENGTH`, documented as scalar-equivalent sigma. Confirm versus broad per-axis model change.
7. **Huber beta:** choose a config/mechanism for common v2 metre-space transition and state whether v1 retains historical normalized beta; otherwise AC-013 cannot be tested/documented.

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| FACT | v1 scale and immutable court geometry share one current schema. | `src/utils/schema/court.py`; CourtKP20 regression in `tests/unit/utils/schema/test_court.py`. |
| FACT | No inspected BLCS/PLCS/SLCS root config selects normalization. | `configs/{train,generate_dataset,visualize,evaluate,predict_clip}.yaml` defaults inspected. |
| FACT | BLCS has independent generation, projection, standard/tracking loss/metric and predictor scale consumers. | `scene_generator.py`, `ball_physics.py`, `DifferentiableProjection`, `training/*`, `inference/*`. |
| FACT | Tracking gravity is a fixed normalized literal. | `src/tasks/blcs/configs/loss/tracking.yaml` and `training/tracking_losses.py`. |
| FACT | PLCS canonical pose is metres while translation is normalized. | `generate_dataset/scene_generator.py`, `data/targets.py`, `utils/geometry/court_pose.py`. |
| FACT | PLCS renderer independently applies v1 dimensions. | `visualization/rendering/scene_renderer.py::_world_positions`. |
| FACT | SLCS normalizes SceneResult metres and uses mean-scale scalar uncertainty. | `slcs/data/dataset.py`, `model_io/adapter.py`, `training/metrics.py`, `evaluation/evaluate.py`. |
| FACT | New dataset/checkpoint normalization metadata checks do not exist in inspected writers/loaders. | `BaseDatasetWriter`, BLCS/PLCS writer/meta types, `SceneDatasetBase`, model checkpoint loaders. |
| FACT | SceneResult is metre-valued integrated output. | `src/tennis_scene/schema.py`, `pipeline/orchestrator.py`. |
| INFERENCE | Resolver plus explicit injection is safer than mutable process-global selection. | Multiple direct imports/defaults and independent tracking paths. |
| INFERENCE | Root plus every-scene metadata is the smallest fail-closed mixed-artifact guard. | Root-only cannot detect replaced scenes; scene-only cannot prove set homogeneity. |
| INFERENCE | v2 requires distinct default loss/matching configs while v1 retains current values. | AC-013/014; existing BLCS tracking `[1,1,0.5]`. |
| UNKNOWN | Intended scalar SLCS uncertainty transform under anisotropic v1. | Existing mean rule lacks a documented statistical derivation. |
| UNKNOWN | Whether normalized velocity is persisted data or model-output-only. | Writer stores world velocity; predictor/docs support normalized velocity. |
| UNKNOWN | Exact legacy numeric tolerance/golden commands and AC-019 artifact/run locations. | Frozen issue requires evidence but repository has not defined it. |
