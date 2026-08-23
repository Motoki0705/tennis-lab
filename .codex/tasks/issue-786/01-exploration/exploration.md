# Exploration

- Issue: #786
- Attempt: 2
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Scope and Issue interpretation

The Validator RETURN cannot be repaired by injecting a normalization section into saved configs. Both named metadata-free files predate current configuration and, for BLCS, model-module state naming. AC-003/004 require a finite, checkpoint-identity-bound legacy *inference migration*, not a relaxation of current training configuration parsing or a generic merge of current Hydra defaults.

Named legacy inputs are `/home/kamimura/projects/tennis-lab/ckpt/blcs/run-mono3d-blcs-bcast-v3-simfix-epoch189.ckpt` and `/home/kamimura/projects/tennis-lab/ckpt/plcs/run-mono3d-plcs-bcast-v2-simfix-epoch199.ckpt`, with roots `/home/kamimura/projects/tennis-lab/data/{blcs_broadcast,plcs_broadcast}`. They remain explicit-v1-only. Any other checkpoint SHA, any metadata-bearing checkpoint, metadata-free v2, malformed state/config, or conflict with an already present field must fail closed.

## Relevant files and symbols

| Area | Verified finding / ownership boundary |
|---|---|
| BLCS current load | `blcs/model_io/checkpoints.py::load_checkpoint_runtime` currently injects only normalization; `inference/{predictor,tracking_predictor}.py` then composes current model-I/O and Lightning-loads strict. It cannot load the named file. |
| BLCS strict config/model | `blcs/configuration.py::parse_model_config` exact-checks axial fields; `models/blcs_multiview_axial_model.py` now registers `stages.*` and `output_head.*`. |
| PLCS current load | `plcs/model_io/court_coordinate_checkpoint.py::prepare_plcs_checkpoint_config` injects only normalization; `PLCSPredictor` then calls `PLCSLightningModule.load_from_checkpoint`, whose constructor invokes `PLCSTrainingConfig.from_config`. |
| PLCS strict config/model | `plcs/configuration.py::{PLCSModelConfig,PLCSTrainingConfig}` and `configuration_contracts.py::PLCSPathConfig` require current exact model/data/runtime-root structures. `PLCSMultiViewAxialSplitModel` state is compatible after a finite model-profile repair. |
| PLCS public scene read | `generate_dataset/io/scene_loader.py::load_scene` accepts `None`; callers are `visualization/io/scene.py::load_scene_bundle` (via orchestrator) and `scripts/analysis/visualize_rotation_error_samples.py::_score_scene`. |
| Required test surfaces | Existing integration smoke uses `np.load` + `torch.nn.Identity`; predictor tests only pin v2 scale. Dataset/model chains are `BallTrajectoryDataset`/`BLCSModel` and `SceneDataset`/`PLCSMultiViewModel`. |

## Entry points and execution paths

1. Current BLCS path: predictor load → checkpoint config injection → `compose_blcs_trajectory_model_io` → strict Lightning state load. It first fails typed config and would subsequently fail strict state keys.
2. Current PLCS path: predictor load → `prepare_plcs_checkpoint_config` → `PLCSLightningModule.load_from_checkpoint` → `PLCSTrainingConfig.from_config`. It first fails current seven-root path validation; adding only root extensions does not reach a usable inference model.
3. Dataset classes already call `validate_dataset_court_coordinate_contract` over every split scene. The standalone PLCS loader is the AC-015 bypass.
4. Visualization has a resolved normalization in `RuntimeConfig`; mandatory forwarding therefore stays within repository-owned boundaries.

## Data, configuration, and interface contracts

### Verified complete BLCS named-checkpoint migration

Bind the adapter to the exact checkpoint content SHA-256 (computed before config/state use) and the legacy metadata-free + explicit-v1 gate. The saved model is `blcs_multiview_axial`, `hidden_dim=512`, `num_layers=8`, `num_heads=8`, and **`data.num_court_kp=14`**.

| Saved difference | Deterministic target / source | Conflict or negative condition |
|---|---|---|
| missing `court_coordinate_normalization` | add `{version: "v1"}` | only all-missing metadata plus explicit v1. |
| missing `model.num_court_tokens` | add 14, derived only from saved `data.num_court_kp`; require positive and equality | absent/non-int/nonpositive/different existing value fails. |
| `model.ffn_dim: null` | replace with 1408 only after every expected checkpoint FFN `w1` has shape `(1408,512)`, `w2` `(512,1408)`, and `w3` `(1408,512)` | inconsistent stacks or any nonmatching shape fails. |
| `model.rope_dim: null` | replace with 64 from the named BLCS axial-base legacy profile (`hidden_dim=512`, heads=8, profile rope_dim=64); state does not encode this nonpersistent RoPE frequency dimension | no SHA/profile match, non-null different value, invalid even/head-dimension rule, or profile hash change fails. |
| legacy `model.rope_theta: 10000.0` | remove only for this named axial profile: current `parse_model_config` deliberately excludes it and current axial construction uses only `rope_theta_time`/`rope_theta_camera`, both saved as 10000/1000 | any noncanonical name/profile/value fails; never delete unknown keys generally. |
| state names `model.camera_layers.{i}.0.*`, `model.time_layers.{i}.0.*`, `model.position_head.*`, `model.velocity_head.*` | map respectively to `model.stages.{i}.camera_layers.0.*`, `model.stages.{i}.time_layers.0.block.*`, `model.output_head.position.*`, `model.output_head.velocity.*` for i=0..7 | only exact one-to-one matched key set/shapes; duplicate/missing/unexpected key fails. |

Read-only construction with that exact config and state-key mapping produced `strict=True` success: 141 mapped state tensors, zero missing and zero unexpected. Without state migration, strict load reports all current `stages.*`/`output_head.*` missing and all saved names unexpected. Therefore current Lightning `load_from_checkpoint(strict=True)` cannot itself be the legacy BLCS mechanism; use a dedicated legacy inference loader that constructs the model-I/O binding, applies the verified map, then retains strict equality checks.

### Verified PLCS named-checkpoint migration

The saved model is `plcs_multiview_axial_split`, `data.num_court_kp=14`, `hidden_dim=512`, heads=8, `num_layers=0`, `num_task_layers=6`, but `ffn_dim=null`; its state has six rotation camera/time and six pose camera/time stacks, each FFN width 1408. A read-only construct after model-profile amendment loaded all 219 model tensors with `strict=True`, zero missing/unexpected.

| Saved difference | Deterministic target / source | Conflict or negative condition |
|---|---|---|
| missing normalization | explicit v1 only | same metadata gate as BLCS. |
| `model.ffn_dim: null` | 1408 from every saved FFN tensor shape | inconsistent shape fails. |
| legacy `model.rope_theta:10000` | remove only for named axial-split profile; current fields permit only time/camera RoPE bases (both saved 1000) | profile/value mismatch fails. |
| missing `model.rot_num_task_layers` / `pose_num_task_layers` | add 6/6 from exact counted `rot_*` and `pose_*` state stacks; require equality with saved `num_task_layers=6` | absent/inconsistent state or different existing values fails. |
| legacy `paths.smplh_model_path` | do **not** feed it to `RuntimePathRoots`; preserve it only as provenance and validate it equals the identity-bound expected absolute legacy location. Current inference external asset becomes role-relative `smplx/smplh` under an explicit seven-root profile. | arbitrary absolute path, different existing seven-root mapping, or unavailable required resolved asset fails. |
| missing current runtime sections | An identity-bound PLCS legacy-inference profile must explicitly carry: seven `paths` roots; `external_assets.smplh_model_path`; `qualitative` fps/style/view_3d; current required training/run/loss additions; and chunked generation fields if, and only if, the generic training parser is used. | A generic merge of current `train.yaml` defaults is forbidden because it changes saved data/loss/training semantics and admits future drift. |

The generic `PLCSTrainingConfig` route requires more than the three originally reported sections: old `paths` is rejected by `RuntimePathRoots` before root additions; old root has legacy `camera`/`motion_sources` but no `generation` despite `data.backend="chunked"`; old data mode `sequence` conflicts with current multiview requirement; old training has now-unknown `scheduler` and lacks `warmup_epochs`, `steps_per_epoch`, optimizer/compile/matmul/TF32/GAN and qualitative-selection fields; old loss lacks Huber/smoothness/velocity fields; run lacks `init_weights` and `test_after_fit`. These are training/runtime drift, not model inference facts.

Accordingly the safe target is **not** to manufacture a modern training config. Implement a named-checkpoint `LegacyPLCSInferenceProfile` that parses only the exact model I/O, v1 contract, explicitly owned roots, and inference-required metrics/loss parameters; directly constructs `PLCSMultiViewAxialSplitModel` + `PLCSModelIOAdapter`, then strict-loads the verified model state. Keep `PLCSTrainingConfig` closed for train/resume/evaluate. If reuse of `PLCSLightningModule` is mandatory, a separately versioned complete profile must enumerate every field/value and source in code/tests; it may not be composed by merging current defaults.

### Frozen-base and parity consequence

Frozen base revision `59e3b166c2d010d5e62be52c2be76d98a94af0e0` has the same strict BLCS axial parser requiring int `ffn_dim`/`rope_dim` and `num_court_tokens`, and its PLCS path roots also require the seven-root mapping. Base predictor loaders had no checkpoint migration adapters: BLCS directly called `compose_blcs_trajectory_model_io(saved_config)`; PLCS directly Lightning-loaded saved config. Thus frozen-base production cannot load either named checkpoint and cannot generate the proposed “pre-change baseline” through its normal paths.

AC-003 parity must instead use a reproducible, explicitly versioned legacy-reference loader/profile (or independently archived pre-refactor source plus its dependency lock), generate and commit immutable expected outputs, then compare the new named migration against those outputs. Until such a reference is supplied and executed, parity is NOT VERIFIED; do not claim base-revision goldens.

### AC-015, AC-020, and AC-021 repairs

Make `load_scene(..., court_coordinate_normalization: CourtCoordinateNormalization)` and `load_scene_bundle(..., court_coordinate_normalization: CourtCoordinateNormalization)` mandatory; update the two production callers and test public missing/unknown/mixed failures. AC-020 must use complete temp scene roots, actual dataset class + collation + tiny actual task model + task loss/metric + metre decode + BLCS projection/PLCS renderer for both versions. AC-021 adds default-v1 fixed-one output assertions for BLCS standard/tracking position+velocity and PLCS standard/tracking translation, alongside existing v2 assertions.

## Existing tests and fixtures

- `tests/integration/tasks/test_court_coordinate_normalization_smoke.py` currently cannot prove AC-020 because it uses `Identity`.
- `tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py` is the direct AC-015 boundary test location.
- BLCS and PLCS standard/tracking predictor files already contain fixed-output models suitable for explicit default-v1 physical-scale assertions.
- New checkpoint migration tests must use synthetic checkpoint config/state fragments for all negative branches and the two real files only in an artifact-mounted integration lane. They must assert identity, parsed profile, state map, strict load, v2 rejection, and conflict rejection.

## Invariants and compatibility constraints

- No generic defaults merge, no generic unknown-key deletion, no tensor-shape inference except the explicitly named FFN/state-stack checks above, and no mutation of old files.
- Named migration profiles are keyed by checkpoint content hash and model name/profile; all conditions precede model execution.
- Strict model-state equality remains mandatory after the finite BLCS key map and PLCS profile amendment.
- New metadata remains fully validated; metadata-free is explicit-v1-only. PLCS canonical pose and public outputs stay metre-valued.

## Risks and likely impact radius

1. The original proposed config-only BLCS repair is acceptance-invalid: strict state names changed.
2. RoPE dimension is not state-encoded. BLCS `rope_dim=64` needs a pinned historical profile/hash; otherwise correct numerical parity is unknowable.
3. Current PLCS Lightning construction is a training contract, so making it accept incomplete legacy config would silently broaden train/resume behavior. Keep legacy inference separate.
4. There is no executable frozen-base baseline. The Issue's parity acceptance needs a supplied archival reference or a consciously approved replacement evidence strategy; fabricated golden values are prohibited.
5. Mandatory PLCS loader validation is a public signature break, but the only repository callers already have owned runtime contracts.

## Unresolved questions

1. **Missing authority/evidence for AC-003:** what immutable pre-refactor reference (commit/environment/image or previously captured outputs) is authoritative for the two named checkpoint numerical goldens? Frozen base cannot load them.
2. **Architecture provenance:** approve a checkpoint-SHA-bound BLCS legacy profile that pins `rope_dim=64` to the named axial-base profile, or provide archived training composition proving it. State shape cannot determine that value.
3. **Scope choice:** approve dedicated inference-only legacy loaders/profile objects (recommended). Requiring generic current Lightning/train composition would require a much larger fully enumerated migration and risks altering non-inference semantics.

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| FACT | BLCS saved `data.num_court_kp` is 14, not 20; ffn/rope are null and `rope_theta` is present. | Direct `torch.load` inspection of named file. |
| FACT | BLCS finite config plus exact state-key map strict-loads 141/141 tensors. | Read-only model construction and `load_state_dict(strict=True)`. |
| FACT | PLCS model-profile amendment strict-loads 219/219 tensors. | Read-only `PLCSMultiViewAxialSplitModel` construction and strict load. |
| FACT | PLCS old single `paths.smplh_model_path` fails current `RuntimePathRoots` schema before external-assets/qualitative parsing. | `RuntimePathRoots.from_mapping` exact seven-root schema and saved config inspection. |
| FACT | Frozen base's normal predictor paths have no legacy adapter and its strict parsers reject the saved drift. | `git show 59e3b166:.../{configuration,inference/predictor}.py`. |
| INFERENCE | A SHA-bound direct inference loader is the smallest fail-closed repair. | Strict model state is compatible after finite maps, while training-config drift is broad and non-inference. |
| UNKNOWN | Authoritative pre-refactor numerical reference outputs/environment. | Neither frozen base nor current normal loader can load these files. |
