# Exploration

- Issue: #786
- Attempt: 2
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Scope and Issue interpretation

Attempt 2 narrowly repairs Validator RETURN findings for AC-003, AC-004, AC-015, AC-020, and AC-021. The existing v1/v2 mathematical contract remains intact. This attempt must make two real metadata-free v1 checkpoints composable without weakening strict typed configuration, remove every public PLCS scene-load bypass, and replace structural smoke/predictor gaps with executable task evidence.

The real artifacts are `/home/kamimura/projects/tennis-lab/ckpt/blcs/run-mono3d-blcs-bcast-v3-simfix-epoch189.ckpt`, `/home/kamimura/projects/tennis-lab/ckpt/plcs/run-mono3d-plcs-bcast-v2-simfix-epoch199.ckpt`, and metadata-free legacy roots `/home/kamimura/projects/tennis-lab/data/blcs_broadcast` and `/home/kamimura/projects/tennis-lab/data/plcs_broadcast`. They are legacy v1 only when the caller explicitly supplies v1; v2 must stop in the metadata gate before config/model/data interpretation.

## Relevant files and symbols

| Area | Verified fact and repair boundary |
|---|---|
| BLCS checkpoints | `src/tasks/blcs/model_io/checkpoints.py::{load_checkpoint_runtime,_inject_legacy_v1_config}` only injects normalization. The real checkpoint's `model` lacks `num_court_tokens`; `parse_model_config` requires it, while saved `data.num_court_kp` is 20. |
| PLCS checkpoints | `src/tasks/plcs/model_io/court_coordinate_checkpoint.py::{_stored_config,prepare_plcs_checkpoint_config}` only injects normalization. `PLCSTrainingConfig.from_config` strictly requires root `external_assets` and `qualitative`; the real checkpoint lacks both but retains `paths.smplh_model_path`. |
| Shared guards | `src/tasks/base/{data,model_io}/court_coordinate_contract.py` implement all-missing legacy=v1-only and v2 rejection. |
| PLCS public reads | `generate_dataset/io/scene_loader.py::load_scene` has optional contract. Production callers are `visualization/io/scene.py::load_scene_bundle` (via `visualization/orchestrator.py::run_visualization`) and `scripts/analysis/visualize_rotation_error_samples.py::_score_scene`. |
| Actual loaders | `BallTrajectoryDataset` and PLCS `SceneDataset` inherit `SceneDatasetBase`; its constructor validates every split scene when supplied a contract. BLCS needs ball/court arrays, `ball_pos_norm`, `ball_vel_world`; PLCS needs human/court arrays, `position`, `rotation`, and valid `build_coco17_world_targets` input. |
| CPU models | Actual `BLCSModel` and `PLCSMultiViewModel` can be tiny: hidden_dim=8, heads=2, ffn_dim=16, layers=0, rope_dim=4, one view/two frames/20 court tokens. Their real adapters are in each task's `model_io/factory.py`. |
| Inadequate tests | `test_court_coordinate_normalization_smoke.py::test_blcs_plcs_cpu_flow_stays_meter_valued_through_projection_and_render` loads a standalone `.npy` then calls `torch.nn.Identity`; existing standard/tracking predictor tests pin v2, not v1/default, metre scaling. |

## Entry points and execution paths

1. BLCS: `BLCSPredictor.load_from_checkpoint`/tracking → `load_checkpoint_runtime` → `compose_blcs_trajectory_model_io` → `BLCSLightningModule.load_from_checkpoint(strict=True)`. A legacy adapter must yield a complete, exact current config before compose.
2. PLCS: standard/tracking predictor checkpoint load → `load_plcs_checkpoint_mapping` → `prepare_plcs_checkpoint_config` → strict Lightning load, whose constructor calls `PLCSTrainingConfig.from_config`. The adapter belongs before that typed boundary.
3. PLCS dataset: `SceneDataset` → `SceneDatasetBase` → `validate_dataset_court_coordinate_contract(root, contract, scene_paths=all_paths)` → payload build. It already validates all split paths; standalone `load_scene` bypasses it when omitted.
4. Visualization: Hydra → `build_runtime_config` resolves a contract → `run_visualization` → `load_scene_bundle` → `load_scene`. The runtime already owns a non-optional contract. The analysis script already passes its contract.

## Data, configuration, and interface contracts

### Exact fail-closed legacy typed-config adapter

Use task-local helpers only after the shared checkpoint validator returned `legacy_metadata_free=True` and explicit runtime v1. Deep-copy the saved config and apply a finite allowlist; never merge current training defaults wholesale.

| Checkpoint | Exact admitted amendment | Validation / rejection |
|---|---|---|
| BLCS real v1 | Add `court_coordinate_normalization: {version: v1}` and `model.num_court_tokens: 20` only when absent. | Derive 20 only from legacy `data.num_court_kp`, require integer >0, parse the amended model, and require equality. Existing different value, missing/invalid source, metadata-bearing file, or non-v1 runtime raises a named config/contract error. Never infer tensor shape or use arbitrary default. |
| PLCS real v1 | Add normalization, `external_assets: {smplh_model_path: <saved paths.smplh_model_path>}`, and the exact canonical `qualitative` schema (fps/style/view_3d) from owned PLCS `configs/qualitative/default.yaml`, only when absent. | Existing sections must exact-validate and never be overwritten. Missing/invalid legacy path, invalid canonical qualitative composition, unknown keys, metadata-bearing checkpoint, or non-v1 runtime raises. Pin the inserted mapping in tests; do not use an empty mapping or guessed path. |

Direct inspection establishes the BLCS saved model is `blcs_multiview_axial`, hidden_dim=512/layers=8/heads=8 and otherwise has current parsed axial fields; PLCS is `plcs_multiview_axial_split` with `data.num_court_kp=20` and only the two root extensions absent. Thus a finite adapter is justified; relaxing `_exact` or `MissingConfigurationKeyError` globally is not.

### Executable AC-003/004 parity evidence

Add a focused integration regression using both named checkpoint paths and one explicitly named deterministic scene from each legacy `test.txt`, `augment=False`, CPU and fixed seeds. Execute the real loaded predictor, task loss, and metric. Commit a small golden fixture generated at frozen base revision `59e3b166c2d010d5e62be52c2be76d98a94af0e0`: input selection, normalized prediction, denormalized metre prediction (plus BLCS velocity), scalar loss, and metre metric fields. Compare CPU float32 arrays/scalars at `atol=1e-5, rtol=0`, require finite outputs and strict checkpoint loading.

Also assert each real checkpoint under explicit v2 raises `MissingCourtCoordinateMetadataError` before composition, and synthetic forbidden/mismatched amendments raise. The frozen tree has no pre-change golden values; generating and committing them from the named base revision is required implementation evidence, not something to invent here.

### Mandatory PLCS public load contract

Make `load_scene(filepath, *, court_coordinate_normalization: CourtCoordinateNormalization)` mandatory; make `load_scene_bundle` mandatory and always forward it. `run_visualization` already has `RuntimeConfig.court_coordinate_normalization`; `_score_scene` already passes one. Update direct loader tests to explicitly pass v1 and add missing/unknown/mixed root-scene failures through `load_scene`, not only generic validator tests. Future callers must resolve/configure an explicit runtime version: plain filesystem loading is no longer a public bypass.

### Actual CPU chain for AC-020

Create temp root/scene fixtures with complete v1/v2 metadata, two frames, one camera, complete arrays, and `[2,2]`/`[1,1]`/`camera_mode="first"` composed task configs with workers=0. For PLCS include `human_kp_3d.npy` (or valid raw target input) so its real target builder succeeds.

- BLCS: `BallTrajectoryDataset` → `collate_multiview_trajectories` → tiny actual `BLCSModel` bound through the matching real adapter → `trajectory_position_loss`/`BLCSMetrics` → denormalize → `DifferentiableProjection`.
- PLCS: `SceneDataset` → `collate_plcs_batch` → tiny actual `PLCSMultiViewModel` bound through `PLCSModelIOAdapter` → `PLCSLoss`/`PLCSMetrics` → metre decode → `PLCSSceneRenderer`.

For v1 and v2 assert actual model type/output shape, finite loss/metrics, restored physical positions, and finite projection/render coordinates. No `np.load` plus `Identity`/fixed model can satisfy AC-020.

### v1/default predictor assertions for AC-021

Keep v2 cases and add default-v1, `denormalize=True`, fixed-one-output tests:

- BLCS `inference/test_predictor.py`: position `(5.485,11.885,1.07)` and velocity twice it.
- BLCS `inference/test_tracking_predictor.py`: every query/frame equals the v1 tuple.
- PLCS `inference/test_predictor.py`: use `_FixedRotationModel(position_value=1.0)` and assert `position_meters` equals the v1 tuple.
- PLCS `inference/test_tracking_predictor.py`: every query/frame `position_meters` equals v1.

Name each a default-v1 physical-scale regression. `denormalize=False`, zero output, or shape-only checks do not pin this contract.

## Existing tests and fixtures

- Current files: `tests/integration/tasks/test_court_coordinate_normalization_smoke.py`, `tests/unit/tasks/base/data/test_court_coordinate_contract.py`, BLCS `data/test_dataset.py`, both BLCS predictor tests, PLCS `generate_dataset/io/test_scene_loader.py`, and both PLCS predictor tests.
- Base contract tests cover generic metadata validation; AC-015 needs the public PLCS loader boundary specifically.
- Fixed predictor models are suitable for AC-021 but not AC-020; actual models/loaders are required there.
- Legacy roots are approximately 80 MB (BLCS) and 1.1 GB (PLCS), so AC-003 uses one deterministic real sample and goldens; AC-020 uses small temp fixtures.

## Invariants and compatibility constraints

- Preserve strict typed config validation, strict state loading, and no overwrite. Legacy mutation is in-memory, finite, explicit-v1-only, and auditable.
- New roots/scenes must carry complete identical metadata. All-missing is v1 legacy only; partial/unknown/mixed/mismatch fail before use.
- PLCS canonical/root-relative pose stays metres; global translation alone is normalized. Public outputs remain metres.
- v2 must reject real metadata-free files before legacy config adaptation.
- CPU smoke must use actual task dataset and model classes.

## Risks and likely impact radius

1. Broad OmegaConf merge would admit drift. Limit path/value/condition and test negative cases.
2. PLCS qualitative defaults can drift; version the legacy adapter and pin its exact inserted mapping.
3. Real-checkpoint tests need declared external artifacts. They may skip only outside required evidence environments; a skip is not AC-003/004 PASS evidence.
4. PLCS fixture construction can under-specify `build_coco17_world_targets`; build it from a known loader-compatible format and assert the real loader/model was reached.
5. Mandatory loader argument intentionally breaks external untyped callers, but owned repository callers already possess contracts and the bypass violates AC-015.

## Unresolved questions

None requiring product authority. Reproducible base-revision golden generation is an implementation prerequisite; if the named artifacts are unavailable for verification, AC-003/004 must remain NOT VERIFIED/RETURN rather than be skipped as PASS.

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| FACT | BLCS real checkpoint lacks normalization and `model.num_court_tokens`, but has `data.num_court_kp=20`. | Direct CPU `torch.load(..., weights_only=False)` inspection; Validator's `MissingConfigurationKeyError`. |
| FACT | PLCS real checkpoint lacks normalization, `external_assets`, and `qualitative`, but has `paths.smplh_model_path`. | Direct checkpoint inspection; `PLCSTrainingConfig.from_config` required fields. |
| FACT | Shared metadata gate rejects metadata-free checkpoint under v2. | Base model-I/O contract and Validator direct attempts. |
| FACT | All repository production PLCS `load_scene` calls are listed above; visualization retains optional bypass despite resolved runtime contract. | `rg "load_scene\\(" src/tasks/plcs --glob '*.py'`; visualization orchestrator. |
| FACT | Nominal CPU smoke uses `np.load` and `torch.nn.Identity`. | Integration smoke test lines 191-196. |
| FACT | Existing v2 predictors assert scale; v1/default cases omit it. | Four predictor test files. |
| INFERENCE | `data.num_court_kp` is the only safe source for missing BLCS token count. | Existing parser requires equality; saved value is 20. |
| INFERENCE | Finite PLCS adapter is safer than optional strict config fields. | Missing fields are runtime extensions; relaxed parser admits arbitrary configs. |
| UNKNOWN | Pre-change numerical golden values/hashes are absent. | Frozen tree and Validator evidence contain no base-revision output fixture. |
