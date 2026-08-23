# Exploration

- Issue: #786
- Attempt: 3
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Scope and Issue interpretation

Fresh AC-017-only exploration: prevent v2 PLCS datasets/checkpoints from being published over, or under the identity of, existing v1 artifacts. Failure must precede the first destination write, including `config.yaml`, a checkpoint, or a scene directory. Do not infer version from bytes or modify legacy v1 files.

The narrow persistent boundaries are the standalone generator dataset root and PLCS training run root (whose `logs/version_N/checkpoints/*.ckpt` children publish checkpoints). Dynamic chunk caches are not selected persistent artifacts. The compact scene-pipeline PLCS producer is a fixed mutable workspace owner; it requires an explicit in/out-of-scope decision rather than a silent assumption.

Default/v1 remains backward-compatible: existing unqualified `data/plcs` and `outputs/plcs/${model.name}` spellings may parse, and an existing empty ordinary destination remains allowed. A selected v2 persistent root needs a canonical v2 token in a parsed relative path component. Shipped names use `*_norm_v2`; a compatible predicate therefore examines each `Path.parts` component for an exact delimiter-bounded `norm_v2` token, never `"v2" in str(path)`. It rejects `norm_v20`, `normalization_v2`, `norm-v1/misnamed-v2`, and tokens outside the configured artifact-relative path. A mandatory new `norm-v2` component would be a policy/layout migration not specified by the frozen Issue.

## Relevant files and symbols

| Boundary | Verified fact / owner |
|---|---|
| Generator CLI | `src/tasks/plcs/scripts/generate_dataset.py::main` calls `output_dir.mkdir(..., exist_ok=True)` and `OmegaConf.save(.../config.yaml)` before `PLCSDatasetWriter`. These are first writes. |
| Generation config | `src/tasks/plcs/generate_dataset/config.py::PLCSGenerationConfig.from_config` resolves `run.output_dir` via DATA role but has no v2-name or occupancy check. `resolve_generation_paths` is also used by chunks. |
| Writer | `PLCSDatasetWriter` inherits `BaseDatasetWriter`; base constructor creates root/`scenes` with `exist_ok=True`, and `save_scene` creates `scenes/<id>` with `exist_ok=True` before writing JSON/NPY. |
| Chunk caller | `src/tasks/plcs/data/chunk_manager.py::_PLCSChunkGenerator.__call__` constructs `PLCSDatasetWriter(chunk_dir)`. |
| Training | `scripts/train.py` -> `PLCSTrainingConfig` -> `PLCSTrainingRunner` -> `BaseTrainingRunner.run`; `BaseRunConfig` resolves OUTPUT role but only rejects empty text. Runner mkdirs/saves config before logger/callback checkpoints. |
| Existing authority | `court_coordinate_materializer.py::_validate_paths` rejects existing target, stages then rechecks before rename. `BLCSDatasetWriter.__init__` rejects non-empty roots. |
| Compact producer | `run_scene_pipeline.py` -> `ScenePipelineRunner` -> `PLCSStageHandler`; registry fixes owner `datasets/plcs`, and `StagePublisher.publish` atomically exchanges an existing owner. `PLCSV5ReusablePublicationValidator` does not check normalization. |

## Entry points and execution paths

1. Standalone: Hydra `plcs.scripts.generate_dataset:main` -> `PLCSGenerationConfig.from_config` -> current mkdir/config write -> writer -> parallel scenes -> `save_scene` -> splits/meta. Required order: typed v2-name validation -> writer occupancy validation -> config save and all later writes.
2. Chunk cache: training -> `PLCSChunkManager` -> `_PLCSChunkGenerator` -> writer -> scenes/meta. It should receive writer occupancy safety, not durable artifact-name policy.
3. Training: Hydra `scripts.train:main` -> `PLCSTrainingConfig.from_config` -> runner -> current mkdir/config save -> TensorBoard `logs/version_N` -> fixed checkpoint child. Name validation must happen in the PLCS typed boundary, before runner execution; resume/init are input validation, not publication naming.
4. Compact: `run_scene_pipeline` creates `.transactions/plcs_dataset/snapshot`, `PLCSStageHandler` assembles/validates, then `StagePublisher` installs/exchanges `workspace/datasets/plcs`. `PLCSStageParameters` defaults v1 and `PLCSDatasetConfiguration.build_stage_parameters` does not pass a selectable normalization; there is no compact v2 owner/name today.

## Data, configuration, and interface contracts

| Artifact kind | Narrow name contract | Non-overwrite boundary |
|---|---|---|
| Standalone dataset `run.output_dir` (DATA) | v1/default keeps current spelling; v2 requires parsed `norm_v2` token. `generate_dataset_norm_v2.yaml` `plcs_broadcast_norm_v2` passes; v1 config remains `plcs_broadcast_norm_v1`. | `PLCSGenerationConfig.from_config` rejects bad v2 before `main`; writer rejects any non-empty root before root/scenes/config writes. Empty root allowed. `save_scene` must exclusively create its new scene directory so collision fails before JSON/NPY writes. Move config save after writer construction. |
| PLCS training `run.output_dir` (OUTPUT) | v1/default keeps `plcs/${model.name}`; v2 requires same predicate. `train_norm_v2.yaml` `plcs/baseline_norm_v2/${model.name}` passes. | PLCS-owned check in `PLCSTrainingConfig.from_config` rejects before `BaseTrainingRunner.run` mkdir/config/logger/checkpoint actions. Callback filename checks are too late. |
| Writer/chunk root | No version predicate: no selected durable identity. | Non-empty-root and exclusive-scene checks apply; fresh chunk root remains valid. |
| Materialized copy | Existing explicit source/target contracts. | Preserve no-existing-target + staging/recheck/rename authority. |
| Compact `datasets/plcs` | No satisfiable v2 name contract exists: fixed owner is unqualified and v2 is not selectable. | Intentional atomic replacement conflicts with AC-017 if compact artifacts are included; versioned owner/layout and reuse changes would be necessary. |

`CourtCoordinateNormalization.version` selects the predicate. Existing `PathResolver` first rejects absolute/root-prefixed values; the predicate then inspects only configured relative `Path.parts`. Metadata remains independently injected/validated by dataset and checkpoint contracts; names do not substitute for metadata.

Current configs: default generator `plcs`; `generate_dataset_norm_v1|v2.yaml` `plcs_broadcast_norm_v1|v2`; default train `plcs/${model.name}`; `train_norm_v1|v2.yaml` `plcs/baseline_norm_v1|v2/${model.name}`; data v1 is metadata-free `plcs_broadcast`, v2 is `plcs_broadcast_norm_v2`.

Required preservation tests: (a) malformed v2 generation config raises with no destination/config; (b) non-empty version-qualified root containing sentinel `config.yaml`, root metadata, and scene byte fails through CLI-before-workers with all bytes identical; (c) post-construction scene collision leaves sentinel unchanged; (d) v2 training under v1-labelled output fails at `PLCSTrainingConfig.from_config` before runner/config/checkpoint writes, preserving representative config/checkpoint bytes; (e) control cases: empty qualified v2 root, unqualified default v1, and all shipped v1/v2 configs succeed. Retain materializer target-exists coverage.

## Existing tests and fixtures

- `tests/unit/tasks/plcs/test_configuration.py` already tests DATA/OUTPUT role resolution and is the natural PLCS train v2-token/default-v1 surface.
- `tests/unit/tasks/plcs/test_configuration_contracts.py` owns direct `PLCSGenerationConfig.from_config` tests.
- There is no dedicated PLCS writer I/O test. Add `tests/unit/tasks/plcs/generate_dataset/io/test_dataset_io.py` for root/scene sentinel behavior. Existing fresh-root users are `test_multi_object_scene_generator.py` and `tests/integration/tasks/tracking/test_training_smoke.py`.
- `tests/unit/tasks/base/training/test_runner.py` owns generic lifecycle, but this policy must not be put in generic `BaseRunConfig` because other tasks lack PLCS normalization authority.
- `tests/integration/tasks/test_court_coordinate_normalization_smoke.py` already proves materializer target refusal/source bytes, not primary PLCS generation/training.
- Compact coverage: `tests/unit/synthetic_data_generation/pipeline/{test_publication,test_runner}.py`, `dataset/plcs/test_assembler.py`, and `tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py`; they test staged replacement/reuse, not AC-017 preservation/v2 owner naming.

## Invariants and compatibility constraints

- Bad v2 labels fail before destination mutation; post-write validation/cleanup is insufficient.
- Non-empty means any entry including dotfiles. Do not delete/reuse it; empty ordinary destination is allowed.
- Keep v2 naming check PLCS-owned after normalization/run typing, not in generic base config.
- Keep writer free of version-name policy because chunks use it; writer owns occupancy only.
- Preserve materializer staging; do not route fresh generation through it.
- A v1 token never makes a selected v2 output safe; existing metadata remains a separate runtime guard.

## Risks and likely impact radius

1. Writer-before-config ordering is mandatory; avoid treating partial datasets as complete when moving it.
2. Raw substring checks admit lookalikes; strict literal `norm-v2` component checks break shipped `_norm_v2` configs. Test the component-local grammar.
3. Callback-level checks are too late because runner config/logs already exist.
4. Writer fail-closed behavior changes reruns: callers must choose a new root rather than merge.
5. Compact PLCS has intentional replacement semantics; inclusion in AC-017 requires registry/workspace/report/reuse-layout work beyond primary direct consumers.

## Unresolved questions

1. Does AC-017 include compact scene-pipeline `workspace/datasets/plcs`? If yes, approve a version-qualified owner/layout; its current fixed owner cannot preserve v1 while publishing v2. If no, explicitly record its mutable scene-workspace semantics as out of the baseline dataset/checkpoint contract.
2. Is delimiter-bounded `norm_v2` inside a path component approved, or must paths migrate to a dedicated `norm-v2` component? Frozen Issue requires identification but not separator/layout.

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| FACT | Generator writes config before writer. | `src/tasks/plcs/scripts/generate_dataset.py::main` lines 48–59. |
| FACT | PLCS writer/base accept non-empty roots and existing scene directories then write files. | `dataset_io.py::save_scene`; `base/data/dataset_writer.py::__init__`, `_write_scene_files`. |
| FACT | Training writes root config before logger/checkpoint publication. | `BaseRunConfig.from_mapping`; `base/training/runner.py::run`, `build_logger`, `build_callbacks`. |
| FACT | Materializer is no-overwrite staged; BLCS writer rejects non-empty root. | `court_coordinate_materializer.py`; `blcs/.../dataset_io.py::BLCSDatasetWriter.__init__`. |
| FACT | Shipped PLCS v2 names use `_norm_v2`, not literal `norm-v2` component. | `configs/generate_dataset_norm_v2.yaml`, `train_norm_v2.yaml`. |
| FACT | Compact PLCS fixed owner atomically replaces; reusable validation omits normalization. | `pipeline/registry.py`, `publication.py::StagePublisher.publish`, `reuse.py`. |
| FACT | Compact config does not pass selectable normalization; stage parameters default v1. | `synthetic_data_generation/configuration.py::build_stage_parameters`; `dataset/plcs/handler.py::PLCSStageParameters`. |
| INFERENCE | PLCS typed v2-name checks plus writer occupancy checks are the smallest fail-closed repair without policy leakage to chunks/other tasks. | Existing ownership boundaries above. |
| UNKNOWN | Compact AC-017 inclusion and exact token grammar/layout. | Frozen Issue has no fixed-owner or delimiter authority. |
