# Validation

- Issue: #786
- Attempt: 2
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`

## Inspection scope and revision

- Independently recomputed the current candidate before semantic inspection; it exactly matched the supplied sealed identity `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9` and still matches after validation.
- Inspected final `HEAD=c1cc2369c037f2cd5695ec5feae9d63d7fcda4cd`, frozen base `59e3b166c2d010d5e62be52c2be76d98a94af0e0`, and `origin/main=179dac756aef137c9a35b1025ce76f0a31023648`. Merge `64ea1b5a99bacd5ec7f8ab4f356333835eaa9de9` has Issue parent `2661f3a80b56d5b2e1d44106162ba199cfaf45b0` and upstream parent `179dac756aef137c9a35b1025ce76f0a31023648`.
- The complete frozen-base final-tree diff is 334 files, 17,291 insertions, and 887 deletions. The Issue-authored final scope relative to the merged upstream main is 268 files, 12,180 insertions, and 492 deletions. Upstream workflow, track-query ablation, padding, and CUDA-operator changes were distinguished from the normalization implementation, while their merged final-tree interactions were still tested.
- Authority was limited to frozen `issue.json` and exactly rendered `issue.md`; inspection used current code, config, documentation, generated datasets, checkpoints, fixtures, tests, merge history, and direct runtime probes. No production file, test, fixture, Issue snapshot, workflow state, or GitHub resource was modified.

## Acceptance checklist verification

| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | PASS | `src/utils/schema/court_normalization.py::_normalization_definition` is the sole version-to-scale mapping, derives v1 from immutable court dimensions, returns isotropic v2, and raises `UnknownCourtCoordinateNormalizationVersionError`; resolver tests passed. |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | PASS | `CourtCoordinateNormalizationConfig` is the shared typed boundary. BLCS, PLCS, SLCS, and tennis-scene generation/training/evaluation/inference roots compose explicit `court_coordinate_normalization=v1` and accept the v2 override; all ten Hydra boundary pairs passed. |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | PASS | All inspected roots default to v1. Frozen-base representative datasets, genuine Torch checkpoints, and golden outputs replay BLCS/PLCS load, inference, loss, and aggregate/batch metrics with `atol=1e-5`; the parity test passed and fixture hashes match their frozen manifest. |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | PASS | `validate_dataset_court_coordinate_contract_documents` and `validate_checkpoint_court_coordinate_contract` accept a wholly metadata-free artifact only under explicit v1. Integration probes prove v2 rejection occurs before `np.load` or Lightning state restoration. |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | PASS | Exact metadata parsing rejects unknown or noncanonical scales; root/scene/runtime mismatch guards and BLCS, PLCS, and SLCS checkpoint config/hooks fail before resume or inference state consumption. Unit mismatch matrices and predictor tests passed. |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | PASS | Shared NumPy/Torch conversion validates trailing XYZ and broadcasts over arbitrary leading shapes. Tests cover multiple ranks, dtypes, devices, positions, and velocities for both versions; a direct float32 probe observed maximum error `1.9073486328125e-06m`. |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | PASS | Physical constants remain `5.485m`, `11.885m`, and `1.07m`; direct v2 normalization gives sideline `±0.4615061001`, baseline `±1.0`, and post `0.0900294489`. Geometry/aspect-ratio tests passed. |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | PASS | `BLCSSceneGenerator` and `BallPhysics`, dataset velocity targets, standard/tracking predictors, `DifferentiableProjection`, `BLCSMetrics`, and physical tracking metrics all receive or use the resolved contract. Targeted BLCS plus two-version CPU integration tests passed. |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | PASS | Standard gravity uses `ballistic_second_difference(gravity, dt, contract.scale_xyz[2])`. `resolve_tracking_gravity_target` preserves v1 `-0.01` and derives v2 from the formula; unit and merged model-config tests passed. |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PASS | PLCS generation/targets, standard and tracking predictors, standard and physical tracking metrics, court-pose integration, tennis-scene PLCS component, and both renderer paths use the selected contract. The complete two-version PLCS CPU smoke rendered both 3D and top-down figures. |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | PASS | Generation normalizes only `court_trans`; target integration and rendering denormalize only translation and rotate/add unchanged canonical metre offsets. Both-version target/generation/court-pose tests passed. |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | PASS | SLCS dataset, evaluation, metric, adapter, inference, and uncertainty conversion use the runtime contract. Tennis-scene components and `SceneResult` keep player/ball arrays in metres and attach normalization only as validated provenance; focused SLCS and SceneResult tests passed. |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | PASS | BLCS default axis weights are null and PLCS position loss is unweighted; both derive v2 normalized beta as `1.0m / 11.885m`. Equal `0.5m` single-axis errors produce equal loss in both tasks, and README/config comments document the shared `1.0m` transition. |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | PASS | BLCS v2 standard loss is uniform and tracking selects `[1,1,1]` for both supervised loss and Hungarian cost; v1 retains `[1,1,0.5]` and beta 1. PLCS uses no axis weight and shares its v2 physical beta with matching, while v1 beta remains 1. Tests assert these exact defaults. |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | PASS | Canonical metadata stores schema, version, scale, `m`, and `m/s`; BLCS/PLCS writers inject identical root/scene values, and standard plus compact-production loaders reject missing, unknown, mixed, or mismatched documents. All 2,000 inspected v2 scenes matched their roots. |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | PASS | BLCS/PLCS/SLCS save hooks persist root metadata; inference loaders restore or validate it against saved config/runtime before state load. Checkpoint round-trip/mismatch tests passed. Representative `.ckpt.bin` fixtures are valid Torch ZIP archives, load as Lightning mappings, retain exact SHA-256, and correctly exercise metadata-free v1 behavior. |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | FAIL | Materializer and shipped baseline names are version-qualified, but the primary PLCS generator violates non-overwrite: `scripts/generate_dataset.py` creates an existing output and overwrites `config.yaml` before writer construction, while `PLCSDatasetWriter` accepts non-empty roots and existing scene directories. A direct probe showed v2 config accepts `run.output_dir=plcs_broadcast`, a v2 writer accepts a non-empty legacy root, and v2 training accepts an output path containing `norm-v1`; therefore neither dataset nor checkpoint naming/preservation is fail-closed. |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | PASS | Separate local `blcs_broadcast_norm_v2` and `plcs_broadcast_norm_v2` artifacts each contain 1,000 scenes and complete v2 metadata. All-scene independent array/hash scans matched 2,000 manifest entries with maxima `3.814697265625e-06m` and `9.5367431640625e-07m`. |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | PASS | Four run nodes and their group record completed BLCS/PLCS v1/v2 training with aggregate and X/Y/Z metre metrics. BLCS records `2.405233→2.338450m`; PLCS records `0.469883→0.313734m`, with its resumed/batch-changed limitation explicit. Knowledge validation reports zero errors. |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | PASS | `test_actual_blcs_plcs_cpu_flow_reaches_projection_and_render` passed for v1 and v2, loading real fixtures and executing CPU model forward, loss, metric, denormalization, BLCS projection, PLCS 3D/top-down render, and metre-valued `SceneResult` integration. |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | PASS | The 232-test focused matrix covers shared schema/config/checkpoints, court pose, every named BLCS and PLCS consumer, SLCS conversion/uncertainty, merged model configs, v1 parity, and two-version smoke. An additional 62 compact synthetic-PLCS tests passed. |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | PASS | BLCS, PLCS, SLCS, tennis-scene, and synthetic-PLCS docs cover formulas, v1 default, units, legacy/mismatch behavior, naming, and migration commands, consistently pointing mathematical authority to `src/utils/schema/court_normalization.py` and metadata authority to the shared artifact contract. |

## Code evidence

- Mathematical authority: `src/utils/schema/court_normalization.py`; unchanged physical geometry: `src/utils/schema/court.py`; typed selection: `src/tasks/base/configuration.py`.
- Dataset/checkpoint contracts: `src/tasks/base/data/court_coordinate_contract.py`, `court_coordinate_materializer.py`, and `src/tasks/base/model_io/court_coordinate_contract.py`.
- BLCS consumers: generator physics/scene I/O, standard/tracking data, losses, projection, metrics, predictors, Lightning hooks, and versioned configs under `src/tasks/blcs`.
- PLCS consumers: generator/targets, standard/tracking losses and matching, predictors/metrics, checkpoint binding, court-pose geometry, renderer, compact synthetic generation, and versioned configs under `src/tasks/plcs` and `src/synthetic_data_generation/dataset/plcs`.
- SLCS and integration consumers: normalization/data/evaluation/adapter/inference/metrics/checkpoints under `src/tasks/slcs`, and metre-valued provenance contracts under `src/tennis_scene`.
- AC-017 defect path: `src/tasks/plcs/scripts/generate_dataset.py::main`, `src/tasks/plcs/generate_dataset/io/dataset_io.py::PLCSDatasetWriter`, `src/tasks/plcs/generate_dataset/config.py::PLCSGenerationConfig`, and PLCS training output-path validation omit the non-empty/version-qualified protections already present for the materializer and BLCS.

## Runtime and test evidence

- Candidate identity: `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` → exact sealed SHA-256 before and after inspection.
- Focused normalization/merged-boundary suite → `232 passed in 28.76s`.
- Compact synthetic PLCS suite → `62 passed in 17.79s`.
- Existing v2 artifact scan validated every root/scene contract and every manifest hash: BLCS 1,000 scenes, max error `3.814697265625e-06m`; PLCS 1,000 scenes, max error `9.5367431640625e-07m`; zero shape/hash mismatches.
- Frozen fixture bytes: BLCS checkpoint `100421` bytes, SHA-256 `69af21b3f8008ab7f53708e1d03346113aafa49c857a5465ad6e6da86f80a5e7`; PLCS checkpoint `71896` bytes, SHA-256 `6c212eec6bbe616b498000928733318b19f76e4ad957a680963621c672ca841d`. Both are ZIP-formatted Torch archives and load as Lightning checkpoint dictionaries with 43/45 state keys.
- AC-017 direct read-only/temporary probe → v2 generation accepted legacy-named `data/plcs_broadcast`; v2 training accepted `outputs/plcs/i786/norm-v1/misnamed-v2`; a v2 `PLCSDatasetWriter` accepted a non-empty temporary legacy root without rejecting its sentinel.
- `.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py` → 181 nodes, zero errors, four unrelated pre-existing warnings.

## Regression and repository-rule checks

- `.venv/bin/ruff check src tests` → PASS, all checks passed.
- `.venv/bin/mypy --follow-imports=skip src tests` → PASS, no issues in 1,120 source files.
- Changed script convention reviewer over all twelve non-`__init__` Issue scripts → PASS; each has the required overview/Usage/Notes, Hydra config, and no argparse.
- `git diff --check` for both frozen-base and `origin/main...HEAD` code scopes, excluding immutable workflow artifacts → PASS. The unrestricted command reports only the exactly rendered frozen `issue.md` terminal blank line; changing that hash-bound authority is prohibited and production/test code is clean.
- Full Issue fixture and code tests preserve the legacy v1 numerical contract after the `64ea1b5a` main merge. The only acceptance defect found is the reproducible PLCS overwrite/misnaming path in AC-017.

## Final verdict

RETURN

## RETURN exploration questions

1. Should every PLCS dataset-generation entrypoint reject a non-empty output directory before writing even `config.yaml`, and require a `norm-v2` version token whenever v2 is selected, matching the BLCS/materializer fail-closed boundary?
2. Should PLCS v2 training likewise reject checkpoint/output roots that are not version-qualified as v2 so a v2 checkpoint cannot be published under a v1 artifact name?
3. Which focused regression tests will prove that failed v2 generation/training attempts leave representative existing v1 dataset/checkpoint bytes unchanged while normal version-qualified v1 and v2 paths still succeed?
