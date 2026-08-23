# Tests

- Issue: #786
- Attempt: 1
- Test cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:f6b6168ccdb5acb6ce84295b07af8895c58692650fb2649d759be8f6e894ecf7`

## Candidate identity

The independent test pass began from the Preflight PASS candidate `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3`. Test-only additions, corrections, and formatting produced the post-test candidate `sha256:f6b6168ccdb5acb6ce84295b07af8895c58692650fb2649d759be8f6e894ecf7`. Every final canonical result in `test-checks.json` is bound to that post-test candidate. No production source, configuration, plan, or `checks.json` file was changed by the Test Writer.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | `test_court_normalization.py` fixes both immutable scale tuples, resolver identity, unknown-version rejection, and invalid shape behavior; `unit-contract` passed. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | `test_hydra_boundaries_default_to_v1_and_explicitly_compose_v2` covers BLCS/PLCS/SLCS and tennis-scene entry configs; task unit suites and `integration-normalization` passed. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | Hydra default assertions plus legacy predictor/loss/metric/checkpoint regressions in BLCS, PLCS, and SLCS; `preflight-regression` passed 125 tests. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | Base data/model-I/O contract tests exercise metadata-free v1 acceptance and v2 rejection, including a metadata-free dataset root; `unit-contract`, `unit-blcs`, and `unit-plcs` passed. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | Root/scene runtime mismatch matrix and checkpoint restore/write mismatch cases cover both version and scale; all four unit groups and integration passed. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | NumPy and Torch tests cover arbitrary leading dimensions, float dtype/device preservation, both versions, and invalid last dimensions; maximum-error assertions use `1e-5m`. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | Court-pose and schema tests assert v2 landmark coordinates and unchanged metre dimensions; `unit-contract` passed. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | BLCS physics, predictor, tracking predictor, standard/tracking metric, and cross-task projection smoke tests cover the full consumer chain for v1/v2; `unit-blcs` and integration passed. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | Ball-physics and tracking loss/metric tests assert version-dependent normalized gravity and metre-equivalent second differences; `unit-blcs` passed. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PLCS scene generation, target, loss, predictor/tracking predictor, metric, and renderer integration cases run both contracts; `unit-plcs` and integration passed. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | Scene-generator and world-target tests assert identical canonical arrays across v1/v2 and only translation scaling; metre pose renderer assertions passed. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | SLCS metric, adapter, and predictor tests cover position/uncertainty conversion; `test_schema.py` and integration assert metre-preserving `SceneResult` provenance. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | BLCS and PLCS loss cases inject equal physical axis errors and assert equal v2 loss plus the common 1m Huber transition; unit and integration checks passed. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | BLCS/PLCS default-loss and matching cases assert uniform v2 position weights while legacy v1 numeric regressions retain configured weighting. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | Base data contract and integration materialization tests cover serialization and missing/unknown/mixed/mismatched root-scene combinations with unit validation. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | Base model-I/O and BLCS checkpoint tests assert write, restore, legacy, mismatch, and predictor binding behavior; canonical unit groups passed. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | Materializer integration asserts versioned destination naming, refusal to overwrite, and root/scene metadata identity; checkpoint filename/metadata guards are covered by model-I/O tests. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | Integration materializes BLCS and PLCS v2 fixtures separately and asserts stored-normalized to physical-world round trips at `1e-5m`; generation unit cases agree. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | `group-i786-normalization-v1-v2` links four completed run nodes with axis-wise and aggregate metre metrics. `knowledge-graph` reported 181 nodes, 0 errors, and 4 unrelated existing warnings. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | `test_both_versions_cpu_dataset_model_loss_metric_projection_render_and_metre_outputs` executes the requested CPU sequence for both tasks and versions; `integration-normalization` passed. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | The new and expanded normalization suites cover every enumerated consumer and all focused groups passed, but the required repository-wide `full-pytest` authority failed under its CUDA-hidden/private-NHT-missing environment, so the mandatory test gate is not closed. | FAIL |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | Public contract documentation references plus schema/config default assertions are covered by `unit-contract`; `knowledge-graph` and `precommit-all` passed. | PASS |

## Independent adversarial risk model

| Risk perspective | Failure mode challenged | Oracle and coverage |
|---|---|---|
| Contract identity and numeric boundaries | Mutable/wrong scales, unknown versions, bad `(...,3)` inputs, dtype/device drift, or non-invertible conversion | Frozen AC-001/006/007; NumPy/Torch schema and court-pose cases for both versions |
| Artifact state combinations | Missing, mixed, unknown, or mismatched root/scene/checkpoint metadata silently falls back or crosses versions | Frozen AC-004/005/015/016/017; exhaustive base data/model-I/O matrices and materializer cases |
| Changed consumer skew | One BLCS/PLCS/SLCS caller continues using a fixed legacy scale while peers use the selected contract | Frozen AC-008/010/012/021; generator, loss, matching, predictor, metric, projection, renderer, adapter, and integrated-scene tests |
| Physical semantics | Gravity, velocity, canonical pose, uncertainty, or output metres are scaled as positions or scaled twice | Frozen AC-009/011/012/013/014; metre-oracle gravity/loss/canonical/uncertainty/`SceneResult` assertions |
| Legacy regression | Default v1, metadata-free v1, prior loss weights, predictor output, or checkpoint restore changes numerically | Frozen AC-003/004/014; legacy assertions plus 125-test `preflight-regression` |
| Repository/data boundary | Shared normalization metadata/schema imports are rejected by stale architectural policy, or an overly broad task-internal allowlist is introduced | Established repository configuration/public-boundary invariants; exact-symbol and exact-module policy updates, focused 8-test pass, and final full-suite passage of those nodes |
| Repository-wide baseline | Issue-local tests pass while unrelated callers or infrastructure regress | Required `full-pytest`; it reached 3206 passes but exposed unresolved CUDA/private-NHT environment failures described below |

## Independent AT probes

No machine-recorded `AT-*` probe was executable. The repository-pinned `manage_issue_task.py --help` has no `run-test-probe` subcommand, and the workflow sources contain no probe implementation. Parent authority explicitly prohibited syncing or modifying the workflow tool and directed that independent cases remain supporting evidence rather than fabricated machine records. Accordingly, no `AT-*` ID was assigned; the independent perspectives above were executed through added test cases, focused pytest iteration, and the canonical `run-check` IDs.

## Tests added or changed

Added:

- `tests/unit/utils/schema/test_court_normalization.py`
- `tests/unit/tasks/base/data/test_court_coordinate_contract.py`
- `tests/unit/tasks/base/model_io/test_court_coordinate_contract.py`
- `tests/unit/tasks/blcs/generate_dataset/simulation/test_ball_physics.py`
- `tests/unit/tasks/plcs/generate_dataset/test_scene_generator.py`
- `tests/unit/tasks/slcs/training/test_metrics.py`
- `tests/unit/tennis_scene/test_schema.py`
- `tests/integration/tasks/test_court_coordinate_normalization_smoke.py`

Expanded:

- BLCS: `test_predictor.py`, `test_tracking_predictor.py`, `test_checkpoints.py`, `test_losses.py`, `test_tracking_matching.py`, and `test_tracking_metrics.py`
- PLCS: `test_targets.py`, `test_predictor.py`, `test_tracking_predictor.py`, and `test_losses.py`
- SLCS: `test_predictor.py` and `test_adapter.py`
- Shared geometry: `tests/unit/utils/geometry/test_court_pose.py`
- Exact repository-policy accommodation for the new public boundaries: `tests/unit/utils/configuration/test_contracts.py` permits only `CourtCoordinateNormalization`, and `tests/e2e/synthetic_data_generation/test_removed_architecture.py` permits only the public `src.tasks.base.data` re-export.

## Normal, boundary, invalid, and regression cases

- Normal cases: v1/v2 resolver use, Hydra default/switching, BLCS/PLCS generation and inference, SLCS metre conversion, checkpoint restore, dataset materialization, and integrated CPU flow.
- Boundary cases: arbitrary leading dimensions ending in three coordinates, NumPy/Torch dtype and device retention, court endpoints/net height, equal physical errors on each axis, one-metre Huber transition, root-relative pose invariance, and versioned artifact naming/non-overwrite.
- Invalid cases: unknown versions, invalid array shapes, metadata-free v2 use, missing/unknown/mixed/mismatched root and scene metadata, version/scale/unit mismatches, checkpoint runtime/metadata mismatches, and incompatible restore/write attempts.
- Regression cases: metadata-free legacy v1 datasets/checkpoints, legacy v1 weights and predictions, prior metric/loss outputs, physical court dimensions, production metre arrays, and narrow public import/configuration-authority invariants.

## Canonical command results

All commands were invoked through `manage_issue_task.py run-check .codex/tasks/issue-786 test <id>` on candidate `sha256:f6b6168ccdb5acb6ce84295b07af8895c58692650fb2649d759be8f6e894ecf7`.

| Canonical ID | Outcome | Machine log |
|---|---|---|
| `unit-contract` | PASS — 56 passed in 10.79s | `logs/canonical-test-unit-contract.log` |
| `unit-blcs` | PASS — 33 passed in 10.50s | `logs/canonical-test-unit-blcs.log` |
| `unit-plcs` | PASS — 25 passed in 13.64s | `logs/canonical-test-unit-plcs.log` |
| `unit-slcs` | PASS — 35 passed in 14.88s | `logs/canonical-test-unit-slcs.log` |
| `integration-normalization` | PASS — 14 passed in 13.09s | `logs/canonical-test-integration-normalization.log` |
| `preflight-regression` | PASS — 125 passed in 13.32s | `logs/canonical-test-preflight-regression.log` |
| `knowledge-graph` | PASS — 181 nodes, 0 errors, 4 pre-existing warnings | `logs/canonical-test-knowledge-graph.log` |
| `precommit-all` | PASS — ruff, mypy, task-script-reviewer | `logs/canonical-test-precommit-all.log` |
| `full-pytest` | FAIL — 14 failed, 3206 passed, 61 skipped, 15 warnings, 2 errors in 715.10s | `logs/canonical-test-full-pytest.log` |

## Commands and exact outcomes

- `manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` returned `sha256:f6b6168ccdb5acb6ce84295b07af8895c58692650fb2649d759be8f6e894ecf7` before the final canonical run.
- An initial focused direct invocation without `PYTHONPATH=.` produced two collection errors (`ModuleNotFoundError: src`); this was an invocation-environment error, not a candidate failure.
- `PYTHONPATH=. .venv/bin/pytest -q tests/unit/tasks/plcs/generate_dataset/test_scene_generator.py tests/unit/utils/configuration/test_contracts.py::test_task_local_boundaries_keep_task_local_configuration_authority tests/e2e/synthetic_data_generation/test_removed_architecture.py::test_canonical_production_imports_only_public_task_boundaries` passed all 8 cases in 38.55s.
- During test iteration, the added PLCS generator assertion was corrected to compare against an explicitly broadcast expected array, stale repository-policy expectations were narrowed to the exact new shared public symbols, and a mypy annotation was added. No production behavior was changed.

## Failures encountered

The required final `full-pytest` check failed only outside the Issue #786 behavior surface:

- CUDA is deliberately hidden by the canonical check (`CUDA_VISIBLE_DEVICES=""`), while one B00 acceptance test, four real ACCAD/SMPL-H coordinate tests, and two ACCAD fixture setups explicitly require CUDA. Two FakeTensor AOT backward tests also fail with `hasPrimaryContext expects a valid device index` when CUDA is hidden.
- `third_party/nht/configs/production.yaml` is absent. Seven synthetic-data configuration tests therefore fail at the prerequisite path check before reaching their intended assertions.
- The final full-suite run contains no failures in the added/changed normalization tests or in the two narrowly updated repository-policy tests.

## Untested risks and reasons

- A real-GPU full-suite normalization regression was not executable because the frozen `full-pytest` environment hides CUDA. CPU coverage for both versions passed.
- Private NHT-backed configuration behavior beyond the prerequisite path check was not executable because the required private config is absent from this worktree.
- The PLCS v1/v2 training comparison is single-seed and not a controlled one-variable experiment: the recorded v2 run changes batch/continuation conditions. The knowledge node states this limitation; it does not invalidate the required recorded baseline but limits causal interpretation.
- No formal `AT-*` machine artifact could be produced because the pinned workflow tool lacks the required command, as recorded above.

## Final test verdict

RETURN

## RETURN implementation findings

- Affected authority: required canonical `full-pytest`, whose manifest row is associated with AC-001 through AC-022. No Issue #786 production assertion failed, so production code must not be changed in response to this RETURN.
- Action required before another Test Writer cycle: reconcile the canonical baseline environment with its own tests. In particular, either authorize a `full-pytest` environment that exposes CUDA or explicitly separate the GPU-required baseline from the CPU check, and provide the expected private NHT config/assets (or revise that unrelated baseline authority through planning). Then rerun every required canonical ID on one stable candidate.
