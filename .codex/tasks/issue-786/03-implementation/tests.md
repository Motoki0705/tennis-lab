# Tests

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`

## Candidate identity

Test cycle 1 began from the matching Preflight PASS candidate `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`. The Test Writer added only tests and committed test fixtures, including representative metadata-free BLCS/PLCS legacy artifacts generated from frozen base commit `59e3b166c2d010d5e62be52c2be76d98a94af0e0`. No production source, configuration, documentation, knowledge, plan, `checks.json`, Issue/state, or reviewer artifact was modified. The post-test candidate is `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`; every final canonical result in `test-checks.json` is bound to this candidate.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | Resolver mapping, immutability, unknown-version, malformed-shape, and geometry landmark tests in `unit-contract`; complete regression in `full-pytest`. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | Hydra composition and task binding for BLCS/PLCS/SLCS/tennis-scene default v1 and explicit v2; task checks and `integration-normalization`. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | New exact frozen-base BLCS/PLCS metadata-free checkpoint/dataset parity covers actual loader, collation, Lightning restore, predictor, loss, and metrics against immutable CPU goldens at `atol=1e-5, rtol=0`; new standard/tracking predictor tests fix default-v1 metre scale. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | New legacy parity tests prove metadata-free v1 acceptance and v2 rejection before Lightning state restore or NumPy payload access; PLCS root/scene missing-metadata matrix and existing cross-task guards also pass. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | Runtime/root/scene/checkpoint mismatch matrices cover version, scale, unit, partial, and mixed contracts, including public PLCS root/scene entry points; focused task and integration checks pass. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | NumPy/Torch arbitrary-leading-shape, dtype/device, position/velocity, invalid-shape, and `1e-5m` round-trip cases for both versions in `unit-contract`, task checks, and full regression. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | v2 landmark and aspect assertions use unchanged physical court constants; `unit-contract` and `preflight-regression` pass. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | BLCS physics/generation, datasets, standard/tracking predictors, projection/loss, and metric tests cover both versions; new legacy parity also checks normalized model velocity against persisted world m/s via the v1 contract. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | Gravity formula, v1 literal, v2 derived target, physical acceleration, and metre-equivalence tests in `unit-blcs` and integration pass. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PLCS generation, targets, loss, predictor, metric, canonical/world, and renderer coverage passes; the strengthened smoke uses real model/collator/Lightning paths plus 3D and top-down rendering for both versions. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | v1/v2 target/generation invariance and strengthened real-model render flow prove canonical metre pose remains unchanged while only position translation is version-normalized. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | SLCS dataset/adapter/predictor/metric/checkpoint/scalar-uncertainty coverage and strengthened `SceneResult` metre-preservation smoke pass. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | BLCS/PLCS equal-axis physical-error cases and common 1.0m Huber-transition assertions pass in task and integration checks. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | Standard/tracking loss and Hungarian cases assert uniform v2 behavior; exact frozen-base parity and predictor physical-scale tests retain v1 behavior. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | Exact metadata serialization and root/scene missing, unknown, partial, mixed, unit, scale, and version failures are covered; new PLCS cases prove failure before scene payload loading. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | Shared/task checkpoint write, restore, legacy, mismatch, and predictor-binding tests pass; new actual Lightning legacy restore and pre-restore v2 rejection strengthen the boundary. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | Materializer destination identity, non-overwrite, version-qualified naming, root/scene identity, and checkpoint guards pass in contract and integration checks. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | Separate bounded BLCS/PLCS v2 materialization fixtures and generation round trips at `1e-5m` pass in integration/task checks. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | Four completed run nodes and their comparison group retain axis-wise and aggregate metre metrics; `knowledge-graph` validates 181 nodes with 0 errors. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | Strengthened smoke replaces generic NumPy/Identity substitutes with versioned copied datasets, actual task datasets/collators, tiny actual BLCS/PLCS models, Lightning forward/loss/metrics, metre decode, real projection, and PLCS 3D/top-down render for v1 and v2. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | Six focused groups pass 174 tests, repository hooks pass, and the canonical GPU-visible full suite passes 3245 tests. | PASS |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | Public contract/default assertions, shared-schema-linked task/tennis-scene documentation, repository hooks, knowledge graph validation, and full regression pass. | PASS |

## Tests added or changed

- `tests/integration/tasks/test_legacy_v1_checkpoint_parity.py`: seven adversarial tests verify frozen-base provenance/hashes, metadata-free v1 dataset and checkpoint identity, actual BLCS/PLCS datasets and collators, strict Lightning restoration, actual predictors, loss/metrics, immutable CPU goldens, and fail-before-payload v2 rejection.
- `tests/fixtures/issue_786/legacy_v1_representative/`: exact frozen-base metadata-free BLCS/PLCS checkpoints, immutable goldens, loader-compatible datasets, a hash manifest, and the exact fixture generator source stored as `generate_representative.py.txt` so it is preserved without becoming active test Python.
- `tests/integration/tasks/test_court_coordinate_normalization_smoke.py`: replaces the nominal NumPy/Identity smoke with actual datasets, collators, tiny BLCS/PLCS models, Lightning loss/metrics, metre decode, differentiable projection, PLCS 3D/top-down drawing, and `SceneResult` metre preservation for v1/v2.
- `tests/unit/tasks/blcs/inference/test_predictor.py` and `tests/unit/tasks/blcs/inference/test_tracking_predictor.py`: add default-v1 physical-scale regression assertions for standard and tracking inference.
- `tests/unit/tasks/plcs/inference/test_predictor.py` and `tests/unit/tasks/plcs/inference/test_tracking_predictor.py`: add default-v1 physical-scale regression assertions for standard and tracking inference.
- `tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py`: adds all-missing v2, root-present/scene-missing, root-missing/scene-present, and v2-artifact-under-v1-runtime cases, each proving contract rejection before NumPy scene payload use.

No production source, configuration, documentation, knowledge, plan, `checks.json`, Issue/state, or reviewer content was changed by this Test Writer.

## Normal, boundary, invalid, and regression cases

### Independent bounded risk model

| Risk perspective | Failure mode challenged | Authority and coverage |
|---|---|---|
| Frozen-base representative legacy v1 parity | Current defaults appear v1-compatible but real metadata-free dataset/checkpoint load, model state, inference, loss, or metrics drift from the frozen base | `BASELINE_REGRESSION`: frozen commit `59e3b166c2d010d5e62be52c2be76d98a94af0e0`, exact artifact/tree hashes, and immutable goldens; seven new actual-pipeline parity tests pass at `atol=1e-5, rtol=0` |
| Legacy artifacts under v2 | Metadata-free checkpoint or dataset reaches Lightning state restoration or NumPy payload access before the contract guard rejects it | `ISSUE_CONTRACT`: AC-004/AC-005 and the prohibition on inference from shape/value range; sentinels prove rejection before state/array payload use |
| Nominal smoke substitution | A generic `Identity`/NumPy test passes while real task dataset, collation, model forward, Lightning loss/metric, projection, or rendering is broken | `ISSUE_CONTRACT`: AC-020's named end-to-end CPU flow; strengthened smoke executes actual bounded BLCS/PLCS components for v1/v2 |
| Default-v1 inference scale | Standard or tracking predictor silently defaults to v2 or returns normalized units instead of physical metres | `PUBLIC_CONTRACT` and AC-003/AC-008/AC-010: new BLCS/PLCS standard/tracking predictor scale assertions plus frozen parity |
| PLCS public metadata bypass | Root and scene metadata combinations are accepted through the public loader or rejected only after arrays have been consumed | `ISSUE_CONTRACT`: AC-004/AC-005/AC-015; four new root/scene matrix cases with payload sentinels |
| PLCS translation-only normalization | Canonical root-relative metre pose is rescaled together with normalized translation, corrupting world pose and render output | `ISSUE_CONTRACT`: AC-010/AC-011; actual target, world-pose, and 3D/top-down render coverage for both versions |
| BLCS velocity unit boundary | Persisted physical `ball_vel_world` in m/s is compared to or consumed as normalized model velocity, or decoded with the wrong scale | `ISSUE_CONTRACT`: AC-008 and metadata unit contract; legacy parity denormalizes the current v1 model target before exact comparison to the frozen physical golden |
| Cross-task and repository regression | Focused normalization tests pass while shared schema/config, SLCS, unrelated callers, static checks, or GPU-visible baseline regresses | `REPO_INVARIANT` and `BASELINE_REGRESSION`: all required focused checks, repository hooks, knowledge validation, and 3245-test GPU-visible full suite pass on one candidate |

### Executable case classes

- Normal: v1/v2 contract selection; actual BLCS/PLCS dataset load, collation, model forward, Lightning loss/metric, metre decode, projection/render; standard/tracking predictors; checkpoint restore; SLCS metre conversion; `SceneResult` metre output.
- Boundary: exact frozen commit/hash provenance, metadata-free legacy artifacts, arbitrary `(...,3)` shapes, one-axis physical errors, common Huber transition, canonical-pose invariance, normalized velocity versus persisted m/s, root-only/scene-only metadata, and actual renderer output.
- Invalid: unknown/malformed contracts, metadata-free v2, v2 artifact under v1 runtime, partial/mixed/mismatched root-scene metadata, unit/scale/version mismatch, incompatible checkpoint restore/write, and invalid materializer identity.
- Regression: composed default v1, exact representative frozen-base inference/loss/metric goldens, predictor physical scale, physical court dimensions, v1 axis behavior, v2 isotropy, repository static hooks, knowledge evidence, and complete GPU-visible suite.

### Independent AT probes

No machine-recorded `AT-*` row was executable. The repository-pinned schema-v5 `manage_issue_task.py --help` exposes no `run-test-probe` subcommand, and repository workflow scripts contain no implementation of that command. Inventing an `AT-*` record would be unsupported and irreproducible. The independently derived adversarial perspectives above were therefore executed through candidate-bound canonical checks, including the focused legacy parity check and complete suite, rather than recorded as false `AT-*` rows.

## Canonical command results

Every command was invoked through `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test <check-id>` on candidate `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`.

| Canonical ID | Outcome | Machine log |
|---|---|---|
| `unit-contract` | PASS — 56 passed in 9.13s | `logs/canonical-test-unit-contract.log` |
| `legacy-v1-checkpoint-parity` | PASS — 7 passed in 9.42s | `logs/canonical-test-legacy-v1-checkpoint-parity.log` |
| `unit-blcs` | PASS — 37 passed in 9.22s | `logs/canonical-test-unit-blcs.log` |
| `unit-plcs` | PASS — 25 passed in 8.65s | `logs/canonical-test-unit-plcs.log` |
| `unit-slcs` | PASS — 35 passed in 9.30s | `logs/canonical-test-unit-slcs.log` |
| `integration-normalization` | PASS — 14 passed in 9.81s | `logs/canonical-test-integration-normalization.log` |
| `preflight-regression` | PASS — 127 passed in 10.06s | `logs/canonical-test-preflight-regression.log` |
| `knowledge-graph` | PASS — 181 nodes, 0 errors, 4 pre-existing warnings | `logs/canonical-test-knowledge-graph.log` |
| `precommit-all` | PASS — ruff, mypy, task-script-reviewer | `logs/canonical-test-precommit-all.log` |
| `full-pytest` | PASS — 3245 passed, 53 skipped, 19 warnings in 689.68s | `logs/canonical-test-full-pytest.log` |

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` returned `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0` before writing this artifact.
- All ten required canonical check IDs passed on that stable post-test candidate, and `test-checks.json` records exit code 0 and matching candidate identity for every result.
- `legacy-v1-checkpoint-parity` passed seven tests using actual CPU task paths and immutable frozen-base artifacts.
- `precommit-all` passed ruff, mypy, and task-script-reviewer.
- `full-pytest` ran with the canonical GPU 0 authority and passed 3245 tests with the established 53 skips.

## Failures encountered

The first legacy-parity execution failed because the new test compared the current normalized BLCS model velocity target directly with the frozen physical `target_velocity_world_mps` golden. The test oracle was corrected to denormalize the current v1 target before comparing it with the persisted m/s value. The rerun passed. This was a Test Writer unit-boundary error, not a production defect or RETURN authority; production code was not changed.

## Untested risks and reasons

- No machine-recorded `AT-*` probes could be created because the repository-pinned schema-v5 manager lacks `run-test-probe`; all executable adversarial perspectives were covered by candidate-bound canonical checks instead.
- Large archived checkpoints that the frozen base itself cannot load remain intentionally outside legacy migration/parity scope; the representative fixtures are exact frozen-base artifacts whose loader compatibility is proven on both sides of the baseline.
- Training quality was not regenerated by CPU tests. AC-019 is evidenced by committed completed knowledge runs and their metre-valued comparison, validated by the canonical knowledge check.
- The complete regression required GPU authority and passed under the canonical GPU 0 environment recorded by `full-pytest`; no additional GPU topology or multi-GPU behavior was required by the frozen Issue contract.

## Final test verdict

PASS

## RETURN implementation findings

None
