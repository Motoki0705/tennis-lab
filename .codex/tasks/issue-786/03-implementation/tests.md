# Tests

- Issue: #786
- Attempt: 6
- Test cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`

## Candidate identity

The post-test candidate is `sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`, identical to the attempt-6 Preflight candidate before and after all test-stage commands. The focused repaired boundary is `tests/integration/tasks/test_court_coordinate_normalization_documentation.py::_read`: it binds `Path.read_text(encoding="utf-8")` to a local `text: str` and returns that value without a cast, ignore, fallback, exception handling, or runtime transformation.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | `unit-contract` exercises the resolver, exact scales, invalid versions, and round trips; `preflight-regression`, `precommit-all`, and `full-pytest` pass. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | `unit-blcs`, `unit-plcs`, `unit-slcs`, `integration-normalization`, and `preflight-regression` pass their v1/v2 composition and propagation cases. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | `legacy-v1-checkpoint-parity` passes 8 frozen representative parity cases; task unit suites and `integration-normalization` pass. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | `legacy-v1-checkpoint-parity`, `unit-contract`, task model-I/O tests, and `integration-normalization` pass the v1 acceptance/v2 rejection matrix. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | `unit-contract`, task unit suites, `integration-normalization`, and `preflight-regression` pass mismatch and malformed-contract rejection cases. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | `unit-contract` and all three task unit suites pass shape/broadcast and metre round-trip coverage; integration smoke passes. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | `unit-contract` and `preflight-regression` pass exact physical-constant and normalized landmark/aspect assertions. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | `unit-blcs` passes 39 generation/loss/predictor/metric/model-I/O cases; `integration-normalization` passes the real CPU chain for both versions. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | `unit-blcs` gravity and tracking tests plus `integration-normalization` pass for v1 and v2. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | `unit-plcs` passes 27 target/generation/loss/predictor cases; `integration-normalization`, `preflight-regression`, and `full-pytest` cover integration/render regressions. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | `unit-plcs` and `integration-normalization` pass canonical-pose/metre invariance and translation-only scale assertions. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | `unit-slcs` passes 35 dataset/adapter/predictor/metric/SceneResult cases; integration and regression suites pass. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | `unit-blcs`, `unit-plcs`, `integration-normalization`, and `normalization-documentation` pass physical-error symmetry, beta boundary, and documentation assertions. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | Both task unit suites and integration smoke pass v2 uniform-weight and v1 legacy-weight regressions. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | `unit-contract`, task loader tests, and `integration-normalization` pass metadata serialization and fail-loud root/scene validation cases. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | `unit-contract`, BLCS/PLCS/SLCS model-I/O and predictor cases, legacy parity, and integration smoke pass. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | `plcs-artifact-preservation` passes 77 name/occupancy/byte-preservation cases; `unit-contract`, integration smoke, pre-commit, and full suite pass. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | BLCS/PLCS task suites and `integration-normalization` pass separate-root materialization and metre reconstruction cases. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | `knowledge-graph` validates 181 nodes with 0 errors, including the Issue #786 v1/v2 BLCS/PLCS run/group evidence; `precommit-all` and `full-pytest` pass. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | `integration-normalization` passes all 14 real task CPU-smoke cases; task unit and full regression suites pass. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | All focused unit/integration checks pass. Broad `candidate-python-mypy` reports 0 issues in 1124 files and isolated `documentation-test-mypy` reports 0 issues in the repaired file; `precommit-all` and 3393-test full regression pass. | PASS |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | `normalization-documentation` passes 4 routing/content/anti-duplication assertions; `unit-contract`, `unit-slcs`, `knowledge-graph`, `precommit-all`, and full suite pass. | PASS |

## Tests added or changed

No source test or fixture was added or changed during this Test Writer stage. The candidate already contains the repair-local change in `tests/integration/tasks/test_court_coordinate_normalization_documentation.py::_read`: the direct UTF-8 `read_text` result is assigned to local `text: str` before return. The Test Writer inspected that exact boundary and retained all existing documentation paths and assertions. Only this `tests.md` record and the workflow-owned `test-checks.json`/raw logs were replaced or generated; workflow artifacts are excluded from the candidate fingerprint.

## Normal, boundary, invalid, and regression cases

The bounded repair-local risk model was:

| Risk | Authority and oracle | Evidence | Outcome |
|---|---|---|---|
| Isolated `--follow-imports=skip` analysis treats imported `PROJECT_ROOT`/path behavior as `Any` and exposes `no-any-return`. | AC-021 plus repository strict-mypy configuration; the one-file hook-equivalent command must succeed without suppressions. | `_read` declares local `text: str`; `documentation-test-mypy` reports no issues in 1 source file. | PASS |
| A cast might satisfy isolated analysis but become redundant under broad analysis. | AC-021 and `warn_redundant_casts = true`; both isolated and broad commands are required oracles. | No `cast` import/call exists in the documentation test; `candidate-python-mypy` reports no issues in 1124 files. | PASS |
| A `type: ignore`, fallback, or mypy/config weakening could hide the original contract leak. | Repository fail-loud and strict-typing invariants; runtime behavior and configuration must remain explicit. | No ignore/fallback/exception handling occurs in `_read`; neither `pyproject.toml` nor `.pre-commit-config.yaml` is changed in the candidate; `precommit-all` passes its mypy hook. | PASS |
| The annotation could accidentally change UTF-8 decoding, path resolution, failure propagation, or documentation assertions. | Python `Path.read_text` public contract and repository no-silent-fallback invariant. | `_read` still evaluates `(PROJECT_ROOT / relative_path).read_text(encoding="utf-8")` exactly once and returns the same object; there is no exception interception. `normalization-documentation` executes all 4 assertion groups across 13 routed entry points and passes. | PASS |
| The narrow repair could mask wider normalization or repository regressions. | Full Issue checklist and established repository baseline. | All focused contract/task/integration checks, hook checks, and `full-pytest` pass on the identical candidate. | PASS |

State records `adversarial_testing_mode="LEGACY"`; therefore `run-test-probe` is forbidden by the workflow and no machine-recorded `AT-*` perspective was executable. Independent repair-local coverage is recorded above using the 14 frozen canonical checks and direct contract inspection, without migrating state or inventing additional requirements.

Normal cases covered v1/v2 resolver use, task propagation, documentation routing, publication into missing/empty destinations, representative inference/loss/metric paths, and both broad and isolated static analysis. Boundary cases covered arbitrary `(..., 3)` shapes, physical Huber transitions, exact v2 court landmarks, empty publication roots, and the local annotation boundary under two mypy import scopes. Invalid cases covered unknown/missing/mixed/mismatched metadata, v2 access to metadata-free artifacts, occupied publication paths, missing wrapped entrypoints, and unknown normalization versions. Regression cases covered frozen v1 checkpoint/data goldens, task-specific canonical/metre semantics, full pre-commit, and the complete repository pytest suite.

## Canonical command results

| Check ID | Exit | Machine result | Exact outcome |
|---|---:|---|---|
| `unit-contract` | 0 | `logs/canonical-test-unit-contract.log` | PASS — 56 passed in 14.51s. |
| `plcs-artifact-preservation` | 0 | `logs/canonical-test-plcs-artifact-preservation.log` | PASS — 77 passed in 22.68s. |
| `legacy-v1-checkpoint-parity` | 0 | `logs/canonical-test-legacy-v1-checkpoint-parity.log` | PASS — 8 passed in 12.37s. |
| `unit-blcs` | 0 | `logs/canonical-test-unit-blcs.log` | PASS — 39 passed in 12.25s. |
| `unit-plcs` | 0 | `logs/canonical-test-unit-plcs.log` | PASS — 27 passed in 11.69s. |
| `unit-slcs` | 0 | `logs/canonical-test-unit-slcs.log` | PASS — 35 passed in 14.24s. |
| `integration-normalization` | 0 | `logs/canonical-test-integration-normalization.log` | PASS — 14 passed in 11.37s. |
| `preflight-regression` | 0 | `logs/canonical-test-preflight-regression.log` | PASS — 167 passed in 10.88s. |
| `knowledge-graph` | 0 | `logs/canonical-test-knowledge-graph.log` | PASS — 181 nodes, 0 errors, 4 unrelated warnings. |
| `normalization-documentation` | 0 | `logs/canonical-test-normalization-documentation.log` | PASS — 4 passed in 3.40s. |
| `candidate-python-mypy` | 0 | `logs/canonical-test-candidate-python-mypy.log` | PASS — no issues in 1124 source files. |
| `documentation-test-mypy` | 0 | `logs/canonical-test-documentation-test-mypy.log` | PASS — no issues in 1 source file. |
| `precommit-all` | 0 | `logs/canonical-test-precommit-all.log` | PASS — ruff, mypy, and task script reviewer passed. |
| `full-pytest` | 0 | `logs/canonical-test-full-pytest.log` | PASS — 3393 passed, 78 skipped, 18 warnings in 727.82s. |

All 14 mandatory test-stage rows in `03-implementation/test-checks.json` are bound to candidate `sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`, have exit code 0, and record `PASS`.

## Commands and exact outcomes

Every command was executed through `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test <check-id>` so the frozen argv, cwd, and environment came only from `02-planning/checks.json`.

- `unit-contract`, `plcs-artifact-preservation`, `legacy-v1-checkpoint-parity`, `unit-blcs`, `unit-plcs`, `unit-slcs`, `integration-normalization`, `preflight-regression`, and `normalization-documentation`: exit 0 with 427 focused pytest cases passed in aggregate.
- `knowledge-graph`: exit 0; 181 nodes validated with 0 errors and 4 pre-existing missing-issue warnings for unrelated BLCS compile nodes.
- `candidate-python-mypy`: exit 0; `Success: no issues found in 1124 source files`.
- `documentation-test-mypy`: exit 0; `Success: no issues found in 1 source file`.
- `precommit-all`: exit 0; ruff (fix), mypy, and task script reviewer all passed without changing the candidate fingerprint.
- `full-pytest`: exit 0; 3393 passed, 78 skipped, 18 warnings in 727.82s.
- Candidate fingerprint before checks and after checks: `sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`.

## Failures encountered

None. All mandatory checks passed on their first Test Writer execution.

## Untested risks and reasons

No repair-local AC-021 risk remains untested: the exact local annotation was covered by both isolated and broad hook-equivalent mypy, its UTF-8 runtime/assertion path by the documentation integration test, and the absence of silent fallback/config weakening by direct inspection plus pre-commit. The repository-wide suite skipped 78 tests according to established markers/environment; none is a supported repair-local oracle for the documentation-test typing bridge, and all 14 required checks passed.

## Final test verdict

PASS

## RETURN implementation findings

None
