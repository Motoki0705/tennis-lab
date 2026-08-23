# Tests

- Issue: #786
- Attempt: 1
- Test cycle: 2
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`

## Candidate identity

Test cycle 2 began from the matching Preflight PASS candidate `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`. The frozen Tester RETURN finding was a direct test-only typing defect in `_metadata()` in `tests/unit/tasks/base/data/test_court_coordinate_contract.py`: a cast can be redundant when mypy analyzes all files, while a direct chained return can become `Any` when the commit hook analyzes the staged test with `--follow-imports=skip`. The helper now assigns the unchanged `to_dict()` result to an explicit `dict[str, object]` local before returning it. No production, config, documentation, knowledge, plan, checks, state, or reviewer content was changed by this Test Writer. The post-test candidate is `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`, and every final canonical result in `test-checks.json` is bound to it.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | Resolver mapping, immutability, unknown-version, malformed-shape, geometry-golden tests; `unit-contract` and `full-pytest`. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | Hydra composition for BLCS/PLCS/SLCS/tennis-scene default v1 and explicit v2; task checks and `integration-normalization`. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | Default-v1 composition plus legacy predictor/loss/metric/dataset/checkpoint goldens; `preflight-regression` and `full-pytest`. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | Root/scene/checkpoint metadata-free v1 acceptance and v2 rejection, including absent root `meta.json`; contract/task/integration checks. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | Runtime/root/scene/checkpoint version, scale, unit, partial, and mixed mismatch matrices; all focused groups. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | NumPy/Torch arbitrary-leading-shape, dtype/device, position/velocity, invalid-shape, and `1e-5m` round-trip cases for both versions. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | v2 landmark/aspect assertions against unchanged physical court constants; `unit-contract`. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | BLCS physics/generation, dataset, predictors, projection/loss, and standard/tracking metric tests for both versions; `unit-blcs` and integration. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | Gravity formula, v1 literal, v2 derived target, physical-acceleration, and metre-equivalence cases; `unit-blcs`. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PLCS generation, targets, loss, predictor, metric, canonical/world, and renderer cases; `unit-plcs`, integration, and full suite. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | v1/v2 generation and target invariance plus world-pose/renderer tests prove only translation is normalized. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | SLCS dataset/adapter/predictor/metrics/checkpoint/scalar-uncertainty coverage and `SceneResult` metre/provenance invariants; `unit-slcs` and integration. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | BLCS/PLCS equal-axis physical-error cases and common 1.0m Huber transition assertions; task and integration checks. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | BLCS/PLCS standard/tracking loss and Hungarian cases assert uniform v2 behavior and retained v1 goldens. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | Exact metadata serialization plus root/scene missing, unknown, partial, mixed, unit, scale, and version failures before payload use. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | Shared and task checkpoint write/restore/legacy/mismatch/predictor-binding tests across BLCS/PLCS/SLCS. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | Materializer destination identity, non-overwrite, version-qualified naming, root/scene identity, and checkpoint guards; integration and contract checks. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | Separate bounded BLCS/PLCS v2 materialization fixtures and generation round trips at `1e-5m`; integration/task checks. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | Four completed run nodes and their comparison group carry axis-wise and aggregate metre metrics; `knowledge-graph` validated 181 nodes with 0 errors. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | Cross-task CPU flow executes both tasks and both versions through persisted input, model, loss, metric, denormalization, projection, and render; `integration-normalization`. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | Six Issue-focused groups passed 288 tests, repository hooks passed, and the GPU-visible full suite passed 3230 tests. | PASS |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | Public contract/default assertions, repository hooks, shared-schema-linked task/tennis-scene docs, and `knowledge-graph`; canonical checks passed. | PASS |

## Independent adversarial risk model

| Risk perspective | Failure mode challenged | Oracle and coverage |
|---|---|---|
| Frozen Tester RETURN closure | `_metadata()` passes one mypy import mode but fails the other: an exact typed import makes a cast redundant, while skipped imported bodies make a direct return `Any` | Repository mypy contract and frozen Tester finding: an explicit `dict[str, object]` local removes both context-sensitive paths; single-file `--follow-imports=skip` and all-files `precommit-all` mypy both passed |
| Repair behavioral equivalence | A typing-only repair accidentally changes serialized metadata or weakens its exact schema assertion | Public `CourtCoordinateNormalizationMetadata.to_dict()` contract and AC-015: the helper's runtime expression is unchanged, its 18 file-local cases passed, and `unit-contract` passed all 56 cases |
| Repair ownership and identity | Test cycle 2 mutates production/config/docs or leaves canonical results bound to the pre-repair candidate | Workflow ownership/candidate invariant: the only cycle-2 content repair is the test helper; all nine machine results bind the final `sha256:12a847...` candidate |
| Complete Issue regression | Repair-local closure masks a regression in shared schema/config, BLCS, PLCS, SLCS, materialization, knowledge evidence, or legacy v1 behavior | Frozen AC-001 through AC-022, public interfaces, and repository baseline: six focused groups passed 288 cases, repository hooks passed, and all 3230 repository tests passed |
| Baseline environment preservation | Full regression silently hides mandatory CUDA acceptance or loses the worktree-local NHT authority fixed after Tester cycle 1 | Frozen checks/plan and baseline behavior: `full-pytest` ran with GPU 0 authority and completed 3230 passes with the established 53 skips; no check argv/environment was reconstructed |
| Repair-local discovery bound | A fresh cycle restarts open-ended semantic discovery after the previous RETURN instead of closing the frozen defect | Workflow bound: adversarial work was limited to the direct typing repair, its static/runtime contexts, candidate identity, and complete canonical regression; no new production expectation was invented |

## Independent AT probes

No machine-recorded `AT-*` row was executable. The repository-pinned `manage_issue_task.py --help` has no `run-test-probe` subcommand, and neither its scripts nor workflow sources contain a probe implementation. Inventing an `AT-*` record would therefore be irreproducible and unsupported. The single-file mypy and focused test executions are reported only as supporting commands, not as independent RETURN authority.

## Tests added or changed

- `tests/unit/tasks/base/data/test_court_coordinate_contract.py`: `_metadata()` now stores the unchanged `to_dict()` result in an explicitly typed `dict[str, object]` local and returns that local. This closes both all-files and staged-file mypy contexts without an ignore, cast, or behavioral change.

All pre-existing Issue coverage in the canonical test files was preserved. No production source, configuration, documentation, knowledge, plan, `checks.json`, state, or reviewer artifact was edited by this Test Writer.

## Normal, boundary, invalid, and regression cases

- Normal: exact v1/v2 resolver/config selection; BLCS/PLCS generation, predictors, losses, metrics, projection/rendering; SLCS metre conversions; checkpoint restore; materialization; training-evidence graph.
- Boundary: arbitrary `(...,3)` NumPy/Torch shapes and dtypes/devices, physical court landmarks, one-axis metre errors, 1.0m Huber transition, canonical-pose invariance, scalar uncertainty, non-overwrite identity, and metadata helper typing under complete/skipped imports.
- Invalid: unknown version/shape, metadata-free v2, partial/mixed/unknown/mismatched root-scene metadata, unit/scale/version mismatch, incompatible checkpoint use/write, and source/destination materialization identity.
- Regression: composed default v1, legacy metadata-free v1, historical v1 loss/metric/predictor behavior, physical court/SceneResult units, GPU-required acceptance, NHT-backed baseline, all-files mypy, and staged-file `--follow-imports=skip` mypy.

## Canonical command results

Every command was invoked through `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test <check-id>` on candidate `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`.

| Canonical ID | Outcome | Machine log |
|---|---|---|
| `unit-contract` | PASS — 56 passed in 9.98s | `logs/canonical-test-unit-contract.log` |
| `unit-blcs` | PASS — 33 passed in 9.86s | `logs/canonical-test-unit-blcs.log` |
| `unit-plcs` | PASS — 25 passed in 9.48s | `logs/canonical-test-unit-plcs.log` |
| `unit-slcs` | PASS — 35 passed in 11.41s | `logs/canonical-test-unit-slcs.log` |
| `integration-normalization` | PASS — 14 passed in 11.61s | `logs/canonical-test-integration-normalization.log` |
| `preflight-regression` | PASS — 125 passed in 11.21s | `logs/canonical-test-preflight-regression.log` |
| `knowledge-graph` | PASS — 181 nodes, 0 errors, 4 pre-existing warnings | `logs/canonical-test-knowledge-graph.log` |
| `precommit-all` | PASS — ruff, mypy, task-script-reviewer | `logs/canonical-test-precommit-all.log` |
| `full-pytest` | PASS — 3230 passed, 53 skipped, 19 warnings in 787.70s | `logs/canonical-test-full-pytest.log` |

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` returned `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40` before and after all canonical checks.
- `.venv/bin/python -m pytest -q tests/unit/tasks/base/data/test_court_coordinate_contract.py` passed 18 tests in 15.32s after the typed-local repair.
- `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip tests/unit/tasks/base/data/test_court_coordinate_contract.py` reported `Success: no issues found in 1 source file`, reproducing the commit-hook-shaped single-file import context.
- All nine required canonical IDs passed on one stable post-test candidate. `precommit-all` supplies the all-files mypy context; `full-pytest` exercised the GPU-visible repository baseline.

## Failures encountered

None in the final cycle-2 canonical or supporting executions. The frozen direct test-only typing defect was closed before canonical evidence was recorded.

## Untested risks and reasons

The recorded PLCS v1/v2 training comparison remains single-seed and its v2 continuation/batch conditions limit causal interpretation. AC-019 requires a completed metre-valued comparison and the graph validates that evidence, but regenerating training runs is outside Test Writer ownership and the repair-local cycle-2 scope. No executable risk remained in the authorized typing-repair scope after both mypy contexts and all nine canonical checks passed.

## Final test verdict

PASS

## RETURN implementation findings

None
