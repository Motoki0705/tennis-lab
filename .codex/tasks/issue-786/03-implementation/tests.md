# Tests

- Issue: #786
- Attempt: 2
- Test cycle: 2
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`

## Candidate identity

- Preflight PASS input candidate: `sha256:0d52bf2f739ab9f989ed8d64df18e5e7d5d53dc4b386cd493dc4633f710b87bb`.
- Post-test candidate: `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`.
- HEAD: `7aa35c1a68a2216c0808fb2dd6fc82ba7906bcfb`; merge commit `64ea1b5a99bacd5ec7f8ab4f356333835eaa9de9` has parents `2661f3a80b56d5b2e1d44106162ba199cfaf45b0` and current-main `179dac756aef137c9a35b1025ce76f0a31023648`.
- The fingerprint change is exclusively from the three Tester-owned test changes listed below; production and frozen planning authority were not edited.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | `unit-contract` exercised resolver values, unknown versions, shape validation, and contract metadata; `full-pytest` passed repository-wide. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | `integration-normalization`, `unit-blcs`, `unit-plcs`, and `unit-slcs` passed; the added merged BLCS strict-config cross-product composes both versions for standard and ablation tracking models. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | `legacy-v1-checkpoint-parity` passed 8 tests against frozen-base checkpoint/dataset/golden bytes and numeric inference/loss/metre metrics; `preflight-regression` passed 127 tests. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | `legacy-v1-checkpoint-parity` covered explicit-v1 replay and v2 rejection before Lightning state or array payload load; `unit-contract` covered the shared legacy gate. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | `unit-contract` and `integration-normalization` passed mismatch, partial, mixed, missing, and unknown metadata cases; task predictor checks passed. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | `unit-contract` passed NumPy/Torch arbitrary-leading-shape round trips for both versions; task and integration suites preserved the same contract. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | `unit-contract` passed court landmark/aspect assertions and `preflight-regression` retained physical court geometry. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | `unit-blcs` passed 39 tests and `integration-normalization` passed the two-version real dataset/model/loss/metric/denormalization/projection flow. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | `unit-blcs` covered standard and tracking gravity formulas; the added merge-local strict-config test verifies v1 literal and v2 derived gravity through both BLCS tracking model families. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | `unit-plcs` passed 27 tests and `integration-normalization` passed both-version actual PLCS model, loss, metric, denormalization, and renderer flow; merged tracking predictor checkpoint scaling passed. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | `unit-plcs`, `unit-contract`, and `integration-normalization` passed canonical-pose invariance and translation-only scaling assertions. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | `unit-slcs` passed 35 tests and `integration-normalization` retained versioned SLCS conversion and metre-valued `SceneResult`. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | `unit-blcs`, `unit-plcs`, and `integration-normalization` passed equal-axis physical error and 1m Huber-transition assertions; merge-local BLCS config coverage verifies the selected beta. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | Task loss/matching tests passed; the new strict BLCS family test asserts v1 `[1,1,0.5]` and v2 `[1,1,1]` for standard and ablation configuration paths. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | `unit-contract` and `integration-normalization` passed metadata creation plus root/scene missing, unknown, mixed, and mismatch rejection before payload use. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | `unit-contract`, `unit-blcs`, and `unit-plcs` passed metadata write/restore/mismatch behavior; enhanced checkpoint-path tests assert exact ablation binding and metre scaling for v1 and v2. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | `legacy-v1-checkpoint-parity` passed exact fixture hashes and PR-safe `.ckpt.bin` naming; `integration-normalization` passed non-overwrite/version-qualified materialization. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | `integration-normalization` passed isolated BLCS/PLCS v2 materialization, output separation, metadata, and physical round-trip assertions. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | `knowledge-graph` validated 181 nodes with 0 errors, including four Issue-786 run nodes and the v1/v2 group; the nodes record physical aggregate and per-axis metre metrics. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | `integration-normalization` passed 14 actual-task CPU smoke/materialization tests; `full-pytest` also passed the complete merged suite. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | All eight focused test/contract canonical checks passed, the three merge-local tests were added, and `full-pytest` passed 3331 tests with 78 skips. | PASS |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | `precommit-all`, `knowledge-graph`, `preflight-regression`, and `full-pytest` passed repository documentation/config/schema invariants. | PASS |

## Independent adversarial test design

| Risk | Authority / oracle | Adversarial perspective | Executed evidence | Outcome |
|---|---|---|---|---|
| R-01 | Issue AC-002, AC-008, AC-009, AC-013, AC-014, AC-020, AC-021 | Merge conflict could validate upstream `blcs_track_query_ablation` while dropping normalization-era strict keys or selecting different loss/gravity/generator behavior from `blcs_track_query`. | Added a v1/v2 × standard/ablation Hydra boundary cross-product; `full-pytest` and `precommit-all` passed. | PASS |
| R-02 | Issue AC-003, AC-010, AC-016 and public tracking predictor APIs | Checkpoint restoration could recover the upstream ablation architecture but lose checkpoint normalization, silently use default v1 for v2, or bind the wrong model/adapter pair. | Enhanced BLCS and PLCS checkpoint restoration tests for v1/v2; each runs normalized and denormalized predictions and asserts exact per-axis metre scaling plus exact ablation model/adapter binding. `unit-blcs` and `unit-plcs` passed. | PASS |
| R-03 | Issue AC-003, AC-004, AC-017 and repository PR artifact policy | Merge or test edits could rewrite opaque frozen-base `.ckpt.bin`/golden fixtures or reintroduce forbidden checkpoint suffixes. | Existing parity test hashes every representative artifact/tree and rejects `.ckpt/.pt/.pth/.pkl`; `legacy-v1-checkpoint-parity` passed. Git comparison to the pre-merge first parent was byte-clean for the fixture tree. | PASS |
| R-04 | Baseline regression against merged `origin/main` and AGENTS.md complete-test requirement | Normalization changes could pass focused checks while regressing newly merged ablation/CUDA/workflow callers elsewhere in the repository. | Candidate-bound `full-pytest` passed 3331 tests with 78 skips using the canonical GPU-visible environment; `precommit-all` passed Ruff, mypy, and task script reviewer. | PASS |
| R-05 | Issue AC-004, AC-005, AC-015, AC-016 and fail-closed repository invariant | Missing/unknown/mixed dataset or checkpoint contracts could reach array/state loading or silently infer a version. | `unit-contract`, `legacy-v1-checkpoint-parity`, and `integration-normalization` passed fail-before-payload/state negative cases. | PASS |
| R-06 | Issue AC-006 through AC-014, AC-018, AC-020 | Round trips alone could hide wrong consumer-specific units, canonical-pose rescaling, anisotropic v2 loss, or wrong gravity/projection/render scale. | Task unit suites and actual-model integration smoke passed both versions through loss, metrics, metre decode, projection/render, and materialization. | PASS |

## Independently derived adversarial tests

None — this task is a loaded schema-v5 task retained as schema 6 with `adversarial_testing_mode = "LEGACY"`; the authoritative workflow explicitly preserves its legacy Tester contract, and `run-test-probe` fail-closes instead of permitting machine-recorded `AT-*` rows. The independent perspectives are therefore recorded above and executed through Tester-owned tests plus candidate-bound canonical `run-check` checks.

## Adversarial probe results

None — no `test-probes.json` was created because legacy adversarial mode does not permit `run-test-probe`. An attempted pre-execution `AT-001` invocation exited 2 with `adversarial test probes require a schema-v6 task` before running a perspective; after parent authority resolution it was not treated as evidence or assigned an AT row.

## Tests added or changed

| Path | Added/changed coverage |
|---|---|
| `tests/integration/tasks/blcs/test_model_configs.py` | Adds standard/ablation × v1/v2 strict Hydra composition, generator equality, selected scale, tracking axis weights, Huber beta, and gravity target coverage. |
| `tests/unit/tasks/blcs/inference/test_tracking_predictor.py` | Expands ablation checkpoint restoration to v1/v2 and verifies exact BLCS model/adapter dispatch plus normalized-to-metre output scaling. |
| `tests/unit/tasks/plcs/inference/test_tracking_predictor.py` | Expands ablation checkpoint restoration to v1/v2 and verifies exact PLCS model/adapter dispatch plus normalized-to-metre output scaling. |

No production code, frozen Issue/state, plan, or `checks.json` was modified.

## Normal, boundary, invalid, and regression cases

- Normal: v1 and v2 contract resolution, composed Hydra selection, real BLCS/PLCS/SLCS data/model paths, normalized and metre prediction outputs, checkpoint metadata restore, dataset materialization, and baseline evidence validation.
- Boundary: arbitrary leading `(...,3)` NumPy/Torch shapes, exact physical landmark scales, equal physical errors at the Huber knee, short CPU scenes, root/scene metadata combinations, and both standard and upstream ablation tracking model families.
- Invalid: unknown version, missing/partial/mixed/mismatched root/scene/checkpoint contracts, metadata-free artifact under v2, payload/state access ordering, non-uniform v2 loss defaults, version-unqualified overwrite targets, and PR-unsafe checkpoint suffixes.
- Regression: exact frozen-base v1 checkpoint/dataset/golden replay, unchanged opaque fixture bytes across the merge, existing v1 configuration/loss/metric behavior, newly merged ablation binding, complete current-main repository tests, Ruff, mypy, and script policy.

## Canonical command results

| Check ID | Exact final outcome |
|---|---|
| `unit-contract` | PASS, exit 0, 56 passed. |
| `legacy-v1-checkpoint-parity` | PASS, exit 0, 8 passed. |
| `unit-blcs` | PASS, exit 0, 39 passed. |
| `unit-plcs` | PASS, exit 0, 27 passed. |
| `unit-slcs` | PASS, exit 0, 35 passed. |
| `integration-normalization` | PASS, exit 0, 14 passed. |
| `preflight-regression` | PASS, exit 0, 127 passed. |
| `knowledge-graph` | PASS, exit 0, 181 nodes, 0 errors, 4 unrelated warnings. |
| `precommit-all` | PASS, exit 0; Ruff, mypy, and task script reviewer passed. |
| `full-pytest` | PASS, exit 0, 3331 passed, 78 skipped. |

All final results in `test-checks.json` are bound to `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test unit-contract` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test legacy-v1-checkpoint-parity` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test unit-blcs` → final exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test unit-plcs` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test unit-slcs` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test integration-normalization` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test preflight-regression` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test knowledge-graph` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test precommit-all` → exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test full-pytest` → exit 0.

## Failures encountered

- During Tester test development, the first `unit-blcs` run used a metadata-free synthetic v1 checkpoint without supplying the explicit legacy-v1 runtime required by AC-004. It failed one newly edited test before production execution. The test was corrected to model a new checkpoint with v1 metadata; the separate frozen-base parity suite remains the authority for metadata-free explicit-v1 behavior. The final candidate-bound rerun passed all 39 `unit-blcs` tests.
- No final AC row, canonical check, or independently derived risk perspective failed.

## Untested risks and reasons

- The Tester did not rerun the four long GPU training baselines. AC-019 is covered by the checked-in reproducibility bundles/run nodes and the canonical knowledge-graph validator; this cycle tests repository behavior and evidence integrity, not experiment reproduction.
- The PLCS v1/v2 operational comparison has documented batch/resume differences, so it is baseline evidence rather than causal evidence for normalization quality. The Issue requires recording the comparison, not proving causal improvement.
- External archived checkpoints already architecture-incompatible with the frozen base were not treated as normalization compatibility oracles. The committed frozen-base representative artifacts provide the bounded baseline authority required by AC-003/004.

## Final test verdict

PASS

## RETURN implementation findings

None
