# Tests

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`

## Candidate identity

- Frozen base revision: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Pre-test production candidate recorded in state: `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`.
- Post-test candidate: `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`.
- Candidate delta is test-only: the two representative metadata-free Torch fixtures were byte-preservingly renamed from forbidden `.ckpt` paths to `.ckpt.bin`; their committed generator/manifest and all test references were updated; one repair-local filename-policy assertion was added. No production, planning, Issue/state, preflight, seal, or validation content was changed.
- Complete base-diff path inventory contains 266 paths and no path ending `.ckpt`, `.pt`, `.pth`, or `.pkl`.

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | `unit-contract` (56 passed) fixes resolver mappings, conversions, geometry, and unknown-version rejection; `full-pytest` passed. | PASS |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | `integration-normalization` (14 passed), `unit-blcs`, `unit-plcs`, `unit-slcs`, and `preflight-regression` passed their config/task-boundary cases. | PASS |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | `legacy-v1-checkpoint-parity` (8 passed) loaded both renamed byte-identical frozen-base checkpoints through real BLCS/PLCS loaders and replayed inference, losses, metre metrics, and golden arrays at `atol=1e-5`; `preflight-regression` and `full-pytest` passed. | PASS |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | `legacy-v1-checkpoint-parity` exercised explicit-v1 acceptance and both checkpoint/dataset v2 rejection before state/array loading; `unit-contract` passed. | PASS |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | `unit-contract` plus BLCS/PLCS/SLCS task checks and `integration-normalization` passed mismatch/rejection matrices. | PASS |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | `unit-contract` passed shape-generic NumPy/Torch round trips; task and integration checks passed downstream conversions. | PASS |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | `unit-contract` passed v2 landmark/aspect and unchanged physical-geometry assertions. | PASS |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | `unit-blcs` (37 passed) and `integration-normalization` passed generation, decode, projection, predictor, and metric cases for both versions. | PASS |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | `unit-blcs` and `integration-normalization` passed standard/tracking gravity formula and CPU-flow cases. | PASS |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | `unit-plcs` (25 passed), `integration-normalization`, and `full-pytest` passed PLCS generation/targets/predictors/metrics/render flow. | PASS |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | `unit-plcs` and `integration-normalization` passed canonical-metre invariance and translation-only normalization cases. | PASS |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | `unit-slcs` (35 passed), `integration-normalization`, and `full-pytest` passed scale/adapter/predictor/uncertainty/SceneResult metre cases. | PASS |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | `unit-blcs`, `unit-plcs`, and `integration-normalization` passed equal-physical-error and shared-Huber-boundary cases; `precommit-all` passed documentation/static policy. | PASS |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | `unit-blcs`, `unit-plcs`, `integration-normalization`, and `preflight-regression` passed v2 uniform and v1 legacy-weight behavior. | PASS |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | `unit-contract`, task loader checks, `integration-normalization`, and `legacy-v1-checkpoint-parity` passed metadata persistence and fail-before-payload rejection. | PASS |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | `unit-contract`, BLCS/PLCS/SLCS checkpoint checks, `integration-normalization`, and real legacy checkpoint parity passed. | PASS |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | `unit-contract` and `integration-normalization` passed non-overwrite/naming/metadata cases; fixture packaging repair retained legacy-v1 identity and exact checkpoint hashes. | PASS |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | `integration-normalization` passed bounded BLCS/PLCS materialization and physical round-trip evidence; task checks passed. | PASS |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | `knowledge-graph` validated 181 nodes with 0 errors (4 unrelated existing warnings); `full-pytest` passed repository evidence consumers. | PASS |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | `integration-normalization` passed 14 real-loader/model CPU smoke cases; `legacy-v1-checkpoint-parity` replayed both real frozen-base task pipelines; `full-pytest` passed. | PASS |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | All scoped canonical suites passed: 56 contract, 37 BLCS, 25 PLCS, 35 SLCS, 14 integration, 8 legacy parity, and 127 baseline-regression cases; full suite passed 3246 tests. | PASS |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | `precommit-all` passed Ruff/mypy/script-reviewer, `knowledge-graph` had 0 errors, and `full-pytest` passed documentation/config contract tests. | PASS |

## Tests added or changed

- Renamed `blcs_representative_legacy_v1.ckpt` to `blcs_representative_legacy_v1.ckpt.bin` without changing its 100421 bytes or SHA-256 `69af21b3f8008ab7f53708e1d03346113aafa49c857a5465ad6e6da86f80a5e7`.
- Renamed `plcs_representative_legacy_v1.ckpt` to `plcs_representative_legacy_v1.ckpt.bin` without changing its 71896 bytes or SHA-256 `6c212eec6bbe616b498000928733318b19f76e4ad957a680963621c672ca841d`.
- Updated every fixture consumer in `tests/integration/tasks/test_legacy_v1_checkpoint_parity.py` to use `.ckpt.bin`, preserving real BLCS/PLCS v1 replay, v2 fail-fast rejection, and explicit-v1 binding. Added `test_representative_fixture_uses_pr_safe_checkpoint_filenames`, making stale forbidden checkpoint suffixes inside the committed representative fixture fail canonically.
- Updated the committed generator provenance and its manager-owned source copy to emit `.ckpt.bin`; both now have SHA-256 `63901c0ffee71a0eea2d79d0f3365444544b1e272cccb89d73b1cf61ad40edae`. Updated manifest path references and generator hash while retaining both checkpoint hashes and byte counts exactly.

## Normal, boundary, invalid, and regression cases

### Bounded repair-local adversarial risk model

| Risk | Oracle/authority | Adversarial case and result |
|---|---|---|
| A forbidden model suffix remains in the PR diff after only the obvious files are renamed. | Repository packaging invariant supplied by the assignment. | Complete 266-path base-diff inventory was scanned for names ending `.ckpt`, `.pt`, `.pth`, or `.pkl`; no match. The new fixture-root assertion also passed. |
| A rename changes or truncates serialized checkpoint bytes. | Frozen-base regression and AC-003/AC-004 legacy parity. | Old HEAD blobs and new `.ckpt.bin` files were SHA-256 compared; BLCS remained `69af…a5e7` and PLCS remained `6c21…41d`, with unchanged manifest byte counts. |
| Checkpoint loaders or Lightning infer format from the filename extension. | Current public loader interfaces accept `Path`/`str` and delegate to `torch.load`; AC-003/AC-004 require behavior, not a suffix. | Both `.ckpt.bin` files passed direct `torch.load`, BLCS/PLCS checkpoint composition, Lightning restore, inference, loss, metre metrics, and golden replay in `legacy-v1-checkpoint-parity`. |
| Tests pass through one reference while generator or manifest provenance remains stale. | Fixture manifest/provenance contract and exact-hash assertions. | All consumer, generator, and manifest references were inventoried; committed and manager-owned generator copies are byte-identical, and the manifest/exact-bytes test passed. |
| Packaging repair accidentally weakens metadata-free v2 rejection or v1-only semantics. | AC-003/AC-004 Issue contract. | The same renamed files passed explicit-v1 load and rejected v2 before Lightning state restore/array payload read; golden replay remained within `1e-5`. |
| A test-only rename masks a wider repository regression. | Baseline regression invariant and frozen canonical manifest. | All ten required test-stage canonical checks passed on one fingerprint, including `precommit-all` and 3246-test `full-pytest`. |

### Independent `AT-*` perspectives

No `AT-*` row was executable: the checked-in schema-v5 manager does not expose the required `run-test-probe` subcommand (`manage_issue_task.py --help` lists only the canonical state/check commands, and invoking `run-test-probe --help` returns “invalid choice”). Because production/workflow-tool edits are outside Test Writer ownership, no probe authority was invented or hand-written. Repair-local checks were therefore incorporated into the canonical `legacy-v1-checkpoint-parity` suite and recorded below as non-AT inspection evidence.

### Case classes

- Normal: both renamed checkpoints load and replay real v1 inference/loss/metre-metric goldens.
- Boundary: exact checkpoint byte counts and SHA-256 values, generator SHA-256, base revision, deterministic runtime metadata, and `1e-5` parity tolerance remain fixed.
- Invalid: v2 runtime continues to reject metadata-free checkpoints before state restore and datasets before array payload loading; any forbidden suffix under the representative fixture fails.
- Regression: explicit-v1 binding, v1/v2 unit/integration contracts, configuration/static checks, knowledge evidence, and the full repository suite all pass.

## Canonical command results

| Check ID | Exact outcome | Machine record |
|---|---|---|
| `unit-contract` | PASS, exit 0; 56 passed. | `logs/canonical-test-unit-contract.log` |
| `legacy-v1-checkpoint-parity` | PASS, exit 0; 8 passed. | `logs/canonical-test-legacy-v1-checkpoint-parity.log` |
| `unit-blcs` | PASS, exit 0; 37 passed. | `logs/canonical-test-unit-blcs.log` |
| `unit-plcs` | PASS, exit 0; 25 passed. | `logs/canonical-test-unit-plcs.log` |
| `unit-slcs` | PASS, exit 0; 35 passed. | `logs/canonical-test-unit-slcs.log` |
| `integration-normalization` | PASS, exit 0; 14 passed. | `logs/canonical-test-integration-normalization.log` |
| `preflight-regression` | PASS, exit 0; 127 passed. | `logs/canonical-test-preflight-regression.log` |
| `knowledge-graph` | PASS, exit 0; 181 nodes, 0 errors, 4 unrelated existing warnings. | `logs/canonical-test-knowledge-graph.log` |
| `precommit-all` | PASS, exit 0; Ruff, mypy, task script reviewer passed. | `logs/canonical-test-precommit-all.log` |
| `full-pytest` | PASS, exit 0; 3246 passed, 53 skipped, 19 warnings in 708.58s. | `logs/canonical-test-full-pytest.log` |

Every row is bound by `03-implementation/test-checks.json` to candidate `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf` with the canonical invocation digest from `02-planning/checks.json`.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` → `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`.
- For each of `unit-contract`, `legacy-v1-checkpoint-parity`, `unit-blcs`, `unit-plcs`, `unit-slcs`, `integration-normalization`, `preflight-regression`, `knowledge-graph`, `precommit-all`, and `full-pytest`: `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 test <check-id>` → exit 0/PASS; exact argv/cwd/environment and logs are machine-recorded in `test-checks.json`.
- `git show HEAD:<old-checkpoint-path> | sha256sum` and `sha256sum <renamed-checkpoint-paths>` → identical old/new BLCS and PLCS digests listed above.
- `git diff --name-only 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- | awk '/\.(ckpt|pt|pth|pkl)$/{print; bad=1} END{exit bad}'` → exit 0 with no output.
- `cmp` on committed versus manager-owned generator provenance → exit 0; both SHA-256 `63901c0f…edae`.
- `manage_issue_task.py run-test-probe --help` → exit 2, unsupported subcommand; therefore no machine-recordable independent `AT-*` probe was claimed.

## Failures encountered

None in the repair or any canonical check. The only unavailable mechanism was the repository manager's absent `run-test-probe` subcommand; this is recorded as the substantive reason no `AT-*` probes were executable, not as an implementation failure.

## Untested risks and reasons

- The external PR-creation uploader was not rerun by the Test Writer because PR packaging is parent-owned. Its exact rejected suffix condition is covered by the complete base-diff path scan and the canonical fixture filename assertion.
- GitHub-side policy behavior beyond the supplied filename-ending rule is not local Issue authority and was not inferred.

## Final test verdict

PASS

## RETURN implementation findings

None
