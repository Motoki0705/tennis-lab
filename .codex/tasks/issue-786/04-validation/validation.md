# Validation

- Issue: #786
- Attempt: 2
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`

## Inspection scope and revision

- Sealed worktree: `/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-normalization-v2`
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`
- Candidate branch / HEAD: `feat/issue-786-normalization-v2` / `f89bb34ee8fe0b3ae21c36368f1d30a70668c529`
- Scope: the frozen `issue.json` / exactly rendered `issue.md`, the sealed candidate's code, configuration, tests, generated datasets/checkpoints/knowledge records, and applicable repository/skill contracts. No implementation, planning, review, test, seal, prior-validation, Issue-comment, or inherited-verdict artifact was used as acceptance evidence.
- The canonical candidate fingerprint was recomputed before validation and again after writing this report; both matched the sealed identity above.

## Acceptance checklist verification

| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | PASS | `src/utils/schema/court_normalization.py` is the sole immutable resolver, returns the exact two tuples, and raises `UnknownCourtCoordinateNormalizationVersionError`; schema tests passed. Physical dimensions remain defined separately in `src/utils/schema/court.py`. |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | PASS | Shared `court_coordinate_normalization` Hydra groups and typed configs are consumed at BLCS/PLCS/SLCS boundaries. Composition smoke tests covered BLCS/PLCS generation/training/visualization, SLCS train/evaluate/predict, and the tennis-scene pipeline with default `v1` and explicit `v2`. |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | PASS | Hydra composition asserts `v1` defaults. `tests/integration/tasks/test_legacy_v1_checkpoint_parity.py` replays a hash-verified metadata-free BLCS/PLCS fixture generated at frozen base `59e3b166...` and matches dataset load, inference, loss, and metrics to golden values at `1e-5`. |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | PASS | Shared dataset/checkpoint contract readers admit absent metadata only for explicit `v1`; unit and legacy-parity integration tests prove `v2` rejects the same legacy datasets/checkpoints before array/state use. |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | PASS | Strict parsers in `src/tasks/base/data/court_coordinate_contract.py` and checkpoint contract modules compare runtime, dataset, and checkpoint version/scale. Resume/init, evaluation/metric, predictor/adapter mismatch tests for BLCS, PLCS, and SLCS passed. |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | PASS | The shared NumPy/Torch API accepts arbitrary leading dimensions and enforces trailing size 3. Parametrized round-trip tests for both versions and all three task consumers passed; complete real v2 datasets round-tripped within `3.81469726562e-06 m` (BLCS) and `9.53674316406e-07 m` (PLCS). |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | PASS | Resolver/court-schema tests derive these values from unchanged `HALF_DOUBLES_WIDTH=5.485`, `HALF_LENGTH=11.885`, and `NET_HEIGHT_POST=1.07`; `v2` uses `11.885` on every axis. |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | PASS | Generator, BLCS datasets/adapters/predictors, differentiable projection, and standard/tracking metrics receive the same resolved contract; targeted generator, dataset, predictor, adapter, projection, and metric tests passed for both versions. |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | PASS | Ball physics/loss helpers derive gravity as `-g*dt^2/scale_z`; tracking retains the literal `-0.01` only for legacy `v1` and derives the `v2` value. Physics, loss, and tracking-matching tests exercise both contracts. |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PASS | PLCS scene generation, target reconstruction, standard/tracking predictors and metrics, court-pose integration, and 3D/top-down scene rendering all receive the resolved contract. Targeted unit tests plus the CPU integration smoke passed for both versions. |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | PASS | PLCS generation/target code normalizes only court translation; canonical/root-relative pose arrays remain metre-valued. Scene-loader, scene-generator, target, predictor, and render tests assert unchanged pose data and versioned translation. |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | PASS | SLCS data/evaluation/metrics/adapters/predictor use the selected contract; scalar uncertainty uses the legacy mean scale for `v1` and the common isotropic scale for `v2`. `SceneResult` keeps metre arrays, and provenance attachment is tested not to mutate them. |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | PASS | `v2` uses unweighted equal-axis Smooth L1 and canonical `beta = 1.0 m / 11.885 m`; loss tests inject equal physical axis errors and verify equal values/transition. BLCS/PLCS README and config comments document the physical transition. |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | PASS | BLCS tracking selects legacy `[1,1,0.5]` only for `v1` and `[1,1,1]` for `v2`, sharing the weights with Hungarian cost; PLCS position loss/matching has no axis weighting. Config and matching/loss tests verify `v1` preservation and `v2` isotropy. |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | PASS | Canonical metadata writers persist exact version, scale, `position_unit=m`, and `velocity_unit=m/s` at root and scene level. Loader tests reject missing, unknown, mixed, noncanonical, and runtime-mismatched contracts; both 1000-scene v2 datasets passed exhaustive loader validation. |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | PASS | Lightning save/load hooks and checkpoint adapters persist/restore the canonical contract and validate it before state use. BLCS/PLCS/SLCS checkpoint and predictor tests passed; inspected v1/v2 baseline checkpoints contain matching config/metadata. |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | PASS | The materializer refuses overwrite, requires version-qualified output, validates source, and publishes atomically. Actual datasets use `*_norm_v2`; actual baselines are separated below `outputs/{blcs,plcs}/i786/norm-v1\|norm-v2/...`; root/scene/checkpoint metadata distinguishes contracts. |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | PASS | `data/blcs_broadcast_norm_v2` and `data/plcs_broadcast_norm_v2` each contain 1000 independently named scenes. Exhaustive denormalization against stored generation-world values gave max errors `3.81469726562e-06 m` and `9.53674316406e-07 m`; manifests record the same counts/errors with `1e-5 m` tolerance. |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | PASS | Version-qualified completed checkpoints were inspected (BLCS v1/v2 epochs 99/95; PLCS v1/v2 epochs 86/92). `knowledge/nodes/group-i786-normalization-v1-v2.md` and four run nodes record physical integrated and X/Y/Z metrics for both versions (plus endpoint/angle); PLCS's batch/resume confound is explicitly disclosed. |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | PASS | `tests/integration/tasks/test_court_coordinate_normalization_smoke.py` executes the complete requested BLCS and PLCS chains for `v1` and `v2`, including BLCS projection and PLCS 3D/top-down render, on CPU; it passed in the 202-test run. |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | PASS | The targeted 202-test suite covered shared schema/config/court pose, all enumerated BLCS and PLCS consumers, SLCS adapters/predictor/metrics, tennis-scene schema, full v1/v2 smoke, and frozen-base v1 parity; all passed. |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | PASS | BLCS, PLCS, SLCS, and tennis-scene READMEs plus Hydra comments document formulas, `v1` default, metres/metres-per-second, legacy rules, fail-fast mismatch, version-qualified naming/migration commands, and point to `src/utils/schema/court_normalization.py` (with the base metadata schema) as authority. |

## Code evidence

- `src/utils/schema/court_normalization.py` defines the closed `v1`/`v2` mapping, immutable contract, validation, and arbitrary-shape NumPy/Torch position/velocity transforms. `src/utils/schema/court.py` retains the unchanged physical court constants and exposes legacy aliases only.
- `src/tasks/base/data/court_coordinate_contract.py` is the canonical strict artifact-metadata layer. The corresponding checkpoint layers validate before payload/state restore and are wired through BLCS, PLCS, SLCS, and tennis-scene entry points.
- BLCS propagation was inspected through generation/simulation, dataset loading, model I/O, predictor/tracking predictor, differentiable projection, loss/matching, and standard/tracking metrics. Position and velocity use the selected XYZ scale, and gravity uses the selected Z scale.
- PLCS propagation was inspected through scene generation/loading, target construction, checkpoint/model I/O, standard/tracking inference and metrics, canonical-to-world court-pose composition, and 3D/top-down rendering. Only world translation is normalized; canonical pose remains metres.
- SLCS propagation was inspected through data/model I/O, evaluation, training metrics, predictor, uncertainty conversion, and tennis-scene `SceneResult` provenance. Public player/ball arrays stay physical metres.
- The materialization implementation validates the source contract, prohibits overwrites and ambiguous output names, uses atomic publication, writes root/scene canonical metadata and a verification manifest, and checks physical round-trip error.
- Documentation in `src/tasks/{blcs,plcs,slcs}/README.md` and `src/tennis_scene/README.md` consistently routes formulas and mappings to the shared resolver and metadata schema instead of duplicating an executable definition.

## Runtime and test evidence

- Targeted command:

  ```text
  .venv/bin/python -m pytest -q tests/unit/utils/schema/test_court_normalization.py tests/unit/utils/geometry/test_court_pose.py tests/unit/tasks/base/data/test_court_coordinate_contract.py tests/unit/tasks/base/model_io/test_court_coordinate_contract.py tests/unit/tasks/blcs/data/test_dataset.py tests/unit/tasks/blcs/data/test_tracking_dataset.py tests/unit/tasks/blcs/generate_dataset/simulation/test_ball_physics.py tests/unit/tasks/blcs/inference/test_predictor.py tests/unit/tasks/blcs/inference/test_tracking_predictor.py tests/unit/tasks/blcs/model_io/test_adapters.py tests/unit/tasks/blcs/model_io/test_checkpoints.py tests/unit/tasks/blcs/training/test_losses.py tests/unit/tasks/blcs/training/test_tracking_matching.py tests/unit/tasks/blcs/training/test_tracking_metrics.py tests/unit/tasks/plcs/data/test_targets.py tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py tests/unit/tasks/plcs/generate_dataset/test_scene_generator.py tests/unit/tasks/plcs/inference/test_predictor.py tests/unit/tasks/plcs/inference/test_tracking_predictor.py tests/unit/tasks/plcs/training/test_losses.py tests/unit/tasks/plcs/visualization/io/test_scene.py tests/unit/tasks/slcs/inference/test_predictor.py tests/unit/tasks/slcs/model_io/test_adapter.py tests/unit/tasks/slcs/training/test_metrics.py tests/unit/tennis_scene/test_schema.py tests/integration/tasks/test_court_coordinate_normalization_smoke.py tests/integration/tasks/test_legacy_v1_checkpoint_parity.py
  ```

  Outcome: `202 passed in 11.26s`.

- Direct exhaustive artifact diagnostic loaded all 1000 scenes from each of `data/blcs_broadcast_norm_v2` and `data/plcs_broadcast_norm_v2`, validated root/scene contracts, and compared every stored normalized coordinate after denormalization with generation-world values. Outcomes: BLCS `3.81469726562e-06 m`; PLCS `9.53674316406e-07 m`, both below `1e-5 m`.
- Dataset root and per-scene metadata observed for both datasets: `schema_version=1`, `version=v2`, `scale_xyz=[11.885,11.885,11.885]`, `position_unit=m`, `velocity_unit=m/s`.
- Inspected trained checkpoints: BLCS v1 epoch 99 / step 1200, BLCS v2 epoch 95 / step 1152, PLCS v1 epoch 86 / step 17400, PLCS v2 epoch 92 / step 13591. Each is version-qualified and its config/metadata matches its selected contract.
- Recorded physical comparison metrics:

  | Model | Version | Integrated [m] | X [m] | Y [m] | Z [m] | Additional physical metric |
  |---|---:|---:|---:|---:|---:|---:|
  | BLCS | v1 | 2.4052329063 | 0.5272797942 | 2.1393527985 | 0.4554491341 | endpoint 4.6605615616 m |
  | BLCS | v2 | 2.3384502253 | 0.4341706164 | 2.1221547100 | 0.5046099059 | endpoint 4.3641868015 m |
  | PLCS | v1 | 0.4698834717 | 0.1916563213 | 0.3854689598 | 0.0446767062 | angle 52.0611 degrees |
  | PLCS | v2 | 0.3137337267 | 0.1715996265 | 0.2071425468 | 0.0663859546 | angle 63.6290 degrees |

## Regression and repository-rule checks

- Ruff over all 142 changed Python files: `All checks passed!`.
- Mypy over all 142 changed Python files with `--follow-imports=skip`: `Success: no issues found in 142 source files`.
- Repository script-convention reviewer over all 12 changed files under script directories: PASS.
- Frozen-base legacy fixture provenance/hash checks passed, and the v1 parity test detects numerical drift in representative data, inference, loss, and metric paths.
- No silent inference, value-range guessing, automatic v1-to-v2 conversion, overwrite path, or relevant regression was found in the inspected candidate.

## Final verdict

PASS

## RETURN exploration questions

None
