# Feasibility

- Issue: #786
- Attempt: 3
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Allowed and prohibited changes

- Allowed: versioned court-coordinate schema/configuration; BLCS, PLCS, and SLCS normalization consumers; dataset/checkpoint metadata and compatibility validation; version-aware generation, training, evaluation, inference, projection, rendering, metrics, losses, matching, physics priors, tests, configuration, README/schema documentation, versioned v2 artifacts, and v2 baseline evidence.
- Allowed: a documented legacy-v1 compatibility path for metadata-free artifacts when runtime explicitly selects v1, plus new v1/v2 regression fixtures and CPU smoke coverage.
- Prohibited: changing physical court/net dimensions, court axes/origin, camera/world units, UV/intrinsic normalization, SMPL Y-up to court Z-up semantics, canonical-pose root-relative metre coordinates, production metre arrays, BLCS shot/apex policy, or automatically converting v1 model weights into v2 weights.
- Prohibited: inferring a normalization version from values/shapes, accepting runtime/dataset/checkpoint mismatches, overwriting legacy artifacts, or silently treating metadata-free artifacts as v2.

## Required checks and baseline

- Baseline: run focused existing schema/geometry, BLCS loss/metric/inference, PLCS generation/loss/inference, and SLCS adapter/metric tests before or during the implementation review to prove v1 behavior from base revision `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Required static checks: Ruff and mypy on all changed Python paths using repository configuration; script-convention checks if any script under `src/**/scripts/` is changed.
- Required unit checks: version resolver/config validation, v1/v2 round trips, v2 aspect ratio, legacy and mismatch behavior, metadata persistence, checkpoint restore/rejection, BLCS gravity/projection/metric/predictor paths, PLCS target/predictor/renderer/loss paths, and SLCS scale/uncertainty paths.
- Required integration checks: v1 and v2 CPU pipelines for dataset load through denormalized projection/render output, with test fixtures rather than GPU training.
- Required artifact evidence: versioned v2 BLCS/PLCS dataset samples with metre round-trip error <=1e-5m and a locally queued v2 training/evaluation run recorded with physical-metre comparisons. GPU work must use the shared training queue.

## Breaking-change and compatibility impact

- A direct mutation of the global scale would break all metadata-free datasets and checkpoints; the versioned resolver and explicit contract propagation avoid this conflict.
- The composed default remains v1. Existing configurations and legacy artifacts retain their old numeric meaning; v2 is opt-in and carries explicit metadata.
- Old artifacts cannot prove their own version. They are accepted only when runtime explicitly selects v1; a v2 runtime rejects missing metadata. This is a deliberate compatibility rule, not value-based fallback.
- New checkpoints and datasets bind version and scale. Resume, evaluation, and inference reject version/scale mismatches before model/data use.
- BLCS gravity targets, tracking weights, PLCS render conversion, and SLCS covariance/uncertainty scaling are semantic consumers rather than simple constants and require version-aware tests.
- Dataset generation and baseline training are long-running but feasible through versioned output locations and the required shared training queue; no immutable repository or Issue constraint prevents them.

## Acceptance checklist feasibility

| ID | Issue checklist item | Verdict | Required change and evidence |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | FEASIBLE | Add a typed/versioned resolver in shared court schema and unit-test both mappings plus unknown rejection. |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | FEASIBLE | Add a shared config group/default and inject the resolved contract through all task entrypoints; verify composed configs and focused integration tests. |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | FEASIBLE | Keep v1 as composed default and add golden/regression assertions for representative loss, metric, and inference conversions. |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | FEASIBLE | Implement explicit legacy-v1 handling gated by runtime version and tests for v1 acceptance/v2 rejection. |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | FEASIBLE | Centralize compatibility validation and cover every mismatch combination in unit/integration tests. |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | FEASIBLE | Provide shape-generic helpers and parameterized round-trip tests across scalar batches and temporal/player dimensions. |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | FEASIBLE | Test normalized landmarks against immutable physical constants and assert the constants remain unchanged. |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | FEASIBLE | Thread the contract through BLCS generator/model/inference/metrics and add version-parameterized focused tests. |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | FEASIBLE | Resolve height scale by contract, replace/derive fixed targets per version, and assert the physical formula. |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | FEASIBLE | Use the injected contract instead of direct dimension multipliers and cover target, predictor, metric, and renderer paths. |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | FEASIBLE | Keep canonical-pose arrays untouched and add regression tests separating translation normalization from local metre pose. |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | FEASIBLE | Thread version through SLCS consumers, scale covariance/uncertainty correctly, and verify SceneResult units. |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | FEASIBLE | Define/document v2 beta semantics and add equal-physical-error loss/transition tests. |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | FEASIBLE | Make weights version-specific or explicit task policy; test v2 defaults and v1 regression. |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | FEASIBLE | Extend writer/schema and loader validation; test valid and malformed root/scene combinations. |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | FEASIBLE | Persist the contract in model configuration/checkpoint metadata and validate it in loaders/predictors. |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | FEASIBLE | Use separate configured output paths/names and assert artifact identifiers plus metadata. |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | FEASIBLE | Generate bounded versioned samples/full configured datasets, validate saved values against world sources, and retain reproducible commands/evidence. |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | FEASIBLE | Queue v2 baseline training/evaluation through the shared training queue and record run IDs/configs/physical metrics without overwriting v1 results. |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | FEASIBLE | Add bounded test fixtures and two-version CPU smoke coverage independent of GPU availability. |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | FEASIBLE | Add/extend tests in repository-prescribed locations and run canonical focused/full suites. |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | FEASIBLE | Update the canonical README/schema docs and make all secondary comments link/reference the single contract. |

## Constraint conflicts

None

## Final feasibility verdict

PASS

## Blocker resolution required

None
