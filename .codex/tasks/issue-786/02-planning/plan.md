# Plan

- Issue: #786
- Attempt: 2
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Acceptance checklist mapping

| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | Add an immutable typed resolver/value object and shape-generic conversion helpers; retain current constants as documented v1 aliases only. | `unit-contract`, `precommit-all`, `full-pytest`. |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | Add a shared Hydra group with explicit v1 default and inject the resolved contract through every task boundary rather than a mutable global. | Config-composition cases in `integration-normalization`; task unit checks. |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | At frozen base `59e3b166...`, deterministically generate small metadata-free BLCS/PLCS checkpoints and loader-compatible dataset fixtures with actual task models; record prediction/loss/metre-metric goldens. The current v1 path must load those unchanged artifacts and match at explicit tolerances. Archived checkpoints already incompatible with the frozen base are documented but not auto-migrated. | `legacy-v1-checkpoint-parity`, existing v1 goldens, and v2 metadata rejection. |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | Keep the shared all-missing legacy gate: frozen-base metadata-free representative artifacts load only under explicit v1 and fail before config/state use under v2. Do not add architecture/config migration for older files already incompatible at the base revision. | `legacy-v1-checkpoint-parity`, `unit-contract`, and real metadata-free artifact rejection matrix. |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | Bind the same contract to runtime, root/scene metadata and checkpoint config; validate before array/model use. | Cross-product mismatch tests in `unit-contract` and integration smoke. |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | Use broadcast-safe NumPy/Torch helpers backed by the typed scale. | Parameterized property examples in `unit-contract` plus task checks. |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | Keep physical constants/CourtKP geometry unchanged and derive v2 only in normalization resolver. | Landmark/aspect assertions in `unit-contract`. |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | Inject contract into BallPhysics, datasets, loss projection buffer, metrics and both predictors; world velocity remains m/s while model normalized velocity uses selected scale. | `unit-blcs`, `integration-normalization`, v2 materialization evidence. |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | Resolve standard height scale from contract; preserve the rounded v1 tracking literal, derive the v2 target from gravity/frame_dt/scale_z, and document tolerance/source. | Gravity formula cases in `unit-blcs` and CPU smoke. |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | Thread contract through legacy/production normalized position, target building, metrics, predictors, court-pose translation and render conversion. | `unit-plcs`, `integration-normalization`, v2 materialization evidence. |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | Limit contract application to position/translation and retain canonical/world metre arrays bitwise/numerically. | Canonical invariance and target/world tests in `unit-plcs`/`unit-contract`. |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | Inject contract into SLCS loading, adapter, metrics, evaluation and predictor; preserve scalar uncertainty semantics using mean(scale) for v1 and S for v2; keep SceneResult metre-valued. | `unit-slcs` and integration SceneResult assertions. |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | Keep v1 beta behavior; define v2 default physical beta as 1.0m (`beta_norm=1/11.885`) and expose/document it through loss config. | Equal-error/boundary tests in `unit-blcs`, `unit-plcs`, integration smoke. |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | Split version-specific defaults: v1 retains historical weights; v2 uses uniform axis weights unless an explicitly documented task policy overrides them. | Loss/matching config and numeric tests in task unit checks. |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | Retain complete root/scene metadata and make the PLCS public `load_scene`/visualization bundle contract mandatory so no owned load path can bypass validation. | Public-loader missing/unknown/mixed/mismatch tests plus `unit-contract` and materialization evidence. |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | Persist the composed/resolved contract in Lightning config and add shared extraction/validation used by BLCS/PLCS/SLCS loaders/predictors. | Checkpoint round-trip/mismatch tests in `unit-contract` and task checks. |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | Use `norm-v1`/`norm-v2` qualified output configs and reject input/output identity in the materializer. | Artifact-name/metadata assertions plus recorded commands. |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | Add a Hydra materialization entrypoint that copies legacy scenes into separate v2 roots, recomputes only normalized position from physical/world or v1-resolved values, and records hashes/contracts. | `integration-normalization` bounded fixture plus full materialization validation log/manifest. |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | Run controlled v1/v2 pairs for BLCS and PLCS with identical architecture/seed/data scenes via shared training queue; register four run nodes and one group node with commands/metrics. | Training-queue completion evidence, run bundles, `knowledge-graph`, physical metric comparison. |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | Replace the standalone NumPy/Identity surrogate with complete two-frame versioned fixtures, real `BallTrajectoryDataset`/PLCS `SceneDataset`, tiny actual `BLCSModel`/`PLCSMultiViewModel` through their adapters, real losses/metrics, and projection/renderer. | `integration-normalization` asserts actual loader/model types and the full CPU chain for both tasks/versions. |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | Keep existing coverage and add default-v1 metre-scale assertions for standard/tracking BLCS position/velocity and PLCS translation alongside v2. | All unit/integration canonical checks and `full-pytest`. |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | Make shared normalization schema/README canonical; task docs/config comments link to it; document Hydra selection, legacy rule, naming and errors. | Documentation inspection, `precommit-all`, `knowledge-graph`. |

## Planned files and symbols

- Shared contract: new `src/utils/schema/court_normalization.py`; compatibility aliases/exports in `src/utils/schema/court.py`; version-aware `src/utils/geometry/court_pose.py`.
- Base artifacts/config: new `src/tasks/base/data/court_coordinate_contract.py`, `src/tasks/base/model_io/court_coordinate_contract.py`, shared typed configuration and Hydra normalization group, root/scene validation hooks in base dataset writer/indexing.
- Dataset materialization: new pure library under `src/tasks/base/data/` plus Hydra entrypoint/config under `src/tasks/base/scripts/` and `src/tasks/base/configs/`; it must follow `script-conventions` and provide BLCS/PLCS adapters without overwriting sources.
- BLCS: generation physics/scene IO and loaders; configuration roots/boundaries; standard/tracking losses, matching, gravity, metrics, projection; standard/tracking predictors and checkpoint loading; task README/config comments and v1/v2 baseline data configs.
- PLCS: legacy and production normalized-position generation/IO/loaders; targets and court-pose calls; standard/tracking loss/matching/metrics/predictors; renderer/analysis; configuration roots/model IO; docs and v1/v2 baseline data configs. Canonical/metre arrays are excluded from scaling.
- SLCS/SceneResult: SLCS data types/loader, adapter, metrics/evaluation/inference and applicable scripts/configs/docs; tennis-scene unit/provenance contract without changing physical output values.
- Formal experiment evidence: `knowledge/nodes/run-i786-*.md`, `knowledge/nodes/group-i786-coordinate-normalization-v2.md`, corresponding small reproducibility bundles under `knowledge/runs/`; large datasets/checkpoints/logs remain outside git.
- Tests are not owned by production Implementers. The independent Test Writer will place pure tests under mirrored `tests/unit/...` paths from `checks.json` and cross-task CPU smoke at `tests/integration/tasks/test_court_coordinate_normalization_smoke.py`.
- Attempt-2 production repair is limited to PLCS public scene-loader propagation. The parent produces raw base-revision representative checkpoint/data goldens in a detached worktree; only the independent Test Writer may add the committed fixture and test changes.

## Implementation topology and ownership

- Attempt-1 staged parallel implementation is complete and retained as the base candidate; attempt 2 does not reopen its shared/BLCS/PLCS/SLCS ownership.
- Production Implementers may edit source/config/docs in their ownership but no tests and no workflow artifacts. They must preserve edits from other agents and use the shared core contract rather than create task-local resolvers.
- The parent is the sole implementation integrator: resolves cross-task config composition, performs repository-wide fixed-scale inventory, writes `implementation.md`, runs materialization/training operations, and registers knowledge evidence.
- Independent Preflight Reviewer, Test Writer, Seal Reviewer and Issue-only Validator remain sequential and own only their prescribed artifacts/tests.
- Attempt 2 is sequential: one bounded production Implementer owns only the PLCS loader contract; the parent owns detached-base representative artifact generation and final integration. No production worker edits tests.

## Independent test work unit

- After production Preflight PASS, one independent Test Writer owns all test changes. Unit tests mirror `src/` only for the new shared/base pure modules and high-value task logic; renderer/model-forward flows live in the single integration smoke.
- Required cases: v1/v2 mappings and unknown version; shape-generic NumPy/Torch round trip; aspect ratio; v1 golden behavior; metadata-free v1 and v2 rejection; root/scene/checkpoint mismatch matrix; BLCS velocity/gravity/projection/metric/predictor; PLCS translation versus canonical metres and renderer; SLCS scalar uncertainty/SceneResult metres; v2 equal physical loss/beta; materializer non-overwrite and physical round trip; both-version CPU task flow.
- Attempt-2 required cases add frozen-base metadata-free representative checkpoint/data parity, mandatory PLCS public-loader metadata rejection, actual task dataset/model CPU chains, and default-v1 standard/tracking predictor physical scaling. A fixture-generation failure is not a skip-based PASS.

## Canonical verification commands

- `unit-contract`: shared resolver, geometry, dataset and checkpoint contract.
- `unit-blcs`: BLCS generation, gravity/loss/matching/metric/predictor/checkpoint.
- `unit-plcs`: PLCS targets, generation, loss and both predictors.
- `unit-slcs`: SLCS dataset/adapter/predictor/metrics and SceneResult units.
- `integration-normalization`: cross-task v1/v2 CPU smoke and bounded materialization.
- `legacy-v1-checkpoint-parity`: frozen-base metadata-free representative BLCS/PLCS checkpoint inference, loss, and metre-metric parity plus explicit v2 rejection before config/state use.
- `preflight-regression`: existing v1-facing schema/config/loss/metric/predictor tests that must stay green before the Test Writer adds new coverage.
- `knowledge-graph`: formal v1/v2 baseline run/group nodes.
- `precommit-all`: Ruff, mypy, script reviewer and repository hooks.
- `full-pytest`: complete repository regression; authorized for Test and Seal after tests are finalized. Unlike the Issue-specific CPU smoke, this repository-wide suite exposes GPU 0 because the existing baseline contains mandatory CUDA acceptance tests that fail rather than skip when CUDA is hidden.

## Ordered execution plan

1. Preserve attempt-1 implementation and keep strict typed config/model state plus shared metadata gates unchanged; do not auto-migrate archived checkpoints already incompatible at the frozen base.
2. Make PLCS public scene loading require and forward a resolved normalization contract across every owned caller.
3. In a detached worktree at frozen base `59e3b166...`, deterministically create small metadata-free representative checkpoints and dataset fixtures using actual BLCS/PLCS task classes, then capture normalized/metre prediction, task loss, and metre metrics with provenance. A non-reproducible base execution is an explicit blocker, never fabricated evidence.
4. Parent integrates implementation and evidence handoffs, records exact adapter amendments/golden provenance in `implementation.md`, and runs attempt-2 discovery Preflight using only frozen Validator-return categories.
5. After Preflight PASS, a fresh Test Writer commits the base goldens, adds strict positive/negative real-checkpoint parity, replaces the surrogate CPU smoke with actual dataset/model chains, adds mandatory PLCS loader failures and v1/default predictor scale assertions, then runs every test-stage canonical check.
6. Repeat Seal and Issue-only validation on one stable candidate. Only Validator PASS may proceed to PR packaging/finalization.

## Validation strategy

- Contract validation is fail-closed: unknown/partial/mixed/mismatch cases must raise before reading model tensors or scene arrays.
- v1 is checked by numeric golden/regression evidence, not only round trip. v2 is checked by aspect ratio, equal physical loss, physical gravity and metre reconstruction.
- Repository-wide deterministic `rg` inventory confirms remaining `COURT_COORD_SCALE_*` references are physical geometry, documented v1 aliases, or explicitly version-resolved compatibility paths.
- Attempt-2 Preflight may additionally inspect only frozen-base representative artifact provenance/current-v1 parity feasibility, explicit-v2 metadata rejection, mandatory PLCS scene-loader call graph, actual tiny-model/loader feasibility, and default-v1 predictor scale consumers. These are frozen Validator-return closure categories, not an open-ended campaign.
- Operational evidence uses identical physical scenes and controlled v1/v2 training pairs; comparisons are reported only in physical metres, never raw normalized loss alone.
- Final Validator receives the frozen Issue and sealed candidate only and must substantively verify all 22 ACs.

## Non-goals and prohibited changes

- Do not alter physical court/net dimensions, court axes/origin, camera/world/UV conventions, SMPL/canonical pose units, production metre arrays, or SceneResult physical units.
- Do not change BLCS shot/apex generation policy, repair the old 28m lob distribution, auto-convert model weights, infer versions from values, overwrite legacy artifacts, or make v2 a process-global mutable singleton.
- Do not broaden SLCS scalar uncertainty into a per-axis prediction head; preserve the scalar convention while making its scale version-aware.

## Risks, rollback, and open decisions

- Frozen decisions: v1 remains the composed default; new v2 runs opt in explicitly. New artifacts carry identical root/scene/checkpoint metadata; all-missing legacy is accepted only under v1. Stored `ball_vel_world` remains m/s; only model normalized velocity uses scale. v1 scalar SLCS uncertainty keeps mean(scale), v2 uses HALF_LENGTH.
- Frozen loss decisions: v1 retains normalized Smooth L1 beta and legacy tracking weights/literal gravity target; v2 uses a 1.0m physical Huber transition, uniform default position/matching axes, and derived gravity target. Any different task weight requires separate explicit justification.
- Frozen data decision: deterministic materialization from identical legacy scenes is used instead of re-simulation so normalization is the only baseline variable; v1 data is never modified.
- Rollback is configuration-level: select v1 and legacy artifacts. Removal of v1 is not part of this Issue.
- Main risks are config paths that omit the shared group, default arguments capturing v1, scene subsets escaping metadata validation, PLCS renderer/canonical coupling, SLCS uncertainty units, and long training. Preflight may use only the bounded diagnostics listed in Validation strategy plus config-composition, metadata-mutation, unit-round-trip, checkpoint-mismatch and materialization-smoke categories.
- Tester cycle 1 exposed a baseline-environment coverage gap: `full-pytest` combined `CUDA_VISIBLE_DEVICES=""` with existing mandatory CUDA tests and lacked the private NHT submodule config. The repaired authority keeps all normalization smokes CPU-only, exposes only GPU 0 for the repository-wide test, and requires a worktree-local non-symlink NHT config. This changes test environment authority only, not Issue production behavior or acceptance scope.
- Attempt-2 risks are confusing already-obsolete archived files with normalization compatibility, fixture provenance drift, and a nominal smoke that substitutes a generic model. Mitigations are no legacy architecture conversion, committed frozen-base artifact/golden hashes and generation provenance, explicit v1/v2 metadata guards, and actual task class assertions. If frozen-base representative artifacts cannot be generated and replayed, AC-003/004 remain NOT VERIFIED and the task blocks rather than silently skipping.
