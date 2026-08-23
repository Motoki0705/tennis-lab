# Plan

- Issue: #786
- Attempt: 1
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Acceptance checklist mapping

| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | Add an immutable typed resolver/value object and shape-generic conversion helpers; retain current constants as documented v1 aliases only. | `unit-contract`, `precommit-all`, `full-pytest`. |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | Add a shared Hydra group with explicit v1 default and inject the resolved contract through every task boundary rather than a mutable global. | Config-composition cases in `integration-normalization`; task unit checks. |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | Preserve v1 aliases, constructor defaults, legacy config values, beta, tracking weights and checkpoint behavior. | Golden v1 assertions in all unit checks and both-version CPU smoke. |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | Central compatibility validator accepts all-missing metadata only for explicitly resolved v1 and rejects it for v2. | `unit-contract` malformed/legacy matrix and `integration-normalization`. |
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
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | Write identical contract metadata at root and each new scene; index construction validates every selected scene before payload use. | Metadata mutation matrix in `unit-contract` and materialization smoke. |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | Persist the composed/resolved contract in Lightning config and add shared extraction/validation used by BLCS/PLCS/SLCS loaders/predictors. | Checkpoint round-trip/mismatch tests in `unit-contract` and task checks. |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | Use `norm-v1`/`norm-v2` qualified output configs and reject input/output identity in the materializer. | Artifact-name/metadata assertions plus recorded commands. |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | Add a Hydra materialization entrypoint that copies legacy scenes into separate v2 roots, recomputes only normalized position from physical/world or v1-resolved values, and records hashes/contracts. | `integration-normalization` bounded fixture plus full materialization validation log/manifest. |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | Run controlled v1/v2 pairs for BLCS and PLCS with identical architecture/seed/data scenes via shared training queue; register four run nodes and one group node with commands/metrics. | Training-queue completion evidence, run bundles, `knowledge-graph`, physical metric comparison. |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | Build bounded synthetic/versioned fixtures and lightweight task models for both versions; renderer remains a task integration smoke rather than file-mirrored unit test. | `integration-normalization`. |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | Independent Test Writer owns mirrored pure-logic tests and task-level integration smoke according to `test-structure`. | All unit/integration canonical checks and `full-pytest`. |
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

## Implementation topology and ownership

- Topology is staged parallel execution. First, one core Implementer owns only shared schema/base contract/config/materialization primitives. After its terminal handoff, three Implementers run in parallel with non-overlapping ownership: BLCS; PLCS plus normalized production compact data; SLCS plus tennis-scene metre boundary/docs.
- Production Implementers may edit source/config/docs in their ownership but no tests and no workflow artifacts. They must preserve edits from other agents and use the shared core contract rather than create task-local resolvers.
- The parent is the sole implementation integrator: resolves cross-task config composition, performs repository-wide fixed-scale inventory, writes `implementation.md`, runs materialization/training operations, and registers knowledge evidence.
- Independent Preflight Reviewer, Test Writer, Seal Reviewer and Issue-only Validator remain sequential and own only their prescribed artifacts/tests.

## Independent test work unit

- After production Preflight PASS, one independent Test Writer owns all test changes. Unit tests mirror `src/` only for the new shared/base pure modules and high-value task logic; renderer/model-forward flows live in the single integration smoke.
- Required cases: v1/v2 mappings and unknown version; shape-generic NumPy/Torch round trip; aspect ratio; v1 golden behavior; metadata-free v1 and v2 rejection; root/scene/checkpoint mismatch matrix; BLCS velocity/gravity/projection/metric/predictor; PLCS translation versus canonical metres and renderer; SLCS scalar uncertainty/SceneResult metres; v2 equal physical loss/beta; materializer non-overwrite and physical round trip; both-version CPU task flow.

## Canonical verification commands

- `unit-contract`: shared resolver, geometry, dataset and checkpoint contract.
- `unit-blcs`: BLCS generation, gravity/loss/matching/metric/predictor/checkpoint.
- `unit-plcs`: PLCS targets, generation, loss and both predictors.
- `unit-slcs`: SLCS dataset/adapter/predictor/metrics and SceneResult units.
- `integration-normalization`: cross-task v1/v2 CPU smoke and bounded materialization.
- `preflight-regression`: existing v1-facing schema/config/loss/metric/predictor tests that must stay green before the Test Writer adds new coverage.
- `knowledge-graph`: formal v1/v2 baseline run/group nodes.
- `precommit-all`: Ruff, mypy, script reviewer and repository hooks.
- `full-pytest`: complete repository regression; authorized for Test and Seal after tests are finalized.

## Ordered execution plan

1. Implement and review the shared immutable resolver, conversion helpers, typed config, metadata/checkpoint validators and Hydra materialization primitive with v1 compatibility preserved.
2. In parallel, migrate BLCS, PLCS and SLCS/tennis-scene consumers to explicit contract injection; add version-specific configs/defaults and canonical documentation references.
3. Parent integrates all handoffs, removes unintended active uses of fixed v1 constants, verifies config composition, and records implementation behavior/deviations.
4. Materialize `data/blcs_broadcast_norm_v2` and `data/plcs_broadcast_norm_v2` from the same legacy scenes without overwriting v1; validate root/scene metadata and physical round trips.
5. Read and use `training-queue`; enqueue controlled v1/v2 BLCS and PLCS baseline pairs serially with identical model/seed/splits except normalization/data path, then wait without starting unrelated work.
6. Register each finished run with `knowledge-control`, create the comparison group, record physical metre axis/aggregate metrics and validate the graph.
7. Complete `implementation.md`, then run the independent discovery Preflight. Any RETURN follows the bounded repair/closure rules before Test Writer.
8. After Test Writer PASS, run Seal, Issue-only validation, create the PR, bind final PR head/checks, and finalize.

## Validation strategy

- Contract validation is fail-closed: unknown/partial/mixed/mismatch cases must raise before reading model tensors or scene arrays.
- v1 is checked by numeric golden/regression evidence, not only round trip. v2 is checked by aspect ratio, equal physical loss, physical gravity and metre reconstruction.
- Repository-wide deterministic `rg` inventory confirms remaining `COURT_COORD_SCALE_*` references are physical geometry, documented v1 aliases, or explicitly version-resolved compatibility paths.
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
