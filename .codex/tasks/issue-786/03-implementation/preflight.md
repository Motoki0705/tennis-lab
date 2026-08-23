# Production preflight

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:ef780395bce0ef3208e173408a2ef1ab4880604d10fee427197ffc8fe270314d`

## Candidate identity

- Review mode: **Discovery**. `state.toml` is at `phase = "implementation"` with an empty `preflight_verdict` and `preflight_cycle = 0`; this is the first Preflight for attempt 2. The review scope is the complete approved implementation and only the frozen attempt-2 categories: frozen-base representative v1 provenance/parity feasibility, explicit-v2 metadata rejection, the mandatory PLCS scene-loader call graph, actual task-chain feasibility, and default-v1 predictor scale consumers.
- Branch/head: `feat/issue-786-normalization-v2` / `3ccea609409d930a4e2a7dbf4083a6f014ae7ff3`.
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Recomputed with `candidate-fingerprint`: `sha256:ef780395bce0ef3208e173408a2ef1ab4880604d10fee427197ffc8fe270314d`; this matches `implementation.md` and both generated preflight check results. No reviewer source, test, plan, Issue, state, or other workflow artifact was edited.
- The complete non-workflow diff is 221 paths (7,155 additions, 457 deletions): shared normalization/base contracts, BLCS/PLCS/SLCS and scene integration, documentation/knowledge evidence, and the existing test suite. `git diff --check` passes.

## Changed scope

- The implementation integrates an immutable v1/v2 contract, explicit configuration propagation, dataset/checkpoint metadata gates, non-overwriting materialization, task consumers, and metre-valued scene outputs. The attempt-2 PLCS boundary changes `src/tasks/plcs/generate_dataset/io/scene_loader.py::load_scene` and `src/tasks/plcs/visualization/io/scene.py::load_scene_bundle` to require a resolved contract and validate metadata before payload reads.
- The parent handoff explicitly excludes architecture migration for the two archived BLCS/PLCS checkpoints that were already unloadable at the frozen base. That exclusion is consistent with the frozen plan; this review does not turn those pre-existing architecture/configuration failures into a new requirement.
- The changed public PLCS signatures leave four existing unit-test callers without the required argument. This is a candidate consistency failure, not an ownership collision; it is recorded below with the canonical mypy evidence.

## Deterministic policy checks

- `candidate-fingerprint .codex/tasks/issue-786`: **PASS**; current candidate is the identity recorded above.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: **PASS**; no whitespace errors.
- The frozen Issue, exploration, plan, checks manifest, implementation, repository guidance, state, current code, previous in-place preflight artifact, and complete diff were read before review. No diagnostic category was added and no open-ended fuzzing was run.
- Repository guidance requires `.venv/bin/python`, pytest, Ruff/mypy/task-script review, and no silent compatibility fallback. The candidate obeys the typed contract at current PLCS scene boundaries, but the legacy BLCS compatibility path fails before model construction as described below.

## Focused checks

- **Frozen-base representative provenance and parity (AC-003, AC-004):** `logs/legacy-base-goldens/representative/manifest.json` binds generation to a clean worktree at the exact frozen base, Python/NumPy/Torch seed 786, CPU float32 deterministic execution, actual `BallTrajectoryDataset`/`SceneDataset`, actual `BLCSLightningModule`/`PLCSLightningModule`, strict checkpoint reload, and immutable checkpoint/dataset/golden hashes. The archived named checkpoints remain separately recorded as unloadable at the frozen base; no migration is required by this review. On the current candidate, explicit-v2 rejection works for both representative checkpoints. The representative PLCS checkpoint loads under a resolved v1 contract and its actual dataset/model prediction matches the recorded meter golden exactly (`max absolute difference = 0.0`). The representative BLCS checkpoint is metadata-free and reaches `load_checkpoint_runtime` under explicit v1, but model composition then fails with `ConfigAttributeError: Key 'position_huber_beta_v1' is not in struct (full_key: training.position_huber_beta_v1)`. It therefore cannot produce current-candidate v1 inference/loss/metric parity. This is an implementation gap in the metadata-free v1 checkpoint path, not an archived-checkpoint architecture issue.
- **Explicit-v2 metadata rejection (AC-004, AC-005, AC-015, AC-016):** bounded direct checks against the representative metadata-free BLCS checkpoint, PLCS checkpoint, and PLCS scene returned `MissingCourtCoordinateMetadataError` before model state or scene payload use when v2 was selected. The messages identify metadata-free artifacts as legacy-v1-only. This category passes.
- **Mandatory PLCS scene-loader call graph (AC-010, AC-015):** `load_scene` requires `CourtCoordinateNormalization`, derives the dataset root, and calls `validate_dataset_court_coordinate_contract` before opening `meta.json`, `scalars.json`, or any `.npy`. `load_scene_bundle` requires and forwards the same resolved object. Every production caller found by `rg` forwards a contract: `visualization/orchestrator.py` forwards `RuntimeConfig.court_coordinate_normalization`, and `scripts/analysis/visualize_rotation_error_samples.py` forwards its resolved runtime contract. The generic `SceneDatasetBase` path independently validates root/selected-scene metadata before header/payload indexing. The boundary itself passes; existing tests omit the new required argument and cause the canonical precommit failure below.
- **Actual tiny-model/loader feasibility (AC-020, AC-021):** the frozen-base representative PLCS fixture was loaded by the current `SceneDataset`, collated, run through the actual `PLCSPredictor`, and denormalized to meter outputs successfully. The analogous current BLCS path is blocked by the missing legacy v1 training fields above. The committed normalization integration smoke remains a `torch.nn.Identity` plus direct loss/metric/projection/renderer exercise; it does not establish the planned actual BLCS/PLCS dataset-to-model chain or task-class assertions. The implementation handoff already assigns replacement of this nominal smoke to the independent Test Writer after a successful Preflight; this review records the feasibility blocker instead of treating the surrogate as proof.
- **Default-v1 predictor scale consumers (AC-003, AC-008, AC-010, AC-011, AC-021):** standard and tracking BLCS predictors and standard/tracking PLCS predictors all call the resolved contract's `denormalize_position` (and BLCS velocity equivalent), with no predictor-local `COURT_COORD_SCALE_XYZ` use. The actual PLCS representative output confirms v1 meter scaling. BLCS predictor execution could not be numerically completed because its metadata-free v1 checkpoint cannot pass current config composition. No additional fixed-scale predictor consumer was found in the bounded inventory.

## Canonical command results

- `preflight-regression`: **PASS**, exit 0; `125 passed in 9.84s`. Result is generated by `manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`, bound to the candidate above, and recorded in `logs/canonical-preflight-preflight-regression.log`.
- `precommit-all`: **FAIL**, exit 1. Ruff and task-script reviewer pass; mypy reports four missing named arguments introduced by the mandatory PLCS boundary:
  - `tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py:25,47` calls `load_scene` without `court_coordinate_normalization`.
  - `tests/unit/tasks/plcs/visualization/io/test_scene.py:18,28` calls `load_scene_bundle` without `court_coordinate_normalization`.
  Result is generated by `manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`, bound to the candidate above, and recorded in `logs/canonical-preflight-precommit-all.log`.
- `preflight-checks.json` contains both required preflight-stage IDs, current candidate fingerprints, invocation digests, and the exact outcomes above. No canonical check was skipped.

## Baseline comparison

- Frozen-base `59e3b166c2d010d5e62be52c2be76d98a94af0e0` representative generation is reproducible and clean for the small fixtures; the manifest records actual task classes and strict reloads. Its separate `blocked-diagnostics.log` proves that the large archived BLCS/PLCS checkpoints and the legacy PLCS filename layout were not loadable through frozen-base public APIs. The plan correctly freezes those as out-of-scope architecture/configuration drift.
- Relative to that baseline, the current candidate successfully preserves the representative PLCS v1 path and explicit v2 metadata guard but fails the representative BLCS v1 path because the new strict parser requires normalization-only training fields absent from the unchanged metadata-free checkpoint config. The candidate therefore does not yet satisfy the frozen v1 parity premise for both tasks.
- The current required preflight baseline is not green: `preflight-regression` passes, while `precommit-all` fails solely on the four unupdated PLCS loader test calls. This is directly caused by the mandatory public signature and must be resolved before a Preflight PASS.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786`: PASS; `sha256:ef780395bce0ef3208e173408a2ef1ab4880604d10fee427197ffc8fe270314d`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`: PASS; exit 0, 125 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`: FAIL; exit 1, four mypy call-argument errors listed above.
- Bounded inline checkpoint/scene matrix using the representative files under `.codex/tasks/issue-786/logs/legacy-base-goldens/representative`: metadata-free v2 rejection PASS for BLCS checkpoint, PLCS checkpoint, and PLCS scene; PLCS resolved-v1 load/dataset/model/prediction PASS with exact meter golden parity; BLCS resolved-v1 load FAIL at `training.position_huber_beta_v1` before model execution.
- `rg` call-graph inventory over `src/tasks/plcs` and `tests`: all production PLCS scene-loader callers forward a resolved contract; four existing test calls do not.
- No production or test files were edited by this reviewer. The only generated machine artifacts are the canonical preflight results and raw logs owned by the stage.

## Final production preflight verdict

RETURN

## RETURN implementation findings

1. **R-001 — affected AC-003, AC-004, AC-020, AC-021 (metadata-free BLCS v1 compatibility):** the unchanged frozen-base representative BLCS checkpoint is accepted by the explicit-v1 metadata gate but cannot be composed by the current candidate because `load_checkpoint_runtime` injects only `court_coordinate_normalization`; strict current BLCS parsing then requires `training.position_huber_beta_v1` (and the companion normalization loss fields). Provide one narrowly scoped, in-memory legacy-v1 configuration overlay for fields introduced by this normalization change, while preserving strict model/state parsing and refusing the archived architecture-drift checkpoints. Add the planned real BLCS representative parity/loss/metric test after the repair; it must prove unchanged checkpoint bytes, actual task classes, and explicit v2 rejection.
2. **R-002 — affected AC-010, AC-015 (mandatory PLCS boundary test callers):** update the four existing PLCS loader test calls to pass an already resolved v1 contract (and retain metadata rejection assertions) so the required `precommit-all` check passes. The public production signatures and call graph must remain mandatory; do not restore an optional/silent fallback.

These are one bounded repair bundle: restore normalization-only legacy-v1 BLCS composition, align the mandatory PLCS test callers, then rerun both canonical preflight checks and a fresh closure review. The known frozen-base archived-checkpoint architecture/configuration drift is not part of the repair.
