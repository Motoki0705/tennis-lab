# Production preflight

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`

## Candidate identity

- Review mode: **Closure**. Before review, `state.toml` recorded `phase = "implementation"`, `preflight_verdict = "RETURN"`, and `preflight_cycle = 1`; therefore the prior RETURN findings R-001 and R-002 are the complete frozen worklist. This review verifies only those findings, every canonical preflight-stage check, and direct repair-local regressions.
- Branch/head: `feat/issue-786-normalization-v2` / `5dd25cce380b488396c94110df150e00bb20d270`.
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Recomputed with `candidate-fingerprint .codex/tasks/issue-786`: `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`. The fingerprint matches `implementation.md` and both generated canonical preflight results.
- The complete non-workflow diff is 223 paths (7,262 additions, 464 deletions); `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'` passes. The reviewer made no production or test edits.

## Changed scope

- The candidate retains the shared immutable v1/v2 contract, strict dataset/checkpoint metadata gates, version propagation, task consumers, and non-overwriting materializer from the integrated implementation.
- The bounded repair adds only the normalization-era in-memory BLCS legacy-v1 overlay in `src/tasks/blcs/model_io/checkpoints.py`, its conflict/byte-preservation unit coverage, and explicit v1 arguments in the four existing PLCS scene-loader test calls. The public PLCS production signatures remain mandatory.
- The overlay supplies only `court_coordinate_normalization.version = v1`, `training.position_huber_beta_v1 = 1.0`, and `training.position_huber_transition_m_v2 = 1.0` when those normalization-era fields are absent. Existing values are checked for exact expected float values; conflicting values fail closed. No checkpoint bytes or metadata-bearing path is rewritten.
- The known archived BLCS/PLCS files remain outside the normalization compatibility promise because their frozen-base configuration/state or filename contracts are already incompatible. No architecture migration was added.

## Deterministic policy checks

- Frozen Issue, exploration, plan, checks manifest, implementation, repository guidance, state, prior RETURN artifact, current source, and complete base diff were read before review. No new diagnostic/mutation category or open-ended fuzzing was introduced.
- Closure scope was frozen to R-001 (metadata-free explicit-v1 BLCS composition/parity, overlay conflicts/bytes, explicit-v2 rejection, and archived architecture-drift refusal) and R-002 (four explicit-v1 PLCS test callers with mandatory production signatures), plus canonical checks and repair-local regressions.
- `git status --short` is clean after the bounded diagnostics and canonical runs; the only authored stage artifacts are this replacement and manager-owned `preflight-checks.json`/raw logs.

## Focused checks

- **R-001 — representative metadata-free BLCS v1 composition/parity (AC-003, AC-004, AC-020, AC-021):** The representative manifest binds generation to clean frozen base `59e3b166c2d010d5e62be52c2be76d98a94af0e0`, CPU float32 deterministic execution, actual `BallTrajectoryDataset`, `BLCSLightningModule`, and `BLCSPredictor`, with checkpoint SHA-256 `69af21b3f8008ab7f53708e1d03346113aafa49c857a5465ad6e6da86f80a5e7` and golden SHA-256 `0f494e30e07c99e3f59feba113093530a87c25334a216eea9bee33e9a32397e6`. Current explicit-v1 loading reports `legacy_metadata_free = True`, composes the actual `BLCSMultiViewAxialModel`, and strict Lightning state loading succeeds. Replayed inputs, prediction, loss, and metric arrays match the frozen golden with `max_abs_diff = 0.0` for all non-velocity fields and all loss/metric fields. The current dataset exposes velocity in normalized model units; converting that field back to the golden's declared world m/s representation gives `max_abs_diff = 1.1920928955078125e-07 m/s`, below the frozen `1e-5` tolerance. The checkpoint SHA-256 is unchanged before/after replay.
- **R-001 — narrow overlay conflicts and v2 guard:** `tests/unit/tasks/blcs/model_io/test_checkpoints.py` covers both introduced training fields, rejects each conflicting value, and asserts the metadata-free checkpoint bytes are unchanged. The representative metadata-free BLCS and PLCS checkpoints both reject explicit v2 with `MissingCourtCoordinateMetadataError` before composition/state use. The focused repair suite passes 10/10.
- **R-001 — archived architecture-drift refusal:** Explicit-v1 runtime qualification of the archived BLCS checkpoint reaches composition but fails with the frozen-configuration error `MissingConfigurationKeyError: Missing required configuration key(s): model.num_court_tokens`; no state-key/config migration occurs. This agrees with the frozen `blocked-diagnostics.log` evidence of 132 missing and 132 unexpected architecture keys. The candidate therefore preserves strict refusal of the archived architecture-drift file while accepting only the base-compatible representative.
- **R-002 — mandatory PLCS scene-loader callers (AC-010, AC-015):** `inspect.signature` shows `court_coordinate_normalization` is required (no default) on both `load_scene` and `load_scene_bundle`. Both production callers forward their resolved contract, and the four existing unit-test calls now pass `resolve_court_coordinate_normalization("v1")`; metadata/file-boundary assertions remain intact. The focused repair suite passes 10/10.

## Canonical command results

- `preflight-regression`: **PASS**, exit 0, `127 passed in 10.04s`; generated by `manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`, candidate-bound in `03-implementation/preflight-checks.json`, raw log `logs/canonical-preflight-preflight-regression.log`.
- `precommit-all`: **PASS**, exit 0; Ruff, mypy, and task-script reviewer all passed; generated by `manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`, candidate-bound in `03-implementation/preflight-checks.json`, raw log `logs/canonical-preflight-precommit-all.log`.
- Every required check in `02-planning/checks.json` authorized for the `preflight` stage was executed through `run-check`; both results bind candidate `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f` and exit 0.

## Baseline comparison

- The frozen-base representative bundle is provenance-bound to a tracked-clean frozen-base worktree and supplies the actual task-class checkpoint/golden oracle. The separate archived-diagnostics manifest records why the named large checkpoints cannot be treated as normalization-only legacy artifacts: BLCS has strict architecture/state-key drift and both files have typed configuration/layout blockers.
- Relative to the previous discovery RETURN, R-001's missing `training.position_huber_beta_v1`/companion-field composition failure is closed by the two-field in-memory overlay; R-002's four missing PLCS arguments are closed without making either production signature optional. The prior `preflight-regression` result was 125 passed with a precommit failure; the current canonical results are 127 passed and precommit PASS.
- No candidate production behavior beyond the frozen repair bundle was reviewed or accepted in closure mode. Remaining planned fixture/test expansion belongs to the post-Preflight Test Writer and is not a closure finding.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` → `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression` → PASS, exit 0, 127 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all` → PASS, exit 0; Ruff/mypy/task-script reviewer passed.
- `.venv/bin/python -m pytest -q tests/unit/tasks/blcs/model_io/test_checkpoints.py tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py tests/unit/tasks/plcs/visualization/io/test_scene.py` → PASS, 10 passed in 11.72s.
- Independent CPU replay of the representative BLCS fixture/checkpoint with explicit v1, strict state load, predictor decode, supervised loss, and metrics → PASS; non-velocity and loss/metric golden deltas 0.0, world-velocity conversion delta `1.1920928955078125e-07`, checkpoint bytes unchanged.
- Explicit-v2 representative checkpoint matrix → PASS; BLCS and PLCS each raised `MissingCourtCoordinateMetadataError` before composition.
- Archived BLCS composition probe under explicit v1 → PASS as a refusal; `MissingConfigurationKeyError` for missing `model.num_court_tokens`, with no architecture migration.
- `inspect.signature` plus `rg` caller inventory → PASS; both public normalization parameters are mandatory, all production callers forward a resolved contract, and all four repaired test callers pass explicit v1.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'` → PASS; `candidate-fingerprint` after all checks remains unchanged.

## Final production preflight verdict

PASS

## RETURN implementation findings

None
