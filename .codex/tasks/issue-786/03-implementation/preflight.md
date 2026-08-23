# Production preflight

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:1bd9e244065866a58dbbdf4a677745eb262d3c64a006f656019fb4021dfebaf4`

## Candidate identity

State was read before substantive review: `phase = "implementation"`,
`preflight_verdict = ""`, `preflight_cycle = 0`, and `test_cycle = 1`; this is
the repair-local review immediately before pending test cycle 2, in Discovery
mode. The manager's current artifact contract expects the completed test cycle
(`1`) until the parent records the preflight transition to pending cycle 2.
The previous
preflight artifact was read but its PASS is stale for this merged candidate and
state does not freeze its findings. This review is bounded to the frozen
repair-local worklist: the three final-tree merge resolutions, preservation of
the normalization contracts on the merged APIs, separation of upstream-main
additions from Issue-authored scope, canonical preflight checks, and direct
merge-local regressions.

- Branch/head: `feat/issue-786-normalization-v2` /
  `64ea1b5a99bacd5ec7f8ab4f356333835eaa9de9` (`merge: synchronize main for
  issue 786`).
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- `candidate-fingerprint .codex/tasks/issue-786` recomputed
  `sha256:1bd9e244065866a58dbbdf4a677745eb262d3c64a006f656019fb4021dfebaf4`,
  matching the assigned candidate identity.
- The complete Issue non-workflow diff is 323 paths from the frozen base;
  `git diff --check` passes. The PR-shaped diff from `origin/main` is 266
  paths: 11 workflow paths, 27 knowledge evidence paths, 164 `src/**` paths,
  63 `tests/**` paths, and `.gitignore`. No source, test, plan,
  state, or implementation artifact was edited by this reviewer; only this
  artifact and manager-owned preflight results/logs are authored.

## Changed scope

- `64ea1b5a` has exactly the three declared overlaps: `src/tasks/blcs/configuration.py`, `tests/unit/tasks/blcs/inference/test_tracking_predictor.py`, and `tests/unit/tasks/plcs/inference/test_tracking_predictor.py`. The resolved tree has no unmerged index entries and no conflict markers.
- The BLCS parser now keeps strict exact schemas for both `blcs_track_query`
  and `blcs_track_query_ablation`, including the ablation-only FFN/writeback
  axes. The merged tracking-boundary path invokes generator-section validation
  for either model and retains `position_axis_weights_v2`,
  `position_huber_beta_v1`, and `position_huber_transition_m_v2` validation.
  The v1/v2 normalization resolver and loss/gravity helpers remain intact.
- Both merged predictor tests retain the default-v1 metre-scale assertion and
  the v2 metre-scale assertion. Their checkpoint-restoration tests retain the
  upstream exact ablation model/adapter dispatch and now bind explicit
  normalization metadata. The upstream ablation model/config/test additions
  are separable from the Issue normalization changes and were not counted as
  Issue-authored implementation scope.

## Deterministic policy checks

- Frozen Issue, exploration, plan, checks manifest, implementation handoff,
  repository guidance, state, prior preflight, current source, merge commit,
  and complete diffs against both the frozen base and `origin/main` were read.
  No open-ended mutation campaign or new diagnostic category was introduced.
- The frozen bounded checks were applied: strict merged BLCS configuration and
  generator/loss validation, default/v2 tracking predictor physical outputs,
  upstream ablation restoration, conflict-marker/index inspection, diff
  sanity, and the two required canonical preflight checks.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- .
  ':(exclude).codex/tasks'` passed; `git ls-files -u` reports zero entries;
  the marker scan over `src` and `tests` reports zero hits.

## Focused checks

- **Merged conflict-local behavior (AC-002, AC-008, AC-010, AC-013,
  AC-014, AC-020, AC-021):** the focused BLCS/PLCS configuration, ablation,
  and tracking-predictor suite passed `51` tests. It exercises both tracking
  model names, all four ablation configurations, strict nested axes, upstream
  exact model/adapter checkpoint restoration, default-v1 physical scale, and
  v2 physical scale.
- A bounded in-memory configuration composition diagnostic validated both
  `train_tracking` and `train_tracking_chunked`, `track_query` and all four
  `track_query_ablation_{a,b,c,d}` variants under both v1 and v2: `20` variant/
  version validations passed. The same bounded diagnostic rejected missing
  generator sections for both model families and rejected non-uniform v2
  position weights: `8` strict rejection checks passed.
- The three conflict paths were inspected against both merge parents. The
  normalization helper implementations use the selected contract; the
  upstream ablation parser/test behavior is retained; and the default/v2
  assertions are present in the final tests. No unresolved conflict or
  unmerged entry remains.

## Canonical command results

- `preflight-regression`: **PASS**, exit 0, `127 passed`; generated through
  `manage_issue_task.py run-check .codex/tasks/issue-786 preflight
  preflight-regression` and bound to the current candidate.
- `precommit-all`: **RETURN**, exit 1. Ruff and task-script reviewer pass, but
  mypy reports one error at
  `tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py:129`:
  `Non-overlapping identity check (left operand type:
  "type[BLCSTrackQueryAblationModel]", right operand type:
  "type[BLCSTrackQueryModel]") [comparison-overlap]`. The result was generated
  through the canonical `run-check` command and is bound in
  `preflight-checks.json` with raw log
  `logs/canonical-preflight-precommit-all.log`.
- Every required check authorized for the preflight stage was executed through
  `run-check`; one required check is therefore not PASS.

## Baseline comparison

- The previous candidate before merge (`2661f3a8`) recorded preflight,
  test, and seal PASS with precommit PASS. The current merged tree preserves
  the focused normalization and ablation behavior, but the required all-files
  mypy gate now reaches an upstream-main ablation test that was absent from the
  Issue parent and fails its class-identity assertion's static typing rule.
- The failing path is an upstream-main addition: it is absent from
  `2661f3a8` and present in `origin/main`; no Issue normalization code is
  implicated by the diagnostic. It is nevertheless a direct merge-local
  regression for this candidate because the required canonical preflight gate
  is no longer green.
- Focused runtime evidence is positive, but it cannot override a required
  canonical preflight failure. No broader semantic acceptance was inferred
  from the passing focused checks.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` →
  `sha256:1bd9e244065866a58dbbdf4a677745eb262d3c64a006f656019fb4021dfebaf4`.
- `.venv/bin/python -m pytest -q tests/integration/tasks/blcs/test_model_configs.py tests/integration/tasks/plcs/test_model_configs.py tests/integration/tasks/blcs/test_track_query_ablation.py tests/integration/tasks/plcs/test_track_query_ablation.py tests/unit/tasks/blcs/inference/test_tracking_predictor.py tests/unit/tasks/plcs/inference/test_tracking_predictor.py` → PASS, `51 passed`.
- Bounded v1/v2 composition and strict rejection diagnostics → PASS, `20`
  variant/version validations and `8` rejection checks.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression` → PASS, exit 0, `127 passed`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all` → RETURN, exit 1, one mypy `comparison-overlap` error at the upstream ablation test line recorded above.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'` → PASS; `git ls-files -u` → zero; conflict-marker scan → zero.

## Final production preflight verdict

RETURN

## RETURN implementation findings

**F-001 — required precommit gate fails on an upstream-main ablation test
(AC-001 through AC-022 by `precommit-all` authority; direct merged tracking
surfaces AC-002, AC-008, AC-010, AC-013, AC-014, AC-020, and AC-021).**
`tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py:129`
uses a class identity comparison that mypy proves non-overlapping under the
repository's strict configuration. This file is absent from the Issue parent
and present in `origin/main`; the merge therefore exposes the failure to the
required all-files precommit gate even though the focused merged behavior
passes.

Bounded repair bundle: preserve the upstream distinct-architecture assertion
and repair only its typing form (or its narrow annotation) so the repository
mypy hook accepts it; do not remove the assertion, weaken mypy, alter
normalization production code, or reopen the normalization mutation campaign.
Then rerun both canonical preflight checks and a fresh closure Preflight on the
same merge-local worklist.
