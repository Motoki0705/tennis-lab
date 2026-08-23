# Production preflight

- Issue: #786
- Attempt: 1
- Test cycle: 2
- Status: COMPLETE
- Candidate SHA-256: `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`

## Candidate identity

- Review mode: **Discovery**. `state.toml` records `preflight_verdict = "PASS"` for cycle 1, so this is the required fresh production review for the pending Tester cycle 2. Scope was bounded to the approved implementation, the frozen plan categories (config composition, metadata mutation, unit round trip, checkpoint mismatch, and materialization smoke), direct Tester-return repair regressions, and canonical preflight checks.
- Branch/head: `feat/issue-786-normalization-v2` / `45320cb7b9e9ea96c0913592bb4e12cdf8e6ae4c`.
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Recomputed candidate fingerprint: `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d` (matches `implementation.md` and all generated test/preflight results).
- Complete candidate diff against the frozen base is 221 paths, 7,155 additions, and 456 deletions, excluding `.codex/tasks/` workflow artifacts. The current source portion is 164 paths; the current test portion is 29 paths; the knowledge evidence portion is 27 paths.

## Changed scope

- The approved production implementation covers the immutable v1/v2 resolver, explicit BLCS/PLCS/SLCS propagation, fail-closed dataset/checkpoint metadata, non-overwriting materialization, metre-valued scene boundaries, and documentation/knowledge evidence. The complete source diff was inspected against the frozen base; remaining fixed court dimensions are physical geometry or documented v1 aliases, and task consumers receive the selected contract.
- Comparing the current worktree with the cycle-1 Preflight PASS commit `65a67c839bbcc0dd021f466a722768d51f04b5a8`, there are no `src/` changes. The post-PASS candidate changes are test-only: Issue coverage additions plus the three direct Tester-return repairs.
- The three repair-local test changes are bounded and do not weaken an oracle: the B00 test now asserts the v2 dataset schema and all v2 semantic classes, the integration smoke asserts optional `SceneResult.ball_3d` presence before copying it, and the metadata test removes only a redundant static cast.
- The revised environment authority is confined to the repository-wide baseline. `full-pytest` selects `CUDA_VISIBLE_DEVICES=0`; all Issue-specific unit and normalization integration checks remain CPU-only with `CUDA_VISIBLE_DEVICES=""`. The required worktree-local `third_party/nht/configs/production.yaml` is a regular file under this worktree, not a symlink. No Issue production behavior or acceptance scope is changed by this authority repair.

## Deterministic policy checks

- `candidate-fingerprint .codex/tasks/issue-786`: PASS; the recomputed identity above is current.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: PASS; no whitespace errors.
- The frozen Issue, exploration, updated plan, canonical check manifest, implementation handoff, repository guidance, state, prior Preflight artifact, Tester artifact, complete current diff, and generated test evidence were read before review. No diagnostic category was added.
- The complete current test-stage result set is supporting evidence only for this stage: all nine generated results bind candidate `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`; the GPU-visible repository baseline reports `3230 passed, 53 skipped, 19 warnings`.

## Focused checks

- **Approved normalization scope (AC-001 through AC-022):** The resolver is immutable and explicit (`v1=(5.485,11.885,1.07)m`, `v2=(11.885,11.885,11.885)m`); conversion, geometry, BLCS/PLCS/SLCS consumers, metadata/checkpoint guards, and the metre-valued `SceneResult` boundary use the selected contract. Physical court dimensions and canonical/root-relative metre poses remain unchanged. The frozen bounded categories are represented by the Issue unit/integration evidence and the v1 regression check below.
- **Tester-return environment regression:** The canonical manifest keeps `CUDA_VISIBLE_DEVICES=""` for `unit-contract`, `unit-blcs`, `unit-plcs`, `unit-slcs`, and `integration-normalization`, while only `full-pytest` uses `CUDA_VISIBLE_DEVICES="0"`. The complete-suite result demonstrates that the GPU-required baseline is executed rather than hidden; the NHT prerequisite is present at the worktree-local regular-file path.
- **Tester-return test repairs:** The B00 assertion is strengthened by exact `COURT_SCHEMA_V2.dataset_schema` identity and positive coverage for every schema-defined class; the smoke assertion preserves the existing metre-array check; the cast removal leaves the metadata mapping oracle unchanged. No production path or test scope outside the frozen repair bundle changed after cycle-1 PASS.

## Canonical command results

- `preflight-regression`: **PASS**, exit 0; `125 passed`. The generated result is bound to candidate `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d` and is recorded in `logs/canonical-preflight-preflight-regression.log`.
- `precommit-all`: **PASS**, exit 0; Ruff, mypy, and task-script reviewer each passed. The generated result is bound to the same candidate and is recorded in `logs/canonical-preflight-precommit-all.log`.
- `preflight-checks.json` contains both required preflight-stage checks, each with exit code 0, current candidate identity, invocation digest, and verdict `PASS`.

## Baseline comparison

- Cycle-1 production Preflight PASS bound candidate `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3`; its approved production scope has no source changes in the current worktree. The current candidate is `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`, and both required preflight checks pass again.
- Tester cycle 1 returned on the repository-wide baseline because CUDA was hidden and the private NHT configuration was absent. The repaired authority exposes only GPU 0 for `full-pytest`, leaves every Issue-specific CPU check unchanged, and supplies the worktree-local non-symlink NHT configuration. The final supporting full-suite result is `3230 passed, 53 skipped` with no failure or error.
- The direct repairs close only the reported test/environment findings: B00 now validates the v2 artifact schema before semantic coverage, and the two static-typing repairs do not remove runtime assertions. No frozen production or acceptance finding remains open.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786`: PASS; `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`: PASS; exit 0, `125 passed`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`: PASS; exit 0, Ruff/mypy/task-script reviewer passed.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: PASS.
- `git diff --name-only 65a67c839bbcc0dd021f466a722768d51f04b5a8 -- src`: PASS; no production source paths changed since cycle-1 Preflight PASS.
- The existing generated test-stage evidence is candidate-bound and reports PASS for all nine required test checks, including `full-pytest: 3230 passed, 53 skipped, 19 warnings`.

## Final production preflight verdict

PASS

## RETURN implementation findings

None
