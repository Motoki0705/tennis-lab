# Final candidate seal

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Seal cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Base revision: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`
- Candidate SHA-256: `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`

## Candidate identity

The independently recomputed candidate fingerprint is exactly
`sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`,
matching `state.toml`, `tests.md`, and `test-checks.json`. The frozen base is
`59e3b166c2d010d5e62be52c2be76d98a94af0e0`. The state is schema-v5 with
`test_verdict = PASS`, `test_cycle = 1`, and `test_return_count = 0`; the Tester
PASS candidate is the required identity for this seal. Candidate fingerprinting
excludes only `.codex/tasks/` workflow storage and includes all production,
test, fixture, documentation, configuration, and knowledge content.

## Changed-since-test inspection

The current fingerprint equals the Tester PASS fingerprint. `git status
--short` is empty, and `HEAD` is `bd51711223318fe18bbd76349844d6c3338421d7`,
the Tester candidate commit; there are no commits after it. Thus no source,
test, or fixture content changed after Tester PASS. Seal result/log files and
this seal artifact are the only seal-stage workflow writes and are excluded by
the fingerprint contract.

The complete base-to-worktree diff contains 266 paths. The non-workflow
inventory is 255 paths: one `.gitignore` repository-artifact policy change, 27
`knowledge/**` evidence paths, 164 approved `src/**` production/config/docs
paths, and 63 approved `tests/**` unit/integration/e2e/fixture paths. The
remaining 11 paths are the frozen Issue and workflow artifacts under
`.codex/tasks/issue-786/`. No untracked path exists.

The final Tester packaging repair is confined to test ownership: the BLCS and
PLCS representative fixture files are R100 byte-preserving renames from
`.ckpt` to `.ckpt.bin`; the committed generator and manifest references and
the parity-test references were updated; and one fixture filename-policy test
was added. Comparing the pre-repair blobs at `1187a3fc` with the current
`.ckpt.bin` files gives unchanged sizes and digests:

| Fixture | Bytes | SHA-256 |
|---|---:|---|
| `blcs_representative_legacy_v1.ckpt.bin` | 100421 | `69af21b3f8008ab7f53708e1d03346113aafa49c857a5465ad6e6da86f80a5e7` |
| `plcs_representative_legacy_v1.ckpt.bin` | 71896 | `6c212eec6bbe616b498000928733318b19f76e4ad957a680963621c672ca841d` |

The complete pushed path inventory has no path ending `.ckpt`, `.pt`, `.pth`,
or `.pkl`; the fixture-root filename scan is also empty. All changed paths
fall within the Issue's normalization production/configuration/documentation,
formal evidence, test, and fixture scope. No production or test path was
changed by sealing.

## Canonical command results

Every required seal-stage check in `02-planning/checks.json` was executed via
`manage_issue_task.py run-check ... seal <check-id>` against the Tester
candidate. The machine record is schema version 1, stage `seal`, and binds all
results to the candidate above.

| Check ID | Result | Exact outcome | Invocation digest |
|---|---|---|---|
| `unit-contract` | PASS | exit 0; 56 passed in 10.25s | `sha256:3ed4ae83b3de4124b06a061a25a86d03951e6e2af05581134fdee52d26573d76` |
| `legacy-v1-checkpoint-parity` | PASS | exit 0; 8 passed in 14.11s | `sha256:f0fbee638ef03518f51cf6b653eff143032562501537fd1a3430fa218c864937` |
| `unit-blcs` | PASS | exit 0; 37 passed in 13.84s | `sha256:91c5453fbdb390f7c8cbb1295463de9d7082414a9651abb0e1a829f1e0c7b2d9` |
| `unit-plcs` | PASS | exit 0; 25 passed in 10.66s | `sha256:729094dc11098d6875294d4c0a8e329cd32e682e4a6fd08b462bbe911cdef0ae` |
| `unit-slcs` | PASS | exit 0; 35 passed in 10.61s | `sha256:f3730fcb1216ec4c1a68a737cf4ef7739e923c0ccace263857cd5cbc6889a21f` |
| `integration-normalization` | PASS | exit 0; 14 passed in 11.69s | `sha256:393e115292ad3ecd19c97c70d78bd6d9001264cfd88d03b7cfbf7126579bb61c` |
| `preflight-regression` | PASS | exit 0; 127 passed in 11.11s | `sha256:3a4023b90399d612468e51a82fa1078582859b7eb78a66691ebe55fba8eb3686` |
| `knowledge-graph` | PASS | exit 0; 181 nodes, 0 errors, 4 warnings | `sha256:fc39d3529b5dafe9405f680ef63607bb0bbde8d72e77c1b439cfae69879e8446` |
| `precommit-all` | PASS | exit 0; Ruff, mypy, and task-script-reviewer passed | `sha256:a8c9e12dd2478c4cf43f586c9830d54e6b66c561ae1beb54ba56bd1bbfd2b61b` |
| `full-pytest` | PASS | exit 0; 3246 passed, 53 skipped, 19 warnings in 798.53s | `sha256:fedc9bb174546f8c9935561e4733f1372d07168fc0dd38846a0defd3ee52390b` |

## Complete scope inspection

The frozen plan and Issue authorize the versioned court-normalization
contract, its shared/base configuration and metadata gates, BLCS/PLCS/SLCS
and tennis-scene propagation, materialization, documentation, knowledge
evidence, and mirrored unit/integration/e2e tests. The 255 non-workflow paths
are confined to those categories. The fixture repair remains test-only and
does not alter production behavior or fixture bytes. The only changed script
path is the committed fixture generator provenance under `tests/fixtures/`; it
has a module docstring and `precommit-all` passed the repository script review.

Repository-rule and evidence checks are internally consistent:

- `tests.md` is a complete schema-v5 Test Writer artifact for all 22 ACs,
  records PASS, and names the same candidate and all ten test-stage checks.
- `test-checks.json` is schema-v1 complete; every required test-stage result is
  PASS and binds the Tester candidate with the manifest invocation digest.
- The fixture manifest retains both checkpoint hashes and byte counts, and the
  fresh legacy parity check passes direct loading, v1 replay, v2 rejection,
  and the filename policy assertion.
- No forbidden model-suffix path occurs in the complete diff or fixture tree;
  no third-party, generated large output, or unrelated repository path is
  introduced by the repair.
- `artifact-check ... tests` returned `ok`; no semantic mutation, fuzzing,
  parser attack, or open-ended production review was performed at seal.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` — `sha256:f3ae5dd22cc15642bdb54e78f8618a181905a9bbcc8c2c015b664cf1648fe8cf`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-contract` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal legacy-v1-checkpoint-parity` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-blcs` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-plcs` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-slcs` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal integration-normalization` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal preflight-regression` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal knowledge-graph` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal precommit-all` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal full-pytest` — PASS, exit 0.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py artifact-check .codex/tasks/issue-786 tests` — `ok`.

## Final candidate seal verdict

PASS

## RETURN implementation findings

None
