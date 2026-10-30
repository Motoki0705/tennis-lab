# Agent operations e2e preparation

[日本語](agents-e2e-cost.ja.md) · [简体中文](agents-e2e-cost.zh-CN.md)

For [#1524](https://github.com/Motoki0705/tennis-lab/issues/1524), keep the complete
message, parent-session and development inventories and reduce their preparation.
Writable git repositories/worktrees, fake GitHub, registry and SQLite state remain
private to each test. Production code, the CPU selection and shard profile are unchanged.

| Old cases / preparation | Retained destination and defects | Distinct interactions retained |
|---|---|---|
| Board messages: confirmed Herdr display owners for every input, SQL invariant, reaction and retry case | Same nodeids/assertions with `message_world`: real SQLite transactions, registry/notification locks and fake notifier dispatch; no display owner is configured | Every prose/container/fence × post/reply × outside-mention combination remains; self/retired/unassigned mention errors, atomicity, frozen recipients, broadcast intent, delivery keys, concurrency and read/ack/reaction state stay observable |
| Broadcast, roster and representative post/reply/read/ack/rename cases | Full `world` with confirmed live Herdr parents | Runtime-filtered `all`, actual display routing alongside delivery, unread counts, rename identity; the container matrix confirms a fourth active parent so accidental `all` expansion has an extra target |
| Downstream board consumers | Existing exported `world` still configures all three live parents | `test_board_doorbell.py`, `test_board_presence.py`, `test_board_web.py` retain their display/socket/HTTP scenarios |
| Uninstructed/invalid review, interrupted/concurrent resend, parent stop and preparation/prelaunch recovery | Same cases with `single_review_env`: one real worktree, public prepare/request and reviewer subprocess boundary | Every invalid authority/input value, zero launch, consumed-run durability, retry of unconsumed preparation, exactly one invocation under concurrency |
| Changed/incomplete review targets and reviewer/parse/partial/publication/head-race failures | Same cases with `pair_review_env`; two PRs | Distinct heads/base, removed/added target, partial coverage, failed PR projection and retry after failure remain separate cases |
| Head changes before instruction, changes during snapshot collection, peer/base evidence and following main | Same tests prepare two PRs | Real git snapshots and head races, reviewed peer versus remaining PR, unreviewed history and resolution paths |
| Large diff initially prepared/requested, then expanded and prepared/requested again | `test_large_diff_is_available_in_full_without_extra_review` expands before its one public request | Full 15,000-line diff, compact prompt and one reviewer call; stale target refusal remains in the two head/snapshot race tests above |
| Cancellation variants: three merged siblings plus cancelled child 13 | Same six variants of `test_completed_parent_distinguishes_confirmed_cancellation_from_success`: merged child 10 plus cancelled child 11 | Real review/gate merge, confirmed close/result and cancellation receipt together; missing/body/state-reason/open-PR/worker defects still prevent completion. Three-PR ordering stays in unchanged merge-gate and sequential-peer tests |
| Revoked adoption: prepare/request before and after provenance/results setup | Same two cases configure provenance/results before one review request | Parent versus child adoption revocation, three independent PR decisions, eligible siblings merging, exactly one review |
| Rebuild an instructed review for every fixture case | `ReviewSeed` captures each 1/2/3-PR variant once per worker through the real public CLI, then uses the merged `DirectorySeed` helper to copy private files and repair real git worktrees | Request metadata and environment are copied; git objects/refs/indexes, GitHub/registry/review files remain separate. A new isolation test corrupts the original request, output and GitHub state and verifies the restored review still succeeds |

All 679 original cases remain, plus one seed-isolation case (680 total). No input class, post/reply cross-product, failure or retry case is removed. Only
the six cancellation nodeids drop their redundant `-independent` fixture suffix;
all six `change` values remain. The shared `review_env`, `make_review_env` and
`review_setup` defaults remain three PRs for downstream users.

## Saved Actions audit

Run [40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121),
attempt 1, has 12,782 JUnit cases: **12,685 passed, 96 skipped, 1 failed**. The only
failure is the Chinese README overview numeric parity check. There are no other
JUnit failures/errors, duplicate phase reports, reruns or worker restarts in the
four saved `pytest.log`/JUnit/phase-event sets. The Chinese translation now uses
`2D`/`3D` consistently with the English technical terms; the parity assertion is unchanged.

| File | Cases | Setup s | Call s | Teardown s |
|---|---:|---:|---:|---:|
| Board messages | 539 | 43.415 | 545.179 | 0.394 |
| Parent session | 97 | 192.810 | 372.032 | 0.072 |
| Parent development | 43 | 61.148 | 288.234 | 0.029 |

These are historical runner observations, not a comparison with local times.

## Local measurements and validation

| Metric | Before | After |
|---|---:|---:|
| Agent e2e cases | 679 | 680 |
| `test_board_messages.py` setup s | 93.346 | 65.762 |
| `test_board_messages.py` call s | 1136.355 | 181.091 |
| `test_board_messages.py` teardown s | 0.320 | 0.234 |
| `test_parent_session.py` setup s | 298.850 | 26.398 |
| `test_parent_session.py` call s | 603.133 | 431.577 |
| `test_parent_session.py` teardown s | 0.047 | 0.042 |
| `test_parent_development.py` setup s | 22.281 | 16.474 |
| `test_parent_development.py` call s | 526.130 | 295.990 |
| `test_parent_development.py` teardown s | 0.020 | 0.019 |
| Three-file phase sum (s) | 2680.482 | 1017.587 |
| Command wall: three files + README (s) | 2695.062 | 1024.909 |
| git commits | 388 | 203 |
| git worktree add / repair | 223 / 9 | 81 / 55 |
| parent CLI | 929 | 549 |
| review / gate CLI | 102 | 103 |
| board CLI | 196 | 196 |
| actual fake GitHub executions | 31986 | 17065 |
| fake Herdr journal rows | 20025 | 2400 |

Observed three-file phase reduction: **62.04%**. The 40% local phase-sum target is met.

Three temporary faults were applied in both serial versions. Code mentions becoming destinations fail 10/10 cases, draft acceptance fails 1/1, and bypassing cancellation verification fails 2/2; every run exits 1 with zero JUnit errors and a positive application counter. The original early-startup cancellation injector timed out before reaching the fault and is explicitly excluded; its records remain in the JSON.

The current normal controls pass: **723 before / 724 after**, including all **44 README cases** in each. The extra after case verifies seed isolation through a real reviewer invocation. An earlier focused seed run passed 17 cases, and the isolation case passed alone. `.venv/bin/python -m spin lint`, mypy on the three changed files and Ruff format checks pass.

## Reproduction and limits

Use the [measurement runner](README.md) with `--workers 1`, the three e2e paths and README validation:

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/observation --workers 1 -- \
  tests/e2e/agents_ops/test_board_messages.py \
  tests/e2e/agents_ops/test_parent_session.py \
  tests/e2e/agents_ops/test_parent_development.py \
  tests/e2e/development/test_readme_layout.py
```

The operation observer counts actual fake GitHub executions in a separate journal outside copied seeds, so copied preparation logs are not counted again. It records pytest-originated git subprocesses (not descendant
git commands inside CLI processes), parent/board/review-gate CLI launches and the
final fake GitHub/Herdr call journals. It delegates every subprocess unchanged.
[Machine evidence](agents-e2e-evidence.json) includes observer and temporary mutation
sources, exact selectors, phase costs, inventories and all unsuccessful attempts.

One worker, native thread limits of one, hidden CUDA and the locked CPU environment
are held fixed. Other sessions share the WSL host. Samples are n=1 observations,
with unknown cache condition; they establish neither Actions wall savings nor a
whole-CPU median/p95. Fixture costs are included in setup; phase sums describe elapsed test cost rather than CPU time. The parent must refresh the profile and verify the full CPU inventory
and CI performance after integration; this task does not change the profile/workflow.

The comparison uses the updated main including #1527 for both versions. Worker/shard-local seeds repeat their first preparation in each CI shard; one-worker cache hit rates do not predict ten-shard CI savings. The earlier two-worker observations have separate provenance and are not mixed with this pair.

Both serial controls include the README translation fix; “before/after” refers to e2e preparation. Only the assigned three files and README validation run.
