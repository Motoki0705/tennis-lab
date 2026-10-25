# Merge fixture preparation

[日本語](fixture-cost.ja.md) · [简体中文](fixture-cost.zh-CN.md)

For [#1402](https://github.com/Motoki0705/tennis-lab/issues/1402), construct one
reviewed PR for a one-PR decision. Retain three real git worktrees and PRs for
interactions: independent/dependency ordering, partial decisions, successive base
advances, failed/uncertain merges, fix publication, head races and newly added PRs.
Writable git, GitHub state and review records remain separate for each test.

| Old preparation/cases | Retained contract and implementation |
|---|---|
| Three PRs for each rejection/dry-run/single-target case | `review_setup(pr_count=...)` and `make_gate_env` select the smallest required scenario; human holds, current review, head/diff coverage, fork/draft/conflict and no-agent-launch assertions remain |
| Prepare/request a review, change dependencies/branch/path, then prepare/request again | `before_review` finalizes the scenario before the one public review request; the actual CLI and git snapshots still produce the review evidence |
| Four invalid auto-merge values each create git/agreements/three PRs/a review | Four policy cases retain every value; one real CLI case requires rejection before git/gh/agent invocation or private-state creation |
| Parent prepare-error and recovery cases always build three PRs | Keep three for the missing-target interaction and one for independent input errors/recovery; review publication/identity/recovery checks remain |

Source paths: `tests/e2e/agents_ops/test_merge_gate.py` and
`tests/e2e/agents_ops/test_parent_session.py`. The current main contract no longer
requires a separate permission based on changed paths; this optimization does not
restore that retired rule. Normal review, human holds and auto-merge opt-in remain.

## Validation and sample

Both modified files pass: **259 passed**. The split of the four invalid-value
cases into four policy cases and one CLI case adds one case; this is a preparation
reduction, not a claim of fewer total tests.

Two temporary faults were tested on the original and optimized fixtures: accepting
an invalid `AGENT_AUTO_MERGE` value (4/5 assertion failures before/after) and ignoring
a human hold (3/3 failures). All four runs exited 1, without setup/collection errors.

The representative subset covers a draft rejection with absent CI, unset auto-merge
dry-run, a non-main default branch, a multi-PR decision, and invalid auto-merge input.
Both observations used two workers, native thread limits of one, CUDA hidden and
the same dependencies. Other local validation was active; these are n=1 subset
observations, not GitHub runner or whole-CI performance claims.

| Observation | Before | After |
|---|---:|---:|
| Passed sample cases | 8 | 9 |
| Summed setup seconds | 78.155 | 26.636 |
| Summed call seconds | 16.729 | 9.350 |
| Command monotonic seconds | 58.828 | 23.558 |
| Prepared gate fixtures | 8 | 4 |
| git commits / worktree creations | 40 / 24 | 14 / 6 |
| Parent CLI / fake GitHub calls | 82 / 3391 | 34 / 1255 |
| Review prepare / request calls | 9 / 9 | 4 / 4 |

The three-PR interaction still uses three PRs. Counts describe this selected sample,
not total CI. [Machine evidence](fixture-evidence.json) records versions, outcomes,
phase costs, operation counts and mutations.

## Reproduction

Use the [measurement runner](README.md) with `--workers 2`, the merge gate file, and
this pytest selector (one string following `-k`):

```text
(test_rejections_never_merge_or_launch_agent and absent and draft) or test_unset_auto_merge_is_dry_run_without_any_writes or test_repository_default_branch_is_used_instead_of_main or test_three_prs_use_one_run_and_make_individual_decisions or test_auto_merge_flag_rejects_invalid_values or test_auto_merge_policy_rejects_invalid_values
```

For operation counts, a temporary pytest observer wraps `ParentEnv.cli` and
`test_parent_session.run_git` before collection, counts their calls and first
arguments, then sums the fake `calls.jsonl` lines for the observed fixture roots at
session finish. It calls the original functions and leaves their behavior intact.
Keep the observer identical on both versions. Fixture/phase measurements are
available directly from the standard runner without that observer.
