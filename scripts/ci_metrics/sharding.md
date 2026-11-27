# Complete CPU sharding

[日本語](sharding.ja.md) · [简体中文](sharding.zh-CN.md)

The CI workflow runs lint independently and partitions the requested CPU target
over ten ordinary test jobs by default, each with two pytest workers, plus one required system-runtime job. Manual dispatch
accepts `shards=1/2/4/6/8/10` and `workers=2/4` for same-source scheduling comparisons. Worker counts are explicit rather than host-dependent `auto`.
The manual one-job reference has a 120-minute limit; ordinary shards retain 60 minutes. The fixed `Python` job runs with
`always()` and succeeds only when plan, lint, all test jobs and the coverage audit succeed.
It remains the required merge check. The [lane registry and planning contract](README.md#lane-registry-and-planning)
defines PR selection and full main/manual requests. Only PR runs cancel previous runs.

## Assignment and fixture boundaries

Every worker collects the current source using the normal `spin test` CPU marker
expression, including the court dataset integration selected through the lane registry. The shard hook runs after
ordinary marker selection. It journals the CPU universe before lane selection,
then records the `system_deps_nodeids` subset in both that event and the worker manifest. Marked cases always enter R; file deferral still applies to unmarked neighbours. Only R minus that subset is assigned to ordinary shards. The dedicated job selects exactly the marked subset, including unregistered new tests.

`cpu-file-times.json` stores measured summed phase seconds. The greedy assignment
places the most expensive group into the currently lightest shard, with stable
lexical/index tie breaking. Whole files are the default fixture boundary.
The profile can explicitly name `split_files` with independent function fixtures;
those cases use individual `test_seconds`, allowing an expensive file to span jobs.
Before adding a split file, inspect its fixtures and keep shared setup together.
Collection rejects a declared split file if a module/class fixture is later added,
instead of silently splitting that shared setup.
No case is removed because its timing is unknown: unmeasured files receive the
measured file median, and new cases in a split file receive its mean case estimate.
Both kinds of estimates are listed in each manifest. Profile values are scheduling
hints, never selection rules; invalid profiles fail explicitly.

Run one local partition from a fixed checkout:

```bash
.venv/bin/python -m scripts.ci_metrics run --workers 2 --output /absolute/new/shard-0/tests \
  --shard-index 0 --shard-count 10 --shard-weights scripts/ci_metrics/cpu-file-times.json
```

Use ordinary indices 0–9 and index 10 (`--shard-index 10 --shard-count 10`) for the system-runtime job in separate output directories to cover the suite. Generally, index equal to count denotes the runtime job; it does not increase the ordinary CPU count. The unpartitioned
full CPU command in [README](README.md#run-and-recover) remains available. Each
partition reports `full_cpu_baseline=false`; only the complete aggregate can prove
full CPU coverage. Lane selection precedes sharding; timing and `slow` markers do not decide membership.

## Evidence and failure semantics

Each schema-3 `shard-gw*.json` records `system_deps_nodeids` and `universe_nodeids` (U), `requested_nodeids` (R),
`deferred` (D: nodeid to suite id), plan/policy/scope identity, group assignment,
selected nodeids (S_i), profile identity and unknown estimates. A separate
`cpu_universe` event records U after CPU marker selection and before lane selection.
`lane-plan.json` preserves the fixed request. Ordinary measurement artifacts
record actual phases, outcomes, JUnit, source and environment. The aggregate
recomputes the per-shard report from those events and requires:

- all expected shard indices exactly once and every expected worker manifest;
- matching clean source revision, run, profile, fixed lane plan (`plan_id`), policy and universe;
- positive integer string attempts with lane plan attempt ≤ each shard attempt ≤ aggregate job attempt;
- registry/source file inventory matching the fixed plan and independent events matching manifest U;
- U = R disjoint-union D, with every deferred node justified by the registry and plan;
- successful complete observations of the requested scope and executed inventory equal to the selected plan;
- R = S_0 disjoint-union ... disjoint-union S_n; no missing, duplicate or empty shard;
- system marker inventories matching independent pre-selection events and every other shard; ordinary shards contain no marked cases, the runtime job contains every marked case exactly once, and all runtime outcomes are passed (zero skip/xfail);
- successful plan, lint, every matrix job and the system-runtime job; skipped/cancelled jobs are not successes.

Deleting the same case from every manifest is detected against the independent
pre-selection events. Deferred cases are neither skips nor successes. The aggregate
reports `scoped_cpu_success`; `full_cpu_success` additionally requires D to be empty.
The existing marker/explicit exclusions define the CPU universe, not all tests in
every possible environment. Historical schema-1/2 observations remain readable for profiling without claiming the new runtime coverage. Schema-1 artifacts retain `full_nodeids`
and remain readable for profiling; mixing schemas or using them with a lane plan fails.

Preparation failures, invalid partitioning, missing artifacts, failure, cancellation
or a newer HEAD fail `Python`. Intentional CPU exclusions and genuine setup/module
skips retain the existing semantics and remain visible. Matrix fail-fast is disabled
so other shards can still leave evidence. Runtime downloads are bounded and verified as specified in the [runtime preparation contract](README.md#pinned-system-runtimes); failures are never converted into omission.

Partial reruns reuse the fixed lane plan and the latest result of each shard,
including successful shards from earlier attempts, only within the same run,
source HEAD (`source_sha`) and lane plan (`plan_id`). Attempts may differ within
the bounds above; another run, HEAD or plan is rejected.

The plan artifact is `cpu-lane-plan-RUN`; measurements are
`cpu-test-measurements-RUN-shard-N` (N=ordinary shard count for the runtime job). Both upload with `overwrite: true`, replacing
only the rerun plan/shard. Measurements from a failed attempt can be overwritten;
its job logs remain available. GitHub deletes earlier artifacts when rerunning all jobs.
Downloads use the run-wide REST API with explicit `run-id` and `github-token`.
Before downloading, the aggregate pages through the REST inventory and requires
exactly one non-expired plan and one artifact for every expected shard. Missing,
duplicate or extra shard artifacts fail with explicit reasons. Downloads use the
checked artifact IDs, keeping shard directories separate to prevent silent file mixing.

`cpu-test-coverage-RUN-ATTEMPT` retains the aggregate job's attempt in its name.
Its `coverage.json` records `attempt` (aggregate), `plan_attempt`, `shard_attempts`
(shard index to attempt), scope, suite outcomes and failure reasons. The artifact
also includes `artifacts.json`: the checked download set with IDs, names,
`created_at`, `size_in_bytes` and inventory errors. A Python-only rerun can audit
unchanged shards from earlier attempts while retaining this provenance.

## Measurement limits

Scheduling estimates use summed worker phase cost, not runner wall time. Each
extra job repeats checkout, environment preparation and collection and consumes
runner time. Review load balance and memory with the actual runner observations;
use [baseline](baseline.md) for evidence and comparison limitations. Refresh the
profile from complete measurements after material fixture or suite changes.
No runner-count recommendation is inferred from the number of test files alone.

System preparation and pin maintenance are owned by the [runtime contract](README.md#pinned-system-runtimes).

## Regenerate measured weights

Run `update-profile` from the checkout with an authenticated `gh` that can read
Actions artifacts. Use `--latest-main` to search successful main pushes from
newest to oldest. Ineligible runs are reported with reasons. Use `--run-id`
and `--attempt` to reproduce a particular update while its artifact set is
still live.

The command pins the run and HEAD, downloads the lane plan and the latest
coverage and shard archives by artifact ID, checks the aggregate's download
manifest, and rebuilds the worker and JUnit audit. It accepts only complete,
successful main push observations with zero deferred cases and an audited
system-runtime shard. A partial rerun can reuse earlier successful shard
observations under the fixed plan and attempt rules above. Invalid or missing
evidence leaves the output profile intact. Replacing the output is atomic.

Reproduce the current update using the source run's scheduling profile as the
comparison baseline.

```bash
profile_work=$(mktemp -d)
git show 8d629a35289f475e514291167435c333876afdd2:scripts/ci_metrics/cpu-file-times.json > "$profile_work/baseline.json"
.venv/bin/python -m scripts.ci_metrics update-profile --run-id 45955929416 --attempt 1 \
  --baseline "$profile_work/baseline.json" --output "$profile_work/cpu-file-times.json" \
  --report "$profile_work/comparison.json" \
  --split-file tests/e2e/agents_ops/test_board_messages.py \
  --split-file tests/e2e/agents_ops/test_parent_session.py \
  --split-file tests/integration/agents_ops/test_storage_tier_crash_boundaries.py \
  --split-file tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py
```

For a new run, replace `--run-id … --attempt …` with `--latest-main`. The default
comparison baseline is the checked-out `cpu-file-times.json`. Omit the split
options to retain the measured run's split declarations. Repeat `--split-file`
to replace that list after reviewing the fixtures. A declaration with zero
measured cases needs an explicit list update. The JSON report includes the
source and pinned command, the file-level differences, and the full group
assignments and loads. When growth reports show a material increase in unmeasured
files or cases, or persistent load imbalance, review a candidate and publish it
through a normal reviewed PR. The command writes only local files.

For historical or failed-run diagnostics, `profile --root /absolute/artifacts
--output /absolute/diagnostic.json` still rebuilds weights from downloaded
shard directories. It can represent complete failing observations in
`source.counts` and `unsuccessful_cases`. For mixed attempts it uses the highest
shard attempt. It rejects deferred cases.

### Offline comparison of the current profile

The previous profile used [run 41240159671](https://github.com/Motoki0705/tennis-lab/actions/runs/41240159671)
with 883 files and 14,561 cases. The new profile uses
[run 45955929416](https://github.com/Motoki0705/tennis-lab/actions/runs/45955929416)
at commit `8d629a35289f475e514291167435c333876afdd2`. The plan, all 11
measurement archives, and successful coverage agree on the run and HEAD. All
attempts are 1. Of 26,189 cases in 1,160 files, 26,070 passed, 119 had legitimate
skips, and none failed or were deferred. The system job passed all 17 cases.
The refresh adds 282 file entries, updates 878, and drops five entries with no
observed cases. The profile's `source` records the exact artifact IDs and
inventory hash.

A uses the previous weights, B refreshes them with the same case split policy
on the observed inventory, and C also splits the storage crash and recovery
boundary files. The old `test_merge_gate.py` declaration had zero collected
cases because the file supplies helpers. There are two effective splits in B
and four in C. A estimates 281 ordinary files and 28 split cases using
unmeasured defaults. B and C have zero unmeasured ordinary files or split cases.

All values below are summed setup, call and teardown seconds across workers.
A observed is the actual phase cost under A on the source run. B and C estimates
reassign the same observations. This comparison leaves the 18.578 phase seconds
of the system job outside the ordinary shard totals.

| Shard | A estimated | A observed | B estimated | C estimated |
|---|---:|---:|---:|---:|
| 0 | 610.579 | 725.360 | 1809.091 | 1360.607 |
| 1 | 610.580 | 1322.624 | 1310.776 | 1360.609 |
| 2 | 610.579 | 611.440 | 1310.776 | 1360.608 |
| 3 | 610.579 | 1304.961 | 1310.776 | 1360.609 |
| 4 | 610.579 | 1185.038 | 1310.777 | 1360.608 |
| 5 | 610.578 | 1397.866 | 1310.777 | 1360.607 |
| 6 | 610.579 | 1650.712 | 1310.776 | 1360.608 |
| 7 | 610.580 | 844.635 | 1310.778 | 1360.609 |
| 8 | 610.578 | 3311.545 | 1310.777 | 1360.608 |
| 9 | 610.578 | 1235.024 | 1310.778 | 1360.609 |
| Min | 610.578 | 611.440 | 1310.776 | 1360.607 |
| Max | 610.580 | 3311.545 | 1809.091 | 1360.609 |
| Mean | 610.579 | 1358.921 | 1360.608 | 1360.608 |
| Max / mean | 1.000 | 2.437 | 1.330 | 1.000 |

The 4,606 recovery cases cost 1809.091 s and move from shard 8 in A to shard 0
in B. The 1,649 crash cases cost 533.836 s and move from shard 6 to shard 1.
C spreads both files across all ten jobs. The next-largest whole files cost
423.166 s (`test_parent_development.py`, A/B/C shards 1/2/0), 373.922 s
(`test_parent_completion_resume.py`, 8/3/1), and 355.775 s
(`test_court_dataset.py`, 0/4/2). With C, replaying measured ordinary case costs
gives a maximum of 1360.609 s, 58.9% below A's observed maximum. The effect on
the Test step and runner consumption requires a matched Actions comparison.

The two boundary files use function-scoped `tmp_path` and `monkeypatch` and
the global per-case seed fixture. The session-scoped pytest temp allocator
gives each case its own root. `make_lab` puts the repository, mirror, synthetic
`/proc`, journals and lock under that root. Fixture trace hooks run in forked
children that are joined or killed, and pytest restores the path prepend
during teardown. Collection with the real shard plugin accepted all 6,255
boundary cases, assigned each once, and reported zero unmeasured split cases.

## Default selection

The default is ten ordinary shards with two workers each, plus the required
system-runtime job. The original runner measurements, cache conditions, and
sample limits are recorded in [baseline](baseline.md#runner-scheduling-observations).
