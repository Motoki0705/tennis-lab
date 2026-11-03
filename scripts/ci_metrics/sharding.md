# Complete CPU sharding

[日本語](sharding.ja.md) · [简体中文](sharding.zh-CN.md)

The CI workflow runs lint independently and partitions the requested CPU target
over ten test jobs by default, each with two pytest workers. Manual dispatch
accepts `shards=1/2/4/6/8/10` and `workers=2/4` for same-source scheduling comparisons. Worker counts are explicit rather than host-dependent `auto`.
The manual one-job reference has a 120-minute limit; ordinary shards retain 60 minutes. The fixed `Python` job runs with
`always()` and succeeds only when plan, lint, all test jobs and the coverage audit succeed.
It remains the required merge check. The [lane registry and planning contract](README.md#lane-registry-and-planning)
defines PR selection and full main/manual requests. Only PR runs cancel previous runs.

## Assignment and fixture boundaries

Every worker collects the current source using the normal `spin test` CPU marker
expression, including the court dataset integration selected through the lane registry. The shard hook runs after
ordinary marker selection. It journals the CPU universe before lane selection,
then assigns the requested nodeids, including unregistered new tests.

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

Use indices 0–9 in separate output directories to cover the suite. The unpartitioned
full CPU command in [README](README.md#run-and-recover) remains available. Each
partition reports `full_cpu_baseline=false`; only the complete aggregate can prove
full CPU coverage. Lane selection precedes sharding; timing and `slow` markers do not decide membership.

## Evidence and failure semantics

Each schema-2 `shard-gw*.json` records `universe_nodeids` (U), `requested_nodeids` (R),
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
- successful plan, lint and every matrix job; skipped/cancelled jobs are not successes.

Deleting the same case from every manifest is detected against the independent
pre-selection events. Deferred cases are neither skips nor successes. The aggregate
reports `scoped_cpu_success`; `full_cpu_success` additionally requires D to be empty.
The existing marker/explicit exclusions define the CPU universe, not all tests in
every possible environment. Historical schema-1 artifacts retain `full_nodeids`
and remain readable for profiling; mixing schemas or using them with a lane plan fails.

Preparation failures, invalid partitioning, missing artifacts, failure, cancellation
or a newer HEAD fail `Python`. Intentional CPU exclusions and genuine setup/module
skips retain the existing semantics and remain visible. Matrix fail-fast is disabled
so other shards can still leave evidence. Apt network retries and each apt step are
bounded; failure is not converted into test omission.

Partial reruns reuse the fixed lane plan and the latest result of each shard,
including successful shards from earlier attempts, only within the same run,
source HEAD (`source_sha`) and lane plan (`plan_id`). Attempts may differ within
the bounds above; another run, HEAD or plan is rejected.

The plan artifact is `cpu-lane-plan-RUN`; measurements are
`cpu-test-measurements-RUN-shard-N`. Both upload with `overwrite: true`, replacing
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

## Regenerate measured weights

Download the latest `cpu-test-measurements-RUN-shard-*` artifacts from one run, with exactly one artifact directory per shard and the same HEAD and lane plan, then run:

```bash
.venv/bin/python -m scripts.ci_metrics profile --root /absolute/artifacts \
  --output scripts/ci_metrics/cpu-file-times.json
```

The command rebuilds phase/JUnit reports and audits source/run/plan, attempt bounds, all worker manifests, assignment, and the exact full-inventory multiset. For mixed attempts it uses the maximum shard attempt as the aggregate attempt and records `source.plan_attempt` and `source.shard_attempts`. Missing phases, duplicates, dirty/mixed sources, cancellations and collection failures are rejected. Complete test failures remain in the weights and in `source.counts`/`unsuccessful_cases`; generating weights never turns them into passing CI. `Python` still requires successful jobs and observations. The audit also reports `observations_complete` separately from success.

Any deferred case rejects profile generation. A successful subset cannot overwrite
the full CPU scheduling profile or claim a full CPU baseline.

By default, retain the observed `split_files`. Repeat `--split-file tests/path.py` to explicitly replace that list after reviewing fixtures. The collection guard validates those declarations against the current fixture graph before execution. Merge gate, board messages and parent session retain function-fixture splitting. Coordinate training keeps its module-scoped `shared` fixture within one shard. No test bodies or fixture scopes change.

The refreshed profile uses the complete successful main push [run 41240159671](https://github.com/Motoki0705/tennis-lab/actions/runs/41240159671): all ten shards, 883 files, 14,561 cases (14,470 passed / 91 skipped / 0 failed), and 7,543.124 summed phase seconds. The old profile left 54 current files unmeasured, including the 29-case court dataset integration: its median estimate was 0.077 s versus 406.228 s measured. On this same inventory, old weights estimated 676.570–676.572 s per shard, but the measured costs of those assignments were 439.996–1,101.161 s. Reassigning with refreshed weights estimates 754.311–754.313 s across ten shards, with no unmeasured files or split cases. These are overlapping worker phase costs, not wall-time predictions. Reprofile after material test-cost changes.

System preparation verifies existing ffmpeg/rclone versions, installs only missing binaries, and verifies both commands afterward. Each apt command is bounded by 180 seconds (two network retries, 30-second network timeouts); the Actions step is bounded by eight minutes. Seven subprocess scenarios cover existing/missing binaries, update/install failures, timeout, missing output and version failure.

## Default selection

The default is 10 shards with two workers: the smallest measured two-worker configuration meeting the 15-minute completion target with no failures beyond the known README case. Four workers reproduced an additional process-cleanup assertion failure and remain available for manual comparisons. The complete timing/cost/cache/RSS table and sample limitations are owned by [baseline](baseline.md#runner-scheduling-observations). After the parent’s fixture changes merge, regenerate the profile and remeasure whether 6/8 shards suffice.

Retain the three existing split-file declarations. The heaviest whole file is the court dataset integration at 406.228 s, below the refreshed 754.312 s per-shard budget. Adding it to `split_files` was checked against the current fixture graph and estimates the same 754.313 s maximum load, so it adds no balancing benefit for the measured full inventory. `test_parent_development.py`, `test_execution_basis.py` and `test_parent_completion_resume.py` cost 291.218 / 251.793 / 138.501 s and also fit within that budget. No test body or fixture scope changes.
