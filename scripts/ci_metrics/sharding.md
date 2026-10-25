# Complete CPU sharding

[日本語](sharding.ja.md) · [简体中文](sharding.zh-CN.md)

The CI workflow runs lint independently and partitions the complete CPU target
over four test jobs by default, each with two pytest workers. Manual dispatch
accepts `shards=1`, `2` or `4` for same-source scheduling comparisons.
The manual one-job reference has a 120-minute limit; ordinary shards retain 60 minutes. The fixed `Python` job runs with
`always()` and succeeds only when lint, all test jobs and the coverage audit succeed.
CI remains an observation rather than a pre-merge requirement. The existing
triggers are retained for this change; the daily maintenance transition belongs
to [#1349](https://github.com/Motoki0705/tennis-lab/issues/1349).

## Assignment and fixture boundaries

Every worker collects the current source using the normal `spin test` CPU marker
expression and the existing court dataset exclusion. The shard hook runs after
ordinary selection. It assigns every remaining nodeid, including new tests.

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
  --shard-index 0 --shard-count 4 --shard-weights scripts/ci_metrics/cpu-file-times.json
```

Use indices 0–3 in separate output directories to cover the suite. The unpartitioned
full CPU command in [README](README.md#run-and-recover) remains available. Each
partition reports `full_cpu_baseline=false`; only the complete aggregate can prove
full CPU coverage. Sharding is by collected tests, not changed paths or a `slow` filter.

## Evidence and failure semantics

Each `shard-gw*.json` records the full CPU inventory, group assignment, selected
nodeids, profile identity and unknown estimates. Ordinary measurement artifacts
record actual phases, outcomes, JUnit, source and environment. The aggregate
recomputes the per-shard report from those events and requires:

- all expected shard indices exactly once and every expected worker manifest;
- matching clean source revision, workflow run and attempt, profile and full inventory;
- a full CPU request before partitioning, successful complete observations, and
  executed inventory equal to the selected plan;
- the multiset union of executed nodeids equal to the complete CPU inventory;
- successful lint and every matrix job, without treating skipped/cancelled jobs as success.

Preparation failures, invalid partitioning, missing artifacts, failure, cancellation
or a newer HEAD fail `Python`. Intentional CPU exclusions and genuine setup/module
skips retain the existing semantics and remain visible. Matrix fail-fast is disabled
so other shards can still leave evidence. Apt network retries and each apt step are
bounded; failure is not converted into test omission.

Artifacts include run/attempt and shard index. `cpu-test-coverage-*` contains the
aggregate `coverage.json`, including failure reasons. Consumers of daily CI can
join this with each shard's JUnit and manifest using those identities. Rerun **all
jobs** for a self-contained new attempt; successful artifacts from an earlier
attempt are deliberately not mixed into a partial rerun.

## Measurement limits

Scheduling estimates use summed worker phase cost, not runner wall time. Each
extra job repeats checkout, environment preparation and collection and consumes
runner time. Review load balance and memory with the actual runner observations;
use [baseline](baseline.md) for evidence and comparison limitations. Refresh the
profile from complete measurements after material fixture or suite changes.
No runner-count recommendation is inferred from the number of test files alone.
