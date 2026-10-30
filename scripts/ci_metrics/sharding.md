# Complete CPU sharding

[日本語](sharding.ja.md) · [简体中文](sharding.zh-CN.md)

The CI workflow runs lint independently and partitions the complete CPU target
over four test jobs by default, each with two pytest workers. Manual dispatch
accepts `shards=1/2/4/6/8/10` and `workers=2/4` for same-source scheduling comparisons. Worker counts are explicit rather than host-dependent `auto`.
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

## Regenerate measured weights

Download all `cpu-test-measurements-RUN-ATTEMPT-shard-*` artifacts from one attempt into a directory with one artifact directory per shard, then run:

```bash
.venv/bin/python -m scripts.ci_metrics profile --root /absolute/artifacts \
  --output scripts/ci_metrics/cpu-file-times.json
```

The command rebuilds phase/JUnit reports and audits source/run/attempt, all worker manifests, assignment, and the exact full-inventory multiset. Missing phases, duplicates, dirty/mixed sources, cancellations and collection failures are rejected. Complete test failures remain in the weights and in `source.counts`/`unsuccessful_cases`; generating weights never turns them into passing CI. `Python` still requires successful jobs and observations. The audit now also reports `observations_complete` separately from success.

By default, retain the observed `split_files`. Repeat `--split-file tests/path.py` to explicitly replace that list after reviewing fixtures. The collection guard validates those declarations against the current fixture graph before execution. Merge gate, board messages and parent session retain function-fixture splitting. Coordinate training keeps its module-scoped `shared` fixture within one shard; its nine cases cost 579.907 s in the reference observation. No test bodies or fixture scopes change.

The refreshed profile uses [run 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121): 830 files, 12,782 cases (12,685 passed / 96 skipped / 1 failed), 8,527.317 summed phase seconds. It includes the known README.zh-CN failure. Estimated maximum loads are 2,131.830 / 1,421.220 / 1,065.916 s for 4 / 6 / 8 shards; these are overlapping phase costs, not predicted wall times. The previous assignment's measured loads were 2,781 / 1,435 / 1,691 / 2,621 s. Reprofile after the parent's fixture optimizations merge.

System preparation verifies existing ffmpeg/rclone versions, installs only missing binaries, and verifies both commands afterward. Each apt command is bounded by 180 seconds (two network retries, 30-second network timeouts); the Actions step is bounded by eight minutes. Seven subprocess scenarios cover existing/missing binaries, update/install failures, timeout, missing output and version failure.
